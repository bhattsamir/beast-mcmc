/*
 * TorusLandingCountLoggerParser.java
 *
 * @author Samir Bhatt
 */

package dr.evomodelxml.treedatalikelihood.continuous.gpu;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evolution.util.Taxon;
import dr.evomodel.treedatalikelihood.continuous.gpu.TorusLandingCountLogger;
import dr.evomodel.treedatalikelihood.continuous.gpu.TorusLandingGpuLikelihood;
import dr.evomodelxml.treelikelihood.TreeTraitParserUtilities;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import dr.xml.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;

/**
 * Parses a {@link TorusLandingCountLogger}, a child of &lt;mcmc&gt;:
 * <pre>
 * &lt;torusLandingCountLogger id="counts" logEvery="880" startState="176000" outDir="counts_cB"
 *        regions="frac_n1024.bin" regionsMd5="..." devices="4,5,6,7" async="true"&gt;
 *   &lt;torusLandingGpuLikelihood idref="tfl"/&gt;
 * &lt;/torusLandingCountLogger&gt;
 * </pre>
 * The model (tree, tips, landing mixture, Sigma_eps, grid) is read from the referenced likelihood's own XML element
 * with the production parser's reading code (TorusLandingGpuLikelihoodParser), so the counts see exactly the data the
 * chain sees. The region file is the float64 little-endian (3, n, n) array of region_fractions(basins, n) on the jar
 * grid (results_pipeline/labelled_counts.py), checked against regionsMd5.
 *
 * @author Samir Bhatt
 */
public class TorusLandingCountLoggerParser extends AbstractXMLObjectParser {

    public static final String NAME = "torusLandingCountLogger";
    private static final String LOG_EVERY = "logEvery", START = "startState", OUT_DIR = "outDir", REGIONS = "regions",
            REGIONS_MD5 = "regionsMd5", DEVICES = "devices", ASYNC = "async", LOGL_TOL = "logLTol",
            CONSERVED_SD = "conservedSd", CONSERVED_MAX = "conservedMax";

    public String getParserName() { return NAME; }

    public Object parseXMLObject(XMLObject xo) throws XMLParseException {
        final TorusLandingGpuLikelihood lik = (TorusLandingGpuLikelihood) xo.getChild(TorusLandingGpuLikelihood.class);
        XMLObject lxo = null;
        for (int i = 0; i < xo.getChildCount(); i++) {
            final Object raw = xo.getRawChild(i);
            if (raw instanceof Reference) lxo = ((Reference) raw).getReferenceObject();
            else if (raw instanceof XMLObject && ((XMLObject) raw).getNativeObject() == lik) lxo = (XMLObject) raw;
        }
        if (lxo == null || lxo.getNativeObject() != lik) {
            throw new XMLParseException(NAME + " needs the likelihood as a child: <torusLandingGpuLikelihood idref=\"...\"/>");
        }

        // ---- the model, read from the likelihood's element as TorusLandingGpuLikelihoodParser reads it
        final Tree tree = (Tree) lxo.getChild(Tree.class);
        final int nGrid = lxo.getIntegerAttribute("nGrid");
        final List<String> traitNames = new ArrayList<String>();
        final List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        final List<double[]> dList = new ArrayList<double[]>();
        final List<double[][]> sList = new ArrayList<double[][]>();
        final List<Double> wList = new ArrayList<Double>();
        for (int i = 0; i < lxo.getChildCount(); i++) {
            if (!(lxo.getChild(i) instanceof XMLObject)) continue;
            final XMLObject cxo = (XMLObject) lxo.getChild(i);
            if (cxo.getName().equals("residue")) {
                traitNames.add(cxo.getStringAttribute("traitName"));
                precisions.add((MatrixParameter) cxo.getChild(MatrixParameter.class));
            } else if (cxo.getName().equals("jumpComponent")) {
                final Parameter disp = (Parameter) cxo.getElementFirstChild("displacement");
                final MatrixParameter s = (MatrixParameter) cxo.getChild(MatrixParameter.class);
                wList.add(cxo.getAttribute("weight", 1.0));
                dList.add(disp.getParameterValues());
                final double[][] m = new double[2][2];
                for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) m[a][b] = s.getParameterValue(a, b);
                sList.add(m);
            }
        }
        if (precisions.size() != lik.getResidueCount()) throw new XMLParseException("residues differ from the likelihood's");
        for (int r = 0; r < precisions.size(); r++) {
            if (precisions.get(r) != lik.getPrecision(r)) throw new XMLParseException("precision " + r + " is not the likelihood's");
        }
        double wsum = 0.0;
        for (double w : wList) wsum += w;
        final int K = wList.size();
        final double[] w = new double[K];
        final double[][] mu = new double[K][];
        final double[][][] s = new double[K][][];
        for (int k = 0; k < K; k++) { w[k] = wList.get(k) / wsum; mu[k] = dList.get(k); s[k] = sList.get(k); }
        final double[][] sigmaEps = new double[2][2];
        if (lxo.hasAttribute("sigmaEpsilon")) {
            final double v = lxo.getDoubleAttribute("sigmaEpsilon");
            sigmaEps[0][0] = v; sigmaEps[1][1] = v;
        } else {
            final Parameter se = (Parameter) lxo.getElementFirstChild("sigmaEpsilon");
            if (se instanceof MatrixParameter) {
                for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) sigmaEps[a][b] = ((MatrixParameter) se).getParameterValue(a, b);
            } else {
                for (int a = 0; a < 2; a++) sigmaEps[a][a] = se.getParameterValue(se.getDimension() == 1 ? 0 : a);
            }
        }
        final int nodeCount = tree.getNodeCount();
        final double[][][] tipData = new double[precisions.size()][nodeCount][];
        final String[] ids = new String[precisions.size()];
        for (int r = 0; r < precisions.size(); r++) {
            final String tn = traitNames.get(r);
            ids[r] = tn.startsWith("theta_") ? tn.substring("theta_".length()) : tn;
            for (int e = 0; e < tree.getExternalNodeCount(); e++) {
                final NodeRef ext = tree.getExternalNode(e);
                final Taxon taxon = tree.getNodeTaxon(ext);
                tipData[r][ext.getNumber()] = parseVector(taxon == null ? null : taxon.getAttribute(tn), tn, taxon);
            }
        }

        // ---- regions
        final File regions = resolve(xo.getStringAttribute(REGIONS));
        final double[][] frac = readFrac(regions, 3, nGrid);
        final String md5 = md5(regions);
        if (xo.hasAttribute(REGIONS_MD5) && !xo.getStringAttribute(REGIONS_MD5).equalsIgnoreCase(md5)) {
            throw new XMLParseException(regions + ": md5 " + md5 + " differs from regionsMd5 " + xo.getStringAttribute(REGIONS_MD5));
        }

        // ---- settings
        final int[] devices;
        if (xo.hasAttribute(DEVICES)) {
            final String[] tok = xo.getStringAttribute(DEVICES).trim().split("\\s*,\\s*");
            devices = new int[tok.length];
            for (int i = 0; i < tok.length; i++) devices[i] = Integer.parseInt(tok[i]);
        } else if (lxo.hasAttribute("devices")) {
            final String[] tok = lxo.getStringAttribute("devices").trim().split("\\s*,\\s*");
            devices = new int[tok.length];
            for (int i = 0; i < tok.length; i++) devices[i] = Integer.parseInt(tok[i]);
        } else {
            devices = new int[]{lxo.getAttribute("device", 0)};
        }
        final long logEvery = xo.getLongIntegerAttribute(LOG_EVERY);
        final long start = xo.hasAttribute(START) ? xo.getLongIntegerAttribute(START) : 0L;
        final File outDir = resolve(xo.getStringAttribute(OUT_DIR));
        final boolean async = xo.getAttribute(ASYNC, false);
        final double logLTol = xo.getAttribute(LOGL_TOL, 1e-6);
        final double cSd = xo.getAttribute(CONSERVED_SD, 3.0), cMax = xo.getAttribute(CONSERVED_MAX, 8.0);
        final String settings = "nGrid=" + nGrid + " regions=" + regions.getName() + " md5=" + md5 + " devices="
                + java.util.Arrays.toString(devices) + " conservedSd=" + cSd + " conservedMax=" + cMax;
        try {
            return new TorusLandingCountLogger(lik, tree, ids, nGrid, tipData, w, mu, s, sigmaEps, frac, devices, logEvery,
                    start, outDir, async, logLTol, cSd, cMax, settings);
        } catch (IllegalArgumentException e) {
            throw new XMLParseException(NAME + ": " + e.getMessage());
        }
    }

    private static File resolve(String name) {
        final File f = new File(name);
        if (f.isAbsolute()) return f;
        final String prefix = System.getProperty("file.name.prefix");
        return prefix != null ? new File(prefix, name) : f;
    }

    private static double[][] readFrac(File f, int K, int n) throws XMLParseException {
        final long expect = 8L * K * n * n;
        if (f.length() != expect) throw new XMLParseException(f + ": " + f.length() + " bytes, expected " + expect + " for (" + K + ", " + n + ", " + n + ") float64");
        try (FileInputStream in = new FileInputStream(f)) {
            final byte[] raw = new byte[(int) expect];
            int off = 0, r;
            while (off < raw.length && (r = in.read(raw, off, raw.length - off)) > 0) off += r;
            final ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            final double[][] frac = new double[K][n * n];
            for (int a = 0; a < K; a++) for (int i = 0; i < n * n; i++) frac[a][i] = bb.getDouble();
            return frac;
        } catch (IOException e) {
            throw new XMLParseException("cannot read " + f + ": " + e.getMessage());
        }
    }

    private static String md5(File f) throws XMLParseException {
        try (FileInputStream in = new FileInputStream(f)) {
            final MessageDigest md = MessageDigest.getInstance("MD5");
            final byte[] buf = new byte[1 << 20];
            int r;
            while ((r = in.read(buf)) > 0) md.update(buf, 0, r);
            final StringBuilder sb = new StringBuilder();
            for (byte b : md.digest()) sb.append(String.format("%02x", b));
            return sb.toString();
        } catch (Exception e) {
            throw new XMLParseException("cannot checksum " + f + ": " + e.getMessage());
        }
    }

    /** TorusLandingGpuLikelihoodParser.parseVector (production), two angles */
    private static double[] parseVector(Object attr, String traitName, Taxon taxon) throws XMLParseException {
        final String who = taxon == null ? "?" : taxon.getId();
        if (attr == null) throw new XMLParseException("taxon '" + who + "' lacks attribute '" + traitName + "'");
        final String[] toks;
        if (attr instanceof Object[]) {
            final Object[] a = (Object[]) attr;
            toks = new String[a.length];
            for (int i = 0; i < a.length; i++) toks[i] = String.valueOf(a[i]);
        } else if (attr instanceof double[]) {
            final double[] a = (double[]) attr;
            toks = new String[a.length];
            for (int i = 0; i < a.length; i++) toks[i] = Double.toString(a[i]);
        } else {
            toks = attr.toString().trim().split("\\s+");
        }
        if (toks.length != 2) throw new XMLParseException("attribute '" + traitName + "' needs 2 values for " + who);
        final double[] v = new double[2];
        for (int i = 0; i < 2; i++) {
            if (TreeTraitParserUtilities.isMissing(toks[i])) return null;
            v[i] = Double.parseDouble(toks[i]);
            if (Double.isNaN(v[i])) return null;
        }
        return v;
    }

    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{
                AttributeRule.newLongIntegerRule(LOG_EVERY),
                AttributeRule.newLongIntegerRule(START, true),
                AttributeRule.newStringRule(OUT_DIR),
                AttributeRule.newStringRule(REGIONS),
                AttributeRule.newStringRule(REGIONS_MD5, true),
                AttributeRule.newStringRule(DEVICES, true),
                AttributeRule.newBooleanRule(ASYNC, true),
                AttributeRule.newDoubleRule(LOGL_TOL, true),
                AttributeRule.newDoubleRule(CONSERVED_SD, true),
                AttributeRule.newDoubleRule(CONSERVED_MAX, true),
                new ElementRule(TorusLandingGpuLikelihood.class),
        };
    }

    public String getParserDescription() {
        return "Exact expected labelled landing counts of a torusLandingGpuLikelihood, computed on GPUs inside the MCMC.";
    }

    public Class getReturnType() { return TorusLandingCountLogger.class; }
}
