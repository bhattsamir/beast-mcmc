/*
 * TorusLandingGpuLikelihoodParser.java
 *
 * @author Samir Bhatt
 */

package dr.evomodelxml.treedatalikelihood.continuous.gpu;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evolution.util.Taxon;
import dr.evomodel.branchratemodel.BranchRateModel;
import dr.evomodel.treedatalikelihood.continuous.gpu.TorusLandingGpuLikelihood;
import dr.evomodelxml.treelikelihood.TreeTraitParserUtilities;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import dr.xml.*;

import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Parses a {@link TorusLandingGpuLikelihood}: the production landing likelihood
 * (&lt;torusFourierTreeLikelihood jumpLaw="landing"&gt;) for many residues, computed on a GPU.
 * The element takes the production grammar unchanged (residue blocks with a traitName and a precision
 * matrix, jumpRate, jumpComponent mixture, sigmaEpsilon as an attribute or element), plus 'device'
 * (CUDA ordinal, default 0) and 'maxBatch' (residues per GPU pass, default all). The trait, mixture and
 * Sigma_eps reading is the production parser's (TorusFourierTreeLikelihoodParser, commit 06e5709ad).
 * Restrictions: two angles per residue, a fixed tree, no branch-rate model, a fixed Sigma_eps.
 *
 * @author Samir Bhatt
 */
public class TorusLandingGpuLikelihoodParser extends AbstractXMLObjectParser {

    public static final String RESIDUE = "residue";
    public static final String TRAIT_NAME = "traitName";
    public static final String N_GRID = "nGrid";
    public static final String JUMP_LAW = "jumpLaw";
    public static final String JUMP_RATE = "jumpRate";
    public static final String JUMP_COMPONENT = "jumpComponent";
    public static final String ADDITIVE_RATE = "additiveRate";    // combined model, with ADDITIVE_SIZE
    public static final String ADDITIVE_SIZE = "additiveSize";    // additive jump SD (radians)
    public static final String WEIGHT = "weight";
    public static final String DISPLACEMENT = "displacement";
    public static final String SIGMA_EPSILON = "sigmaEpsilon";
    public static final String DEVICE = "device";
    public static final String MAX_BATCH = "maxBatch";
    public static final String DEVICES = "devices";               // comma-separated CUDA ordinals, residues split
    private static final String[] REMOVED = {"nodeNoise", "nodeNoiseSD"};   // node-noise kick, removed 10 Oct 2026
    private static final double MAX_PLAUSIBLE_SIGMA_EPSILON = Math.PI * Math.PI / 4.0;
    private static final String[] XML_RESERVED_ATTRIBUTES = {"id", "idref", "index"};

    public String getParserName() {
        return TorusLandingGpuLikelihood.TORUS_LANDING_GPU_LIKELIHOOD;
    }

    public Object parseXMLObject(XMLObject xo) throws XMLParseException {
        for (String r : REMOVED) {
            if (xo.hasAttribute(r) || xo.hasChildNamed(r)) {
                throw new XMLParseException("'" + r + "': the inherited node-noise kick was removed from this parser "
                        + "(10 Oct 2026; it accumulates down every lineage). This XML asks for a model it no longer has.");
            }
        }
        rejectUnknownAttributes(xo, getSyntaxRules());
        final Tree tree = (Tree) xo.getChild(Tree.class);
        if (xo.getChild(BranchRateModel.class) != null) {
            throw new XMLParseException(getParserName() + " does not support a branch-rate model");
        }
        final int nGrid = xo.getIntegerAttribute(N_GRID);
        if (!xo.getAttribute(JUMP_LAW, "landing").equals("landing")) {
            throw new XMLParseException(getParserName() + " implements jumpLaw=\"landing\" only");
        }
        final int[] devices;
        if (xo.hasAttribute(DEVICES)) {
            if (xo.hasAttribute(DEVICE)) throw new XMLParseException("give either " + DEVICE + " or " + DEVICES + ", not both");
            final String[] tok = xo.getStringAttribute(DEVICES).trim().split("\\s*,\\s*");
            devices = new int[tok.length];
            for (int i = 0; i < tok.length; i++) devices[i] = Integer.parseInt(tok[i]);
        } else {
            devices = new int[]{xo.getAttribute(DEVICE, 0)};
        }

        final List<String> traitNames = new ArrayList<String>();
        final List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        for (int i = 0; i < xo.getChildCount(); i++) {
            if (xo.getChild(i) instanceof XMLObject) {
                final XMLObject cxo = (XMLObject) xo.getChild(i);
                if (cxo.getName().equals(RESIDUE)) {
                    final MatrixParameter precision = (MatrixParameter) cxo.getChild(MatrixParameter.class);
                    if (precision == null) {
                        throw new XMLParseException("a <" + RESIDUE + "> requires a square precision matrixParameter");
                    }
                    if (precision.getRowDimension() != 2 || precision.getColumnDimension() != 2) {
                        throw new XMLParseException(getParserName() + " supports two angles per residue (2x2 precision)");
                    }
                    traitNames.add(cxo.getStringAttribute(TRAIT_NAME));
                    precisions.add(precision);
                }
            }
        }
        if (precisions.isEmpty()) throw new XMLParseException(getParserName() + " requires at least one <" + RESIDUE + ">");
        final int maxBatch = xo.getAttribute(MAX_BATCH, precisions.size());
        final int dim = 2;

        final Parameter jumpRate = (Parameter) xo.getElementFirstChild(JUMP_RATE);
        if (jumpRate.getDimension() != 1 && jumpRate.getDimension() != precisions.size()) {
            throw new XMLParseException("<" + JUMP_RATE + "> must have dimension 1 (shared) or "
                    + precisions.size() + " (one lambda per residue), got " + jumpRate.getDimension());
        }

        // combined model (optional): additive jumps alongside the landings
        final boolean hasRate = xo.hasChildNamed(ADDITIVE_RATE), hasSize = xo.hasChildNamed(ADDITIVE_SIZE);
        if (hasRate != hasSize) {
            throw new XMLParseException("<" + ADDITIVE_RATE + "> and <" + ADDITIVE_SIZE + "> go together");
        }
        final Parameter additiveRate = hasRate ? (Parameter) xo.getElementFirstChild(ADDITIVE_RATE) : null;
        final Parameter additiveSize = hasSize ? (Parameter) xo.getElementFirstChild(ADDITIVE_SIZE) : null;
        if (additiveRate != null) {
            for (Parameter q : new Parameter[]{additiveRate, additiveSize}) {
                if (q.getDimension() != 1 && q.getDimension() != precisions.size()) {
                    throw new XMLParseException("<" + ADDITIVE_RATE + "> and <" + ADDITIVE_SIZE + "> must have dimension 1 "
                            + "or " + precisions.size() + ", got " + q.getDimension());
                }
            }
            for (int i = 0; i < additiveSize.getDimension(); i++) {
                final double v = additiveSize.getParameterValue(i);
                if (!(v > 0.0) || v > 2 * Math.PI) {
                    throw new XMLParseException("<" + ADDITIVE_SIZE + "> is a standard deviation in RADIANS in (0, 2 pi], got "
                            + v + " (degrees? " + Math.toRadians(v) + " rad)");
                }
            }
        }

        // landing mixture: production parser, jumpComponent branch, weights normalised by their sum
        final List<double[]> dList = new ArrayList<double[]>();
        final List<double[][]> sList = new ArrayList<double[][]>();
        final List<Double> wList = new ArrayList<Double>();
        for (int i = 0; i < xo.getChildCount(); i++) {
            if (xo.getChild(i) instanceof XMLObject) {
                final XMLObject cxo = (XMLObject) xo.getChild(i);
                if (cxo.getName().equals(JUMP_COMPONENT)) {
                    final Parameter disp = (Parameter) cxo.getElementFirstChild(DISPLACEMENT);
                    final MatrixParameter s = (MatrixParameter) cxo.getChild(MatrixParameter.class);
                    if (disp == null || s == null) {
                        throw new XMLParseException("<" + JUMP_COMPONENT + "> needs a <" + DISPLACEMENT
                                + "> vector and a covariance matrixParameter");
                    }
                    if (disp.getDimension() != dim) throw new XMLParseException("<" + DISPLACEMENT + "> must have " + dim + " values");
                    checkSquare(s, dim, JUMP_COMPONENT);
                    final double w = cxo.getAttribute(WEIGHT, 1.0);
                    if (w < 0.0) throw new XMLParseException("<" + JUMP_COMPONENT + "> " + WEIGHT + " must be >= 0");
                    wList.add(w);
                    dList.add(disp.getParameterValues());
                    sList.add(matrixOf(s, dim));
                }
            }
        }
        if (wList.isEmpty()) throw new XMLParseException("jumpLaw=landing requires a fixed basin mixture (jumpComponent)");
        final int K = wList.size();
        final double[] jumpW = new double[K];
        final double[][] jumpD = new double[K][];
        final double[][][] jumpS = new double[K][][];
        double wsum = 0.0;
        for (int k = 0; k < K; k++) wsum += wList.get(k);
        if (wsum <= 0.0) throw new XMLParseException("jump mixture weights must sum to a positive value");
        for (int k = 0; k < K; k++) {
            jumpW[k] = wList.get(k) / wsum;
            jumpD[k] = dList.get(k);
            jumpS[k] = sList.get(k);
        }

        // tip data: production parser, trait branch
        final int nExt = tree.getExternalNodeCount();
        final double[][][] tipData = new double[precisions.size()][][];
        for (int r = 0; r < traitNames.size(); r++) {
            final String traitName = traitNames.get(r);
            final double[][] data = new double[nExt][];
            for (int e = 0; e < nExt; e++) {
                final NodeRef ext = tree.getExternalNode(e);
                final Taxon taxon = tree.getNodeTaxon(ext);
                final Object attr = (taxon == null) ? null : taxon.getAttribute(traitName);
                data[ext.getNumber()] = parseVector(attr, dim, traitName, taxon);
            }
            tipData[r] = data;
        }

        // Sigma_eps: production parser and TorusFourierTreeLikelihood.sigmaEpsilonMatrix
        final double[][] sigmaEps = new double[dim][dim];
        Parameter sigmaEpsilon = null;
        if (xo.hasAttribute(SIGMA_EPSILON)) {
            if (xo.hasChildNamed(SIGMA_EPSILON)) {
                throw new XMLParseException("give " + SIGMA_EPSILON + " either as an attribute or as an element, not both");
            }
            final double v = xo.getDoubleAttribute(SIGMA_EPSILON);
            checkVariance(v, SIGMA_EPSILON + " attribute");
            sigmaEps[0][0] = v;
            sigmaEps[1][1] = v;
        } else if (xo.hasChildNamed(SIGMA_EPSILON)) {
            final Object child = xo.getElementFirstChild(SIGMA_EPSILON);
            if (!(child instanceof Parameter)) throw new XMLParseException("<" + SIGMA_EPSILON + "> requires a parameter or matrixParameter");
            sigmaEpsilon = (Parameter) child;
            if (sigmaEpsilon instanceof MatrixParameter) {
                final MatrixParameter mp = (MatrixParameter) sigmaEpsilon;
                checkSquare(mp, dim, SIGMA_EPSILON);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) {
                        final double v = mp.getParameterValue(i, j);
                        checkVariance(i == j ? v : Math.abs(v), "<" + SIGMA_EPSILON + "> entry (" + i + "," + j + ")");
                        sigmaEps[i][j] = v;
                    }
                }
            } else {
                final int sd = sigmaEpsilon.getDimension();
                if (sd != 1 && sd != dim) throw new XMLParseException("<" + SIGMA_EPSILON + "> must have dimension 1 or " + dim);
                for (int i = 0; i < dim; i++) {
                    final double v = sigmaEpsilon.getParameterValue(sd == 1 ? 0 : i);
                    checkVariance(v, "<" + SIGMA_EPSILON + "> entry " + i);
                    sigmaEps[i][i] = v;
                }
            }
        } else {
            throw new XMLParseException("jumpLaw=landing requires positive-definite " + SIGMA_EPSILON);
        }
        if (sigmaEps[0][0] == 0.0 && sigmaEps[1][1] == 0.0 && sigmaEps[0][1] == 0.0 && sigmaEps[1][0] == 0.0) {
            throw new XMLParseException("jumpLaw=landing requires positive-definite " + SIGMA_EPSILON);
        }

        String id = xo.getId();
        if (id == null) id = getParserName();
        try {
            return new TorusLandingGpuLikelihood(id, tree, precisions, tipData, jumpRate, additiveRate, additiveSize,
                    jumpW, jumpD, jumpS, sigmaEps, sigmaEpsilon, nGrid, devices, maxBatch);
        } catch (IllegalArgumentException e) {
            throw new XMLParseException("Invalid landing likelihood: " + e.getMessage());
        }
    }

    // ---- copied from the production parser (TorusFourierTreeLikelihoodParser, commit 06e5709ad) --------

    private static void checkVariance(double v, String where) throws XMLParseException {
        if (!Double.isFinite(v) || !(v >= 0.0)) {
            throw new XMLParseException(where + " must be a finite variance >= 0, got " + v);
        }
        if (v > MAX_PLAUSIBLE_SIGMA_EPSILON) {
            throw new XMLParseException(where + " = " + v + " is a VARIANCE in radians^2, not an "
                    + "angle in degrees or a standard deviation. " + v + " rad^2 is a standard deviation of "
                    + String.format("%.0f", Math.toDegrees(Math.sqrt(v))) + " degrees, which is larger than "
                    + "this check allows (" + MAX_PLAUSIBLE_SIGMA_EPSILON + " rad^2). For s degrees write "
                    + "(s*pi/180)^2 -- e.g. 5 degrees is 0.00761544, 3.5 degrees is 0.00373156339239.");
        }
    }

    private static void rejectUnknownAttributes(XMLObject xo, XMLSyntaxRule[] rules) throws XMLParseException {
        final Set<String> allowed = new HashSet<String>();
        for (String reserved : XML_RESERVED_ATTRIBUTES) allowed.add(reserved);
        for (XMLSyntaxRule rule : rules) {
            if (rule instanceof AttributeRule) allowed.add(((AttributeRule) rule).getName());
        }
        final NamedNodeMap attributes = xo.getAttributes();
        for (int i = 0; i < attributes.getLength(); i++) {
            final Node attribute = attributes.item(i);
            final String name = attribute.getNodeName();
            if (name.startsWith("xmlns") || allowed.contains(name)) continue;
            final StringBuilder known = new StringBuilder();
            for (String a : new java.util.TreeSet<String>(allowed)) {
                known.append(known.length() == 0 ? "" : ", ").append(a);
            }
            throw new XMLParseException("<" + xo.getName() + "> has an attribute '" + name
                    + "' that this parser does not read, so it would be SILENTLY IGNORED. "
                    + "Recognised attributes here: " + known + ". "
                    + "(If you meant the tip observation error, the spelling is '" + SIGMA_EPSILON + "'.)");
        }
        for (XMLSyntaxRule rule : rules) {
            if (!(rule instanceof ElementRule)) continue;
            final ElementRule er = (ElementRule) rule;
            if (er.getName() == null || er.getRules() == null) continue;
            for (int i = 0; i < xo.getChildCount(); i++) {
                final Object child = xo.getChild(i);
                if (child instanceof XMLObject && ((XMLObject) child).getName().equals(er.getName())) {
                    rejectUnknownAttributes((XMLObject) child, er.getRules());
                }
            }
        }
    }

    private static void checkSquare(MatrixParameter s, int dim, String who) throws XMLParseException {
        if (s.getRowDimension() != dim || s.getColumnDimension() != dim) {
            throw new XMLParseException("<" + who + "> covariance must be " + dim + "x" + dim);
        }
    }

    private static double[][] matrixOf(MatrixParameter s, int dim) {
        final double[][] m = new double[dim][dim];
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++) m[i][j] = s.getParameterValue(i, j);
        return m;
    }

    private double[] parseVector(Object attr, int dim, String traitName, Taxon taxon) throws XMLParseException {
        final String who = (taxon == null) ? "?" : taxon.getId();
        if (attr == null) {
            throw new XMLParseException("taxon '" + who + "' lacks attribute '" + traitName
                    + "' (use NA or ? to mark the residue missing)");
        }
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
        if (toks.length != dim) {
            throw new XMLParseException("attribute '" + traitName + "' needs " + dim + " values for " + who);
        }
        final double[] v = new double[dim];
        for (int i = 0; i < dim; i++) {
            if (TreeTraitParserUtilities.isMissing(toks[i])) return null;
            v[i] = Double.parseDouble(toks[i]);
            if (Double.isNaN(v[i])) return null;
        }
        return v;
    }

    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{
                AttributeRule.newIntegerRule(N_GRID),
                AttributeRule.newStringRule(JUMP_LAW, true),
                AttributeRule.newDoubleRule(SIGMA_EPSILON, true),
                AttributeRule.newIntegerRule(DEVICE, true),
                AttributeRule.newIntegerRule(MAX_BATCH, true),
                AttributeRule.newStringRule(DEVICES, true),
                new ElementRule(Tree.class),
                new ElementRule(BranchRateModel.class, true),
                new ElementRule(RESIDUE, new XMLSyntaxRule[]{
                        AttributeRule.newStringRule(TRAIT_NAME),
                        new ElementRule(MatrixParameter.class)
                }, 1, Integer.MAX_VALUE),
                new ElementRule(JUMP_RATE, new XMLSyntaxRule[]{new ElementRule(Parameter.class)}),
                new ElementRule(ADDITIVE_RATE, new XMLSyntaxRule[]{new ElementRule(Parameter.class)}, true),
                new ElementRule(ADDITIVE_SIZE, new XMLSyntaxRule[]{new ElementRule(Parameter.class)}, true),
                new ElementRule(SIGMA_EPSILON, new XMLSyntaxRule[]{new ElementRule(Parameter.class)}, true),
                new ElementRule(JUMP_COMPONENT, new XMLSyntaxRule[]{
                        AttributeRule.newDoubleRule(WEIGHT, true),
                        new ElementRule(DISPLACEMENT, new XMLSyntaxRule[]{new ElementRule(Parameter.class)}),
                        new ElementRule(MatrixParameter.class)
                }, 1, Integer.MAX_VALUE),
        };
    }

    public String getParserDescription() {
        return "The production torus landing likelihood (torusFourierTreeLikelihood, jumpLaw=landing) for many "
                + "residues on one fixed tree, evaluated in batches on a CUDA GPU.";
    }

    public Class getReturnType() {
        return TorusLandingGpuLikelihood.class;
    }
}
