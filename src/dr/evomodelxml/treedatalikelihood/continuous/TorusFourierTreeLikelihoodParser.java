/*
 * TorusFourierTreeLikelihoodParser.java
 *
 * @author Samir Bhatt
 */

package dr.evomodelxml.treedatalikelihood.continuous;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evolution.util.Taxon;
import dr.evomodel.branchratemodel.BranchRateModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.evomodelxml.treelikelihood.TreeTraitParserUtilities;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import dr.xml.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Parses a {@link TorusFourierTreeLikelihood}: a spectral (FFT) Felsenstein pruning
 * likelihood for independent d-dimensional wrapped-Brownian diffusions of backbone
 * dihedrals. Each &lt;residue&gt; block names a d-dimensional tip trait (e.g. phi,
 * psi) read from the taxon attributes and supplies a dim x dim precision matrix; the
 * common dimension d is inferred from the precision matrices.
 *
 * @author Samir Bhatt
 */
public class TorusFourierTreeLikelihoodParser extends AbstractXMLObjectParser {

    public static final String TORUS_FOURIER_LIKELIHOOD = TorusFourierTreeLikelihood.TORUS_FOURIER_LIKELIHOOD;
    public static final String RESIDUE = "residue";
    public static final String TRAIT_NAME = "traitName";
    public static final String N_GRID = "nGrid";
    public static final String JUMP_RATE = "jumpRate";          // compound-Poisson rate lambda (shared)
    public static final String SIGMA_JUMP = "sigmaJump";        // single zero-mean jump covariance (shorthand)
    public static final String JUMP_COMPONENT = "jumpComponent";// one basin-directed mixture component
    public static final String WEIGHT = "weight";
    public static final String DISPLACEMENT = "displacement";   // d_k, a dim-vector parameter

    public String getParserName() {
        return TORUS_FOURIER_LIKELIHOOD;
    }

    public Object parseXMLObject(XMLObject xo) throws XMLParseException {

        final Tree tree = (Tree) xo.getChild(Tree.class);
        final BranchRateModel branchRateModel = (BranchRateModel) xo.getChild(BranchRateModel.class);
        final int nGrid = xo.getAttribute(N_GRID, 64);

        // collect residue specifications and infer the common dimension
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
                    traitNames.add(cxo.getStringAttribute(TRAIT_NAME));
                    precisions.add(precision);
                }
            }
        }
        if (precisions.isEmpty()) {
            throw new XMLParseException(TORUS_FOURIER_LIKELIHOOD + " requires at least one <" + RESIDUE + ">");
        }

        final int dim = precisions.get(0).getRowDimension();
        for (MatrixParameter p : precisions) {
            if (p.getRowDimension() != dim || p.getColumnDimension() != dim) {
                throw new XMLParseException("all <" + RESIDUE + "> precision matrices must be square of the same dimension ("
                        + dim + ")");
            }
        }
        final TorusFourierKernel kernel = new TorusFourierKernel(dim, nGrid);

        // Optional compound-Poisson jump, shared across residues: a sampled rate lambda
        // (<jumpRate>) plus a FIXED mixture of wrapped-normal components {weight, displacement
        // d_k, covariance S_k}. A single <sigmaJump> is the zero-displacement 1-component
        // shorthand. Absent -> pure wrapped BM.
        //
        // IMPORTANT: the displacement vectors and covariance matrices are read ONCE here and held
        // fixed (their numeric values are copied into the likelihood); they are NOT live parameters.
        // Only lambda is sampled. Do not attach MCMC operators to the displacement/covariance
        // parameters expecting the likelihood to update -- it will not. (Basin geometry comes from
        // the Ramachandran fit and is intended to be fixed.)
        Parameter jumpRate = null;
        double[] jumpW = null;
        double[][] jumpD = null;
        double[][][] jumpS = null;
        if (xo.hasChildNamed(JUMP_RATE)) {
            jumpRate = (Parameter) xo.getElementFirstChild(JUMP_RATE);
            final int jd = jumpRate.getDimension();
            if (jd != 1 && jd != precisions.size()) {
                throw new XMLParseException("<" + JUMP_RATE + "> must have dimension 1 (shared) or "
                        + precisions.size() + " (one lambda per residue), got " + jd);
            }
        }

        final List<double[]> dList = new ArrayList<double[]>();
        final List<double[][]> sList = new ArrayList<double[][]>();
        final List<Double> wList = new ArrayList<Double>();
        if (xo.hasChildNamed(SIGMA_JUMP)) {                 // single zero-mean Gaussian shorthand
            final MatrixParameter s = (MatrixParameter) xo.getElementFirstChild(SIGMA_JUMP);
            checkSquare(s, dim, SIGMA_JUMP);
            wList.add(1.0);
            dList.add(new double[dim]);                     // d = 0
            sList.add(matrixOf(s, dim));
        }
        for (int i = 0; i < xo.getChildCount(); i++) {      // basin-directed mixture components
            if (xo.getChild(i) instanceof XMLObject) {
                final XMLObject cxo = (XMLObject) xo.getChild(i);
                if (cxo.getName().equals(JUMP_COMPONENT)) {
                    final Parameter disp = (Parameter) cxo.getElementFirstChild(DISPLACEMENT);
                    final MatrixParameter s = (MatrixParameter) cxo.getChild(MatrixParameter.class);
                    if (disp == null || s == null) {
                        throw new XMLParseException("<" + JUMP_COMPONENT + "> needs a <" + DISPLACEMENT
                                + "> vector and a covariance matrixParameter");
                    }
                    if (disp.getDimension() != dim) {
                        throw new XMLParseException("<" + DISPLACEMENT + "> must have " + dim + " values");
                    }
                    checkSquare(s, dim, JUMP_COMPONENT);
                    final double w = cxo.getAttribute(WEIGHT, 1.0);
                    if (w < 0.0) {
                        throw new XMLParseException("<" + JUMP_COMPONENT + "> " + WEIGHT
                                + " must be >= 0 (mixture weights are a probability distribution)");
                    }
                    wList.add(w);
                    dList.add(disp.getParameterValues());
                    sList.add(matrixOf(s, dim));
                }
            }
        }
        if (!wList.isEmpty()) {
            final int K = wList.size();
            jumpW = new double[K]; jumpD = new double[K][]; jumpS = new double[K][][];
            double wsum = 0.0;
            for (int k = 0; k < K; k++) wsum += wList.get(k);
            if (wsum <= 0.0) throw new XMLParseException("jump mixture weights must sum to a positive value");
            for (int k = 0; k < K; k++) {                   // normalise so phi_nu(0)=1 (valid density)
                jumpW[k] = wList.get(k) / wsum;
                jumpD[k] = dList.get(k);
                jumpS[k] = sList.get(k);
            }
        }

        // read the per-taxon d-vectors for each residue
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

        String id = xo.getId();
        if (id == null) id = TORUS_FOURIER_LIKELIHOOD;
        TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood(id, tree, branchRateModel,
                precisions, tipData, kernel, jumpRate, jumpW, jumpD, jumpS);
        like.setTraitNames(traitNames.toArray(new String[0]));   // name ancestral traits after the tips
        return like;
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

    /** Parse angles in radians, retaining observed coordinates. Missing coordinates
     *  are NaN and are integrated out by the tip kernel; an entirely missing vector
     *  is null and contributes a constant-one message. */
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
        boolean observed = false;
        for (int i = 0; i < dim; i++) {
            if (TreeTraitParserUtilities.isMissing(toks[i])) {
                v[i] = Double.NaN;
            } else {
                try {
                    v[i] = Double.parseDouble(toks[i]);
                } catch (NumberFormatException invalid) {
                    throw new XMLParseException("Invalid angle '" + toks[i] + "' in trait '"
                            + traitName + "' for " + who);
                }
                if (Double.isInfinite(v[i])) {
                    throw new XMLParseException("Infinite angle in trait '" + traitName + "' for " + who);
                }
                observed |= !Double.isNaN(v[i]);
            }
        }
        return observed ? v : null;
    }

    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{
                AttributeRule.newIntegerRule(N_GRID, true),
                new ElementRule(Tree.class),
                new ElementRule(BranchRateModel.class, true),
                new ElementRule(RESIDUE, new XMLSyntaxRule[]{
                        AttributeRule.newStringRule(TRAIT_NAME),
                        new ElementRule(MatrixParameter.class)
                }, 1, Integer.MAX_VALUE),
                new ElementRule(JUMP_RATE, new XMLSyntaxRule[]{
                        new ElementRule(Parameter.class)
                }, true),
                new ElementRule(SIGMA_JUMP, new XMLSyntaxRule[]{
                        new ElementRule(MatrixParameter.class)
                }, true),
                new ElementRule(JUMP_COMPONENT, new XMLSyntaxRule[]{
                        AttributeRule.newDoubleRule(WEIGHT, true),
                        new ElementRule(DISPLACEMENT, new XMLSyntaxRule[]{new ElementRule(Parameter.class)}),
                        new ElementRule(MatrixParameter.class)
                }, 0, Integer.MAX_VALUE),
        };
    }

    public String getParserDescription() {
        return "Spectral (FFT) Felsenstein pruning likelihood for independent d-dimensional wrapped-Brownian "
                + "diffusions of protein backbone dihedrals on a flat torus.";
    }

    public Class getReturnType() {
        return TorusFourierTreeLikelihood.class;
    }
}
