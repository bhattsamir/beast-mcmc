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

import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Parses a {@link TorusFourierTreeLikelihood}: a spectral (FFT) Felsenstein pruning
 * likelihood for independent d-dimensional wrapped-Brownian diffusions of backbone
 * dihedrals. Each &lt;residue&gt; block names a d-dimensional tip trait (e.g. phi,
 * psi) read from the taxon attributes and supplies a dim x dim precision matrix; the
 * common dimension d is inferred from the precision matrices. An optional
 * &lt;sigmaEpsilon&gt; supplies a wrapped-Gaussian tip observation-error covariance
 * (variances in radians^2, default zero = Dirac tips).
 * With jumpLaw="landing", jumps replace the angles with a draw from one fixed
 * mixture shared by this likelihood's residues; positive sigmaEpsilon is required.
 * The existing jumpComponent displacement field then means an ABSOLUTE basin mean,
 * whereas jumpLaw="shift" (the default) interprets it as an added displacement.
 *
 * @author Samir Bhatt
 */
public class TorusFourierTreeLikelihoodParser extends AbstractXMLObjectParser {

    public static final String TORUS_FOURIER_LIKELIHOOD = TorusFourierTreeLikelihood.TORUS_FOURIER_LIKELIHOOD;
    public static final String RESIDUE = "residue";
    public static final String TRAIT_NAME = "traitName";
    public static final String N_GRID = "nGrid";
    public static final String JUMP_LAW = "jumpLaw";
    public static final String JUMP_RATE = "jumpRate";          // compound-Poisson rate lambda (shared)
    public static final String SIGMA_JUMP = "sigmaJump";        // single zero-mean jump covariance (shorthand)
    public static final String JUMP_COMPONENT = "jumpComponent";// one basin-directed mixture component
    public static final String WEIGHT = "weight";
    public static final String DISPLACEMENT = "displacement";   // d_k, a dim-vector parameter
    public static final String SIGMA_EPSILON = "sigmaEpsilon";  // tip observation-error covariance
    /** A variance this large (sd ~ 90 degrees) is a unit mistake, not a measurement. */
    private static final double MAX_PLAUSIBLE_SIGMA_EPSILON = Math.PI * Math.PI / 4.0;
    /** BEAST's XMLParser resolves these itself (see XMLParser.ID/IDREF and its "index" handling);
     *  every other attribute must be one this parser actually reads. */
    private static final String[] XML_RESERVED_ATTRIBUTES = {"id", "idref", "index"};

    public String getParserName() {
        return TORUS_FOURIER_LIKELIHOOD;
    }

    public Object parseXMLObject(XMLObject xo) throws XMLParseException {

        // A MISSPELLED attribute used to be silently ignored, which is how a whole analysis can
        // come back with Sigma_eps quietly absent, plausible numbers and exit 0. Nothing else in
        // BEAST checks this, so check it here: every attribute on this element and on the child
        // elements this parser owns must be one the syntax rules declare.
        rejectUnknownAttributes(xo, getSyntaxRules());

        final Tree tree = (Tree) xo.getChild(Tree.class);
        final BranchRateModel branchRateModel = (BranchRateModel) xo.getChild(BranchRateModel.class);
        final int nGrid = xo.getAttribute(N_GRID, 64);
        final String lawName = xo.getAttribute(JUMP_LAW, "shift");
        final TorusFourierTreeLikelihood.JumpLaw jumpLaw;
        if (lawName.equals("shift")) {
            jumpLaw = TorusFourierTreeLikelihood.JumpLaw.SHIFT;
        } else if (lawName.equals("landing")) {
            jumpLaw = TorusFourierTreeLikelihood.JumpLaw.LANDING;
        } else {
            throw new XMLParseException("jumpLaw must be 'shift' or 'landing', got '" + lawName + "'");
        }

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

        // Optional tip observation error, shared across residues. A tip becomes a wrapped
        // Gaussian of covariance Sigma_eps instead of a Dirac delta. Three shapes are accepted:
        //   sigmaEpsilon="0.00761544"  (attribute)                            isotropic, fixed
        //   <sigmaEpsilon><parameter value="0.00762"/></sigmaEpsilon>          isotropic, s*I
        //   <sigmaEpsilon><parameter value="0.00762 0.00762"/></sigmaEpsilon>  per-axis variances
        //   <sigmaEpsilon><matrixParameter> ... </matrixParameter></sigmaEpsilon>  full dim x dim
        // Values are VARIANCES in radians^2 -- (5 degrees)^2 is 0.0076154... . It is a live
        // Parameter, so an operator may sample it; we fix it in the HA analysis but the sweep
        // needs to ask whether it is identifiable at all.
        //
        // DEFAULT IS ABSENT == ZERO. Every existing XML keeps its exact meaning: the likelihood
        // maps a zero Sigma_eps back onto the original Dirac-tip code path.
        Parameter sigmaEpsilon = null;
        if (xo.hasAttribute(SIGMA_EPSILON)) {
            if (xo.hasChildNamed(SIGMA_EPSILON)) {
                throw new XMLParseException("give " + SIGMA_EPSILON + " either as an attribute "
                        + "(fixed isotropic variance) or as an element (a sampled Parameter), not both");
            }
            final double v = xo.getDoubleAttribute(SIGMA_EPSILON);
            checkVariance(v, SIGMA_EPSILON + " attribute");
            sigmaEpsilon = new Parameter.Default(TORUS_FOURIER_LIKELIHOOD + "." + SIGMA_EPSILON, v);
        } else if (xo.hasChildNamed(SIGMA_EPSILON)) {
            final Object child = xo.getElementFirstChild(SIGMA_EPSILON);
            if (!(child instanceof Parameter)) {
                throw new XMLParseException("<" + SIGMA_EPSILON + "> requires a parameter or matrixParameter");
            }
            sigmaEpsilon = (Parameter) child;
            // The degrees-for-variance guard belongs to the QUANTITY, not to one spelling of it.
            // <sigmaEpsilon><parameter value="5.0"/></sigmaEpsilon> used to sail straight past the
            // attribute-only check and silently drive the likelihood to the uniform limit.
            if (sigmaEpsilon instanceof MatrixParameter) {
                final MatrixParameter mp = (MatrixParameter) sigmaEpsilon;
                checkSquare(mp, dim, SIGMA_EPSILON);
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) {
                        final double v = mp.getParameterValue(i, j);
                        // |cov_ij| <= sqrt(var_i var_j) <= max variance, so one bound covers the
                        // whole matrix; only the diagonal has to be nonnegative.
                        checkVariance(i == j ? v : Math.abs(v), "<" + SIGMA_EPSILON + "> entry ("
                                + i + "," + j + ")");
                    }
                }
            } else {
                final int sd = sigmaEpsilon.getDimension();
                if (sd != 1 && sd != dim) {
                    throw new XMLParseException("<" + SIGMA_EPSILON + "> must have dimension 1 (isotropic) or "
                            + dim + " (per-axis variances), got " + sd);
                }
                for (int i = 0; i < sd; i++) {
                    checkVariance(sigmaEpsilon.getParameterValue(i), "<" + SIGMA_EPSILON + "> entry " + i);
                }
                // A negative variance would make the leaf damping factor exp(-1/2 m'Sigma_eps m)
                // grow without bound, so bound the parameter rather than trust the operator.
                sigmaEpsilon.addBounds(new Parameter.DefaultBounds(Double.POSITIVE_INFINITY, 0.0, sd));
            }
        }

        String id = xo.getId();
        if (id == null) id = TORUS_FOURIER_LIKELIHOOD;
        final TorusFourierTreeLikelihood like;
        if (jumpLaw == TorusFourierTreeLikelihood.JumpLaw.LANDING) {
            if (jumpRate == null || jumpW == null) {
                throw new XMLParseException("jumpLaw=landing requires jumpRate and a fixed basin mixture "
                        + "(jumpComponent or sigmaJump)");
            }
            try {
                like = new TorusFourierTreeLikelihood(id, tree, branchRateModel,
                        precisions, tipData, kernel, jumpRate, jumpW, jumpD, jumpS, sigmaEpsilon, jumpLaw);
            } catch (IllegalArgumentException e) {
                throw new XMLParseException("Invalid landing likelihood: " + e.getMessage());
            }
        } else {
            like = new TorusFourierTreeLikelihood(id, tree, branchRateModel,
                    precisions, tipData, kernel, jumpRate, jumpW, jumpD, jumpS, sigmaEpsilon);
        }
        like.setTraitNames(traitNames.toArray(new String[0]));   // name ancestral traits after the tips
        return like;
    }

    /**
     * One units/range guard for Sigma_eps, applied to EVERY spelling (attribute, scalar element,
     * per-axis element, full matrix element). A value in degrees, or a standard deviation, lands
     * far above pi^2/4 rad^2 and destroys the model in silence: the leaf damping exp(-m'Se m/2)
     * kills every nonzero Fourier mode and the likelihood collapses to the uniform limit
     * -d*log(2*pi), with no warning and exit 0.
     */
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

    /**
     * Fail on any attribute this parser does not read, on this element or on the child elements
     * it owns. BEAST's XMLObject exposes the raw DOM attributes ({@link XMLObject#getAttributes()})
     * but never checks them against the parser's syntax rules, so {@code sigmaEpsilion="..."} is
     * accepted, ignored, and leaves no trace anywhere in the run's output.
     */
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
        for (XMLSyntaxRule rule : rules) {         // recurse into the child elements we own
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
                AttributeRule.newStringRule(JUMP_LAW, true),
                AttributeRule.newDoubleRule(SIGMA_EPSILON, true),
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
                new ElementRule(SIGMA_EPSILON, new XMLSyntaxRule[]{
                        new ElementRule(Parameter.class)
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
                + "diffusions of protein backbone dihedrals on a flat torus. Tips may carry a wrapped-Gaussian "
                + "observation error (<" + SIGMA_EPSILON + ">, variances in radians^2); omitted, tips are "
                + "point observations. jumpLaw='shift' (default) adds the jumpComponent displacement; "
                + "jumpLaw='landing' resets to that absolute basin mean, requires positive sigmaEpsilon, "
                + "and uses the landing process's stationary root distribution. The mixture is fixed "
                + "and shared by all residues in this likelihood.";
    }

    public Class getReturnType() {
        return TorusFourierTreeLikelihood.class;
    }
}
