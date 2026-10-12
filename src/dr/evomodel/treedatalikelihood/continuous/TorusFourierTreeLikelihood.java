/*
 * TorusFourierTreeLikelihood.java
 *
 * @author Samir Bhatt
 */

package dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evolution.tree.TreeTrait;
import dr.evolution.tree.TreeTraitProvider;
import dr.evomodel.branchratemodel.BranchRateModel;
import dr.evomodel.tree.TreeChangedEvent;
import dr.inference.model.*;
import dr.xml.Reportable;

import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

/**
 * Felsenstein pruning likelihood for independent d-dimensional wrapped-Brownian
 * diffusions (one per residue) on a flat torus. Tips may carry a wrapped-Gaussian
 * observation error Sigma_eps (see {@code sigmaEpsilon}); with Sigma_eps absent or zero
 * they are Dirac point observations for shift; landing requires positive Sigma_eps.
 * Optional per-evaluation validation uses a positive reference for the selected law;
 * the reference result replaces an inaccurate FFT likelihood. Ancestral logging draws
 * internal states jointly by a downward pass over the cached FFT partials, with the
 * zero-jump half of every branch kernel evaluated exactly in real space (see
 * TorusFourierKernel, "ancestral sampling"); when a positive reference is present, as under
 * validation, draws come from that reference instead.
 *
 * <p>Partial likelihoods are cached per node and per residue, following BEAST's
 * standard tree-likelihood scheme (cf. {@code AbstractTreeLikelihood} /
 * {@code TreeDataLikelihood}). Each node's grid message is stored in a
 * double-buffered array; an MCMC move flags only the affected nodes/residues
 * dirty, and a likelihood evaluation recomputes a node iff it is flagged or one
 * of its children was recomputed (the dirty path from a changed node to the
 * root). {@code store}/{@code restore} swap buffer indices in O(nodes), so
 * rejected proposals revert without recomputing any grid. A local tree move
 * costs O(depth) node recomputes; a single residue's precision move recomputes
 * only that residue, not all of them.
 *
 * @author Samir Bhatt
 */
public class TorusFourierTreeLikelihood extends AbstractModelLikelihood implements Reportable, TreeTraitProvider {

    public static final String TORUS_FOURIER_LIKELIHOOD = "torusFourierTreeLikelihood";
    public enum JumpLaw { SHIFT, LANDING }
    private static final double MIN_BRANCH = 1.0e-8;   // floor for zero-length branches (replicate tips)
    private static final int PARALLEL_THRESHOLD = 8;   // parallelise the residue loop at/above this many residues

    private final Tree tree;
    private final BranchRateModel branchRateModel;     // nullable -> rate 1
    private final List<MatrixParameter> precisions;    // one dim x dim precision per residue
    private final double[][][] tipData;                // [residue][externalNodeNumber][dim], radians
    private final TorusFourierKernel kernel;
    private final JumpLaw jumpLaw;
    // One fixed landing distribution per likelihood, shared by its residues. Unlike
    // psi, ghat is independent of both the sampled diffusion and the jump rate.
    private final double[] landingGhatRe;
    private final double[] landingGhatIm;
    private final long[][] landingNonpositiveByBranch;
    private final long[][] storedLandingNonpositiveByBranch;
    // Strict tail checking is an audit option. The usual likelihood preserves the
    // requested Fourier discretisation, including signed truncation tails, so its
    // values agree with the standalone used to calibrate practical grid sizes.
    private final boolean strictLandingNumerics = Boolean.getBoolean("torus.landing.strict");
    private final java.util.concurrent.atomic.AtomicLong landingUndefinedEvaluations =
            new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicBoolean landingUndefinedWarning =
            new java.util.concurrent.atomic.AtomicBoolean(false);

    /** Residue evaluations whose landing root integral was non-positive at this nGrid (returned -inf). */
    public long getLandingUndefinedEvaluations() { return landingUndefinedEvaluations.get(); }
    private final java.util.concurrent.atomic.AtomicBoolean landingResolutionWarning =
            new java.util.concurrent.atomic.AtomicBoolean();
    // Compound-Poisson jump, shared across residues: sampled rate lambda (jumpRate) and a FIXED
    // directed mixture of wrapped-normal components {weight w_k, displacement d_k, covariance S_k}
    // from the Ramachandran-basin fit. Nullable; jumpRate <= 0 or jumpS == null gives pure BM.
    // d_k != 0 makes the per-mode multiplier complex (handled by the kernel's complex path).
    private final Parameter jumpRate;
    // Tip observation error. Each tip is a wrapped Gaussian of covariance Sigma_eps rather
    // than a Dirac delta, so its Fourier coefficients are the unit-modulus point-observation
    // phase damped by exp(-1/2 m' Sigma_eps m). Nullable; null, or a parameter sitting at
    // exactly zero, is the Dirac tip and reproduces the pre-tip-error engine bit for bit.
    // Shapes: dim x dim MatrixParameter (full), dimension-dim Parameter (per-axis variances),
    // dimension-1 Parameter (isotropic). Units are VARIANCES in radians^2.
    private final Parameter sigmaEpsilon;
    private final double[] jumpW;      // mixture weights, length K
    private final double[][] jumpD;    // displacements, K x dim
    private final double[][][] jumpS;  // covariances, K x dim x dim
    private final int nResidues;
    private final int n2;
    private final int dim;
    private final int nodeCount;
    // Shared all-ones leaf message for masked (missing) tips: a residue absent at a taxon
    // (tipData[r][tip] == null) contributes a flat message, i.e. its value is marginalised out
    // exactly as internal nodes are. Read-only during the sweep, so it is safe to share.
    private final double[] flatLeaf;

    // ---- cached per-node partials (double-buffered) ----
    // partials[buffer][residue][nodeNumber] is the grid message at the node, i.e.
    // the product over children of their messages propagated up their branches,
    // normalised so its maximum is 1; the pulled-out log scale (summed over the
    // node's whole subtree) is in nodeLogScale. Inner arrays exist for internal
    // nodes only; tip slots stay null (tip messages are made on the fly).
    private final double[][][][] partials;             // [2][nResidues][nodeCount][n2]
    private final int[][] currentIndex;                // [nResidues][nodeCount] -> 0/1
    private final double[][] nodeLogScale;             // [nResidues][nodeCount]
    private final int[][] storedCurrentIndex;
    private final double[][] storedNodeLogScale;
    // A node's buffer index flips at most once between a store and the next store or restore,
    // as in BufferIndexHelper. A second evaluation in the same cycle (MarkovChain's full-evaluation
    // check, or any caller that evaluates twice) then rewrites the proposal's buffer in place
    // instead of flipping back onto the stored one, which a later restore would hand back corrupted.
    private final boolean[][] flipped;                 // [nResidues][nodeCount]

    // ---- dirty bookkeeping ----
    private final boolean[] updateNode;                // [nodeCount], shared across residues (tree/rate moves)
    private final boolean[] residueDirty;              // [nResidues], set when that residue's precision changes
    private final boolean[] storedUpdateNode;
    private final boolean[] storedResidueDirty;

    private double logLikelihood;
    private boolean likelihoodKnown = false;
    private double storedLogLikelihood;

    // Accepted positive quadratures are immutable; store/restore snapshot references.
    /**
     * Per-evaluation validation against the positive log-space reference is OPT-IN.
     * TorusLogSpacePruning enumerates multinomial allocations over the jump mixture, so
     * the 13-component Ramachandran mixture needs tens of thousands of allocations: on
     * the 21-taxon lysozyme tree that turns a ~0.1 s likelihood into minutes, or fails
     * outright on the component cap. Enable with -Dtorus.validate=true for audits and
     * regression runs. In production the FFT result is used directly and its accuracy is
     * bounded by a calibrated operating range instead (verify_checks.py F2/J2).
     */
    private static boolean validateEveryEvaluation = Boolean.getBoolean("torus.validate");

    /**
     * Turn per-evaluation validation on or off. Intended for tests and audits; production
     * runs leave it off. Construct a fresh likelihood after changing it, since accepted
     * per-residue values are cached.
     */
    public static void setValidateEveryEvaluation(boolean validate) {
        validateEveryEvaluation = validate;
    }

    public static boolean isValidateEveryEvaluation() {
        return validateEveryEvaluation;
    }

    /**
     * Ancestral sampling source. By default internal states come from the split FFT downward
     * pass: one O(g^2 log g) step per node, usable at production scale, drawing from the same
     * posterior the FFT likelihood defines. If a positive reference already exists for a
     * residue (per-evaluation validation builds one), draws come from it, so they follow the
     * likelihood the chain accepted. -Dtorus.ancestral.reference=true forces the positive
     * reference (small trees only: it is O(g^4) per edge). An UNRESOLVED draw -- one resting on
     * partial values below the FFT's dynamic range -- warns once; -Dtorus.ancestral.strict=true
     * makes it an error.
     */
    private static boolean ancestralReferenceForced() {
        return Boolean.getBoolean("torus.ancestral.reference");
    }

    private static boolean ancestralStrict() {
        return Boolean.getBoolean("torus.ancestral.strict");
    }

    private final java.util.concurrent.atomic.AtomicBoolean ancestralResolutionWarning =
            new java.util.concurrent.atomic.AtomicBoolean(false);

    private final TorusLogSpacePruning.Result[] stableResults;
    private final TorusLogSpacePruning.Result[] storedStableResults;
    // The landing reference uses absolute reset locations and a stationary landing root.
    // It is built only for explicit validation or requested ancestral trait logging.
    private final TorusLandingPruning.Result[] landingStableResults;
    private final TorusLandingPruning.Result[] storedLandingStableResults;
    private final double[] acceptedResidueLogL;
    private final double[] storedAcceptedResidueLogL;

    // ---- ancestral reconstruction (downward joint sampling; populated lazily per logged state) ----
    private String[] traitNames;                       // per-residue trait names (nullable -> default)
    private final double[][][] sampledAnc;             // [residue][nodeNumber][dim], null for tips
    private boolean ancestorsDirty = true;             // resample on next access after any (re)evaluation
    private TreeTrait[] treeTraitArray;
    private boolean storedLikelihoodKnown = false;

    // Instrumentation counts candidate FFT partial recomputes. Reference validation
    // currently repeats the changed residue, so it has additional full-tree cost.
    private final java.util.concurrent.atomic.AtomicLong nodeRecomputes = new java.util.concurrent.atomic.AtomicLong();
    public long getNodeRecomputeCount() { return nodeRecomputes.get(); }
    public void resetRecomputeCount() { nodeRecomputes.set(0L); }

    public TorusFourierTreeLikelihood(String name, Tree tree, BranchRateModel branchRateModel,
                                      List<MatrixParameter> precisions, double[][][] tipData,
                                      TorusFourierKernel kernel) {
        this(name, tree, branchRateModel, precisions, tipData, kernel, null, null, null, null);
    }

    public TorusFourierTreeLikelihood(String name, Tree tree, BranchRateModel branchRateModel,
                                      List<MatrixParameter> precisions, double[][][] tipData,
                                      TorusFourierKernel kernel,
                                      Parameter jumpRate, double[] jumpW, double[][] jumpD,
                                      double[][][] jumpS) {
        this(name, tree, branchRateModel, precisions, tipData, kernel,
                jumpRate, jumpW, jumpD, jumpS, null);
    }

    /** As above, with a tip observation-error covariance parameter (null = Dirac tips). */
    public TorusFourierTreeLikelihood(String name, Tree tree, BranchRateModel branchRateModel,
                                      List<MatrixParameter> precisions, double[][][] tipData,
                                      TorusFourierKernel kernel,
                                      Parameter jumpRate, double[] jumpW, double[][] jumpD,
                                      double[][][] jumpS, Parameter sigmaEpsilon) {
        this(name, tree, branchRateModel, precisions, tipData, kernel,
                jumpRate, jumpW, jumpD, jumpS, sigmaEpsilon, JumpLaw.SHIFT);
    }

    /** Select the jump law explicitly. For LANDING, jumpD contains absolute basin
     * means, and a strictly positive observation-error covariance is required. */
    public TorusFourierTreeLikelihood(String name, Tree tree, BranchRateModel branchRateModel,
                                      List<MatrixParameter> precisions, double[][][] tipData,
                                      TorusFourierKernel kernel,
                                      Parameter jumpRate, double[] jumpW, double[][] jumpD,
                                      double[][][] jumpS, Parameter sigmaEpsilon, JumpLaw jumpLaw) {
        super(name);
        if (jumpLaw == null) throw new IllegalArgumentException("jumpLaw must be shift or landing");
        this.jumpLaw = jumpLaw;
        this.sigmaEpsilon = sigmaEpsilon;
        this.tree = tree;
        this.branchRateModel = branchRateModel;
        this.precisions = precisions;
        this.tipData = tipData;
        this.kernel = kernel;
        this.jumpRate = jumpRate;
        this.jumpW = jumpW;
        this.jumpD = jumpD;
        this.jumpS = jumpS;
        this.nResidues = precisions.size();
        this.n2 = kernel.getGridCount();
        this.dim = kernel.getDimension();
        if (jumpLaw == JumpLaw.LANDING) {
            if (jumpRate == null || (jumpRate.getDimension() != 1
                    && jumpRate.getDimension() != nResidues)) {
                throw new IllegalArgumentException("landing requires jumpRate with dimension 1 or nResidues");
            }
            this.landingGhatRe = new double[n2];
            this.landingGhatIm = new double[n2];
            kernel.computeGhat(jumpW, jumpD, jumpS, landingGhatRe, landingGhatIm);
            if (tree.isExternal(tree.getRoot())) {
                throw new IllegalArgumentException("landing requires a tree with an internal root");
            }
        } else {
            this.landingGhatRe = null;
            this.landingGhatIm = null;
        }
        this.nodeCount = tree.getNodeCount();
        this.landingNonpositiveByBranch = jumpLaw == JumpLaw.LANDING ? new long[nResidues][nodeCount] : null;
        this.storedLandingNonpositiveByBranch = jumpLaw == JumpLaw.LANDING ? new long[nResidues][nodeCount] : null;
        this.stableResults = new TorusLogSpacePruning.Result[nResidues];
        this.storedStableResults = new TorusLogSpacePruning.Result[nResidues];
        this.landingStableResults = jumpLaw == JumpLaw.LANDING ? new TorusLandingPruning.Result[nResidues] : null;
        this.storedLandingStableResults = jumpLaw == JumpLaw.LANDING ? new TorusLandingPruning.Result[nResidues] : null;
        this.acceptedResidueLogL = new double[nResidues];
        Arrays.fill(this.acceptedResidueLogL, Double.NaN);   // "not yet computed" sentinel
        this.storedAcceptedResidueLogL = new double[nResidues];
        this.flatLeaf = new double[n2];
        Arrays.fill(this.flatLeaf, 1.0);

        for (int r = 0; r < nResidues; r++) {
            if (tipData[r].length != tree.getExternalNodeCount()) {
                throw new IllegalArgumentException("tipData residue " + r + " has "
                        + tipData[r].length + " taxa, tree has " + tree.getExternalNodeCount());
            }
            final MatrixParameter p = precisions.get(r);
            if (p.getRowDimension() != dim || p.getColumnDimension() != dim) {
                throw new IllegalArgumentException("precision " + r + " must be " + dim + "x" + dim);
            }
        }

        // Allocate the double-buffered partial store. Only internal nodes hold a
        // grid; tip slots remain null. This is the bulk of the memory footprint:
        // 2 * nResidues * nInternal * n2 doubles.
        this.partials = new double[2][nResidues][nodeCount][];
        for (int i = 0; i < nodeCount; i++) {
            if (!tree.isExternal(tree.getNode(i))) {
                for (int r = 0; r < nResidues; r++) {
                    partials[0][r][i] = new double[n2];
                    partials[1][r][i] = new double[n2];
                }
            }
        }
        this.currentIndex = new int[nResidues][nodeCount];
        this.nodeLogScale = new double[nResidues][nodeCount];
        this.storedCurrentIndex = new int[nResidues][nodeCount];
        this.storedNodeLogScale = new double[nResidues][nodeCount];
        this.flipped = new boolean[nResidues][nodeCount];

        this.updateNode = new boolean[nodeCount];
        this.residueDirty = new boolean[nResidues];
        this.storedUpdateNode = new boolean[nodeCount];
        this.storedResidueDirty = new boolean[nResidues];
        this.sampledAnc = new double[nResidues][nodeCount][];
        updateAllNodes();
        Arrays.fill(residueDirty, true);

        if (tree instanceof Model) addModel((Model) tree);
        if (branchRateModel != null) addModel(branchRateModel);
        for (MatrixParameter p : precisions) addVariable(p);
        // Only the rate lambda is sampled (mixture components are fixed). jumpRate may be
        // per-residue (dim = nResidues, one lambda each -> a move dirties only that residue) or
        // shared (dim 1 -> dirties all). See handleVariableChangedEvent.
        if (jumpRate != null) addVariable(jumpRate);
        // Sigma_eps is shared by every residue, so a change to it dirties all of them
        // (handleVariableChangedEvent's catch-all branch).
        if (sigmaEpsilon != null) {
            final int sd = sigmaEpsilon.getDimension();
            if (sigmaEpsilon instanceof MatrixParameter) {
                final MatrixParameter mp = (MatrixParameter) sigmaEpsilon;
                if (mp.getRowDimension() != dim || mp.getColumnDimension() != dim) {
                    throw new IllegalArgumentException("sigmaEpsilon matrix must be " + dim + "x" + dim);
                }
            } else if (sd != 1 && sd != dim) {
                throw new IllegalArgumentException("sigmaEpsilon must have dimension 1 (isotropic) or "
                        + dim + " (per-axis variances), got " + sd);
            }
            sigmaEpsilonMatrix();      // fail at construction, not mid-chain, on a bad Sigma_eps
            addVariable(sigmaEpsilon);
        }
        if (jumpLaw == JumpLaw.LANDING) requireLandingObservationError();
        logSigmaEpsilon(name);
        if (jumpLaw == JumpLaw.LANDING) {
            Logger.getLogger("dr.evomodel").info("TorusFourierTreeLikelihood '" + name
                    + "': JUMP_LAW = landing (fixed absolute basin means, stationary landing root), nGrid="
                    + kernel.getGridSize() + ", strict numerical audit=" + strictLandingNumerics);
        }
    }

    public JumpLaw getJumpLaw() { return jumpLaw; }

    /** Counts nonpositive spectral values, without replacing them by a positive floor. */
    public long getLandingNonpositiveCount() { return kernel.getLandingNonpositiveCount(); }

    /** Current branch audit indexed by residue and child node number (the root slot is zero).
     * These are cells that a positivity floor would change; landing leaves them untouched. */
    public long[][] getLandingBranchNonpositiveCounts() {
        if (jumpLaw != JumpLaw.LANDING) return new long[0][];
        getLogLikelihood();
        final long[][] out = new long[nResidues][];
        for (int r = 0; r < nResidues; r++) out[r] = landingNonpositiveByBranch[r].clone();
        return out;
    }

    private double[][] requireLandingObservationError() {
        final double[][] s = sigmaEpsilonMatrix();
        if (s == null) {
            throw new IllegalArgumentException("jumpLaw=landing requires positive-definite sigmaEpsilon");
        }
        for (double[] row : s) for (double value : row) {
            if (!Double.isFinite(value)) throw new IllegalArgumentException("landing sigmaEpsilon must be finite");
        }
        return s;
    }

    /**
     * State the effective tip observation error ONCE, at construction, for every instance --
     * including the Sigma_eps-free one.
     *
     * A misspelled XML attribute is not the only way to lose Sigma_eps, and until this line
     * existed NOTHING in a run's output recorded whether tip error was applied: a Dirac run and
     * a Sigma_eps run produced output that was identical except for the likelihood values
     * themselves. A 51-hour array could come back silently Dirac with plausible numbers.
     * Grep the run log for SIGMA_EPS to settle it.
     */
    private void logSigmaEpsilon(String name) {
        final Logger logger = Logger.getLogger("dr.evomodel");
        final double[][] s = sigmaEpsilonMatrix();
        final StringBuilder sb = new StringBuilder();
        sb.append("TorusFourierTreeLikelihood '").append(name).append("': SIGMA_EPS ");
        if (s == null) {
            sb.append("= NONE (Dirac point tips)");
            if (sigmaEpsilon != null) {
                sb.append(" -- a sigmaEpsilon parameter IS attached but its value is exactly zero");
            }
        } else {
            sb.append("APPLIED, tip observation-error covariance in rad^2 = [");
            for (int i = 0; i < dim; i++) {
                sb.append(i == 0 ? "" : "; ");
                for (int j = 0; j < dim; j++) sb.append(j == 0 ? "" : " ").append(s[i][j]);
            }
            sb.append("]  (per-axis sd in degrees:");
            for (int i = 0; i < dim; i++) {
                sb.append(String.format(" %.4f", Math.toDegrees(Math.sqrt(s[i][i]))));
            }
            sb.append(")");
        }
        logger.info(sb.toString());
    }

    /**
     * Tip observation-error covariance at the current parameter values, or null when it is
     * exactly zero. Returning null rather than a zero matrix is deliberate: the kernel's null
     * path is literally the original Dirac-tip arithmetic, so a model without tip error -- or
     * with the parameter sitting at zero -- reproduces the pre-tip-error engine bit for bit.
     */
    private double[][] sigmaEpsilonMatrix() {
        if (sigmaEpsilon == null) return null;
        final double[][] s = new double[dim][dim];
        boolean nonZero = false;
        if (sigmaEpsilon instanceof MatrixParameter) {
            final MatrixParameter mp = (MatrixParameter) sigmaEpsilon;
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) {
                    s[i][j] = mp.getParameterValue(i, j);
                    if (s[i][j] != 0.0) nonZero = true;
                }
            }
        } else {
            final boolean isotropic = sigmaEpsilon.getDimension() == 1;
            for (int i = 0; i < dim; i++) {
                s[i][i] = sigmaEpsilon.getParameterValue(isotropic ? 0 : i);
                if (s[i][i] != 0.0) nonZero = true;
            }
        }
        if (!nonZero) return null;
        checkCovariance(s);
        return s;
    }

    /**
     * A tip-error covariance must be symmetric positive definite. The leaf damping factor is
     * exp(-1/2 m' Sigma_eps m), which GROWS without bound in any direction where that quadratic
     * form is negative; nothing downstream would fail, the likelihood would just be wrong. So
     * check here and throw.
     */
    private static void checkCovariance(double[][] s) {
        final int n = s.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (s[i][j] != s[j][i]) {
                    throw new IllegalArgumentException("sigmaEpsilon must be symmetric: (" + i + "," + j
                            + ")=" + s[i][j] + " but (" + j + "," + i + ")=" + s[j][i]);
                }
            }
        }
        final double[][] L = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = s[i][j];
                for (int k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
                if (i == j) {
                    if (!(sum > 0.0)) {
                        throw new IllegalArgumentException("sigmaEpsilon must be positive definite "
                                + "(a covariance in radians^2); Cholesky failed on axis " + i);
                    }
                    L[i][i] = Math.sqrt(sum);
                } else {
                    L[i][j] = sum / L[j][j];
                }
            }
        }
    }

    /** Current jump rate for residue r, clamped at 0. jumpRate may be per-residue (dimension =
     *  nResidues, one lambda each) or shared (dimension 1). A negative sampled lambda is not a
     *  valid Poisson rate (would explode the multiplier), so it is clamped; also bound it >=0 in XML. */
    private double jumpRateValue(int r) {
        if (jumpRate == null) return 0.0;
        final int i = (jumpRate.getDimension() == nResidues) ? r : 0;
        final double lam = jumpRate.getParameterValue(i);
        return lam > 0.0 ? lam : 0.0;
    }

    // ---- Likelihood ----

    public Model getModel() { return this; }

    public double getLogLikelihood() {
        if (!likelihoodKnown) {
            logLikelihood = calculateLogLikelihood();
            likelihoodKnown = true;
        }
        return logLikelihood;
    }

    public void makeDirty() {
        likelihoodKnown = false;
        updateAllNodes();
        Arrays.fill(residueDirty, true);
        ancestorsDirty = true;   // ancestral sample is stale until the next evaluation
    }

    public String getReport() {
        final double[][] s = sigmaEpsilonMatrix();
        final StringBuilder sb = new StringBuilder(getClass().getSimpleName());
        sb.append(": logL = ").append(getLogLikelihood()).append(", SIGMA_EPS ");
        if (s == null) {
            sb.append("= NONE (Dirac point tips)");
        } else {
            sb.append("= [");
            for (int i = 0; i < dim; i++) {
                sb.append(i == 0 ? "" : "; ");
                for (int j = 0; j < dim; j++) sb.append(j == 0 ? "" : " ").append(s[i][j]);
            }
            sb.append("] rad^2");
        }
        return sb.toString();
    }

    private double calculateLogLikelihood() {
        if (jumpLaw == JumpLaw.LANDING) return calculateLandingLogLikelihood();
        final NodeRef root = tree.getRoot();
        // Residues are independent; each residue reads/writes only its own slice of
        // the partial/scale/index arrays, the dirty flags are read-only during the
        // sweep, and the kernel allocates per call, so the residue loop parallelises
        // safely. Above a threshold we use a parallel stream (helps the p=2 sweeps).
        // Compute each residue's log-likelihood in parallel (each writes only its own
        // slot, so this is order-independent), then reduce in fixed index order. A parallel
        // stream .sum() reduces in thread-scheduling order, whose floating-point rounding is
        // not reproducible run-to-run; a fixed-order serial sum keeps the parallel compute
        // speedup while making the total deterministic (seed-reproducible).
        // Mixture components (jumpW/jumpD/jumpS) are fixed instance fields; the rate lambda may be
        // per-residue (jumpRateValue(r)), so it is read inside the per-residue loop.
        // Sigma_eps is shared by all residues and constant across the sweep; resolve it once.
        final double[][] sigmaEps = sigmaEpsilonMatrix();
        final double[] residueLogL = new double[nResidues];
        java.util.stream.IntStream stream = java.util.stream.IntStream.range(0, nResidues);
        if (nResidues >= PARALLEL_THRESHOLD) stream = stream.parallel();
        stream.forEach(r -> {
            final double[][] sigma = invertPrecision(precisions.get(r));
            final double jr = jumpRateValue(r);
            double[] psiRe = null, psiIm = null;
            if (jr > 0.0 && jumpS != null) {     // precompute the branch-independent jump exponent once per residue
                psiRe = new double[n2];
                psiIm = new double[n2];
                kernel.computePsiComplex(sigma, jr, jumpW, jumpD, jumpS, psiRe, psiIm);
            }
            boolean changed = updateNodePartial(r, root, sigma, psiRe, psiIm, jr, sigmaEps);
            if (changed || Double.isNaN(acceptedResidueLogL[r])) {
                final int rn = root.getNumber();
                double fft = nodeLogScale[r][rn]
                        + Math.log(kernel.rootIntegral(partials[currentIndex[r][rn]][r][rn]));
                if (validateEveryEvaluation) {
                    // A smooth-looking tip or a positive FFT result does not certify
                    // the pruning tail. Validate the completed likelihood itself.
                    TorusLogSpacePruning.Result stable = TorusLogSpacePruning.calculate(
                            tree, this::branchLength, tipData[r], sigma, jr, jumpW, jumpD, jumpS,
                            sigmaEps);
                    acceptedResidueLogL[r] = Double.isFinite(fft)
                            && Math.abs(fft - stable.logLikelihood) <= TorusLogSpacePruning.LOG_TOLERANCE
                            ? fft : stable.logLikelihood;
                    stableResults[r] = stable;
                } else {
                    acceptedResidueLogL[r] = fft;
                    stableResults[r] = null;   // built on demand if ancestral states are asked for
                }
            }
            residueLogL[r] = acceptedResidueLogL[r];
        });
        double total = 0.0;
        for (int r = 0; r < nResidues; r++) total += residueLogL[r];   // fixed-order, deterministic
        // Clear dirty flags only after the whole sweep has consumed them.
        Arrays.fill(updateNode, false);
        Arrays.fill(residueDirty, false);
        ancestorsDirty = true;          // partials refreshed -> ancestral sample is stale
        return total;
    }

    /**
     * Ensure node's cached partial (and subtree log-scale) for residue r are current,
     * recomputing iff the node is flagged, the residue's precision changed, or a child
     * was recomputed. Returns whether this node was recomputed, so the parent can
     * propagate the dirty path upward. Mirrors prune_loglik on a cache miss.
     */
    private boolean updateNodePartial(int r, NodeRef node, double[][] sigma, double[] psiRe, double[] psiIm,
                                      double jumpRate, double[][] sigmaEps) {
        final int nn = node.getNumber();
        final int childCount = tree.getChildCount(node);

        boolean childRecomputed = false;
        for (int c = 0; c < childCount; c++) {
            final NodeRef child = tree.getChild(node, c);
            if (!tree.isExternal(child)) {
                childRecomputed |= updateNodePartial(r, child, sigma, psiRe, psiIm, jumpRate, sigmaEps);
            } else {
                // Tips hold no cached partial (the leaf message is rebuilt on the fly),
                // so their own dirty flag must be consulted here: a sampled tip-date or
                // external-branch change alters this node's leafKernel and must force the
                // parent to recompute.
                childRecomputed |= updateNode[child.getNumber()];
            }
        }

        if (!(updateNode[nn] || residueDirty[r] || childRecomputed)) {
            return false;   // cache hit: reuse stored partial + scale
        }

        // Write into the alternate buffer; the current one stays intact for restore. Once a node
        // has flipped in this cycle, its current buffer IS the alternate, so rewrite it in place.
        final int target = flipped[r][nn] ? currentIndex[r][nn] : 1 - currentIndex[r][nn];
        final double[] values = partials[target][r][nn];
        Arrays.fill(values, 1.0);
        final boolean hasJump = psiRe != null;
        double scale = 0.0;
        for (int c = 0; c < childCount; c++) {
            final NodeRef child = tree.getChild(node, c);
            final double bl = branchLength(child);
            final double[] msg;
            if (tree.isExternal(child)) {
                final double[] y = tipData[r][child.getNumber()];
                if (y == null) {                          // masked: residue absent at this taxon
                    msg = flatLeaf;                       // flat -> tip value marginalised out
                } else {
                    // Tip observation error enters HERE and nowhere else: only terminal branches
                    // carry it, and internal propagation is untouched.
                    msg = hasJump ? kernel.leafKernelFromPsi(y, bl, psiRe, psiIm, sigma, jumpRate, sigmaEps)
                                  : kernel.leafKernel(y, sigma, bl, sigmaEps);
                }
            } else {
                final int cn = child.getNumber();
                final double[] cv = partials[currentIndex[r][cn]][r][cn];
                msg = hasJump ? kernel.propagateFromPsi(cv, bl, psiRe, psiIm)
                              : kernel.propagate(cv, sigma, bl);
                scale += nodeLogScale[r][cn];   // carry the child's subtree scale
            }
            double max = 0.0;
            for (int k = 0; k < n2; k++) {
                values[k] *= msg[k];
                if (values[k] > max) max = values[k];
            }
            if (!(max > 0.0)) {
                // Mark this FFT candidate unusable. The positive reference below
                // still evaluates the state; numerical failure is not evidence of
                // a vanishing-probability proposal.
                nodeLogScale[r][nn] = Double.NEGATIVE_INFINITY;
                System.arraycopy(flatLeaf, 0, values, 0, n2);  // leave a defined (uniform) buffer,
                currentIndex[r][nn] = target;                  // never a half-accumulated product
                flipped[r][nn] = true;
                return true;
            }
            final double inv = 1.0 / max;
            for (int k = 0; k < n2; k++) values[k] *= inv;
            scale += Math.log(max);
        }
        nodeLogScale[r][nn] = scale;
        currentIndex[r][nn] = target;   // point at the freshly written buffer
        flipped[r][nn] = true;
        nodeRecomputes.incrementAndGet();
        return true;
    }

    /** Landing has its own root, branch operator, and optional positive reference. */
    private double calculateLandingLogLikelihood() {
        final double[][] sigmaEps = requireLandingObservationError();
        final NodeRef root = tree.getRoot();
        java.util.stream.IntStream stream = java.util.stream.IntStream.range(0, nResidues);
        if (nResidues >= PARALLEL_THRESHOLD) stream = stream.parallel();
        stream.forEach(r -> {
            final double[][] sigma = invertPrecision(precisions.get(r));
            final double lambda = jumpRate.getParameterValue(jumpRate.getDimension() == 1 ? 0 : r);
            if (!Double.isFinite(lambda) || lambda < 0.0) {
                throw new IllegalArgumentException("landing jumpRate must be finite and nonnegative");
            }
            final boolean changed = updateLandingNodePartial(r, root, sigma, lambda, sigmaEps);
            if (changed || Double.isNaN(acceptedResidueLogL[r])) {
                final int rn = root.getNumber();
                final double rootMass = kernel.rootIntegralLanding(
                        partials[currentIndex[r][rn]][r][rn], sigma, lambda, landingGhatRe, landingGhatIm);
                final double fft = rootMass > 0.0 && Double.isFinite(rootMass)
                        ? nodeLogScale[r][rn] + Math.log(rootMass) : Double.NaN;
                if (validateEveryEvaluation) {
                    final TorusLandingPruning.Result reference = landingReference(r, sigma, lambda, sigmaEps);
                    landingStableResults[r] = reference;
                    acceptedResidueLogL[r] = Double.isFinite(fft)
                            && Math.abs(fft - reference.logLikelihood) <= TorusLandingPruning.LOG_TOLERANCE
                            ? fft : reference.logLikelihood;
                } else {
                    if (!Double.isFinite(fft)) {
                        // The root integral is non-positive or non-finite: at this nGrid the
                        // landing likelihood is UNDEFINED for this parameter value (Fourier
                        // truncation makes the root mass ring negative). Throwing here would kill
                        // an MCMC thread on an ordinary proposal; instead return -inf so the move
                        // is rejected -- what the shift path already does for an unusable FFT
                        // state, and the BEAST convention. The posterior is thereby restricted to
                        // the region the chosen grid can evaluate; every occurrence is counted.
                        if (strictLandingNumerics) {
                            throw new ArithmeticException("Invalid landing root integral for residue " + r
                                    + " at nGrid=" + kernel.getGridSize()
                                    + "; check grid convergence and covariance conditioning");
                        }
                        landingUndefinedEvaluations.incrementAndGet();
                        if (landingUndefinedWarning.compareAndSet(false, true)) {
                            Logger.getLogger("dr.evomodel").warning("TorusFourierTreeLikelihood '" + getId()
                                    + "': landing likelihood undefined (non-positive root integral) for residue "
                                    + r + " at nGrid=" + kernel.getGridSize() + "; returning -inf so the state is"
                                    + " rejected. Further occurrences are counted, not logged"
                                    + " (getLandingUndefinedEvaluations); -Dtorus.landing.strict=true throws instead.");
                        }
                        acceptedResidueLogL[r] = Double.NEGATIVE_INFINITY;
                    } else {
                        acceptedResidueLogL[r] = fft;
                    }
                    landingStableResults[r] = null;
                }
                stableResults[r] = null;
            }
        });
        double total = 0.0;
        for (int r = 0; r < nResidues; r++) total += acceptedResidueLogL[r];
        Arrays.fill(updateNode, false);
        Arrays.fill(residueDirty, false);
        ancestorsDirty = true;
        return total;
    }

    private boolean updateLandingNodePartial(int r, NodeRef node, double[][] sigma,
                                              double lambda, double[][] sigmaEps) {
        final int nn = node.getNumber();
        boolean changed = false;
        for (int c = 0; c < tree.getChildCount(node); c++) {
            final NodeRef child = tree.getChild(node, c);
            changed |= tree.isExternal(child) ? updateNode[child.getNumber()]
                    : updateLandingNodePartial(r, child, sigma, lambda, sigmaEps);
        }
        if (!(changed || updateNode[nn] || residueDirty[r])) return false;
        final int target = flipped[r][nn] ? currentIndex[r][nn] : 1 - currentIndex[r][nn];
        final double[] values = partials[target][r][nn];
        Arrays.fill(values, 1.0);
        double scale = 0.0;
        for (int c = 0; c < tree.getChildCount(node); c++) {
            final NodeRef child = tree.getChild(node, c);
            final int cn = child.getNumber();
            final double[] message;
            if (tree.isExternal(child)) {
                final double[] theta = tipData[r][cn];
                message = theta == null ? flatLeaf : kernel.leafKernelLanding(theta, sigmaEps,
                        sigma, branchLength(child), lambda, landingGhatRe, landingGhatIm);
            } else {
                message = kernel.propagateLanding(partials[currentIndex[r][cn]][r][cn],
                        sigma, branchLength(child), lambda, landingGhatRe, landingGhatIm);
                scale += nodeLogScale[r][cn];
            }
            // Preserve spectral values exactly: flooring small or negative tails here
            // would silently change the model. Normalize by absolute maximum, matching
            // the standalone Fourier reference even on an under-resolved grid.
            double max = 0.0;
            long nonpositive = 0;
            double messageMin = 0.0, messageMax = 0.0;
            for (int k = 0; k < n2; k++) {
                values[k] *= message[k];
                max = Math.max(max, Math.abs(values[k]));
                if (message[k] <= 0.0) nonpositive++;
                messageMin = Math.min(messageMin, message[k]);
                messageMax = Math.max(messageMax, Math.abs(message[k]));
            }
            landingNonpositiveByBranch[r][cn] = nonpositive;
            // Negative lobes diagnose Fourier truncation. Whether the requested grid
            // is adequate for inference is calibrated at the likelihood level, not
            // decided solely by a pointwise tail sign. Never floor these values.
            if (messageMin < -1e-12 * messageMax) {
                if (strictLandingNumerics) {
                    throw new ArithmeticException("Under-resolved landing branch for residue " + r + ", child " + cn
                            + ", nGrid=" + kernel.getGridSize() + ": min=" + messageMin + ", maxAbs=" + messageMax
                            + "; verify likelihood accuracy for this parameter regime");
                }
                if (landingResolutionWarning.compareAndSet(false, true)) {
                    Logger.getLogger("dr.evomodel").warning("Landing likelihood at nGrid=" + kernel.getGridSize()
                            + " has negative Fourier tail values. Values are preserved without clipping; "
                            + "check accuracy in the intended parameter regime. "
                            + "Use -Dtorus.landing.strict=true to reject these cases in an audit.");
                }
            }
            if (!(max > 0.0) || !Double.isFinite(max)) {
                throw new ArithmeticException("Invalid landing partial at residue " + r + ", node " + nn
                        + ", nGrid=" + kernel.getGridSize() + "; check grid convergence and covariance conditioning");
            }
            for (int k = 0; k < n2; k++) values[k] /= max;
            scale += Math.log(max);
        }
        nodeLogScale[r][nn] = scale;
        currentIndex[r][nn] = target;
        flipped[r][nn] = true;
        nodeRecomputes.incrementAndGet();
        return true;
    }

    private TorusLandingPruning.Result landingReference(int r, double[][] sigma,
                                                         double lambda, double[][] sigmaEps) {
        return TorusLandingPruning.calculate(tree, this::branchLength, tipData[r], sigma,
                lambda, jumpW, jumpD, jumpS, sigmaEps, Math.min(128, kernel.getGridSize()));
    }

    /**
     * Joint downward sample of residue r's internal angles under the LANDING law from the
     * cached FFT partials. The root is drawn from (stationary landing law x root partial); each
     * internal child is drawn given its parent cell from
     *     [ e^{-lambda t} WN(child - parent; t Sigma) + (1 - e^{-lambda t}) r_t(child) ] x childPartial,
     * with the zero-jump wrapped normal evaluated EXACTLY (winding sum) and the broad reset law
     * r_t on the FFT. The dense positive reference (TorusLandingPruning, O(g^4) per edge) is used
     * when already present or forced; it cannot serve real trees at production grids.
     */
    private void sampleAncestralLanding(int r) {
        if (landingStableResults[r] != null || ancestralReferenceForced()) {
            if (landingStableResults[r] == null) {
                landingStableResults[r] = landingReference(r, invertPrecision(precisions.get(r)),
                        jumpRate.getParameterValue(jumpRate.getDimension() == 1 ? 0 : r),
                        requireLandingObservationError());
            }
            landingStableResults[r].sample(sampledAnc[r]);
            return;
        }
        final double[][] sigma = invertPrecision(precisions.get(r));
        final double lambda = jumpRate.getParameterValue(jumpRate.getDimension() == 1 ? 0 : r);
        final NodeRef root = tree.getRoot();
        final int rootNumber = root.getNumber();
        requireUsableFftPartials(r, rootNumber);
        java.util.Arrays.fill(sampledAnc[r], null);
        final long unresolvedBefore = kernel.getAncestralUnresolvedDraws();

        final double[] stationary = kernel.landingStationaryDensity(
                sigma, lambda, landingGhatRe, landingGhatIm);
        final int rootCell = kernel.sampleFromProduct(
                stationary, partials[currentIndex[r][rootNumber]][r][rootNumber],
                dr.math.MathUtils.nextDouble());
        if (rootCell < 0) throw noAncestralMass("the root", r);
        sampledAnc[r][rootNumber] = kernel.anglesAt(rootCell);
        sampleLandingChildren(r, root, rootCell, sigma, lambda);
        checkAncestralResolution(r, unresolvedBefore);
    }

    /** Draw every internal descendant of `node`, given that `node` was sampled into `parentCell`. */
    private void sampleLandingChildren(int r, NodeRef node, int parentCell,
                                       double[][] sigma, double lambda) {
        for (int c = 0; c < tree.getChildCount(node); c++) {
            final NodeRef child = tree.getChild(node, c);
            if (tree.isExternal(child)) continue;          // tips are observed, never sampled
            final int cn = child.getNumber();
            final double t = branchLength(child);
            final double noJump = Math.exp(-lambda * t);
            final double jumpProbability = -Math.expm1(-lambda * t);
            final double[] diffusion = kernel.branchDiffusionKernel(sigma, t);
            for (int k = 0; k < n2; k++) diffusion[k] *= noJump;
            double[] reset = null;
            if (jumpProbability > 0.0) {
                reset = kernel.landingResetDensity(sigma, t, lambda, landingGhatRe, landingGhatIm);
                for (int k = 0; k < n2; k++) reset[k] *= jumpProbability;
            }
            final int cell = kernel.sampleChildCell(diffusion, reset, parentCell,
                    partials[currentIndex[r][cn]][r][cn], dr.math.MathUtils.nextDouble());
            if (cell < 0) throw noAncestralMass("node " + cn, r);
            sampledAnc[r][cn] = kernel.anglesAt(cell);
            sampleLandingChildren(r, child, cell, sigma, lambda);
        }
    }

    private double branchLength(NodeRef node) {
        double bl = tree.getNodeHeight(tree.getParent(node)) - tree.getNodeHeight(node);
        if (branchRateModel != null) bl *= branchRateModel.getBranchRate(tree, node);
        return bl > MIN_BRANCH ? bl : MIN_BRANCH;
    }

    /** Invert a dim x dim precision matrix to a covariance, by Gauss-Jordan elimination. */
    private double[][] invertPrecision(MatrixParameter p) {
        final int d = dim;
        final double[][] a = new double[d][2 * d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) a[i][j] = p.getParameterValue(i, j);
            a[i][d + i] = 1.0;
        }
        for (int col = 0; col < d; col++) {
            int piv = col;
            for (int rr = col + 1; rr < d; rr++) {
                if (Math.abs(a[rr][col]) > Math.abs(a[piv][col])) piv = rr;
            }
            if (Math.abs(a[piv][col]) < 1.0e-300) {
                throw new RuntimeException("Singular precision matrix");
            }
            final double[] tmp = a[col]; a[col] = a[piv]; a[piv] = tmp;
            final double diag = a[col][col];
            for (int j = 0; j < 2 * d; j++) a[col][j] /= diag;
            for (int rr = 0; rr < d; rr++) {
                if (rr != col) {
                    final double f = a[rr][col];
                    if (f != 0.0) for (int j = 0; j < 2 * d; j++) a[rr][j] -= f * a[col][j];
                }
            }
        }
        final double[][] inv = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) inv[i][j] = a[i][d + j];
        }
        return inv;
    }

    // ---- Model bookkeeping: flag only the dirty path / dirty residue ----

    protected void handleModelChangedEvent(Model model, Object object, int index) {
        if (model == tree) {
            if (object instanceof TreeChangedEvent) {
                final TreeChangedEvent ev = (TreeChangedEvent) object;
                if (ev.isNodeChanged()) {
                    // A node event (height change, attach/detach) alters the branches
                    // incident to this node; flag it and its children, and the
                    // child-recomputed cascade carries the update up to the root.
                    updateNodeAndChildren(ev.getNode());
                } else if (ev.isTreeChanged()) {
                    updateAllNodes();
                }
                // other tree events (e.g. trait changes) do not affect this likelihood
            }
            // Non-TreeChangedEvent objects from the tree (e.g. the raw node-height
            // Parameter passthrough that BEAST also fires alongside the TreeChangedEvent)
            // are deliberately ignored: the TreeChangedEvent in the same firing is
            // authoritative and does the dirtying. Acting on the passthrough here would
            // re-flag the whole tree and defeat the dirty-path caching. (This mirrors
            // dr.evomodel.treedatalikelihood.TreeDataLikelihood, which only acts on
            // TreeChangedEvent.)
        } else if (model == branchRateModel) {
            // As TreeDataLikelihood: the index is the number of the node whose branch rate changed,
            // -1 when every rate may have changed. A flagged node is recomputed together with the
            // path above it, and its parent re-propagates the branch with the new rate.
            if (index >= 0 && index < nodeCount) {
                updateNode[index] = true;
            } else {
                updateAllNodes();
            }
        } else {
            updateAllNodes();
        }
        likelihoodKnown = false;
        ancestorsDirty = true;
    }

    protected void handleVariableChangedEvent(Variable variable, int index, Parameter.ChangeType type) {
        final int r = precisionIndex(variable);
        if (r >= 0) {
            residueDirty[r] = true;      // only this residue's diffusion changed
        } else if (variable == jumpRate && jumpRate.getDimension() == nResidues && index >= 0) {
            residueDirty[index] = true;  // per-residue lambda: only that residue's rate changed
        } else {
            // shared lambda, sigmaEpsilon (shared by every residue), bulk change (index<0), or anything else
            Arrays.fill(residueDirty, true);
        }
        likelihoodKnown = false;
        ancestorsDirty = true;
    }

    private int precisionIndex(Variable variable) {
        for (int r = 0; r < nResidues; r++) {
            if (precisions.get(r) == variable) return r;
        }
        return -1;
    }

    private void updateNodeAndChildren(NodeRef node) {
        if (node == null) { updateAllNodes(); return; }
        updateNode[node.getNumber()] = true;
        for (int i = 0; i < tree.getChildCount(node); i++) {
            updateNode[tree.getChild(node, i).getNumber()] = true;
        }
    }

    private void updateAllNodes() { Arrays.fill(updateNode, true); }

    protected void storeState() {
        if (jumpLaw == JumpLaw.LANDING) {
            System.arraycopy(landingStableResults, 0, storedLandingStableResults, 0, nResidues);
            for (int r = 0; r < nResidues; r++) {
                System.arraycopy(landingNonpositiveByBranch[r], 0, storedLandingNonpositiveByBranch[r], 0, nodeCount);
            }
        }
        storedLogLikelihood = logLikelihood;
        storedLikelihoodKnown = likelihoodKnown;
        System.arraycopy(stableResults, 0, storedStableResults, 0, nResidues);
        System.arraycopy(acceptedResidueLogL, 0, storedAcceptedResidueLogL, 0, nResidues);
        // Also snapshot the dirty flags. Without this, a store taken while the state is
        // still dirty (e.g. before the first evaluation) would, after a proposal clears
        // the flags, restore buffer indices that point at never-computed buffers. Saving
        // and restoring the flags keeps the restored state internally consistent and
        // independent of the store/evaluate/restore call ordering.
        System.arraycopy(updateNode, 0, storedUpdateNode, 0, nodeCount);
        System.arraycopy(residueDirty, 0, storedResidueDirty, 0, nResidues);
        for (int r = 0; r < nResidues; r++) {
            System.arraycopy(currentIndex[r], 0, storedCurrentIndex[r], 0, nodeCount);
            System.arraycopy(nodeLogScale[r], 0, storedNodeLogScale[r], 0, nodeCount);
            Arrays.fill(flipped[r], false);
        }
    }

    protected void restoreState() {
        if (jumpLaw == JumpLaw.LANDING) {
            System.arraycopy(storedLandingStableResults, 0, landingStableResults, 0, nResidues);
            for (int r = 0; r < nResidues; r++) {
                System.arraycopy(storedLandingNonpositiveByBranch[r], 0, landingNonpositiveByBranch[r], 0, nodeCount);
            }
        }
        logLikelihood = storedLogLikelihood;
        likelihoodKnown = storedLikelihoodKnown;
        System.arraycopy(storedStableResults, 0, stableResults, 0, nResidues);
        System.arraycopy(storedAcceptedResidueLogL, 0, acceptedResidueLogL, 0, nResidues);
        System.arraycopy(storedUpdateNode, 0, updateNode, 0, nodeCount);
        System.arraycopy(storedResidueDirty, 0, residueDirty, 0, nResidues);
        // Swap the buffer indices and scales back; the partial data itself was never
        // overwritten in place (recompute wrote the alternate buffer), so this reverts
        // every recomputed node in O(nodes) without touching the grids.
        for (int r = 0; r < nResidues; r++) {
            System.arraycopy(storedCurrentIndex[r], 0, currentIndex[r], 0, nodeCount);
            System.arraycopy(storedNodeLogScale[r], 0, nodeLogScale[r], 0, nodeCount);
            Arrays.fill(flipped[r], false);
        }
        ancestorsDirty = true;          // reverted partials -> ancestral sample is stale
    }

    protected void acceptState() { }

    // ====================== ancestral reconstruction (TreeTraitProvider) ======================
    // The spectral pruning marginalises internal states, so ancestral angles are recovered with a
    // downward joint-sampling pass over the cached inward partials: sample the root from its grid
    // marginal, then each internal child from K_t(child-parent) * child-subtree-partial. Sampled
    // once per logged state (lazily, on first getTrait access) using the MCMC's RNG, so the chain
    // yields the posterior distribution of ancestral (phi,psi) at every node.

    /** Per-residue trait names (so the logged ancestral trait matches the tip trait); set by the parser. */
    public void setTraitNames(String[] names) {
        this.traitNames = names;
        this.treeTraitArray = null;     // rebuild lazily with the new names
    }

    private void ensureAncestors() {
        if (!ancestorsDirty) {
            return;
        }
        getLogLikelihood();             // make the inward partials current
        for (int r = 0; r < nResidues; r++) {
            sampleAncestral(r);
        }
        ancestorsDirty = false;
    }

    /** Joint-sample residue r's internal-node angles into sampledAnc[r] (tips kept null). */
    private void sampleAncestral(int r) {
        if (jumpLaw == JumpLaw.LANDING) {
            sampleAncestralLanding(r);
            return;
        }
        if (stableResults[r] != null || ancestralReferenceForced()) {
            if (stableResults[r] == null) {
                stableResults[r] = TorusLogSpacePruning.calculate(
                        tree, this::branchLength, tipData[r], invertPrecision(precisions.get(r)),
                        jumpRateValue(r), jumpW, jumpD, jumpS, sigmaEpsilonMatrix());
            }
            stableResults[r].sample(sampledAnc[r]);
            return;
        }
        sampleAncestralShift(r);
    }

    /**
     * Joint downward sample of residue r's internal angles under the SHIFT law from the cached
     * FFT partials. The stationary law of a wrapped Levy process is uniform, so the root is drawn
     * from its partial alone; each internal child is drawn given its parent cell from
     * [e^{-lambda t} WN(t Sigma) + D_t](child - parent) * childPartial.
     */
    private void sampleAncestralShift(int r) {
        final double[][] sigma = invertPrecision(precisions.get(r));
        final double jr = jumpRateValue(r);
        double[] psiRe = null, psiIm = null;
        if (jr > 0.0 && jumpS != null) {
            psiRe = new double[n2];
            psiIm = new double[n2];
            kernel.computePsiComplex(sigma, jr, jumpW, jumpD, jumpS, psiRe, psiIm);
        }
        final NodeRef root = tree.getRoot();
        final int rn = root.getNumber();
        requireUsableFftPartials(r, rn);
        java.util.Arrays.fill(sampledAnc[r], null);
        final long unresolvedBefore = kernel.getAncestralUnresolvedDraws();
        final int rootCell = kernel.sampleFromProduct(flatLeaf, partials[currentIndex[r][rn]][r][rn],
                dr.math.MathUtils.nextDouble());
        if (rootCell < 0) throw noAncestralMass("the root", r);
        sampledAnc[r][rn] = kernel.anglesAt(rootCell);
        sampleShiftChildren(r, root, rootCell, sigma, jr, psiRe, psiIm);
        checkAncestralResolution(r, unresolvedBefore);
    }

    private void sampleShiftChildren(int r, NodeRef node, int parentCell, double[][] sigma,
                                     double jr, double[] psiRe, double[] psiIm) {
        for (int c = 0; c < tree.getChildCount(node); c++) {
            final NodeRef child = tree.getChild(node, c);
            if (tree.isExternal(child)) continue;          // tips are observed, never sampled
            final int cn = child.getNumber();
            final double t = branchLength(child);
            final double[] transition = kernel.branchDiffusionKernel(sigma, t);
            final double[] jump = kernel.shiftJumpKernel(sigma, t, jr, psiRe, psiIm);
            if (jump != null) {
                final double noJump = Math.exp(-jr * t);
                for (int k = 0; k < n2; k++) transition[k] = noJump * transition[k] + jump[k];
            }
            final int cell = kernel.sampleChildCell(transition, null, parentCell,
                    partials[currentIndex[r][cn]][r][cn], dr.math.MathUtils.nextDouble());
            if (cell < 0) throw noAncestralMass("node " + cn, r);
            sampledAnc[r][cn] = kernel.anglesAt(cell);
            sampleShiftChildren(r, child, cell, sigma, jr, psiRe, psiIm);
        }
    }

    /** The FFT partials are a usable sampling basis only if no node's product underflowed
     *  (a failed node leaves a flat placeholder buffer and a -inf scale that reaches the root). */
    private void requireUsableFftPartials(int r, int rootNumber) {
        if (!Double.isFinite(nodeLogScale[r][rootNumber])) {
            throw new ArithmeticException("Ancestral sampling for residue " + r + ": the FFT partials"
                    + " are not usable (a node product underflowed at nGrid=" + kernel.getGridSize()
                    + "); enable -Dtorus.validate=true or -Dtorus.ancestral.reference=true");
        }
    }

    private ArithmeticException noAncestralMass(String where, int r) {
        return new ArithmeticException("Ancestral sampling found no positive mass at " + where
                + " for residue " + r + " at nGrid=" + kernel.getGridSize()
                + "; the grid does not resolve this conditional");
    }

    /** An unresolved draw rests on partial values below the FFT's dynamic range (J2 regime). */
    private void checkAncestralResolution(int r, long unresolvedBefore) {
        if (kernel.getAncestralUnresolvedDraws() == unresolvedBefore) return;
        final String message = "Ancestral draw for residue " + r + " relies on partial values below"
                + " the FFT's ~16-decade dynamic range at nGrid=" + kernel.getGridSize()
                + "; this state lies outside the production path's operating range";
        if (ancestralStrict()) throw new TorusLogSpacePruning.AccuracyException(message);
        if (ancestralResolutionWarning.compareAndSet(false, true)) {
            Logger.getLogger("dr.evomodel").warning("TorusFourierTreeLikelihood '" + getId() + "': "
                    + message + ". Further occurrences are counted, not logged"
                    + " (TorusFourierKernel.getAncestralUnresolvedDraws); use"
                    + " -Dtorus.ancestral.strict=true to reject them.");
        }
    }


    /** Force a fresh ancestral draw from the CURRENT partials, without re-pruning.
     *  For validation harnesses that need many draws from one posterior. */
    public void forceAncestralResample() {
        ancestorsDirty = true;
        ensureAncestors();
    }

    public TreeTrait[] getTreeTraits() {
        if (treeTraitArray == null) {
            treeTraitArray = new TreeTrait[nResidues];
            for (int r = 0; r < nResidues; r++) {
                final int rr = r;
                final String nm = (traitNames != null && traitNames[r] != null) ? traitNames[r] : ("theta_" + r);
                treeTraitArray[r] = new TreeTrait.DA() {
                    public String getTraitName() { return nm; }
                    public Intent getIntent() { return Intent.NODE; }
                    public double[] getTrait(Tree t, NodeRef node) {
                        ensureAncestors();
                        return sampledAnc[rr][node.getNumber()];
                    }
                };
            }
        }
        return treeTraitArray;
    }

    public TreeTrait getTreeTrait(String key) {
        for (TreeTrait tt : getTreeTraits()) {
            if (tt.getTraitName().equals(key)) return tt;
        }
        return null;
    }
}
