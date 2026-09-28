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

/**
 * Felsenstein pruning likelihood for independent d-dimensional wrapped-Brownian
 * jump-diffusions (one per residue) on a flat torus. Production uses FFT propagation
 * and split wrapped-normal tips. Positive log-space validation is opt-in; analytic
 * HMC gradients target the production FFT calculation only.
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
    private static final double MIN_BRANCH = 1.0e-8;   // floor for zero-length branches (replicate tips)
    private static final int PARALLEL_THRESHOLD = 8;   // parallelise the residue loop at/above this many residues

    private final Tree tree;
    private final BranchRateModel branchRateModel;     // nullable -> rate 1
    private final List<MatrixParameter> precisions;    // one dim x dim precision per residue
    private final double[][][] tipData;                // [residue][externalNodeNumber][dim], radians
    private final TorusFourierKernel kernel;
    // Compound-Poisson jump, shared across residues: sampled rate lambda (jumpRate) and a FIXED
    // directed mixture of wrapped-normal components {weight w_k, displacement d_k, covariance S_k}
    // from the Ramachandran-basin fit. Nullable; jumpRate <= 0 or jumpS == null gives pure BM.
    // d_k != 0 makes the per-mode multiplier complex (handled by the kernel's complex path).
    private final Parameter jumpRate;
    private final double[] jumpW;      // mixture weights, length K
    private final double[][] jumpD;    // displacements, K x dim
    private final double[][][] jumpS;  // covariances, K x dim x dim
    private final int nResidues;
    private final int n2;
    private final int dim;
    private final int nodeCount;
    // Shared all-ones leaf message for entirely missing tips: a residue absent at a taxon
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
    // During a proposal, preserve the stored buffer even when HMC evaluates many
    // intermediate parameter values. All writes use the other buffer until accept/restore.
    private boolean stateStored = false;

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

    private final TorusLogSpacePruning.Result[] stableResults;
    private final TorusLogSpacePruning.Result[] storedStableResults;
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
        super(name);
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
        this.nodeCount = tree.getNodeCount();
        this.stableResults = new TorusLogSpacePruning.Result[nResidues];
        this.storedStableResults = new TorusLogSpacePruning.Result[nResidues];
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
            for (double[] tip : tipData[r]) {
                if (tip == null) continue;
                if (tip.length != dim) throw new IllegalArgumentException("Tip observation dimension mismatch");
                for (double value : tip) if (Double.isInfinite(value))
                    throw new IllegalArgumentException("Tip angles must be finite or NaN for missing coordinates");
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
        return getClass().getSimpleName() + ": logL = " + getLogLikelihood();
    }

    public int getResidueCount() { return nResidues; }

    public MatrixParameter getResiduePrecision(int residue) { return precisions.get(residue); }

    public Parameter getJumpRateParameter() { return jumpRate; }

    /**
     * Derivatives of one residue's production log likelihood with respect to its
     * two precision diagonals, precision correlation, and jump rate, in that order.
     * This read-only sweep does not alter the likelihood/ancestral caches. The
     * rate derivative is residue-specific: a shared-rate sampler would need to
     * sum this component across residues.
     */
    public double[] getResidueGradient(int residue) {
        if (validateEveryEvaluation) {
            throw new IllegalStateException("Torus HMC gradients require torus.validate=false; "
                    + "they differentiate the FFT likelihood, not adaptive reference quadrature");
        }
        if (dim != 2 || !(precisions.get(residue) instanceof CompoundSymmetricMatrix)) {
            throw new IllegalArgumentException("Torus HMC requires a 2x2 compoundSymmetricMatrix precision");
        }
        CompoundSymmetricMatrix precision = (CompoundSymmetricMatrix) precisions.get(residue);
        if (!precision.asCorrelation() || precision.isCholesky()) {
            throw new IllegalArgumentException("Torus HMC requires asCorrelation=true and isCholesky=false");
        }
        double q1 = precision.getDiagonalParameter().getParameterValue(0);
        double q2 = precision.getDiagonalParameter().getParameterValue(1);
        double rho = precision.getOffDiagonalParameter().getParameterValue(0);
        if (!(q1 > 0.0) || !(q2 > 0.0) || !Double.isFinite(q1) || !Double.isFinite(q2)
                || !(Math.abs(rho) < 1.0)) {
            throw new ArithmeticException("Invalid precision in torus gradient");
        }
        final double[][] sigma = invertPrecision(precision);
        final double rate = jumpRateValue(residue);
        if (!Double.isFinite(rate) || rate < 0.0) {
            throw new ArithmeticException("Invalid jump rate in torus gradient");
        }
        final double[][][] dPrecision = new double[4][2][2];
        dPrecision[0][0][0] = 1.0;
        dPrecision[1][1][1] = 1.0;
        dPrecision[0][0][1] = dPrecision[0][1][0] = 0.5 * rho * Math.sqrt(q2 / q1);
        dPrecision[1][0][1] = dPrecision[1][1][0] = 0.5 * rho * Math.sqrt(q1 / q2);
        dPrecision[2][0][1] = dPrecision[2][1][0] = Math.sqrt(q1) * Math.sqrt(q2);
        final double[][][] dSigma = new double[4][2][2];
        // d(Q^-1) = -Q^-1 (dQ) Q^-1.
        for (int p = 0; p < 3; p++) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int a = 0; a < 2; a++) {
                        for (int b = 0; b < 2; b++) {
                            dSigma[p][i][j] -= sigma[i][a] * dPrecision[p][a][b] * sigma[b][j];
                        }
                    }
                }
            }
        }
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
            if (!Double.isFinite(sigma[i][j]))
                throw new ArithmeticException("Covariance overflow in torus gradient");
            for (int p = 0; p < 4; p++) if (!Double.isFinite(dSigma[p][i][j]))
                throw new ArithmeticException("Covariance derivative overflow in torus gradient");
        }
        TorusFourierKernel.GradientContext context = kernel.prepareGradientContext(
                sigma, rate, jumpW, jumpD, jumpS,
                dSigma, new double[]{0.0, 0.0, 0.0, 1.0});
        GradientPartial root = gradientPartial(residue, tree.getRoot(), context);
        double total = 0.0;
        for (double v : root.values) total += v;
        if (!(total > 0.0) || !Double.isFinite(total)) {
            throw new ArithmeticException("Nonfinite torus likelihood in gradient evaluation");
        }
        double[] result = root.logScaleGradient.clone();
        for (int p = 0; p < result.length; p++) {
            double derivative = 0.0;
            for (double v : root.gradients[p]) derivative += v;
            result[p] += derivative / total;
            if (!Double.isFinite(result[p])) {
                throw new ArithmeticException("Nonfinite torus likelihood gradient");
            }
        }
        return result;
    }

    private static final class GradientPartial {
        final double[] values;
        final double[][] gradients;
        final double[] logScaleGradient = new double[4];

        GradientPartial(int size) {
            values = new double[size];
            Arrays.fill(values, 1.0);
            gradients = new double[4][size];
        }
    }

    /** Forward differentiation, with the same per-child normalization as updateNodePartial. */
    private GradientPartial gradientPartial(int residue, NodeRef node,
                                            TorusFourierKernel.GradientContext context) {
        GradientPartial partial = new GradientPartial(n2);
        for (int c = 0; c < tree.getChildCount(node); c++) {
            NodeRef child = tree.getChild(node, c);
            TorusFourierKernel.ValueGradient message;
            if (tree.isExternal(child)) {
                double[] observation = tipData[residue][child.getNumber()];
                if (observation == null) continue; // neutral message and zero derivative
                message = kernel.leafKernelWithGradient(observation, branchLength(child), context);
            } else {
                GradientPartial childPartial = gradientPartial(residue, child, context);
                message = kernel.propagateWithGradient(childPartial.values, childPartial.gradients,
                        branchLength(child), context);
                for (int p = 0; p < 4; p++) {
                    partial.logScaleGradient[p] += childPartial.logScaleGradient[p];
                }
            }
            double max = 0.0;
            int peakIndex = 0;
            for (int k = 0; k < n2; k++) {
                final double previous = partial.values[k];
                for (int p = 0; p < 4; p++) {
                    partial.gradients[p][k] = partial.gradients[p][k] * message.values[k]
                            + previous * message.gradients[p][k];
                }
                partial.values[k] = previous * message.values[k];
                if (partial.values[k] > max) {
                    max = partial.values[k];
                    peakIndex = k;
                }
            }
            if (!(max > 0.0) || !Double.isFinite(max)) {
                throw new ArithmeticException("Torus subtree underflow or overflow in gradient evaluation");
            }
            for (int p = 0; p < 4; p++) {
                double dLogScale = partial.gradients[p][peakIndex] / max;
                partial.logScaleGradient[p] += dLogScale;
                for (int k = 0; k < n2; k++) {
                    partial.gradients[p][k] = partial.gradients[p][k] / max
                            - (partial.values[k] / max) * dLogScale;
                }
            }
            for (int k = 0; k < n2; k++) partial.values[k] /= max;
        }
        return partial;
    }

    private double calculateLogLikelihood() {
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
            boolean changed = updateNodePartial(r, root, sigma, psiRe, psiIm, jr);
            if (changed || Double.isNaN(acceptedResidueLogL[r])) {
                final int rn = root.getNumber();
                double fft = nodeLogScale[r][rn]
                        + Math.log(kernel.rootIntegral(partials[currentIndex[r][rn]][r][rn]));
                if (validateEveryEvaluation) {
                    // A smooth-looking tip or a positive FFT result does not certify
                    // the pruning tail. Validate the completed likelihood itself.
                    TorusLogSpacePruning.Result stable = TorusLogSpacePruning.calculate(
                            tree, this::branchLength, tipData[r], sigma, jr, jumpW, jumpD, jumpS);
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
                                      double jumpRate) {
        final int nn = node.getNumber();
        final int childCount = tree.getChildCount(node);

        boolean childRecomputed = false;
        for (int c = 0; c < childCount; c++) {
            final NodeRef child = tree.getChild(node, c);
            if (!tree.isExternal(child)) {
                childRecomputed |= updateNodePartial(r, child, sigma, psiRe, psiIm, jumpRate);
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

        // Every intermediate HMC evaluation writes the same proposal buffer. The
        // node is rebuilt from its children, so its previous proposal value is not
        // needed; the stored state must remain intact until accept/restore.
        final int writeIndex = 1 - (stateStored ? storedCurrentIndex[r][nn] : currentIndex[r][nn]);
        final double[] values = partials[writeIndex][r][nn];
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
                    msg = hasJump ? kernel.leafKernelFromPsi(y, bl, psiRe, psiIm, sigma, jumpRate)
                                  : kernel.leafKernel(y, sigma, bl);
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
                currentIndex[r][nn] = writeIndex; // never a half-accumulated product
                return true;
            }
            final double inv = 1.0 / max;
            for (int k = 0; k < n2; k++) values[k] *= inv;
            scale += Math.log(max);
        }
        nodeLogScale[r][nn] = scale;
        currentIndex[r][nn] = writeIndex;
        nodeRecomputes.incrementAndGet();
        return true;
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
                throw new ArithmeticException("Singular precision matrix");
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
            updateAllNodes();   // a rate change rescales every branch
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
            Arrays.fill(residueDirty, true);   // shared lambda, bulk change (index<0), or anything else
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
        }
        stateStored = true;
    }

    protected void restoreState() {
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
        }
        ancestorsDirty = true;          // reverted partials -> ancestral sample is stale
        stateStored = false;
    }

    protected void acceptState() { stateStored = false; }

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
        // Always sample from the accepted positive quadrature, including directed
        // jumps. FFT tail clipping must not leak into ancestral conditionals. With
        // per-evaluation validation off, that reference is built here on demand, so its
        // cost is paid only when ancestral states are actually requested.
        if (stableResults[r] == null) {
            stableResults[r] = TorusLogSpacePruning.calculate(
                    tree, this::branchLength, tipData[r], invertPrecision(precisions.get(r)),
                    jumpRateValue(r), jumpW, jumpD, jumpS);
        }
        stableResults[r].sample(sampledAnc[r]);
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
