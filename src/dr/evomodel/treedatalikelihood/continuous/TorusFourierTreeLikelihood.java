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
 * internal states jointly by a downward pass over the FFT partials (formed from the cached
 * messages), with the zero-jump half of every branch kernel evaluated exactly in real space (see
 * TorusFourierKernel, "ancestral sampling"); when a positive reference is present, as under
 * validation, draws come from that reference instead.
 *
 * <p>The message each node sends up its branch is cached per node and per residue in a
 * double-buffered array, following BEAST's standard tree-likelihood scheme (cf.
 * {@code TreeDataLikelihood}, whose flag semantics it shares: a flagged node's branch changed).
 * An MCMC move flags only the affected branches and residues, and an evaluation recomputes a
 * message iff its branch is flagged, a message below it was recomputed, or its residue changed;
 * an unchanged sibling on the path to the root is reused, not re-propagated, and a tip kernel is
 * rebuilt only when its own branch changes. A buffer flips at most once per store/restore cycle,
 * so {@code store}/{@code restore} swap buffer indices in O(nodes) and rejected proposals revert
 * without recomputing any grid, however many times the proposal was evaluated. A local tree move
 * costs O(depth) propagations; a single residue's precision move recomputes only that residue.
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
    // Shift law: the mixture's characteristic function phi(m), so psi is built from it per residue
    // instead of re-summing the mixture at every mode. Built on first use (a mixture that is never
    // used is never read, as before) and rebuilt whenever the contents of the mixture arrays, which
    // the constructor holds by reference, differ from those it was built from.
    private double[] jumpPhiRe;
    private double[] jumpPhiIm;
    private double[] jumpPhiW;                         // copies of the mixture jumpPhi was built from
    private double[][] jumpPhiD;
    private double[][][] jumpPhiS;
    private final int nResidues;
    private final int n2;
    private final int dim;
    private final int nodeCount;
    // Shared all-ones leaf message for masked (missing) tips: a residue absent at a taxon
    // (tipData[r][tip] == null) contributes a flat message, i.e. its value is marginalised out
    // exactly as internal nodes are. Read-only during the sweep, so it is safe to share.
    private final double[] flatLeaf;

    // ---- cached branch messages (double-buffered) ----
    // messages[buffer][residue][nodeNumber] is the message the node sends up its branch: for an
    // observed tip its tip kernel, for an internal node its partial (the product of its children's
    // messages, normalised so its maximum is 1) propagated across the branch. The partial's
    // pulled-out log scale, summed over the node's whole subtree, is in nodeLogScale. A masked tip
    // sends the shared flat leaf and owns no buffer; the root sends nothing.
    //
    // Partials are not stored. A node's partial is formed from its children's current messages when
    // its own message or the root value is recomputed, and on demand for ancestral sampling, always
    // with the same arithmetic in the same order, so it equals what a full recomputation gives.
    // A message is recomputed only when its branch (flag), its subtree, or its residue changed, so
    // an unchanged sibling costs one pass over the grid instead of a propagation, and a tip kernel is
    // rebuilt only when its own branch changes. The flags mean what they mean in TreeDataLikelihood:
    // updateNode[v] = the branch above v changed; tree events flag the node and its children.
    private final double[][][][] messages;             // [2][nResidues][nodeCount][n2]
    private final int[][] messageIndex;                // [nResidues][nodeCount] -> 0/1
    private final int[][] storedMessageIndex;
    // The current buffer holds the message for the node's current branch. False for the root (it
    // sends nothing) and before the first evaluation; a former root becoming a child is therefore
    // recomputed whatever events the move fired.
    private final boolean[][] messageValid;
    private final boolean[][] storedMessageValid;
    private final double[][] nodeLogScale;             // [nResidues][nodeCount]
    private final double[][] storedNodeLogScale;
    // A node's buffer index flips at most once between a store and the next store or restore,
    // as in BufferIndexHelper. A second evaluation in the same cycle (MarkovChain's full-evaluation
    // check, or any caller that evaluates twice) then rewrites the proposal's buffer in place
    // instead of flipping back onto the stored one, which a later restore would hand back corrupted.
    private final boolean[][] messageFlipped;          // [nResidues][nodeCount]

    // ---- landing hot path ----
    // Per residue, the branch-independent tables of the landing operator for its current Sigma and
    // lambda; the kernel hands back the same tables while both are bit-for-bit unchanged.
    private final TorusFourierKernel.LandingTables[] landingTables;
    // The Fourier coefficients of a landing tip depend only on its angles and Sigma_eps, so they are
    // computed once and kept, for residues r < tipCacheResidues (a memory budget, see the
    // constructor). They are checked against the bits of Sigma_eps at every evaluation: a rejected
    // Sigma_eps proposal restores the parameter without an event, so validity cannot rest on events.
    private final double[][][] tipCoefficientRe;       // [residue][node], allocated on first use
    private final double[][][] tipCoefficientIm;
    private final double[][][] tipCoefficientAngles;   // the tip angles they were built from (tipData is
                                                       // held by reference, so its contents are checked)
    private final boolean[][] tipCoefficientValid;
    private double[][] tipCoefficientSigmaEps;         // the Sigma_eps the coefficients were built for
    private final int tipCacheResidues;
    // Bytes of tip-coefficient cache reserved by every landing likelihood in this JVM (never returned).
    private static final java.util.concurrent.atomic.AtomicLong TIP_CACHE_RESERVED =
            new java.util.concurrent.atomic.AtomicLong();

    /** Heap cost of one grid: header plus payload, doubled from 256 KB up (a possible G1 humongous object). */
    private static long gridHeapBytes(int n2) {
        final long bytes = 16L + (long) Double.BYTES * n2;
        return bytes >= (256L << 10) ? 2L * bytes : bytes;
    }

    /** The process-wide tip-coefficient budget: -Dtorus.landing.tipCacheMB, else an eighth of the heap. */
    private static long tipCacheBudget(String name, long maxHeap) {
        final String requested = System.getProperty("torus.landing.tipCacheMB");
        if (requested != null) {
            try {
                final long mb = Long.parseLong(requested.trim());
                if (mb >= 0) return mb >= (Long.MAX_VALUE >> 20) ? Long.MAX_VALUE : mb << 20;
            } catch (NumberFormatException ignored) {
                // fall through to the warning
            }
            Logger.getLogger("dr.evomodel").warning("TorusFourierTreeLikelihood '" + name
                    + "': ignoring -Dtorus.landing.tipCacheMB=" + requested + " (not a nonnegative integer)");
        }
        return maxHeap / 8;
    }

    private static boolean reserveTipCache(long bytes, long budget) {
        while (true) {
            final long current = TIP_CACHE_RESERVED.get();
            if (current + bytes > budget || current + bytes < current) return false;
            if (TIP_CACHE_RESERVED.compareAndSet(current, current + bytes)) return true;
        }
    }

    // ---- branch lengths ----
    // Every branch length is computed once per evaluation, on the calling thread, into branchLengths
    // (the sweep reads them from there, so the tree and the rate model are not called from worker
    // threads). A branch whose length differs bit for bit from the one the cached messages were built
    // with is flagged, whatever events were fired: the cache is then correct even under a model that
    // changes branch lengths without saying so (ScaledTreeLengthRateModel after a tree move, or a rate
    // event whose index is a parameter index rather than a node number). Correctly signalled changes
    // are flagged already, so this changes nothing for them.
    private final double[] branchLengths;              // [nodeCount], this evaluation; NaN for the root
    private final double[] lastBranchLength;           // the lengths the current messages were built with
    private final double[] storedLastBranchLength;
    // Working arrays for one residue's sweep {partial, imaginary part, damping, reset}, pooled so the
    // branch operations allocate nothing. Results never depend on which set a sweep receives.
    private final java.util.concurrent.ConcurrentLinkedDeque<double[][]> sweepScratch =
            new java.util.concurrent.ConcurrentLinkedDeque<double[][]>();

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

        // The double-buffered message store, filled on first use: two grids for every internal node
        // and every observed tip, about twice the per-node partial store it replaced. This is the bulk
        // of the memory footprint: up to 2 * nResidues * nodeCount * n2 doubles (a node keeps its pair
        // after becoming the root). A likelihood that is evaluated but never restored uses one of each
        // pair.
        this.messages = new double[2][nResidues][nodeCount][];
        this.messageIndex = new int[nResidues][nodeCount];
        this.storedMessageIndex = new int[nResidues][nodeCount];
        this.messageValid = new boolean[nResidues][nodeCount];
        this.storedMessageValid = new boolean[nResidues][nodeCount];
        this.nodeLogScale = new double[nResidues][nodeCount];
        this.storedNodeLogScale = new double[nResidues][nodeCount];
        this.messageFlipped = new boolean[nResidues][nodeCount];
        // Heap accounting, conservative: a grid costs its header and payload, doubled from 256 KB up,
        // where G1 may store it as a humongous object in a region twice its size.
        final long gridBytes = gridHeapBytes(n2);
        final long maxHeap = Runtime.getRuntime().maxMemory();
        long messageBytes = 0L;
        final long[] observedTips = new long[nResidues];
        for (int r = 0; r < nResidues; r++) {
            for (double[] tip : tipData[r]) if (tip != null) observedTips[r]++;
            messageBytes += 2L * (observedTips[r] + tree.getInternalNodeCount()) * gridBytes;
        }
        long tipCacheBytes = 0L;
        if (jumpLaw == JumpLaw.LANDING) {
            this.landingTables = new TorusFourierKernel.LandingTables[nResidues];
            // Tip-coefficient cache: 2 grids per observed tip and residue. Residues are cached in index
            // order while (a) the message store and the cache together stay within half the maximum
            // heap and (b) the process-wide cache budget, shared by every likelihood in this JVM, is
            // not exhausted: by default an eighth of the maximum heap, or -Dtorus.landing.tipCacheMB
            // (0 turns the cache off). Reservations are never returned, so a program that keeps
            // building likelihoods soon stops caching instead of multiplying its footprint.
            final long budget = tipCacheBudget(name, maxHeap);
            int cached = 0;
            for (int r = 0; r < nResidues; r++) {
                final long bytes = observedTips[r] * 2L * gridBytes;
                if (messageBytes + tipCacheBytes + bytes > maxHeap / 2 || !reserveTipCache(bytes, budget)) break;
                tipCacheBytes += bytes;
                cached++;
            }
            this.tipCacheResidues = cached;
            this.tipCoefficientRe = new double[cached][nodeCount][];
            this.tipCoefficientIm = new double[cached][nodeCount][];
            this.tipCoefficientAngles = new double[cached][nodeCount][];
            this.tipCoefficientValid = new boolean[cached][nodeCount];
            Logger.getLogger("dr.evomodel").info("TorusFourierTreeLikelihood '" + name
                    + "': landing tip coefficients cached for " + cached + " of " + nResidues
                    + " residues (" + (tipCacheBytes >> 20) + " MB; process-wide budget " + (budget >> 20) + " MB)");
        } else {
            this.landingTables = null;
            this.tipCacheResidues = 0;
            this.tipCoefficientRe = null;
            this.tipCoefficientIm = null;
            this.tipCoefficientAngles = null;
            this.tipCoefficientValid = null;
        }
        final String footprint = "TorusFourierTreeLikelihood '" + name + "': cached branch messages need up to "
                + (messageBytes >> 20) + " MB, tip coefficients " + (tipCacheBytes >> 20) + " MB, of a "
                + (maxHeap >> 20) + " MB maximum heap";
        if (messageBytes + tipCacheBytes > maxHeap * 0.9) {
            Logger.getLogger("dr.evomodel").warning(footprint + "; the run is likely to exhaust the heap (raise -Xmx)");
        } else {
            Logger.getLogger("dr.evomodel").info(footprint);
        }
        this.lastBranchLength = new double[nodeCount];
        this.storedLastBranchLength = new double[nodeCount];
        this.branchLengths = new double[nodeCount];
        Arrays.fill(lastBranchLength, Double.NaN);

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
        prepareBranches();
        ensureJumpCharacteristic();
        boolean anyNodeDirty = false;
        for (boolean flag : updateNode) anyNodeDirty |= flag;
        final boolean nodesDirty = anyNodeDirty;
        forEachResidue(r -> shiftResidue(r, root, sigmaEps, nodesDirty, residueLogL));
        double total = 0.0;
        for (int r = 0; r < nResidues; r++) total += residueLogL[r];   // fixed-order, deterministic
        // Clear dirty flags only after the whole sweep has consumed them.
        Arrays.fill(updateNode, false);
        Arrays.fill(residueDirty, false);
        System.arraycopy(branchLengths, 0, lastBranchLength, 0, nodeCount);
        ancestorsDirty = true;          // partials refreshed -> ancestral sample is stale
        return total;
    }

    /**
     * Run body for every residue: in index order below PARALLEL_THRESHOLD, where the first exception
     * stops the sweep as it always did, and as a parallel stream at or above it. In parallel, a
     * residue's exception is held until every residue has stopped (see rethrowFirst).
     */
    private void forEachResidue(java.util.function.IntConsumer body) {
        if (nResidues < PARALLEL_THRESHOLD) {
            for (int r = 0; r < nResidues; r++) body.accept(r);
            return;
        }
        final Throwable[] failures = new Throwable[nResidues];
        java.util.stream.IntStream.range(0, nResidues).parallel().forEach(r -> {
            try {
                body.accept(r);
            } catch (RuntimeException | Error e) {
                failures[r] = e;           // rethrown below, once every residue has stopped
            }
        });
        rethrowFirst(failures);
    }

    /**
     * Rethrow the failure of the lowest-numbered residue, if any. A residue's exception is held until
     * the whole sweep has stopped, so no residue is still writing its cache when the caller sees it
     * (and, say, restores); the residue reported does not depend on thread scheduling.
     */
    private static void rethrowFirst(Throwable[] failures) {
        for (Throwable failure : failures) {
            if (failure instanceof RuntimeException) throw (RuntimeException) failure;
            if (failure instanceof Error) throw (Error) failure;
        }
    }

    /** This evaluation's branch lengths, computed once, and a flag on every branch whose length changed. */
    private void prepareBranches() {
        final NodeRef root = tree.getRoot();
        for (int i = 0; i < nodeCount; i++) {
            final NodeRef node = tree.getNode(i);
            final double t = node == root ? Double.NaN : branchLength(node);
            if (Double.doubleToRawLongBits(t) != Double.doubleToRawLongBits(lastBranchLength[i])) {
                updateNode[i] = true;
            }
            branchLengths[i] = t;
        }
    }

    /** Shift law: build the mixture's characteristic function if a residue will need its jump exponent
     *  and the table is missing or was built from different mixture contents. */
    private void ensureJumpCharacteristic() {
        if (jumpW == null || jumpD == null || jumpS == null) return;
        boolean needed = false;
        for (int r = 0; r < nResidues && !needed; r++) needed = jumpRateValue(r) > 0.0;
        if (!needed) return;
        if (jumpPhiRe != null && Arrays.equals(jumpPhiW, jumpW) && Arrays.deepEquals(jumpPhiD, jumpD)
                && Arrays.deepEquals(jumpPhiS, jumpS)) {
            return;
        }
        final double[] re = new double[n2], im = new double[n2];
        kernel.computeJumpCharacteristic(jumpW, jumpD, jumpS, re, im);
        jumpPhiW = jumpW.clone();
        jumpPhiD = new double[jumpD.length][];
        for (int k = 0; k < jumpD.length; k++) jumpPhiD[k] = jumpD[k].clone();
        jumpPhiS = new double[jumpS.length][][];
        for (int k = 0; k < jumpS.length; k++) {
            jumpPhiS[k] = new double[jumpS[k].length][];
            for (int i = 0; i < jumpS[k].length; i++) jumpPhiS[k][i] = jumpS[k][i].clone();
        }
        jumpPhiRe = re;
        jumpPhiIm = im;
    }

    /** One residue of calculateLogLikelihood (shift law). */
    private void shiftResidue(int r, NodeRef root, double[][] sigmaEps, boolean nodesDirty, double[] residueLogL) {
        final double[][] sigma = invertPrecision(precisions.get(r));
        final double jr = jumpRateValue(r);
        // Nothing flagged and this residue unchanged: every node would be a cache hit, so the
        // accepted value stands. Skip it before building its jump exponent.
        if (!nodesDirty && !residueDirty[r] && !Double.isNaN(acceptedResidueLogL[r])) {
            residueLogL[r] = acceptedResidueLogL[r];
            return;
        }
        double[] psiRe = null, psiIm = null;
        if (jr > 0.0 && jumpS != null) {     // precompute the branch-independent jump exponent once per residue
            psiRe = new double[n2];
            psiIm = new double[n2];
            if (jumpPhiRe != null) {
                kernel.computePsiFromCharacteristic(sigma, jr, jumpPhiRe, jumpPhiIm, psiRe, psiIm);
            } else {
                kernel.computePsiComplex(sigma, jr, jumpW, jumpD, jumpS, psiRe, psiIm);
            }
        }
        double[][] scratch = sweepScratch.poll();
        if (scratch == null) scratch = new double[4][n2];
        try {
            final Sweep sweep = new Sweep(r, sigma, sigmaEps, jr, psiRe, psiIm, null, scratch);
            final boolean changed = refresh(sweep, root);
            if (changed || Double.isNaN(acceptedResidueLogL[r])) {
                final int rn = root.getNumber();
                if (!changed) formPartial(r, root, sweep.partial, false, null);
                double fft = nodeLogScale[r][rn] + Math.log(kernel.rootIntegral(sweep.partial));
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
        } finally {
            sweepScratch.offer(scratch);
        }
        residueLogL[r] = acceptedResidueLogL[r];
    }

    /** One residue's evaluation context: its parameters and the sweep's working arrays. */
    private final class Sweep {
        final int r;
        final double[][] sigma;
        final double[][] sigmaEps;
        final double jumpRate;                             // lambda_r (both laws)
        final double[] psiRe, psiIm;                       // shift: jump exponent, null without jumps
        final TorusFourierKernel.LandingTables tables;     // landing only
        final double[] partial, im, damping, reset;
        // A message whose computation failed, by node: the failure surfaces when the parent's product
        // reaches that child, as it did when messages were computed inside the parent's loop. A shift
        // product that underflows first stops there, and a later child's failure never surfaces.
        final RuntimeException[] failure;

        Sweep(int r, double[][] sigma, double[][] sigmaEps, double jumpRate, double[] psiRe, double[] psiIm,
              TorusFourierKernel.LandingTables tables, double[][] scratch) {
            this.r = r;
            this.sigma = sigma;
            this.sigmaEps = sigmaEps;
            this.jumpRate = jumpRate;
            this.psiRe = psiRe;
            this.psiIm = psiIm;
            this.tables = tables;
            this.partial = scratch[0];
            this.im = scratch[1];
            this.damping = scratch[2];
            this.reset = scratch[3];
            this.failure = new RuntimeException[nodeCount];
        }
    }

    /**
     * Post-order refresh of residue r's messages in the subtree of `node`. A tip's message is
     * rebuilt when its branch is flagged; an internal node's partial is re-formed when its branch is
     * flagged or any child's message changed, and its message is then propagated; everything is
     * redone when the residue changed. At the root the partial is formed into the sweep's partial
     * array and left there for the root integral. Returns whether node's message (at the root, its
     * partial) was recomputed, which is exactly when the per-node cache used to recompute the node.
     */
    private boolean refresh(Sweep s, NodeRef node) {
        final int r = s.r, nn = node.getNumber();
        if (tree.isExternal(node)) {
            if (!(updateNode[nn] || residueDirty[r] || !messageValid[r][nn])) return false;
            final double[] y = tipData[r][nn];
            messageValid[r][nn] = false;
            if (y != null) {                            // masked tips send the flat leaf
                try {
                    tipMessage(s, node, y);
                } catch (RuntimeException e) {
                    s.failure[nn] = e;
                    return true;
                }
            }
            messageValid[r][nn] = true;
            return true;
        }
        boolean childChanged = false;
        for (int c = 0; c < tree.getChildCount(node); c++) childChanged |= refresh(s, tree.getChild(node, c));
        final boolean partialChanged = childChanged || updateNode[nn] || residueDirty[r];
        if (tree.isRoot(node)) {
            messageValid[r][nn] = false;                // the root sends nothing
            if (!partialChanged) return false;
            formPartial(r, node, s.partial, true, s.failure);
            return true;
        }
        if (!partialChanged && messageValid[r][nn]) return false;
        messageValid[r][nn] = false;
        formPartial(r, node, s.partial, true, s.failure);
        try {
            internalMessage(s, node, s.partial);
        } catch (RuntimeException e) {
            s.failure[nn] = e;
            return true;
        }
        messageValid[r][nn] = true;
        return true;
    }

    /** The buffer node's message is written to: the alternate one, once per store/restore cycle. */
    private int messageTarget(int r, int nn) {
        final int target = messageFlipped[r][nn] ? messageIndex[r][nn] : 1 - messageIndex[r][nn];
        messageIndex[r][nn] = target;
        messageFlipped[r][nn] = true;
        return target;
    }

    private double[] messageSlot(int target, int r, int nn) {
        double[] slot = messages[target][r][nn];
        if (slot == null) messages[target][r][nn] = slot = new double[n2];
        return slot;
    }

    /** The message `tip` sends up its branch, written to the tip's next buffer. */
    private void tipMessage(Sweep s, NodeRef tip, double[] y) {
        final int r = s.r, tn = tip.getNumber();
        final double t = branchLengths[tn];
        final int target = messageTarget(r, tn);
        if (jumpLaw == JumpLaw.LANDING) {
            landingTipMessage(r, tn, y, s.sigma, t, s.sigmaEps, s.tables, messageSlot(target, r, tn),
                    s.im, s.damping, s.reset);
        } else {
            // Tip observation error enters HERE and nowhere else: only terminal branches
            // carry it, and internal propagation is untouched.
            messages[target][r][tn] = s.psiRe != null
                    ? kernel.leafKernelFromPsi(y, t, s.psiRe, s.psiIm, s.sigma, s.jumpRate, s.sigmaEps)
                    : kernel.leafKernel(y, s.sigma, t, s.sigmaEps);
        }
    }

    /** The message internal `node` sends up its branch: its partial propagated across the branch. */
    private void internalMessage(Sweep s, NodeRef node, double[] partial) {
        final int r = s.r, nn = node.getNumber();
        final double t = branchLengths[nn];
        final int target = messageTarget(r, nn);
        if (jumpLaw == JumpLaw.LANDING) {
            kernel.propagateLanding(partial, s.tables, s.sigma, t, landingGhatRe, landingGhatIm,
                    messageSlot(target, r, nn), s.im, s.damping, s.reset);
        } else {
            messages[target][r][nn] = s.psiRe != null ? kernel.propagateFromPsi(partial, t, s.psiRe, s.psiIm)
                    : kernel.propagate(partial, s.sigma, t);
        }
    }

    /**
     * Form residue r's partial at internal `node` into `values`: the product of its children's current
     * messages, renormalised after each child, with the log scales pulled out, in child order. This is
     * the arithmetic the per-node cache applied when it recomputed a node. With `record`, as part of an
     * evaluation, the node's log scale, the recompute count and the landing branch audit are updated
     * and the strict landing checks apply; without it (ancestral sampling) only `values` is written.
     * Shift: a product with no positive value leaves a flat partial and a -inf log scale, as before.
     * `failure` holds the sweep's failed messages (null outside an evaluation); the first one reached,
     * in child order, is thrown, as it was when the child's message was computed at this point.
     */
    private void formPartial(int r, NodeRef node, double[] values, boolean record, RuntimeException[] failure) {
        final int nn = node.getNumber();
        Arrays.fill(values, 1.0);
        double scale = 0.0;
        for (int c = 0; c < tree.getChildCount(node); c++) {
            final NodeRef child = tree.getChild(node, c);
            final int cn = child.getNumber();
            if (failure != null && failure[cn] != null) throw failure[cn];
            final double[] msg;
            if (tree.isExternal(child)) {
                msg = tipData[r][cn] == null ? flatLeaf : messages[messageIndex[r][cn]][r][cn];
            } else {
                msg = messages[messageIndex[r][cn]][r][cn];
                scale += nodeLogScale[r][cn];           // carry the child's subtree scale
            }
            if (jumpLaw == JumpLaw.LANDING) {
                // Preserve spectral values exactly: flooring small or negative tails here
                // would silently change the model. Normalize by absolute maximum, matching
                // the standalone Fourier reference even on an under-resolved grid.
                double max = 0.0;
                long nonpositive = 0;
                double messageMin = 0.0, messageMax = 0.0;
                for (int k = 0; k < n2; k++) {
                    values[k] *= msg[k];
                    max = Math.max(max, Math.abs(values[k]));
                    if (msg[k] <= 0.0) nonpositive++;
                    messageMin = Math.min(messageMin, msg[k]);
                    messageMax = Math.max(messageMax, Math.abs(msg[k]));
                }
                if (record) {
                    landingNonpositiveByBranch[r][cn] = nonpositive;
                    // Negative lobes diagnose Fourier truncation. Whether the requested grid
                    // is adequate for inference is calibrated at the likelihood level, not
                    // decided solely by a pointwise tail sign. Never floor these values.
                    if (messageMin < -1e-12 * messageMax) {
                        if (strictLandingNumerics) {
                            throw new ArithmeticException("Under-resolved landing branch for residue " + r
                                    + ", child " + cn + ", nGrid=" + kernel.getGridSize() + ": min=" + messageMin
                                    + ", maxAbs=" + messageMax + "; verify likelihood accuracy for this parameter regime");
                        }
                        if (landingResolutionWarning.compareAndSet(false, true)) {
                            Logger.getLogger("dr.evomodel").warning("Landing likelihood at nGrid=" + kernel.getGridSize()
                                    + " has negative Fourier tail values. Values are preserved without clipping; "
                                    + "check accuracy in the intended parameter regime. "
                                    + "Use -Dtorus.landing.strict=true to reject these cases in an audit.");
                        }
                    }
                }
                if (!(max > 0.0) || !Double.isFinite(max)) {
                    throw new ArithmeticException("Invalid landing partial at residue " + r + ", node " + nn
                            + ", nGrid=" + kernel.getGridSize() + "; check grid convergence and covariance conditioning");
                }
                for (int k = 0; k < n2; k++) values[k] /= max;
                scale += Math.log(max);
            } else {
                double max = 0.0;
                for (int k = 0; k < n2; k++) {
                    values[k] *= msg[k];
                    if (values[k] > max) max = values[k];
                }
                if (!(max > 0.0)) {
                    // Mark this FFT candidate unusable. The positive reference below
                    // still evaluates the state; numerical failure is not evidence of
                    // a vanishing-probability proposal.
                    System.arraycopy(flatLeaf, 0, values, 0, n2);   // a defined (uniform) partial,
                    if (record) nodeLogScale[r][nn] = Double.NEGATIVE_INFINITY;   // never a half product
                    return;
                }
                final double inv = 1.0 / max;
                for (int k = 0; k < n2; k++) values[k] *= inv;
                scale += Math.log(max);
            }
        }
        if (record) {
            nodeLogScale[r][nn] = scale;
            nodeRecomputes.incrementAndGet();
        }
    }

    /** Landing has its own root, branch operator, and optional positive reference. */
    private double calculateLandingLogLikelihood() {
        final double[][] sigmaEps = requireLandingObservationError();
        refreshTipCoefficients(sigmaEps);
        prepareBranches();
        final NodeRef root = tree.getRoot();
        forEachResidue(r -> {
            final double[][] sigma = invertPrecision(precisions.get(r));
            final double lambda = jumpRate.getParameterValue(jumpRate.getDimension() == 1 ? 0 : r);
            if (!Double.isFinite(lambda) || lambda < 0.0) {
                throw new IllegalArgumentException("landing jumpRate must be finite and nonnegative");
            }
            double[][] scratch = sweepScratch.poll();
            if (scratch == null) scratch = new double[4][n2];
            try {
                landingResidue(r, root, sigma, lambda, sigmaEps, scratch);
            } finally {
                sweepScratch.offer(scratch);
            }
        });
        double total = 0.0;
        for (int r = 0; r < nResidues; r++) total += acceptedResidueLogL[r];
        Arrays.fill(updateNode, false);
        Arrays.fill(residueDirty, false);
        System.arraycopy(branchLengths, 0, lastBranchLength, 0, nodeCount);
        ancestorsDirty = true;
        return total;
    }

    /** Invalidate the cached tip coefficients if Sigma_eps is not bit for bit what they were built for. */
    private void refreshTipCoefficients(double[][] sigmaEps) {
        boolean same = tipCoefficientSigmaEps != null;
        for (int i = 0; i < dim && same; i++) {
            for (int j = 0; j < dim && same; j++) {
                same = Double.doubleToRawLongBits(sigmaEps[i][j])
                        == Double.doubleToRawLongBits(tipCoefficientSigmaEps[i][j]);
            }
        }
        if (same) return;
        for (int r = 0; r < tipCacheResidues; r++) Arrays.fill(tipCoefficientValid[r], false);
        tipCoefficientSigmaEps = new double[dim][];
        for (int i = 0; i < dim; i++) tipCoefficientSigmaEps[i] = sigmaEps[i].clone();
    }

    /** One residue of calculateLandingLogLikelihood, with that sweep's working arrays. */
    private void landingResidue(int r, NodeRef root, double[][] sigma, double lambda, double[][] sigmaEps,
                                double[][] scratch) {
        final TorusFourierKernel.LandingTables tables = kernel.landingTables(landingTables[r], sigma, lambda);
        landingTables[r] = tables;
        final Sweep sweep = new Sweep(r, sigma, sigmaEps, lambda, null, null, tables, scratch);
        final boolean changed = refresh(sweep, root);
        if (changed || Double.isNaN(acceptedResidueLogL[r])) {
            final int rn = root.getNumber();
            if (!changed) formPartial(r, root, sweep.partial, false, null);
            final double rootMass = kernel.rootIntegralLanding(sweep.partial, tables, sigma, landingGhatRe,
                    landingGhatIm, sweep.damping, sweep.reset);
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
    }

    /** The message a landing tip sends up its branch, into `out`, from cached coefficients where the
     *  budget allows. */
    private void landingTipMessage(int r, int tip, double[] theta, double[][] sigma, double t,
                                   double[][] sigmaEps, TorusFourierKernel.LandingTables tables,
                                   double[] out, double[] im, double[] damping, double[] reset) {
        if (r < tipCacheResidues) {
            if (!tipCoefficientValid[r][tip] || !sameBits(theta, tipCoefficientAngles[r][tip])) {
                if (tipCoefficientRe[r][tip] == null) {
                    tipCoefficientRe[r][tip] = new double[n2];
                    tipCoefficientIm[r][tip] = new double[n2];
                }
                tipCoefficientValid[r][tip] = false;
                kernel.landingTipCoefficients(theta, sigmaEps, tipCoefficientRe[r][tip], tipCoefficientIm[r][tip]);
                tipCoefficientAngles[r][tip] = theta.clone();
                tipCoefficientValid[r][tip] = true;
            }
            kernel.leafKernelLandingFromCoefficients(tipCoefficientRe[r][tip], tipCoefficientIm[r][tip],
                    tables, sigma, t, landingGhatRe, landingGhatIm, out, im, damping, reset);
            return;
        }
        kernel.landingTipCoefficients(theta, sigmaEps, out, im);
        kernel.leafKernelLandingFromCoefficients(out, im, tables, sigma, t, landingGhatRe, landingGhatIm,
                out, im, damping, reset);
    }

    private static boolean sameBits(double[] a, double[] b) {
        if (a == null || b == null || a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (Double.doubleToRawLongBits(a[i]) != Double.doubleToRawLongBits(b[i])) return false;
        }
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
        final double[] partial = new double[n2];           // each node's partial, formed when drawn
        formPartial(r, root, partial, false, null);
        final int rootCell = kernel.sampleFromProduct(stationary, partial, dr.math.MathUtils.nextDouble());
        if (rootCell < 0) throw noAncestralMass("the root", r);
        sampledAnc[r][rootNumber] = kernel.anglesAt(rootCell);
        sampleLandingChildren(r, root, rootCell, sigma, lambda, partial);
        checkAncestralResolution(r, unresolvedBefore);
    }

    /** Draw every internal descendant of `node`, given that `node` was sampled into `parentCell`.
     *  `partial` is working space; a child's partial is only needed until the child is drawn. */
    private void sampleLandingChildren(int r, NodeRef node, int parentCell,
                                       double[][] sigma, double lambda, double[] partial) {
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
            formPartial(r, child, partial, false, null);
            final int cell = kernel.sampleChildCell(diffusion, reset, parentCell, partial,
                    dr.math.MathUtils.nextDouble());
            if (cell < 0) throw noAncestralMass("node " + cn, r);
            sampledAnc[r][cn] = kernel.anglesAt(cell);
            sampleLandingChildren(r, child, cell, sigma, lambda, partial);
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
        // A precision matrix may be shared by several residues (an XML idref): every residue holding
        // it has changed, not only the first.
        boolean precision = false;
        for (int r = 0; r < nResidues; r++) {
            if (precisions.get(r) == variable) {
                residueDirty[r] = true;  // this residue's diffusion changed
                precision = true;
            }
        }
        if (precision) {
            // done
        } else if (variable == jumpRate && jumpRate.getDimension() == nResidues && index >= 0) {
            residueDirty[index] = true;  // per-residue lambda: only that residue's rate changed
        } else {
            // shared lambda, sigmaEpsilon (shared by every residue), bulk change (index<0), or anything else
            Arrays.fill(residueDirty, true);
        }
        likelihoodKnown = false;
        ancestorsDirty = true;
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
        System.arraycopy(lastBranchLength, 0, storedLastBranchLength, 0, nodeCount);
        for (int r = 0; r < nResidues; r++) {
            System.arraycopy(messageIndex[r], 0, storedMessageIndex[r], 0, nodeCount);
            System.arraycopy(messageValid[r], 0, storedMessageValid[r], 0, nodeCount);
            System.arraycopy(nodeLogScale[r], 0, storedNodeLogScale[r], 0, nodeCount);
            Arrays.fill(messageFlipped[r], false);
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
        System.arraycopy(storedLastBranchLength, 0, lastBranchLength, 0, nodeCount);
        // Swap the buffer indices and scales back; the stored messages were never overwritten
        // (a recompute writes the alternate buffer, at most one flip per cycle), so this reverts
        // every recomputed branch in O(nodes) without touching the grids.
        for (int r = 0; r < nResidues; r++) {
            System.arraycopy(storedMessageIndex[r], 0, messageIndex[r], 0, nodeCount);
            System.arraycopy(storedMessageValid[r], 0, messageValid[r], 0, nodeCount);
            System.arraycopy(storedNodeLogScale[r], 0, nodeLogScale[r], 0, nodeCount);
            Arrays.fill(messageFlipped[r], false);
        }
        ancestorsDirty = true;          // reverted messages -> ancestral sample is stale
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
        final double[] partial = new double[n2];           // each node's partial, formed when drawn
        formPartial(r, root, partial, false, null);
        final int rootCell = kernel.sampleFromProduct(flatLeaf, partial, dr.math.MathUtils.nextDouble());
        if (rootCell < 0) throw noAncestralMass("the root", r);
        sampledAnc[r][rn] = kernel.anglesAt(rootCell);
        sampleShiftChildren(r, root, rootCell, sigma, jr, psiRe, psiIm, partial);
        checkAncestralResolution(r, unresolvedBefore);
    }

    private void sampleShiftChildren(int r, NodeRef node, int parentCell, double[][] sigma,
                                     double jr, double[] psiRe, double[] psiIm, double[] partial) {
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
            formPartial(r, child, partial, false, null);
            final int cell = kernel.sampleChildCell(transition, null, parentCell, partial,
                    dr.math.MathUtils.nextDouble());
            if (cell < 0) throw noAncestralMass("node " + cn, r);
            sampledAnc[r][cn] = kernel.anglesAt(cell);
            sampleShiftChildren(r, child, cell, sigma, jr, psiRe, psiIm, partial);
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
