/*
 * TorusLandingGpuLikelihood.java
 *
 * @author Samir Bhatt
 */

package dr.evomodel.treedatalikelihood.continuous.gpu;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.tree.TreeChangedEvent;
import dr.inference.model.AbstractModelLikelihood;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Model;
import dr.inference.model.Parameter;
import dr.inference.model.Variable;
import dr.xml.Reportable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Logger;

/**
 * The landing likelihood of {@link dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood}
 * (jumpLaw = landing) for many residues on one fixed tree, computed on one or more GPUs by
 * {@link TorusLandingGpuEngine}. Same model, same parameters (one precision matrix per residue, a jump rate
 * of dimension 1 or one per residue), same value: the production classes are the reference in every test.
 *
 * <p>Several devices: residues are split into contiguous blocks, one engine per device, evaluated in parallel
 * threads. Each residue's value does not depend on which engine computes it beyond rounding.
 *
 * <p>Caching is by value, not by events. A residue's log-likelihood is a function of five numbers: its four
 * precision entries and its jump rate. A value is reused only if those numbers are
 * bit-identical to the ones it was computed at, so no sequence of parameter events, stores and restores can
 * serve a stale value. Residues whose inputs did change are evaluated together.
 *
 * <p>{@link #evaluateProposals} evaluates residues at their CURRENT parameter values without making them the
 * accepted state, and remembers the results by value (used by {@link TorusResidueBlockOperator}).
 *
 * <p><b>Combined model</b> (optional): an additive jump rate and an additive jump size (each of dimension 1 or one
 * per residue; zero-mean isotropic wrapped-normal displacements of standard deviation s_J radians) add additive jumps
 * to the landing jumps. Each Fourier mode's decay rate gains lambda_A (1 - exp(-1/2 s_J^2 |m|^2)); the stationary
 * root and the landing resets follow (paper Appendix B, Prop. 1). The value key is then seven numbers. Without these
 * two parameters the class is the landing likelihood, arithmetic unchanged.
 *
 * <p>Production failures are kept: a non-finite branch message or a partial without a positive maximum throws
 * ArithmeticException from {@link #getLogLikelihood}, as production does, and a non-positive root integral
 * gives -infinity. The tree must not change (fixed-tree analyses only), and Sigma_eps is fixed.
 *
 * @author Samir Bhatt
 */
public class TorusLandingGpuLikelihood extends AbstractModelLikelihood implements Reportable {

    public static final String TORUS_LANDING_GPU_LIKELIHOOD = "torusLandingGpuLikelihood";
    private static final double MIN_BRANCH = 1.0e-8;    // production TorusFourierTreeLikelihood.MIN_BRANCH
    private final int KEY;                              // P00, P01, P10, P11, lambda [, lambda_A, s_J]

    private final Tree tree;
    private final List<MatrixParameter> precisions;
    private final Parameter jumpRate;
    private final Parameter additiveRate;               // nullable (landing model); with additiveSize
    private final Parameter additiveSize;               // jump SD s_J, radians
    private final TorusLandingGpuEngine[] engines;
    private final int[] engineOf;                       // residue -> engine
    private final int[] localIndex;                     // residue -> index within its engine
    private final ExecutorService pool;                 // one thread per engine (null for one engine)
    private final int nResidues;
    private final double[] branchAtConstruction;
    private final Parameter sigmaEpsilonParameter;      // nullable; fixed for the life of the engines

    private final double[] residueLogL;
    private final double[] storedResidueLogL;
    private final double[][] key;
    private final double[][] storedKey;
    private final boolean[] keyValid;
    private final boolean[] storedKeyValid;

    private final double[][] proposalKey;
    private final double[] proposalLogL;
    private final boolean[] proposalValid;

    private double logLikelihood;
    private double storedLogLikelihood;
    private boolean likelihoodKnown = false;
    private boolean storedLikelihoodKnown = false;

    private long residueEvaluations = 0L, valueCacheHits = 0L, proposalCacheHits = 0L;

    /**
     * @param tipData   [residue][external node number][2] angles in radians, null where missing
     * @param w         landing mixture weights, already normalised (production parser: w / sum w)
     * @param devices   CUDA device ordinals; residues are split across them
     */
    public TorusLandingGpuLikelihood(String name, Tree tree, List<MatrixParameter> precisions, double[][][] tipData,
                                     Parameter jumpRate, double[] w, double[][] mu, double[][][] s,
                                     double[][] sigmaEps, Parameter sigmaEpsilonParameter,
                                     int nGrid, int[] devices, int maxBatch) {
        this(name, tree, precisions, tipData, jumpRate, null, null, w, mu, s, sigmaEps, sigmaEpsilonParameter, nGrid,
                devices, maxBatch);
    }

    /** Combined model: additiveRate and additiveSize both null (landing) or both given. */
    public TorusLandingGpuLikelihood(String name, Tree tree, List<MatrixParameter> precisions, double[][][] tipData,
                                     Parameter jumpRate, Parameter additiveRate, Parameter additiveSize,
                                     double[] w, double[][] mu, double[][][] s,
                                     double[][] sigmaEps, Parameter sigmaEpsilonParameter,
                                     int nGrid, int[] devices, int maxBatch) {
        super(name);
        this.tree = tree;
        this.precisions = precisions;
        this.jumpRate = jumpRate;
        this.additiveRate = additiveRate;
        this.additiveSize = additiveSize;
        this.nResidues = precisions.size();
        this.sigmaEpsilonParameter = sigmaEpsilonParameter;
        if (jumpRate == null || (jumpRate.getDimension() != 1 && jumpRate.getDimension() != nResidues)) {
            throw new IllegalArgumentException("landing requires jumpRate with dimension 1 or nResidues");
        }
        if ((additiveRate == null) != (additiveSize == null)) {
            throw new IllegalArgumentException("additiveRate and additiveSize go together");
        }
        if (additiveRate != null) {
            for (Parameter q : new Parameter[]{additiveRate, additiveSize}) {
                if (q.getDimension() != 1 && q.getDimension() != nResidues) {
                    throw new IllegalArgumentException("additive parameters need dimension 1 or nResidues ("
                            + nResidues + "), got " + q.getDimension() + " for " + q.getId());
                }
            }
        }
        this.KEY = additiveRate == null ? 5 : 7;
        for (int r = 0; r < nResidues; r++) {
            final MatrixParameter p = precisions.get(r);
            if (p.getRowDimension() != 2 || p.getColumnDimension() != 2) {
                throw new IllegalArgumentException("precision " + r + " must be 2x2");
            }
            if (tipData[r].length != tree.getExternalNodeCount()) {
                throw new IllegalArgumentException("tipData residue " + r + " has " + tipData[r].length
                        + " taxa, tree has " + tree.getExternalNodeCount());
            }
        }
        if (devices == null || devices.length < 1) throw new IllegalArgumentException("at least one device");
        final int nodeCount = tree.getNodeCount();
        final int[][] children = new int[nodeCount][];
        for (int i = 0; i < nodeCount; i++) {
            final NodeRef node = tree.getNode(i);
            children[i] = new int[tree.getChildCount(node)];
            for (int c = 0; c < children[i].length; c++) children[i][c] = tree.getChild(node, c).getNumber();
        }
        this.branchAtConstruction = branchLengths();

        // contiguous blocks of residues, one per device
        final int E = Math.min(devices.length, nResidues);
        this.engines = new TorusLandingGpuEngine[E];
        this.engineOf = new int[nResidues];
        this.localIndex = new int[nResidues];
        int start = 0;
        for (int e = 0; e < E; e++) {
            final int size = nResidues / E + (e < nResidues % E ? 1 : 0);
            final double[][][] block = Arrays.copyOfRange(tipData, start, start + size);
            for (int k = 0; k < size; k++) { engineOf[start + k] = e; localIndex[start + k] = k; }
            engines[e] = new TorusLandingGpuEngine(nGrid, children, tree.getRoot().getNumber(), branchAtConstruction,
                    block, w, mu, s, sigmaEps, devices[e], Math.min(maxBatch, size));
            start += size;
        }
        this.pool = E > 1 ? Executors.newFixedThreadPool(E, r -> {
            final Thread t = new Thread(r, "torus-gpu");
            t.setDaemon(true);
            return t;
        }) : null;

        this.residueLogL = new double[nResidues];
        this.storedResidueLogL = new double[nResidues];
        this.key = new double[nResidues][KEY];
        this.storedKey = new double[nResidues][KEY];
        this.keyValid = new boolean[nResidues];
        this.storedKeyValid = new boolean[nResidues];
        this.proposalKey = new double[nResidues][KEY];
        this.proposalLogL = new double[nResidues];
        this.proposalValid = new boolean[nResidues];

        if (tree instanceof Model) addModel((Model) tree);
        for (MatrixParameter p : precisions) addVariable(p);
        addVariable(jumpRate);
        if (additiveRate != null) { addVariable(additiveRate); addVariable(additiveSize); }
        if (sigmaEpsilonParameter != null) addVariable(sigmaEpsilonParameter);
        final StringBuilder dev = new StringBuilder();
        for (int e = 0; e < E; e++) dev.append(e == 0 ? "" : ",").append(devices[e]);
        Logger.getLogger("dr.evomodel").info("TorusLandingGpuLikelihood '" + name + "': "
                + (additiveRate == null ? "landing model, " : "COMBINED model (landing + additive jumps, rate '"
                + additiveRate.getId() + "', size '" + additiveSize.getId() + "'), ") + nResidues
                + " residues, nGrid=" + nGrid + ", devices " + dev + ", " + engines[0].getPartialBufferCount()
                + " partial buffers; SIGMA_EPS APPLIED, tip observation-error covariance in rad^2 = [" + sigmaEps[0][0] + " "
                + sigmaEps[0][1] + "; " + sigmaEps[1][0] + " " + sigmaEps[1][1] + "]  (per-axis sd in degrees: "
                + String.format("%.4f %.4f", Math.toDegrees(Math.sqrt(sigmaEps[0][0])), Math.toDegrees(Math.sqrt(sigmaEps[1][1])))
                + ")");
    }

    public int getResidueCount() { return nResidues; }
    public MatrixParameter getPrecision(int r) { return precisions.get(r); }
    public Parameter getJumpRate() { return jumpRate; }
    public boolean isCombined() { return additiveRate != null; }
    public Parameter getAdditiveRate() { return additiveRate; }
    public Parameter getAdditiveSize() { return additiveSize; }
    public TorusLandingGpuEngine[] getEngines() { return engines.clone(); }
    public long getResidueEvaluationCount() { return residueEvaluations; }
    public long getValueCacheHits() { return valueCacheHits; }
    public long getProposalCacheHits() { return proposalCacheHits; }

    // ---- Likelihood ------------------------------------------------------------------------------------

    public Model getModel() { return this; }

    public double getLogLikelihood() {
        if (!likelihoodKnown) {
            logLikelihood = calculateLogLikelihood();
            likelihoodKnown = true;
        }
        return logLikelihood;
    }

    /** Current accepted log-likelihood of one residue (evaluates if needed). */
    public double getResidueLogLikelihood(int r) {
        getLogLikelihood();
        return residueLogL[r];
    }

    public void makeDirty() {
        likelihoodKnown = false;
        Arrays.fill(keyValid, false);
        Arrays.fill(proposalValid, false);
    }

    private double calculateLogLikelihood() {
        int[] batch = new int[nResidues];
        int nb = 0;
        final double[][] cur = new double[nResidues][];
        for (int r = 0; r < nResidues; r++) {
            cur[r] = currentKey(r);
            if (keyValid[r] && Arrays.equals(cur[r], key[r])) {
                valueCacheHits++;
            } else if (proposalValid[r] && Arrays.equals(cur[r], proposalKey[r])) {
                residueLogL[r] = proposalLogL[r];
                System.arraycopy(cur[r], 0, key[r], 0, KEY);
                keyValid[r] = true;
                proposalCacheHits++;
            } else {
                batch[nb++] = r;
            }
        }
        if (nb > 0) {
            batch = Arrays.copyOf(batch, nb);
            final double[] ll = new double[nb];
            final int[] status = new int[nb];
            evaluate(batch, cur, ll, status);
            for (int i = 0; i < nb; i++) {
                final int r = batch[i];
                throwProductionFailure(r, status[i]);
                residueLogL[r] = ll[i];
                System.arraycopy(cur[r], 0, key[r], 0, KEY);
                keyValid[r] = true;
            }
        }
        double total = 0.0;
        for (int r = 0; r < nResidues; r++) total += residueLogL[r];   // fixed order, as production
        return total;
    }

    /**
     * Log-likelihoods of the given residues at their CURRENT parameter values, without making those values
     * the accepted state. Failures are returned, not thrown: logL is NaN and status has
     * {@link TorusLandingGpuEngine#STATUS_NONFINITE_MESSAGE} or {@link TorusLandingGpuEngine#STATUS_INVALID_PARTIAL}
     * where production would have thrown, and -infinity with STATUS_ROOT_UNDEFINED where production returns
     * -infinity.
     */
    public void evaluateProposals(int[] residues, double[] logL, int[] status) {
        final double[][] cur = new double[nResidues][];
        for (int r : residues) cur[r] = currentKey(r);
        evaluate(residues, cur, logL, status);
        for (int i = 0; i < residues.length; i++) {
            final int r = residues[i];
            System.arraycopy(cur[r], 0, proposalKey[r], 0, KEY);
            proposalLogL[r] = logL[i];
            proposalValid[r] = status[i] == TorusLandingGpuEngine.STATUS_OK
                    || status[i] == TorusLandingGpuEngine.STATUS_ROOT_UNDEFINED;
        }
    }

    /** Evaluate residues (global indices) at the given keys, split across the engines, in parallel. */
    private void evaluate(int[] residues, double[][] cur, double[] logL, int[] status) {
        checkTreeUnchanged();
        final int k = residues.length;
        final int E = engines.length;
        final int[] count = new int[E];
        for (int r : residues) count[engineOf[r]]++;
        final int[][] pos = new int[E][];                 // positions in 'residues' handled by engine e
        for (int e = 0; e < E; e++) pos[e] = new int[count[e]];
        final int[] fill = new int[E];
        for (int i = 0; i < k; i++) { final int e = engineOf[residues[i]]; pos[e][fill[e]++] = i; }
        final List<Future<?>> jobs = new ArrayList<Future<?>>();
        for (int e = 0; e < E; e++) {
            if (count[e] == 0) continue;
            final int ee = e;
            final Runnable job = () -> {
                final int m = count[ee];
                final int[] local = new int[m];
                final double[][][] sigma = new double[m][][];
                final double[] lambda = new double[m];
                final double[] lamA = additiveRate == null ? null : new double[m];
                final double[] sJ = additiveRate == null ? null : new double[m];
                for (int j = 0; j < m; j++) {
                    final int r = residues[pos[ee][j]];
                    final double[] c = cur[r];
                    local[j] = localIndex[r];
                    sigma[j] = TorusLandingGpuEngine.invertPrecision(new double[][]{{c[0], c[1]}, {c[2], c[3]}});
                    lambda[j] = c[4];
                    if (lamA != null) { lamA[j] = c[5]; sJ[j] = c[6]; }
                }
                final double[] ll = new double[m];
                final int[] st = new int[m];
                engines[ee].logLikelihood(local, sigma, lambda, lamA, sJ, ll, st);
                for (int j = 0; j < m; j++) { logL[pos[ee][j]] = ll[j]; status[pos[ee][j]] = st[j]; }
            };
            if (pool == null) job.run(); else jobs.add(pool.submit(job));
        }
        for (Future<?> f : jobs) {
            try {
                f.get();
            } catch (InterruptedException | ExecutionException ex) {
                final Throwable c = ex.getCause() == null ? ex : ex.getCause();
                if (c instanceof RuntimeException) throw (RuntimeException) c;
                throw new RuntimeException(c);
            }
        }
        residueEvaluations += k;
    }

    /** The five (combined model: seven) numbers a residue's log-likelihood depends on, read as production reads them. */
    private double[] currentKey(int r) {
        final MatrixParameter p = precisions.get(r);
        if (additiveRate == null) {
            return new double[]{p.getParameterValue(0, 0), p.getParameterValue(0, 1), p.getParameterValue(1, 0),
                    p.getParameterValue(1, 1), jumpRate.getParameterValue(jumpRate.getDimension() == 1 ? 0 : r)};
        }
        return new double[]{p.getParameterValue(0, 0), p.getParameterValue(0, 1), p.getParameterValue(1, 0),
                p.getParameterValue(1, 1), jumpRate.getParameterValue(jumpRate.getDimension() == 1 ? 0 : r),
                additiveRate.getParameterValue(additiveRate.getDimension() == 1 ? 0 : r),
                additiveSize.getParameterValue(additiveSize.getDimension() == 1 ? 0 : r)};
    }

    private static void throwProductionFailure(int r, int status) {
        if ((status & TorusLandingGpuEngine.STATUS_NONFINITE_MESSAGE) != 0) {
            throw new ArithmeticException("Non-finite landing Fourier grid value (residue " + r + ")");
        }
        if ((status & TorusLandingGpuEngine.STATUS_INVALID_PARTIAL) != 0) {
            throw new ArithmeticException("Invalid landing partial at residue " + r
                    + "; check grid convergence and covariance conditioning");
        }
    }

    /** production TorusFourierTreeLikelihood.branchLength without a branch-rate model */
    private double[] branchLengths() {
        final double[] b = new double[tree.getNodeCount()];
        for (int i = 0; i < b.length; i++) {
            final NodeRef node = tree.getNode(i);
            if (tree.isRoot(node)) continue;
            final double bl = tree.getNodeHeight(tree.getParent(node)) - tree.getNodeHeight(node);
            b[i] = bl > MIN_BRANCH ? bl : MIN_BRANCH;
        }
        return b;
    }

    /** Branch length above every node (by node number), as the engines use them. */
    public double[] getBranchLengths() { return branchAtConstruction.clone(); }

    private void checkTreeUnchanged() {
        if (!Arrays.equals(branchLengths(), branchAtConstruction)) {
            throw new UnsupportedOperationException(TORUS_LANDING_GPU_LIKELIHOOD
                    + " supports a fixed tree only; the branch lengths have changed");
        }
    }

    // ---- model bookkeeping: events only clear the 'known' flag; values decide what is recomputed -----

    protected void handleModelChangedEvent(Model model, Object object, int index) {
        if (model == tree && object instanceof TreeChangedEvent) checkTreeUnchanged();
        likelihoodKnown = false;
    }

    protected void handleVariableChangedEvent(Variable variable, int index, Parameter.ChangeType type) {
        if (variable == sigmaEpsilonParameter) {
            throw new UnsupportedOperationException(TORUS_LANDING_GPU_LIKELIHOOD + ": Sigma_eps is fixed (built into the "
                    + "GPU tip coefficients); it must not be sampled");
        }
        likelihoodKnown = false;
    }

    protected void storeState() {
        storedLogLikelihood = logLikelihood;
        storedLikelihoodKnown = likelihoodKnown;
        System.arraycopy(residueLogL, 0, storedResidueLogL, 0, nResidues);
        System.arraycopy(keyValid, 0, storedKeyValid, 0, nResidues);
        for (int r = 0; r < nResidues; r++) System.arraycopy(key[r], 0, storedKey[r], 0, KEY);
    }

    protected void restoreState() {
        logLikelihood = storedLogLikelihood;
        likelihoodKnown = storedLikelihoodKnown;
        System.arraycopy(storedResidueLogL, 0, residueLogL, 0, nResidues);
        System.arraycopy(storedKeyValid, 0, keyValid, 0, nResidues);
        for (int r = 0; r < nResidues; r++) System.arraycopy(storedKey[r], 0, key[r], 0, KEY);
    }

    protected void acceptState() { }

    public String getReport() {
        long undefined = 0L;
        for (TorusLandingGpuEngine e : engines) undefined += e.getRootUndefinedCount();
        return getClass().getSimpleName() + ": logL = " + getLogLikelihood() + ", residues " + nResidues + " on "
                + engines.length + " device(s), GPU residue evaluations " + residueEvaluations + ", value-cache hits "
                + valueCacheHits + ", proposal-cache hits " + proposalCacheHits + ", root undefined " + undefined;
    }
}
