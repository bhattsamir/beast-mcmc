/*
 * TorusLandingGpuEngine.java
 *
 * @author Samir Bhatt
 */

package dr.evomodel.treedatalikelihood.continuous.gpu;

import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftType;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static jcuda.driver.JCudaDriver.*;

/**
 * The production LANDING likelihood of {@link dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood}
 * on a GPU, batched over residues. One call evaluates the log-likelihood of any set of residues, each at its
 * own diffusion covariance and jump rate, on one fixed tree, in a single pass of batched cuFFT transforms
 * and CUDA kernels (torus_landing_kernels.cu).
 *
 * <p>The arithmetic is the production engine's (commit 06e5709ad, the source of the jar that ran HX2 jobs
 * 846999 and 981654), step for step: tips are built from their closed-form Fourier coefficients and sent
 * up their branch in one inverse transform; an internal branch is a full complex forward transform, the
 * landing operator (damping, plus the reset mass summed from the undamped coefficients and added to mode
 * zero), and an inverse transform; children are multiplied into the parent in the production order with a
 * rescale by the maximum absolute value after each child; the root integral is the stationary landing law
 * against the root partial; a non-positive or non-finite root integral gives -infinity. Precision inversion
 * and the covariance checks are copied from the production classes. The only differences are in rounding:
 * cuFFT against BEAST's four1, CUDA against Java transcendental functions, and pairwise compensated against
 * sequential compensated summation.
 *
 * <p>Two production failures are reported per residue rather than thrown, so the caller decides: a non-finite
 * branch message ({@link #STATUS_NONFINITE_MESSAGE}) and a partial whose maximum is not positive
 * ({@link #STATUS_INVALID_PARTIAL}). Production throws an ArithmeticException in both cases.
 * Dimension 2 only (phi, psi).
 *
 * <p>Not thread safe. All calls must come from one thread at a time (the CUDA context is made current on
 * entry to every call).
 *
 * @author Samir Bhatt
 */
public final class TorusLandingGpuEngine implements AutoCloseable {

    public static final int STATUS_OK = 0;
    public static final int STATUS_NONFINITE_MESSAGE = 1;
    public static final int STATUS_INVALID_PARTIAL = 2;
    public static final int STATUS_ROOT_UNDEFINED = 4;

    private static final int BLOCK = 256;
    private static final int DIM = 2;
    private static final double TWO_PI = 2.0 * Math.PI;
    private static final String PTX_RESOURCE = "torus_landing_kernels.ptx";

    private final int n;              // grid points per axis
    private final int nTotal;         // n * n
    private final int nResidues;
    private final int nodeCount;
    private final int root;
    private final int maxBatch;
    private final int nblk;           // blocks per residue in the strided kernels
    private final int[][] children;   // production child order
    private final boolean[] isTip;
    private final double[] branch;    // branch length above each node (root unused)

    // schedule: internal branch or tip branch -> parent, in an order that keeps few partials alive
    private static final int OP_TIP = 0, OP_BRANCH = 1;
    private final int[] opType, opChild, opParent, opBufParent, opBufChild, opTipOrdinal;
    private final boolean[] opFirst;
    private final int rootBuffer, nBuffers;
    private final int[] tipOrdinal;   // node -> 0..nTips-1, or -1

    // host copies used by the checks and by evaluation
    private final double[] ghatRe, ghatIm;

    // device state
    private final int deviceId;
    private final CUcontext context;
    private final CUmodule module;
    private final CUfunction fTipPhases, fTipApply, fApply, fCorrection, fLoad, fCombine, fNormalize, fRootSum,
            fSumReal, fFinalize;
    private final CUdeviceptr dX, dT, dScale, dPart, dBlkMax, dBad, dOut, dOutMean, dGhat, dTipMag, dSig, dLam, dRmap;
    private final CUdeviceptr dTheta, dPresent;
    private final CUdeviceptr[] dP;
    private final CUdeviceptr nullPtr = new CUdeviceptr();
    private final Map<Integer, cufftHandle> plans = new HashMap<Integer, cufftHandle>();
    private boolean closed = false;

    // counters (all evaluations)
    private long evaluations = 0L, rootUndefined = 0L, invalidPartials = 0L, nonfiniteMessages = 0L;

    /**
     * @param n            grid points per axis (power of two >= 4), the production nGrid
     * @param children     children of every node in the production (tree.getChild) order; empty for tips
     * @param root         root node number
     * @param branchLength length of the branch above every node, already floored as production does
     *                     (TorusFourierTreeLikelihood.branchLength: max(height difference x rate, 1e-8))
     * @param tipData      [residue][node][2] observed angles in radians, null where the residue is missing at a
     *                     tip; entries for internal nodes are ignored
     * @param w            landing mixture weights (normalised by the caller exactly as the production parser does)
     * @param mu           landing mixture means (absolute angles)
     * @param s            landing mixture covariances
     * @param sigmaEps     tip observation-error covariance (2 x 2, strictly positive definite)
     * @param deviceId     CUDA device ordinal
     * @param maxBatch     largest number of residues evaluated in one pass (memory scales with it)
     */
    public TorusLandingGpuEngine(int n, int[][] children, int root, double[] branchLength, double[][][] tipData,
                                 double[] w, double[][] mu, double[][][] s, double[][] sigmaEps,
                                 int deviceId, int maxBatch) {
        if (n < 4 || (n & (n - 1)) != 0) throw new IllegalArgumentException("nGrid must be a power of two >= 4");
        if (sigmaEps == null || sigmaEps.length != DIM) throw new IllegalArgumentException("sigmaEps must be 2 x 2");
        this.n = n;
        this.nTotal = n * n;
        this.nodeCount = children.length;
        this.root = root;
        this.children = new int[nodeCount][];
        this.isTip = new boolean[nodeCount];
        for (int v = 0; v < nodeCount; v++) {
            this.children[v] = children[v].clone();
            this.isTip[v] = children[v].length == 0;
        }
        if (isTip[root]) throw new IllegalArgumentException("landing requires a tree with an internal root");
        this.branch = branchLength.clone();
        for (int v = 0; v < nodeCount; v++) {
            if (v == root) continue;
            final double t = branch[v];
            if (!(t > 0.0) || !Double.isFinite(t)) {
                throw new IllegalArgumentException("branch length above node " + v + " must be positive and finite, got "
                        + t + " (production floors it at 1e-8)");
            }
        }
        this.nResidues = tipData.length;
        if (nResidues < 1) throw new IllegalArgumentException("at least one residue");
        this.maxBatch = Math.max(1, Math.min(maxBatch, nResidues));
        this.nblk = Math.max(1, Math.min(128, nTotal / (BLOCK * 4)));
        this.deviceId = deviceId;

        // the fixed landing law, from the production kernel itself (validation included)
        final TorusFourierKernel kernel = new TorusFourierKernel(DIM, n);
        this.ghatRe = new double[nTotal];
        this.ghatIm = new double[nTotal];
        kernel.computeGhat(w, mu, s, ghatRe, ghatIm);
        requireLandingCovariance(sigmaEps, "Landing tip-error covariance");

        // tip magnitudes: production landingTipCoefficients, magnitude = (nTotal / (2 pi)^2) exp(-1/2 m'Se m)
        final double volume = Math.pow(2.0 * Math.PI, DIM);
        final double scale = (double) nTotal / volume;
        final double[] tipMag = new double[nTotal];
        final double[] m = new double[DIM];
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            tipMag[idx] = scale * Math.exp(-0.5 * quadForm(sigmaEps, m));
        }

        // tips: wrapped angles (production wraps with IEEEremainder inside the phase), presence flags
        this.tipOrdinal = new int[nodeCount];
        Arrays.fill(tipOrdinal, -1);
        int nTips = 0;
        for (int v = 0; v < nodeCount; v++) if (isTip[v]) tipOrdinal[v] = nTips++;
        final double[] theta = new double[nTips * nResidues * DIM];
        final byte[] present = new byte[nTips * nResidues];
        for (int v = 0; v < nodeCount; v++) {
            if (!isTip[v]) continue;
            final int k = tipOrdinal[v];
            for (int r = 0; r < nResidues; r++) {
                if (tipData[r].length != nodeCount && v >= tipData[r].length) {
                    throw new IllegalArgumentException("tipData residue " + r + " has no slot for tip node " + v);
                }
                final double[] y = tipData[r][v];
                if (y == null) continue;
                if (y.length != DIM) throw new IllegalArgumentException("tip angles must have 2 values");
                for (int ax = 0; ax < DIM; ax++) {
                    if (!Double.isFinite(y[ax])) throw new IllegalArgumentException("Tip angles must be finite");
                    theta[(k * nResidues + r) * DIM + ax] = Math.IEEEremainder(y[ax], TWO_PI);
                }
                present[k * nResidues + r] = 1;
            }
        }

        // schedule
        final List<int[]> ops = new ArrayList<int[]>();
        final int[] buf = new int[nodeCount];
        Arrays.fill(buf, -1);
        final int[] need = new int[nodeCount];
        computeNeed(root, need);
        final BufferPool pool = new BufferPool();
        generate(root, need, buf, pool, ops);
        final int nOps = ops.size();
        this.opType = new int[nOps];
        this.opChild = new int[nOps];
        this.opParent = new int[nOps];
        this.opBufParent = new int[nOps];
        this.opBufChild = new int[nOps];
        this.opTipOrdinal = new int[nOps];
        this.opFirst = new boolean[nOps];
        for (int i = 0; i < nOps; i++) {
            final int[] o = ops.get(i);
            opType[i] = o[0]; opChild[i] = o[1]; opParent[i] = o[2]; opFirst[i] = o[3] == 1;
            opBufParent[i] = o[4]; opBufChild[i] = o[5]; opTipOrdinal[i] = tipOrdinal[o[1]];
        }
        this.rootBuffer = buf[root];
        this.nBuffers = pool.max;

        // device
        JCudaDriver.setExceptionsEnabled(true);
        JCufft.setExceptionsEnabled(true);
        cuInit(0);
        final CUdevice device = new CUdevice();
        cuDeviceGet(device, deviceId);
        this.context = new CUcontext();
        cuDevicePrimaryCtxRetain(context, device);
        cuCtxSetCurrent(context);
        this.module = new CUmodule();
        cuModuleLoadData(module, loadPtx());
        this.fTipPhases = function("tip_phases");
        this.fTipApply = function("tip_apply");
        this.fApply = function("apply_landing");
        this.fCorrection = function("add_correction");
        this.fLoad = function("load_real");
        this.fCombine = function("combine");
        this.fNormalize = function("normalize");
        this.fRootSum = function("root_sum");
        this.fSumReal = function("sum_real");
        this.fFinalize = function("finalize_sum");

        final long B = this.maxBatch;
        this.dX = alloc(B * nTotal * 16L);
        this.dT = alloc(B * 2L * n * 16L);
        this.dP = new CUdeviceptr[nBuffers];
        for (int i = 0; i < nBuffers; i++) dP[i] = alloc(B * nTotal * 8L);
        this.dScale = alloc((long) nodeCount * B * 8L);
        this.dPart = alloc(B * nblk * 2L * 8L);
        this.dBlkMax = alloc(B * nblk * 8L);
        this.dBad = alloc(B * 4L);
        this.dOut = alloc(B * 8L);
        this.dOutMean = alloc(B * 8L);
        this.dSig = alloc(B * 4L * 8L);
        this.dLam = alloc(B * 8L);
        this.dRmap = alloc(B * 4L);
        final double[] ghat = new double[2 * nTotal];
        for (int i = 0; i < nTotal; i++) { ghat[2 * i] = ghatRe[i]; ghat[2 * i + 1] = ghatIm[i]; }
        this.dGhat = upload(ghat);
        this.dTipMag = upload(tipMag);
        this.dTheta = upload(theta.length == 0 ? new double[1] : theta);
        this.dPresent = alloc(Math.max(1, present.length));
        if (present.length > 0) cuMemcpyHtoD(dPresent, Pointer.to(present), present.length);
    }

    public int getGridSize() { return n; }
    public int getResidueCount() { return nResidues; }
    public int getMaxBatch() { return maxBatch; }
    public int getPartialBufferCount() { return nBuffers; }
    public long getEvaluationCount() { return evaluations; }
    public long getRootUndefinedCount() { return rootUndefined; }
    public long getInvalidPartialCount() { return invalidPartials; }
    public long getNonfiniteMessageCount() { return nonfiniteMessages; }

    /**
     * Log-likelihoods of the listed residues.
     *
     * @param residues residue indices (any order, repeats allowed)
     * @param sigma    [k][2][2] diffusion covariance of residues[k] (see {@link #invertPrecision})
     * @param lambda   [k] jump rate of residues[k]
     * @param logL     out: [k] log-likelihood; -infinity where the root integral is not positive and finite;
     *                 NaN where status is NONFINITE_MESSAGE or INVALID_PARTIAL
     * @param status   out (nullable): [k] STATUS_* bit set
     */
    public void logLikelihood(int[] residues, double[][][] sigma, double[] lambda, double[] logL, int[] status) {
        if (closed) throw new IllegalStateException("engine closed");
        final int k = residues.length;
        if (sigma.length != k || lambda.length != k || logL.length < k) {
            throw new IllegalArgumentException("residues, sigma, lambda and logL must have matching lengths");
        }
        for (int i = 0; i < k; i++) {
            if (residues[i] < 0 || residues[i] >= nResidues) throw new IllegalArgumentException("residue out of range");
            final double lam = lambda[i];
            // production calculateLandingLogLikelihood and requireLandingRate
            if (!Double.isFinite(lam) || lam < 0.0) {
                throw new IllegalArgumentException("landing jumpRate must be finite and nonnegative");
            }
            requireLandingCovariance(sigma[i], "Diffusion covariance");
        }
        cuCtxSetCurrent(context);
        for (int start = 0; start < k; start += maxBatch) {
            final int b = Math.min(maxBatch, k - start);
            evaluateChunk(residues, sigma, lambda, start, b, logL, status);
        }
        evaluations += k;
    }

    private void evaluateChunk(int[] residues, double[][][] sigma, double[] lambda, int start, int B, double[] logL,
                               int[] status) {
        final int[] rmap = new int[B];
        final double[] sig = new double[4 * B];
        final double[] lam = new double[B];
        boolean anyZero = false, anyPositive = false;
        for (int b = 0; b < B; b++) {
            rmap[b] = residues[start + b];
            final double[][] s = sigma[start + b];
            sig[4 * b] = s[0][0]; sig[4 * b + 1] = s[0][1]; sig[4 * b + 2] = s[1][0]; sig[4 * b + 3] = s[1][1];
            lam[b] = lambda[start + b];
            if (lam[b] == 0.0) anyZero = true; else anyPositive = true;
        }
        cuMemcpyHtoD(dRmap, Pointer.to(rmap), 4L * B);
        cuMemcpyHtoD(dSig, Pointer.to(sig), 8L * 4 * B);
        cuMemcpyHtoD(dLam, Pointer.to(lam), 8L * B);
        cuMemsetD32(dBad, 0, B);
        final cufftHandle plan = plan(B);

        for (int i = 0; i < opType.length; i++) {
            final int child = opChild[i], parent = opParent[i];
            final double t = branch[child];
            final CUdeviceptr P = dP[opBufParent[i]];
            final int first = opFirst[i] ? 1 : 0;
            final CUdeviceptr scaleParent = scaleOf(parent);
            if (opType[i] == OP_TIP) {
                final CUdeviceptr theta = (CUdeviceptr) dTheta.withByteOffset((long) opTipOrdinal[i] * nResidues * DIM * 8L);
                final CUdeviceptr present = (CUdeviceptr) dPresent.withByteOffset((long) opTipOrdinal[i] * nResidues);
                launch(fTipPhases, (n + BLOCK - 1) / BLOCK, B, Pointer.to(dT), Pointer.to(theta), Pointer.to(dRmap),
                        Pointer.to(new int[]{n}));
                launch(fTipApply, nblk, B, Pointer.to(dX), Pointer.to(dT), Pointer.to(dTipMag), Pointer.to(dGhat),
                        Pointer.to(dSig), Pointer.to(dLam), Pointer.to(new double[]{t}), Pointer.to(new int[]{n}),
                        Pointer.to(new int[]{nTotal}), Pointer.to(dPart));
                correctionAndInverse(B, plan);
                launch(fCombine, nblk, B, Pointer.to(P), Pointer.to(dX), Pointer.to(present), Pointer.to(dRmap),
                        Pointer.to(new int[]{first}), Pointer.to(new double[]{1.0 / nTotal}), Pointer.to(new int[]{nTotal}),
                        Pointer.to(dBlkMax), Pointer.to(dBad));
                launch(fNormalize, nblk, B, Pointer.to(P), Pointer.to(dBlkMax), Pointer.to(new int[]{nblk}),
                        Pointer.to(scaleParent), Pointer.to(nullPtr), Pointer.to(new int[]{first}),
                        Pointer.to(new int[]{nTotal}), Pointer.to(dBad));
            } else {
                launch(fLoad, nblk, B, Pointer.to(dX), Pointer.to(dP[opBufChild[i]]), Pointer.to(new int[]{nTotal}));
                JCufft.cufftExecZ2Z(plan, dX, dX, JCufft.CUFFT_FORWARD);
                branchOperator(t, B, plan);
                launch(fCombine, nblk, B, Pointer.to(P), Pointer.to(dX), Pointer.to(nullPtr), Pointer.to(dRmap),
                        Pointer.to(new int[]{first}), Pointer.to(new double[]{1.0 / nTotal}), Pointer.to(new int[]{nTotal}),
                        Pointer.to(dBlkMax), Pointer.to(dBad));
                launch(fNormalize, nblk, B, Pointer.to(P), Pointer.to(dBlkMax), Pointer.to(new int[]{nblk}),
                        Pointer.to(scaleParent), Pointer.to(scaleOf(child)), Pointer.to(new int[]{first}),
                        Pointer.to(new int[]{nTotal}), Pointer.to(dBad));
            }
        }

        // root integral
        final CUdeviceptr Proot = dP[rootBuffer];
        if (anyZero) {
            launch(fSumReal, nblk, B, Pointer.to(Proot), Pointer.to(new int[]{nTotal}), Pointer.to(dPart));
            launch1(fFinalize, B, Pointer.to(dPart), Pointer.to(new int[]{nblk}), Pointer.to(dOutMean));
        }
        if (anyPositive) {
            launch(fLoad, nblk, B, Pointer.to(dX), Pointer.to(Proot), Pointer.to(new int[]{nTotal}));
            JCufft.cufftExecZ2Z(plan, dX, dX, JCufft.CUFFT_FORWARD);
            launch(fRootSum, nblk, B, Pointer.to(dX), Pointer.to(dGhat), Pointer.to(dSig), Pointer.to(dLam),
                    Pointer.to(new int[]{n}), Pointer.to(new int[]{nTotal}), Pointer.to(dPart));
            launch1(fFinalize, B, Pointer.to(dPart), Pointer.to(new int[]{nblk}), Pointer.to(dOut));
        }
        final double[] sums = new double[B], means = new double[B], rootScale = new double[B];
        final int[] bad = new int[B];
        cuCtxSynchronize();
        if (anyPositive) cuMemcpyDtoH(Pointer.to(sums), dOut, 8L * B);
        if (anyZero) cuMemcpyDtoH(Pointer.to(means), dOutMean, 8L * B);
        cuMemcpyDtoH(Pointer.to(rootScale), scaleOf(root), 8L * B);
        cuMemcpyDtoH(Pointer.to(bad), dBad, 4L * B);
        for (int b = 0; b < B; b++) {
            int st = STATUS_OK;
            double ll;
            if ((bad[b] & 1) != 0) { st |= STATUS_NONFINITE_MESSAGE; nonfiniteMessages++; }
            if ((bad[b] & 2) != 0) { st |= STATUS_INVALID_PARTIAL; invalidPartials++; }
            if (st != STATUS_OK) {
                ll = Double.NaN;
            } else {
                // production rootIntegralLanding / rootIntegral: sum / nTotal
                final double rootMass = (lam[b] == 0.0 ? means[b] : sums[b]) / nTotal;
                if (rootMass > 0.0 && Double.isFinite(rootMass)) {
                    ll = rootScale[b] + Math.log(rootMass);
                } else {
                    ll = Double.NEGATIVE_INFINITY;
                    st |= STATUS_ROOT_UNDEFINED;
                    rootUndefined++;
                }
            }
            logL[start + b] = ll;
            if (status != null) status[start + b] = st;
        }
    }

    /** apply_landing + add_correction + inverse transform, on dX (internal children only). */
    private void branchOperator(double t, int B, cufftHandle plan) {
        launch(fApply, nblk, B, Pointer.to(dX), Pointer.to(dGhat), Pointer.to(dSig), Pointer.to(dLam),
                Pointer.to(new double[]{t}), Pointer.to(new int[]{n}), Pointer.to(new int[]{nTotal}), Pointer.to(dPart));
        correctionAndInverse(B, plan);
    }

    /** add_correction (reset mass to mode zero) + inverse transform, on dX. */
    private void correctionAndInverse(int B, cufftHandle plan) {
        launch1(fCorrection, B, Pointer.to(dX), Pointer.to(dPart), Pointer.to(new int[]{nblk}), Pointer.to(new int[]{nTotal}));
        JCufft.cufftExecZ2Z(plan, dX, dX, JCufft.CUFFT_INVERSE);
    }

    private CUdeviceptr scaleOf(int node) {
        return (CUdeviceptr) dScale.withByteOffset((long) node * maxBatch * 8L);
    }

    private void launch(CUfunction f, int gridX, int gridY, Pointer... args) {
        cuLaunchKernel(f, gridX, gridY, 1, BLOCK, 1, 1, 0, null, Pointer.to(args), null);
    }

    private void launch1(CUfunction f, int blocks, Pointer... args) {
        cuLaunchKernel(f, blocks, 1, 1, BLOCK, 1, 1, 0, null, Pointer.to(args), null);
    }

    private cufftHandle plan(int B) {
        cufftHandle p = plans.get(B);
        if (p == null) {
            p = new cufftHandle();
            JCufft.cufftPlanMany(p, 2, new int[]{n, n}, null, 1, nTotal, null, 1, nTotal, cufftType.CUFFT_Z2Z, B);
            plans.put(B, p);
        }
        return p;
    }

    private CUfunction function(String name) {
        final CUfunction f = new CUfunction();
        cuModuleGetFunction(f, module, name);
        return f;
    }

    private static CUdeviceptr alloc(long bytes) {
        final CUdeviceptr p = new CUdeviceptr();
        cuMemAlloc(p, bytes);
        return p;
    }

    private static CUdeviceptr upload(double[] a) {
        final CUdeviceptr p = alloc(8L * a.length);
        cuMemcpyHtoD(p, Pointer.to(a), 8L * a.length);
        return p;
    }

    private static byte[] loadPtx() {
        final String override = System.getProperty("torus.gpu.ptx");
        try (InputStream in = override != null ? new java.io.FileInputStream(override)
                : TorusLandingGpuEngine.class.getResourceAsStream(PTX_RESOURCE)) {
            if (in == null) throw new IllegalStateException("PTX resource " + PTX_RESOURCE + " not found next to "
                    + TorusLandingGpuEngine.class.getName());
            final ByteArrayOutputStream out = new ByteArrayOutputStream();
            final byte[] buffer = new byte[8192];
            int r;
            while ((r = in.read(buffer)) > 0) out.write(buffer, 0, r);
            out.write(0);                                   // cuModuleLoadData wants a NUL-terminated image
            return out.toByteArray();
        } catch (IOException e) {
            throw new IllegalStateException("cannot read PTX: " + e.getMessage(), e);
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        cuCtxSetCurrent(context);
        for (cufftHandle p : plans.values()) JCufft.cufftDestroy(p);
        plans.clear();
        final List<CUdeviceptr> all = new ArrayList<CUdeviceptr>(Arrays.asList(dX, dT, dScale, dPart, dBlkMax, dBad, dOut,
                dOutMean, dGhat, dTipMag, dSig, dLam, dRmap, dTheta, dPresent));
        all.addAll(Arrays.asList(dP));
        for (CUdeviceptr p : all) cuMemFree(p);
        cuModuleUnload(module);
        final CUdevice device = new CUdevice();
        cuDeviceGet(device, deviceId);
        cuDevicePrimaryCtxRelease(device);
    }

    // ---- schedule ------------------------------------------------------------------------------------

    /** Buffers alive at once if node's subtree is evaluated with larger subtrees first (Sethi-Ullman). */
    private int computeNeed(int v, int[] need) {
        if (isTip[v]) return need[v] = 0;
        final List<Integer> internal = new ArrayList<Integer>();
        for (int c : children[v]) if (!isTip[c]) { computeNeed(c, need); internal.add(c); }
        internal.sort((a, b) -> Integer.compare(need[b], need[a]));
        int k = 0;
        for (int i = 0; i < internal.size(); i++) k = Math.max(k, need[internal.get(i)] + i);
        // the node's own partial is alive while the remaining children are combined into it
        return need[v] = Math.max(k, internal.size()) + 1;
    }

    private void generate(int v, int[] need, int[] buf, BufferPool pool, List<int[]> ops) {
        final List<Integer> internal = new ArrayList<Integer>();
        for (int c : children[v]) if (!isTip[c]) internal.add(c);
        internal.sort((a, b) -> Integer.compare(need[b], need[a]));
        for (int c : internal) generate(c, need, buf, pool, ops);   // each leaves its partial in buf[c]
        boolean first = true;
        for (int c : children[v]) {                                  // production order for the products
            if (isTip[c]) {
                if (first) buf[v] = pool.take();
                ops.add(new int[]{OP_TIP, c, v, first ? 1 : 0, buf[v], -1});
            } else {
                // the child's partial is copied into the transform buffer before the parent is written,
                // so the parent may reuse the child's buffer
                final int childBuf = buf[c];
                pool.give(childBuf);
                if (first) buf[v] = pool.take();
                ops.add(new int[]{OP_BRANCH, c, v, first ? 1 : 0, buf[v], childBuf});
            }
            first = false;
        }
    }

    private static final class BufferPool {
        final List<Integer> free = new ArrayList<Integer>();
        int max = 0;
        int take() {
            if (!free.isEmpty()) return free.remove(free.size() - 1);
            return max++;
        }
        void give(int b) { free.add(b); }
    }

    // ---- production arithmetic, copied -----------------------------------------------------------------

    /** production TorusFourierKernel.modeOf for dim 2 */
    private void modeOf(int idx, double[] m) {
        final int[] stride = {n, 1};
        for (int ax = 0; ax < DIM; ax++) {
            final int i = (idx / stride[ax]) % n;
            m[ax] = (i < n / 2) ? i : i - n;
        }
    }

    /** production TorusFourierKernel.quadForm */
    private static double quadForm(double[][] s, double[] m) {
        double q = 0.0;
        for (int i = 0; i < DIM; i++) {
            for (int j = 0; j < DIM; j++) q += s[i][j] * m[i] * m[j];
        }
        return q;
    }

    /** production TorusFourierTreeLikelihood.invertPrecision (Gauss-Jordan with partial pivoting) */
    public static double[][] invertPrecision(double[][] p) {
        final int d = DIM;
        final double[][] a = new double[d][2 * d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) a[i][j] = p[i][j];
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

    /** production TorusFourierKernel.requireLandingCovariance: finite, symmetric, positive definite */
    static void requireLandingCovariance(double[][] covariance, String name) {
        if (covariance == null || covariance.length != DIM) {
            throw new IllegalArgumentException(name + " must be a " + DIM + " by " + DIM + " covariance");
        }
        for (int i = 0; i < DIM; i++) {
            if (covariance[i] == null || covariance[i].length != DIM) {
                throw new IllegalArgumentException(name + " row must have length " + DIM);
            }
            for (double entry : covariance[i]) {
                if (!Double.isFinite(entry)) throw new IllegalArgumentException(name + " must be finite");
            }
        }
        final double[][] lower = new double[DIM][DIM];
        for (int i = 0; i < DIM; i++) {
            for (int j = 0; j <= i; j++) {
                final double scale = Math.max(Math.max(Math.abs(covariance[i][j]), Math.abs(covariance[j][i])),
                        Math.sqrt(Math.abs(covariance[i][i])) * Math.sqrt(Math.abs(covariance[j][j])));
                if (Math.abs(covariance[i][j] - covariance[j][i]) > 1e-12 * scale) {
                    throw new IllegalArgumentException(name + " must be symmetric");
                }
                double residual = covariance[i][j];
                for (int k = 0; k < j; k++) residual -= lower[i][k] * lower[j][k];
                if (i == j) {
                    if (!(residual > 0.0) || !Double.isFinite(residual)) {
                        throw new IllegalArgumentException(name + " must be positive definite");
                    }
                    lower[i][j] = Math.sqrt(residual);
                } else {
                    lower[i][j] = residual / lower[j][j];
                }
            }
        }
    }
}
