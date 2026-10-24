/*
 * TorusLandingCountEngine.java
 *
 * @author Samir Bhatt
 */

package dr.evomodel.treedatalikelihood.continuous.gpu;

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
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

import static jcuda.driver.JCudaDriver.*;

/**
 * Exact expected numbers of landing jumps between coarse conformational regions, per branch, for one residue at
 * a time on one GPU: the estimand and numerics of results_pipeline/labelled_counts.py (LabelledCounts.counts),
 * step for step as its validated JAX port results_pipeline/labelled_counts_gpu.py does them.
 *
 * <ul>
 *   <li>The model (Lambda = quad(Sigma, m) + lambda, mu = lambda ghat / Lambda), the landing transform ghat and
 *   the tip damping are computed on the host with numpy's expressions and operation order. The modes of the
 *   half-plane representative set (plus the Nyquist row and column) are sorted by Lambda on the host with a
 *   stable sort, as numpy's argsort(kind="stable").</li>
 *   <li>The device runs the inside (pruning) pass, keeping every propagated message Lt and every internal node's
 *   coefficients c, then the pre-order outside pass: per branch the outside density W, its integral against Lt
 *   (Lb), the series coefficients AR and BR, the closed-form parts, the kept-mode compaction and the counts of
 *   kept modes below every breakpoint candidate.</li>
 *   <li>The host replays labelled_counts._cross's panel plan on those numbers, with the CPU's breakpoint
 *   candidates t/2^k, t - t/2^k and numpy's Gauss-Legendre nodes and weights (exact bits), and the device sums the
 *   double sum over mode pairs on every panel.</li>
 * </ul>
 * Differences from the numpy engine are round-off: cuFFT against pocketfft, CUDA and Java against libm
 * transcendental functions, reduction order. Acceptance is check_labelled_counts_gpu.py's (tree totals, branch
 * totals and changes within 1e-9 of the tree total, log L within 1e-9).
 *
 * <p>The count runs in its own CUDA context (cuCtxCreate), never the primary context the likelihood engine uses,
 * and shares no memory, module or FFT plan with it. Not thread safe: one thread at a time.
 *
 * @author Samir Bhatt
 */
public final class TorusLandingCountEngine implements AutoCloseable {

    public static final int MAX_REGIONS = 6;             // 3 coarse states or the 6 landing basins (kernels exist for 3 and 6)
    public static final double C0 = 40.0, C1 = 1.0;      // labelled_counts.C0, C1

    public static final int STATUS_OK = 0;
    public static final int STATUS_NONFINITE = 1;        // a non-finite value reached the count (logL, Lb or N)
    public static final int STATUS_JMAX = 512;           // a branch needs more breakpoints than JMAX

    private static final int BLOCK = 256, TILE = 1024, NQMAX = 32, KMAX = MAX_REGIONS;
    private static final int JMAX = 48, NC = 2 * JMAX - 1, PMAX = 2 * JMAX;
    private static final int CHUNK = 4096;               // kept modes per cross-term work unit
    private static final double TWO_PI = 2.0 * Math.PI;
    private static final double TP2 = TWO_PI * TWO_PI;   // numpy: TWO_PI ** 2
    private static final String PTX_RESOURCE = "torus_count_kernels.ptx";

    // numpy.polynomial.legendre.leggauss(nq) nodes and weights, exact bits (numpy 2.2.6)
    private static final long[][] GL_X_BITS = {
            {0xbfedd6ca4e80a01eL, 0xbfe528a09655c95eL, 0xbfce8b12d03675c6L, 0x3fce8b12d03675c6L, 0x3fe528a09655c95eL, 0x3fedd6ca4e80a01eL},
            {0xbfeebab1cb0acc66L, 0xbfe97e4ab249f41eL, 0xbfe0d129583284b4L, 0xbfc77ac94f3c7344L, 0x3fc77ac94f3c7344L, 0x3fe0d129583284b4L, 0x3fe97e4ab249f41eL, 0x3feebab1cb0acc66L},
            {0xbfef68f1d8e42e81L, 0xbfecee874ffb88b3L, 0xbfe8a30aeed88f36L, 0xbfe2cb4f05c077f9L, 0xbfd78a8d20a8b19dL, 0xbfc007a5f8f630e4L, 0x3fc007a5f8f630e4L, 0x3fd78a8d20a8b19dL, 0x3fe2cb4f05c077f9L, 0x3fe8a30aeed88f36L, 0x3fecee874ffb88b3L, 0x3fef68f1d8e42e81L},
            {0xbfefa92c264d787eL, 0xbfee39f56616f9b0L, 0xbfebb3403514e483L, 0xbfe82c45dda4726bL, 0xbfe3c5a466d5e8b8L, 0xbfdd50259a43a772L, 0xbfd205cae642337cL, 0xbfb852bd6676a9faL, 0x3fb852bd6676a9faL, 0x3fd205cae642337cL, 0x3fdd50259a43a772L, 0x3fe3c5a466d5e8b8L, 0x3fe82c45dda4726bL, 0x3febb3403514e483L, 0x3fee39f56616f9b0L, 0x3fefa92c264d787eL},
            {0xbfefd892de691982L, 0xbfef30f9f0cbf876L, 0xbfee06585a70aa4dL, 0xbfec5d841864d0f4L, 0xbfea3d74ce0d3700L, 0xbfe7af18edb9ddd6L, 0xbfe4bd2ee5fa1086L, 0xbfe17417bac4d72bL, 0xbfdbc345d81e24b5L, 0xbfd429a8c588e910L, 0xbfc8769542b94f8dL, 0xbfb0660853eda2e8L, 0x3fb0660853eda2e8L, 0x3fc8769542b94f8dL, 0x3fd429a8c588e910L, 0x3fdbc345d81e24b5L, 0x3fe17417bac4d72bL, 0x3fe4bd2ee5fa1086L, 0x3fe7af18edb9ddd6L, 0x3fea3d74ce0d3700L, 0x3fec5d841864d0f4L, 0x3fee06585a70aa4dL, 0x3fef30f9f0cbf876L, 0x3fefd892de691982L},
            {0xbfefe995e70409b6L, 0xbfef8a212714bcdcL, 0xbfeedf5518053baaL, 0xbfedeac0259f7f42L, 0xbfecaea9b4574cbaL, 0xbfeb2e04fd686a13L, 0xbfe96c69481c4bc5L, 0xbfe76e0931d693baL, 0xbfe537a89c487f8aL, 0xbfe2ce9146962ca4L, 0xbfe038862866b29dL, 0xbfdaf76b57c6f8f1L, 0xbfd53d55ce57bdf6L, 0xbfcea0f7e19c094bL, 0xbfc27e0ea717f237L, 0xbfa8bbc8488cc498L, 0x3fa8bbc8488cc498L, 0x3fc27e0ea717f237L, 0x3fcea0f7e19c094bL, 0x3fd53d55ce57bdf6L, 0x3fdaf76b57c6f8f1L, 0x3fe038862866b29dL, 0x3fe2ce9146962ca4L, 0x3fe537a89c487f8aL, 0x3fe76e0931d693baL, 0x3fe96c69481c4bc5L, 0x3feb2e04fd686a13L, 0x3fecaea9b4574cbaL, 0x3fedeac0259f7f42L, 0x3feedf5518053baaL, 0x3fef8a212714bcdcL, 0x3fefe995e70409b6L},
    };
    private static final long[][] GL_W_BITS = {
            {0x3fc5edf601e2dbe2L, 0x3fd716b7b5794c22L, 0x3fddf24d499545eeL, 0x3fddf24d499545eeL, 0x3fd716b7b5794c22L, 0x3fc5edf601e2dbe2L},
            {0x3fb9ea1d04ca0393L, 0x3fcc76fb531d2b91L, 0x3fd413c50a255611L, 0x3fd736360b19933fL, 0x3fd736360b19933fL, 0x3fd413c50a255611L, 0x3fcc76fb531d2b91L, 0x3fb9ea1d04ca0393L},
            {0x3fa8275d9dea6dabL, 0x3fbb60602bce61a2L, 0x3fc47d7258f22d8bL, 0x3fca0163e6b1ab68L, 0x3fcde3155c256aafL, 0x3fcfe40ce6d4f022L, 0x3fcfe40ce6d4f022L, 0x3fcde3155c256aafL, 0x3fca0163e6b1ab68L, 0x3fc47d7258f22d8bL, 0x3fbb60602bce61a2L, 0x3fa8275d9dea6dabL},
            {0x3f9bcddab4b7c200L, 0x3fafdfb1a2c12642L, 0x3fb85c4ee79cc24aL, 0x3fbfe7af2bad3875L, 0x3fc325f61bca3cc0L, 0x3fc5a6ebbb5a75ffL, 0x3fc75f8c77e0c010L, 0x3fc83feae80e4dffL, 0x3fc83feae80e4dffL, 0x3fc75f8c77e0c010L, 0x3fc5a6ebbb5a75ffL, 0x3fc325f61bca3cc0L, 0x3fbfe7af2bad3875L, 0x3fb85c4ee79cc24aL, 0x3fafdfb1a2c12642L, 0x3f9bcddab4b7c200L},
            {0x3f89465bd31121c3L, 0x3f9d375514486f34L, 0x3fa6ab884f57c954L, 0x3fae5c6255d25ed4L, 0x3fb2c6d5c2eff06cL, 0x3fb6108ef504463bL, 0x3fb8fd8936444b23L, 0x3fbb8177ba4a68daL, 0x3fbd91c78acb1b2eL, 0x3fbf25cbce1d1ff8L, 0x3fc01b7117cf8bd8L, 0x3fc060475e763738L, 0x3fc060475e763738L, 0x3fc01b7117cf8bd8L, 0x3fbf25cbce1d1ff8L, 0x3fbd91c78acb1b2eL, 0x3fbb8177ba4a68daL, 0x3fb8fd8936444b23L, 0x3fb6108ef504463bL, 0x3fb2c6d5c2eff06cL, 0x3fae5c6255d25ed4L, 0x3fa6ab884f57c954L, 0x3f9d375514486f34L, 0x3f89465bd31121c3L},
            {0x3f7cbf8bc743c8c4L, 0x3f90aa3c2486971eL, 0x3f9a0060a8532064L, 0x3fa18c5800a355f5L, 0x3fa5ee963a335470L, 0x3faa1c6ae961fc04L, 0x3fae0bd76c924998L, 0x3fb0d9b9a62cac09L, 0x3fb2854103b35e08L, 0x3fb40483e126fd14L, 0x3fb553ee25ebebcdL, 0x3fb6705e18e13ed8L, 0x3fb7572bdb3f6e4fL, 0x3fb8062fc0f6fef9L, 0x3fb87bc776f8c6d5L, 0x3fb8b6d9eaec77aeL, 0x3fb8b6d9eaec77aeL, 0x3fb87bc776f8c6d5L, 0x3fb8062fc0f6fef9L, 0x3fb7572bdb3f6e4fL, 0x3fb6705e18e13ed8L, 0x3fb553ee25ebebcdL, 0x3fb40483e126fd14L, 0x3fb2854103b35e08L, 0x3fb0d9b9a62cac09L, 0x3fae0bd76c924998L, 0x3faa1c6ae961fc04L, 0x3fa5ee963a335470L, 0x3fa18c5800a355f5L, 0x3f9a0060a8532064L, 0x3f90aa3c2486971eL, 0x3f7cbf8bc743c8c4L},
    };
    private static final int[] GL_ORDERS = {6, 8, 12, 16, 24, 32};
    private static final double[][] GL_X = new double[GL_ORDERS.length][], GL_W = new double[GL_ORDERS.length][];

    static {
        for (int g = 0; g < GL_ORDERS.length; g++) {
            if (GL_X_BITS[g].length != GL_ORDERS[g] || GL_W_BITS[g].length != GL_ORDERS[g]) throw new AssertionError("GL table");
            GL_X[g] = new double[GL_ORDERS[g]];
            GL_W[g] = new double[GL_ORDERS[g]];
            for (int j = 0; j < GL_ORDERS[g]; j++) {
                GL_X[g][j] = Double.longBitsToDouble(GL_X_BITS[g][j]);
                GL_W[g][j] = Double.longBitsToDouble(GL_W_BITS[g][j]);
            }
        }
    }

    /** One residue's count. N[v][a][b]: branch above node v (root row zero), pre-jump region a, landing region b. */
    public static final class Result {
        public double[][][] N;
        public double logL, k0;
        public long nPanels, nTerms;
        public int status;
        public double secInside, secTotal;
        /** host-side breakdown (seconds): model + sort, outside launches, waits at the plan sync, host plan, cross */
        public double secModel, secLaunch, secSync, secPlan, secCross, secUpload, secKernel, secWait;
    }

    private final int n, N, S, nodeCount, root, nResidues, ntile, nb, nbs, maxUnits;
    private final long SP;
    private final double h2;
    private final int[][] children;
    private final boolean[] isTip;
    private final double[] branch;
    private final double[][][] tips;                     // [residue][node] -> (phi, psi) radians, null if missing
    private final int[] post, pre;
    private final int[] selIdx;                          // representatives + Nyquist, ascending flat index
    private final double[] selW;
    private final int K;                                 // regions: the partition the counts are labelled by
    private final double[] G;                            // g(R_b)
    private final int[] cslot;                           // node -> slot in the coefficient store, or -1
    private final int nOut;                              // outside densities alive at once (pool size)
    private final double[][] candVal;                    // [node][NC]: t/2^k (k = 1..JMAX), t - t/2^k (k = 2..JMAX)
    private final List<Map<Double, Integer>> candIndex = new ArrayList<Map<Double, Integer>>();
    private final double c0;
    private int cur;                                     // the residue being counted

    private final int deviceId;
    private final CUcontext context;
    private final CUmodule module;
    private final Map<String, CUfunction> fn = new HashMap<String, CUfunction>();
    private final cufftHandle plan;
    private final List<CUdeviceptr> allocated = new ArrayList<CUdeviceptr>();
    private final CUdeviceptr dGhat, dDamp, dRho, dGam, dG, dFlat, dOnes;
    private final CUdeviceptr dLam, dMu, dIdx, dWts, dLs, dMus, dRhoS, dGamG;
    private final CUdeviceptr dLt, dC, dOut, dProd, dW, dXw, dXc, dXo;
    private final CUdeviceptr dAR, dBR, dFlags, dTileCnt, dTileOff, dTot, dLcA, dRcA, dLcB, dRcB;
    private final CUdeviceptr dPart, dScal, dOpScale, dThr, dPlan, dRB;
    private final CUdeviceptr dUp, dUnits, dNodes, dUfirst, dUcount, dWs, dNq, dPartial, dAlpha, dXall;
    private final java.nio.ByteBuffer up;               // host image of dUp (one upload per branch)
    private final CUdeviceptr nullPtr = new CUdeviceptr();
    private boolean closed = false;

    // scalar slots in dScal
    private static final int SC_SC = 0, SC_DOT = 1, SC_ZB = 2, SC_SUMS = 4, SC_TAU = SC_SUMS + 4 * KMAX,
            SC_L = SC_TAU + 2, SC_PI = SC_L + 2, NSCAL = SC_PI + 2 * KMAX;
    // the per-branch readback region dRB: plan of side A, plan of side B, the scalars, Wh(0) (one copy per branch)
    private static final int PL = 2 + 2 * NC, RB_SCAL = 2 * PL, RB_WH0 = RB_SCAL + NSCAL + (NSCAL & 1), RB_N = RB_WH0 + 2;
    // the per-branch upload dUp: Gauss-Legendre nodes, weights, unit ranges, orders, then the work units (int4)
    private static final int UP_NODES = 0, UP_WS = UP_NODES + 8 * PMAX * 2 * NQMAX, UP_UFIRST = UP_WS + 8 * PMAX * NQMAX,
            UP_UCOUNT = UP_UFIRST + 4 * PMAX * 2, UP_NQ = UP_UCOUNT + 4 * PMAX * 2, UP_UNITS = UP_NQ + 4 * PMAX;

    /**
     * @param n        grid points per axis (power of two)
     * @param children children of every node, in the order the products are formed
     * @param root     root node
     * @param branch   branch length above every node, floored as the likelihood floors it (1e-8)
     * @param tipData  [residue][node][2] observed angles in radians, null where missing (internal entries ignored)
     * @param w        landing mixture weights, normalised
     * @param mu       landing mixture means
     * @param s        landing mixture covariances
     * @param sigmaEps tip observation-error covariance
     * @param frac     [K][n * n] share of every grid cell (theta_j = 2 pi j / n, axis 0 = phi) in each region
     * @param c0       labelled_counts C0 (keep threshold of a panel), normally 40
     * @param deviceId CUDA ordinal
     */
    public TorusLandingCountEngine(int n, int[][] children, int root, double[] branch, double[][][] tipData,
                                   double[] w, double[][] mu, double[][][] s, double[][] sigmaEps, double[][] frac,
                                   double c0, int deviceId) {
        if (n < 8 || (n & (n - 1)) != 0) throw new IllegalArgumentException("grid must be a power of two >= 8");
        if (frac.length != 3 && frac.length != 6) throw new IllegalArgumentException("need 3 or 6 region share arrays, got " + frac.length);
        this.K = frac.length;
        this.G = new double[K];
        this.n = n;
        this.N = n * n;
        this.h2 = (TWO_PI / n) * (TWO_PI / n);           // numpy: h = TWO_PI / n; h * h
        this.c0 = c0;
        this.nodeCount = children.length;
        this.root = root;
        this.children = new int[nodeCount][];
        this.isTip = new boolean[nodeCount];
        for (int v = 0; v < nodeCount; v++) {
            this.children[v] = children[v].clone();
            isTip[v] = children[v].length == 0;
        }
        this.branch = branch.clone();
        for (int v = 0; v < nodeCount; v++) {
            if (v != root && !(branch[v] > 0.0 && Double.isFinite(branch[v]))) {
                throw new IllegalArgumentException("branch above node " + v + " must be positive and finite: " + branch[v]);
            }
        }
        this.nResidues = tipData.length;
        this.tips = new double[nResidues][nodeCount][];
        for (int r = 0; r < nResidues; r++) {
            for (int v = 0; v < nodeCount; v++) {
                if (!isTip[v]) continue;
                final double[] y = tipData[r][v];
                if (y != null) {
                    if (y.length != 2 || !Double.isFinite(y[0]) || !Double.isFinite(y[1])) {
                        throw new IllegalArgumentException("tip angles must be two finite numbers (residue " + r + ", node " + v + ")");
                    }
                    tips[r][v] = y.clone();
                }
            }
        }
        // labelled_counts._post_order and its reverse
        final List<Integer> out = new ArrayList<Integer>();
        final Deque<Integer> st = new ArrayDeque<Integer>();
        st.push(root);
        while (!st.isEmpty()) {
            final int v = st.pop();
            out.add(v);
            for (int c : this.children[v]) st.push(c);
        }
        this.pre = new int[nodeCount];
        this.post = new int[nodeCount];
        for (int i = 0; i < nodeCount; i++) { pre[i] = out.get(i); post[i] = out.get(nodeCount - 1 - i); }
        if (out.size() != nodeCount) throw new IllegalArgumentException("children do not form one tree from the root");

        // the representative set (labelled_counts: rep | nyq), ascending flat index
        final int[] sel = new int[N];
        final double[] sw = new double[N];
        int ns = 0;
        for (int i = 0; i < N; i++) {
            final int m1 = mode(i / n), m2 = mode(i % n);
            final boolean nyq = m1 == -(n / 2) || m2 == -(n / 2);
            final boolean rep = (m2 > 0 || (m2 == 0 && m1 > 0)) && !nyq;
            if (rep || nyq) { sel[ns] = i; sw[ns] = nyq ? 1.0 : 2.0; ns++; }
        }
        this.S = ns;
        this.selIdx = Arrays.copyOf(sel, ns);
        this.selW = Arrays.copyOf(sw, ns);
        this.ntile = (S + TILE - 1) / TILE;
        this.SP = (long) ntile * TILE;
        this.nb = Math.max(1, Math.min(1024, (N + BLOCK - 1) / BLOCK));
        this.nbs = Math.max(1, Math.min(1024, (S + BLOCK - 1) / BLOCK));

        // ghat and the tip damping, numpy's expressions (labelled_counts.__init__, _quad)
        final double[] ghat = new double[2 * N];
        final double[] damp = new double[N];
        for (int i = 0; i < N; i++) {
            final double m1 = mode(i / n), m2 = mode(i % n);
            double gr = 0.0, gi = 0.0;
            for (int k = 0; k < w.length; k++) {
                final double arg = m1 * mu[k][0] + m2 * mu[k][1];
                final double e = Math.exp(-quad(s[k], m1, m2));
                gr += w[k] * (e * Math.cos(arg));
                gi += w[k] * (e * Math.sin(arg));
            }
            ghat[2 * i] = gr;
            ghat[2 * i + 1] = gi;
            damp[i] = Math.exp(-quad(sigmaEps, m1, m2));
        }
        for (int i = 0; i < N; i++) {
            double t = 0.0;
            for (int a = 0; a < K; a++) t += frac[a][i];
            if (Math.abs(t - 1.0) > 1e-12) throw new IllegalArgumentException("region shares do not partition cell " + i);
        }

        // coefficient-store slots (internal non-root nodes) and the outside-density pool size
        this.cslot = new int[nodeCount];
        Arrays.fill(cslot, -1);
        int nc = 0;
        for (int v = 0; v < nodeCount; v++) if (!isTip[v] && v != root) cslot[v] = nc++;
        int live = 1, maxLive = 1;                       // Out[root]
        for (int u : pre) {
            if (isTip[u]) continue;
            for (int v : this.children[u]) if (!isTip[v]) { live++; maxLive = Math.max(maxLive, live); }
            live--;
        }
        this.nOut = maxLive;

        // breakpoint candidates and their c0 thresholds (labelled_counts_gpu.LabelledCountsGPU.cand / _thresholds)
        this.candVal = new double[nodeCount][];
        final double[] thr = new double[nodeCount * 2 * NC];
        Arrays.fill(thr, Double.POSITIVE_INFINITY);
        for (int v = 0; v < nodeCount; v++) {
            final Map<Double, Integer> m = new HashMap<Double, Integer>();
            candIndex.add(m);
            if (v == root) continue;
            final double t = branch[v];
            final double[] cv = new double[NC];
            for (int k = 1; k <= JMAX; k++) cv[k - 1] = t / Math.scalb(1.0, k);
            for (int k = 2; k <= JMAX; k++) cv[JMAX + k - 2] = t - t / Math.scalb(1.0, k);
            for (int i = 0; i < NC; i++) {
                if (m.put(cv[i], i) != null) throw new IllegalArgumentException("branch " + v + ": candidates collide");
                thr[(v * 2) * NC + i] = c0 / cv[i];
                thr[(v * 2 + 1) * NC + i] = c0 / (t - cv[i]);
            }
            candVal[v] = cv;
        }
        this.maxUnits = PMAX * 2 * ((S + CHUNK - 1) / CHUNK + 1);

        // ---- device
        this.deviceId = deviceId;
        JCudaDriver.setExceptionsEnabled(true);
        JCufft.setExceptionsEnabled(true);
        cuInit(0);
        final CUdevice device = new CUdevice();
        cuDeviceGet(device, deviceId);
        final CUcontext previous = new CUcontext();
        cuCtxGetCurrent(previous);
        this.context = new CUcontext();
        cuCtxCreate(context, 0, device);                 // its own context: no state shared with the likelihood
        this.module = new CUmodule();
        cuModuleLoadData(module, loadPtx());
        this.plan = new cufftHandle();
        JCufft.cufftPlanMany(plan, 2, new int[]{n, n}, null, 1, N, null, 1, N, cufftType.CUFFT_Z2Z, 1);

        final long G8 = 8L * N, C16 = 16L * N;
        dGhat = upload(ghat);
        dDamp = upload(damp);
        dRho = alloc(K * C16);
        dGam = alloc(K * C16);
        dG = alloc(8L * K);
        final double[] flat = new double[2 * N];
        flat[0] = TP2;
        dFlat = upload(flat);
        final double[] ones = new double[N];
        Arrays.fill(ones, 1.0);
        dOnes = upload(ones);
        dLam = alloc(G8);
        dMu = alloc(C16);
        dIdx = alloc(4L * S);
        dWts = alloc(8L * S);
        dLs = alloc(8L * S);
        dMus = alloc(16L * S);
        dRhoS = alloc(16L * K * S);
        dGamG = alloc(16L * K * S);
        dLt = alloc(G8 * nodeCount);
        dC = alloc(C16 * Math.max(1, nc));
        dOut = alloc(G8 * nOut);
        dProd = alloc(G8);
        dW = alloc(G8);
        dXw = alloc(C16);
        dXc = alloc(C16);
        dXo = alloc(C16);
        dAR = alloc(8L * K * S);
        dBR = alloc(8L * K * S);
        dFlags = alloc(4L * SP);
        dTileCnt = alloc(4L * ntile);
        dTileOff = alloc(4L * ntile);
        dTot = alloc(4L * 2);
        dLcA = alloc(8L * SP);
        dRcA = alloc(8L * K * SP);
        dLcB = alloc(8L * SP);
        dRcB = alloc(8L * K * SP);
        dPart = alloc(8L * 4 * KMAX * Math.max(nb, nbs));
        dRB = alloc(8L * RB_N);
        dPlan = dRB;
        dScal = off(dRB, 8L * RB_SCAL);
        dOpScale = alloc(8L * nodeCount);
        dThr = upload(thr);
        if (UP_UNITS % 16 != 0) throw new AssertionError("upload layout");
        dUp = alloc(UP_UNITS + 16L * maxUnits);
        dNodes = off(dUp, UP_NODES);
        dWs = off(dUp, UP_WS);
        dUfirst = off(dUp, UP_UFIRST);
        dUcount = off(dUp, UP_UCOUNT);
        dNq = off(dUp, UP_NQ);
        dUnits = off(dUp, UP_UNITS);
        up = java.nio.ByteBuffer.allocateDirect(UP_UNITS + 16 * maxUnits).order(java.nio.ByteOrder.nativeOrder());
        dPartial = alloc(8L * maxUnits * NQMAX * KMAX);
        dAlpha = alloc(8L * PMAX * 2 * NQMAX * KMAX);
        dXall = alloc(8L * K * K * nodeCount);

        // residue-invariant region tables: rho_a = h2 FFT(frac_a), g_grid = Re FFT(ghat) / TP2,
        // gam_b = TP2 IFFT(frac_b g_grid) / N, G_b = Re gam_b(0)          (labelled_counts.__init__)
        final CUdeviceptr gGrid = alloc(G8);
        k("k_copy_c", nb, dXo, dGhat, (long) N);
        fft(dXo, JCufft.CUFFT_FORWARD);
        k("k_c2r_div", nb, gGrid, dXo, TP2, (long) N);
        final CUdeviceptr fr = alloc(G8);
        for (int a = 0; a < K; a++) {
            cuMemcpyHtoD(fr, Pointer.to(frac[a]), G8);
            final CUdeviceptr rho = off(dRho, a * C16), gam = off(dGam, a * C16);
            k("k_r2c", nb, rho, fr, (long) N);
            fft(rho, JCufft.CUFFT_FORWARD);
            k("k_scale_c", nb, rho, h2, 1.0, (long) N);
            k("k_mul2_max", nb, dW, fr, gGrid, (long) N, 0, dPart);
            k("k_r2c", nb, gam, dW, (long) N);
            fft(gam, JCufft.CUFFT_INVERSE);
            k("k_scale_c", nb, gam, 1.0 / N, TP2, (long) N);
            final double[] g0 = new double[1];
            cuMemcpyDtoH(Pointer.to(g0), gam, 8L);
            G[a] = g0[0];
        }
        cuMemcpyHtoD(dG, Pointer.to(G), 8L * K);
        cuCtxSynchronize();
        cuCtxSetCurrent(previous);                      // back to the caller's context (none: unbinds)
    }

    public int getGridSize() { return n; }
    public int getResidueCount() { return nResidues; }
    public double[] getRegionMass() { return G.clone(); }
    public int getRegionCount() { return K; }

    /**
     * Counts for one residue at (Sigma, lambda): labelled_counts.LabelledCounts.counts(Sigma, lam, c0, c1, refine).
     */
    public Result count(int residue, double[][] sigma, double lambda, double c1, int refine) {
        if (closed) throw new IllegalStateException("count engine closed");
        if (refine != 1 && refine != 2) throw new IllegalArgumentException("refine must be 1 or 2");
        if (!(lambda > 0.0) || !Double.isFinite(lambda)) throw new IllegalArgumentException("lambda must be finite and > 0");
        for (double[] row : sigma) for (double x : row) if (!Double.isFinite(x)) throw new IllegalArgumentException("Sigma must be finite");
        if (residue < 0 || residue >= nResidues) throw new IllegalArgumentException("residue out of range");
        final long t0 = System.nanoTime();
        cur = residue;
        cuCtxSetCurrent(context);
        final Result res = new Result();
        res.N = new double[nodeCount][K][K];

        // ---- the model on the host, numpy's expressions (labelled_counts._model) and its stable mode sort
        final double[] Lam = new double[N];
        for (int i = 0; i < N; i++) {
            Lam[i] = quad(sigma, mode(i / n), mode(i % n)) + lambda;
            if (!(Lam[i] > 0.0)) throw new IllegalArgumentException("Sigma is not positive semi-definite: some rates Lambda <= 0");
        }
        final double[] key = new double[S];
        for (int j = 0; j < S; j++) key[j] = Lam[selIdx[j]];
        final int[] order = stableArgsortPositive(key);
        final int[] idx = new int[S];
        final double[] wts = new double[S], Ls = new double[S];
        for (int j = 0; j < S; j++) { idx[j] = selIdx[order[j]]; wts[j] = selW[order[j]]; Ls[j] = Lam[idx[j]]; }
        res.secModel = (System.nanoTime() - t0) * 1e-9;
        cuMemcpyHtoD(dLam, Pointer.to(Lam), 8L * N);
        cuMemcpyHtoD(dIdx, Pointer.to(idx), 4L * S);
        cuMemcpyHtoD(dWts, Pointer.to(wts), 8L * S);
        cuMemcpyHtoD(dLs, Pointer.to(Ls), 8L * S);
        final long Nl = N;
        k("k_mu", nb, dMu, dGhat, dLam, lambda, Nl);
        k("k_gather", nbs, dMus, dRhoS, dGamG, dMu, dRho, dGam, dG, dIdx, S, K, Nl);
        for (int a = 0; a < K; a++) {                    // Pi_a = Re(sum rho_a mu) / TP2
            k("k_sum_cmul", nb, off(dRho, a * 16L * N), dMu, Nl, dPart);
            reduce(nb, 2, false, SC_PI + 2 * a);
        }

        // ---- inside pass (labelled_counts._inside), keeping Lt of every non-root node and c of internal nodes
        final double[][] tipY = tips[residue];
        final int[] opBase = new int[nodeCount];         // node -> first slot of its children's scales in dOpScale
        int nOps = 0;
        for (int v : post) {
            if (isTip[v]) {
                final double[] y = tipY[v];
                if (y != null) {
                    k("k_tip_c", nb, dXc, dDamp, y[0], y[1], n);
                    prop(dXc, branch[v], lt(v));
                }
                continue;
            }
            opBase[v] = nOps;
            final int[] ch = children[v];
            for (int i = 0; i < ch.length; i++) {
                k("k_mul_max", nb, dProd, lt(ch[i]), i == 0 ? 1 : 0, Nl, dPart);
                k("k_reduce_parts", 1, dPart, nb, 1, 1, off(dOpScale, 8L * nOps));
                k("k_div_scalar", nb, dProd, off(dOpScale, 8L * nOps), Nl);
                nOps++;
            }
            final CUdeviceptr c = v == root ? dXc : cOf(v);
            k("k_r2c", nb, c, dProd, Nl);
            fft(c, JCufft.CUFFT_FORWARD);
            k("k_scale_c", nb, c, h2, 1.0, Nl);
            if (v != root) prop(c, branch[v], lt(v));
        }
        k("k_sum_cmul", nb, dMu, dXc, Nl, dPart);       // L = Re(sum mu c_root) / TP2
        reduce(nb, 2, false, SC_L);
        final double[] opScale = new double[Math.max(1, nOps)];
        cuMemcpyDtoH(Pointer.to(opScale), dOpScale, 8L * nOps);
        final double[] scal = new double[NSCAL];
        cuMemcpyDtoH(Pointer.to(scal), dScal, 8L * NSCAL);
        final double[] ls = new double[nodeCount];
        for (int v : post) {
            if (isTip[v]) continue;
            double lsv = 0.0;
            final int[] ch = children[v];
            for (int i = 0; i < ch.length; i++) lsv = lsv + (Math.log(opScale[opBase[v] + i]) + ls[ch[i]]);
            ls[v] = lsv;
        }
        final double logL = Math.log(scal[SC_L] / TP2) + ls[root];
        res.logL = logL;
        final double[] Pi = new double[K];
        for (int a = 0; a < K; a++) Pi[a] = scal[SC_PI + 2 * a] / TP2;
        res.secInside = (System.nanoTime() - t0) * 1e-9;

        // ---- outside pass (labelled_counts.counts' pre-order loop)
        final Deque<Integer> free = new ArrayDeque<Integer>();
        for (int i = nOut - 1; i >= 0; i--) free.push(i);
        final int[] outOf = new int[nodeCount];
        final double[] lo = new double[nodeCount];
        outOf[root] = free.pop();
        k("k_copy_c", nb, dXo, dMu, Nl);                 // Out[root] = pi_grid = Re(FFT(mu)) / TP2
        fft(dXo, JCufft.CUFFT_FORWARD);
        k("k_c2r_div", nb, outBuf(outOf[root]), dXo, TP2, Nl);
        lo[root] = 0.0;
        double k0 = 0.0;
        final double[] planA = new double[2 + 2 * NC], planB = new double[2 + 2 * NC], sc = new double[NSCAL],
                wh0 = new double[2];
        final double[][][] I = new double[nodeCount][K][K];
        final double[] LbOf = new double[nodeCount];
        final boolean[] badX = new boolean[nodeCount];
        cuMemsetD32(dXall, 0, 2 * K * K * nodeCount);
        final double[] rb = new double[RB_N];
        for (int u : pre) {
            if (isTip[u]) continue;
            final int[] ch = children[u];
            for (final int v : ch) {
                long tb = System.nanoTime();
                final double t = branch[v];
                // W = Out[u] x Lt of the siblings, rescaled by its max
                int ns = 0;
                for (int s : ch) if (s != v) ns++;
                if (ns == 0) {
                    k("k_copy_max", nb, dW, outBuf(outOf[u]), Nl, dPart);
                } else {
                    int j = 0;
                    for (int s : ch) {
                        if (s == v) continue;
                        k("k_mul2_max", nb, dW, j == 0 ? outBuf(outOf[u]) : dW, lt(s), Nl, j == ns - 1 ? 1 : 0, dPart);
                        j++;
                    }
                }
                reduce(nb, 1, true, SC_SC);
                k("k_div_dot", nb, dW, off(dScal, 8L * SC_SC), lt(v), Nl, dPart);
                reduce(nb, 1, false, SC_DOT);
                // Wh = TP2 ifft2(W)
                k("k_r2c", nb, dXw, dW, Nl);
                fft(dXw, JCufft.CUFFT_INVERSE);
                k("k_scale_c", nb, dXw, 1.0 / N, TP2, Nl);
                // the inside coefficients of v: the tip's, FLAT, or the stored c
                final CUdeviceptr cv;
                if (isTip[v]) {
                    final double[] y = tipY[v];
                    if (y == null) cv = dFlat;
                    else { k("k_tip_c", nb, dXc, dDamp, y[0], y[1], n); cv = dXc; }
                } else cv = cOf(v);
                k("k_sum_cmul", nb, dMu, cv, Nl, dPart);
                reduce(nb, 2, false, SC_ZB);
                k("k_series", nbs, dAR, dBR, dXw, cv, dIdx, dWts, dLs, dMus, dRhoS, dGamG, t, TP2, S, K, dPart);
                reduce(nbs, 4 * K, false, SC_SUMS);
                k("k_tau", 1, off(dScal, 8L * SC_SUMS), K, S, off(dScal, 8L * SC_TAU));
                side(dAR, 0, dLcA, dRcA, v);
                side(dBR, 1, dLcB, dRcB, v);
                if (!isTip[v]) {                         // Out[v] = Re(FFT(a Wh + mu (1 - a) Wh(0))) / TP2
                    outOf[v] = free.pop();
                    k("k_fwd_prep", nb, dXo, dXw, dLam, dMu, t, Nl);
                    fft(dXo, JCufft.CUFFT_FORWARD);
                    k("k_c2r_div", nb, outBuf(outOf[v]), dXo, TP2, Nl);
                }
                long tc = System.nanoTime(); res.secLaunch += (tc - tb) * 1e-9; tb = tc;
                k("k_copy_c", 1, off(dRB, 8L * RB_WH0), dXw, 1L);
                cuMemcpyDtoH(Pointer.to(rb), dRB, 8L * RB_N);            // the branch's only wait for the device
                System.arraycopy(rb, 0, planA, 0, PL);
                System.arraycopy(rb, PL, planB, 0, PL);
                System.arraycopy(rb, RB_SCAL, sc, 0, NSCAL);
                wh0[0] = rb[RB_WH0]; wh0[1] = rb[RB_WH0 + 1];
                tc = System.nanoTime(); res.secSync += (tc - tb) * 1e-9; tb = tc;
                double lw = lo[u];
                for (int s : ch) if (s != v) lw += ls[s];
                lw += Math.log(sc[SC_SC]);
                final double Lb = h2 * sc[SC_DOT];
                double e0 = Lb > 0.0 ? Math.abs(Math.log(Lb) + lw + ls[v] - logL) : Double.POSITIVE_INFINITY;
                if (!Double.isFinite(e0)) e0 = Double.POSITIVE_INFINITY;
                k0 = Math.max(k0, e0);
                if (!isTip[v]) lo[v] = lw;
                final double W1 = wh0[0];
                final double zb = sc[SC_ZB] / TP2;
                for (int a = 0; a < K; a++) {
                    for (int b = 0; b < K; b++) {
                        final double Ainf = W1 * Pi[a], Binf = G[b] * zb;
                        final double FA = sc[SC_SUMS + 2 * K + a], FB = sc[SC_SUMS + 3 * K + b];
                        I[v][a][b] = (Ainf * Binf) * t + Ainf * FB + FA * Binf;
                    }
                }
                LbOf[v] = Lb;
                badX[v] = !cross(v, t, planA, planB, c1, refine, res);
                tc = System.nanoTime(); res.secCross += (tc - tb) * 1e-9; tb = tc;
            }
            free.push(outOf[u]);
        }
        final double[] xall = new double[K * K * nodeCount];
        long tw = System.nanoTime();
        cuMemcpyDtoH(Pointer.to(xall), dXall, 8L * xall.length);           // every branch's double sum X
        res.secWait += (System.nanoTime() - tw) * 1e-9;
        for (int v = 0; v < nodeCount; v++) {
            if (v == root) continue;
            for (int a = 0; a < K; a++) {
                for (int b = 0; b < K; b++) {
                    final double X = badX[v] ? Double.NaN : xall[(v * K + a) * K + b];
                    res.N[v][a][b] = (lambda * (I[v][a][b] + X)) / LbOf[v];   // labelled_counts: lam (I + X) / Lb
                }
            }
        }
        res.k0 = k0;
        if (!Double.isFinite(logL)) res.status |= STATUS_NONFINITE;
        for (double[][] Nv : res.N) for (double[] row : Nv) for (double x : row) if (!Double.isFinite(x)) res.status |= STATUS_NONFINITE;
        res.secTotal = (System.nanoTime() - t0) * 1e-9;
        return res;
    }

    /** Keep, compaction and candidate counts of one side (A: AR, B: BR) of branch v. */
    private void side(CUdeviceptr R, int side, CUdeviceptr Lc, CUdeviceptr Rc, int v) {
        k("k_keep", ntile, R, K, S, off(dScal, 8L * SC_TAU), side, dFlags, dTileCnt);
        k("k_scan_tiles", 1, dTileCnt, ntile, dTileOff, off(dTot, 4L * side));
        k("k_compact", ntile, R, dLs, dFlags, dTileOff, K, S, SP, Lc, Rc);
        k("k_plan", 1, Lc, off(dTot, 4L * side), off(dThr, 8L * ((long) (v * 2 + side) * NC)), NC,
                off(dPlan, 8L * side * (2 + 2 * NC)));
    }

    /** labelled_counts._cross for branch v: the host plan (labelled_counts_gpu._plan) and the device panel sums. */
    /** labelled_counts._cross for branch v, into dXall[v] (zero if no panel); false if v needs more than JMAX breakpoints */
    private boolean cross(int v, double t, double[] pA, double[] pB, double c1, int refine, Result res) {
        final int nA = (int) pA[0], nB = (int) pB[0];
        if (nA == 0 || nB == 0 || !(t > 0.0)) return true;
        final double LAmax = pA[1], LBmax = pB[1];
        final int JA = LAmax * t > c1 ? ceilLog2(LAmax * t / c1) : 0;
        final int JB = LBmax * t > c1 ? ceilLog2(LBmax * t / c1) : 0;
        if (JA > JMAX || JB > JMAX) {
            res.status |= STATUS_JMAX;
            return false;
        }
        final TreeSet<Double> bpSet = new TreeSet<Double>();
        bpSet.add(0.0);
        bpSet.add(t);
        for (int j = 1; j <= JA; j++) bpSet.add(t / Math.scalb(1.0, j));
        for (int j = 1; j <= JB; j++) bpSet.add(t - t / Math.scalb(1.0, j));
        final Double[] bp = bpSet.toArray(new Double[0]);
        final Map<Double, Integer> cidx = candIndex.get(v);
        final int[] pa = new int[PMAX], pb = new int[PMAX], nq = new int[PMAX];
        final double[] nodes = new double[PMAX * 2 * NQMAX], ws = new double[PMAX * NQMAX];
        int np = 0;
        long nterm = 0;
        for (int i = 0; i + 1 < bp.length; i++) {
            final double a = bp[i], b = bp[i + 1];
            final int pai = a == 0.0 ? nA : (int) pA[2 + cidx.get(a)];
            final int pbi = b == t ? nB : (int) pB[2 + cidx.get(b)];
            if (pai == 0 || pbi == 0) continue;
            final double la = a == 0.0 ? LAmax : pA[2 + NC + cidx.get(a)];
            final double lb = b == t ? LBmax : pB[2 + NC + cidx.get(b)];
            final double R = Math.max(la, lb) * (b - a);
            final int q = (R <= 1 ? 6 : R <= 4 ? 8 : R <= 12 ? 12 : 16) * refine;
            final int g = Arrays.binarySearch(GL_ORDERS, q);
            final double[] x = GL_X[g], w = GL_W[g];
            for (int j = 0; j < q; j++) {
                final double s = a + (b - a) * (x[j] + 1) / 2;
                nodes[(np * 2) * NQMAX + j] = s;
                nodes[(np * 2 + 1) * NQMAX + j] = t - s;
                ws[np * NQMAX + j] = w[j] * (b - a) / 2;
            }
            pa[np] = pai; pb[np] = pbi; nq[np] = q;
            nterm += (long) q * (pai + pbi);
            np++;
        }
        res.nPanels += bp.length - 1;
        res.nTerms += nterm;
        if (np == 0) return true;
        long tq = System.nanoTime();
        // work units grouped by Gauss-Legendre order (one kernel launch per order), (p, side) contiguous
        final List<int[]> units = new ArrayList<int[]>();
        final int[] ufirst = new int[PMAX * 2], ucount = new int[PMAX * 2];
        final int[] classStart = new int[GL_ORDERS.length], classCount = new int[GL_ORDERS.length];
        for (int g = 0; g < GL_ORDERS.length; g++) {
            classStart[g] = units.size();
            for (int p = 0; p < np; p++) {
                if (nq[p] != GL_ORDERS[g]) continue;
                for (int side = 0; side < 2; side++) {
                    final int lim = side == 0 ? pa[p] : pb[p];
                    ufirst[p * 2 + side] = units.size();
                    for (int st = 0; st < lim; st += CHUNK) units.add(new int[]{p, side, st, Math.min(CHUNK, lim - st)});
                    ucount[p * 2 + side] = units.size() - ufirst[p * 2 + side];
                }
            }
            classCount[g] = units.size() - classStart[g];
        }
        if (units.size() > maxUnits) throw new IllegalStateException("cross-term work units exceed the buffer");
        res.secPlan += (System.nanoTime() - tq) * 1e-9; tq = System.nanoTime();
        up.clear();
        for (double x : nodes) up.putDouble(x);
        for (double x : ws) up.putDouble(x);
        for (int x : ufirst) up.putInt(x);
        for (int x : ucount) up.putInt(x);
        for (int x : nq) up.putInt(x);
        if (up.position() != UP_UNITS) throw new AssertionError("upload layout");
        for (int[] uu : units) for (int x : uu) up.putInt(x);
        cuMemcpyHtoD(dUp, Pointer.to(up), (long) up.position());          // one upload per branch
        res.secUpload += (System.nanoTime() - tq) * 1e-9; tq = System.nanoTime();
        for (int g = 0; g < GL_ORDERS.length; g++) {
            if (classCount[g] == 0) continue;
            launch(fn("k_cross_" + GL_ORDERS[g] + "_" + K), classCount[g], BLOCK, Pointer.to(dUnits), Pointer.to(new int[]{classStart[g]}),
                    Pointer.to(dNodes), Pointer.to(dLcA), Pointer.to(dRcA), Pointer.to(dLcB), Pointer.to(dRcB),
                    Pointer.to(new long[]{SP}), Pointer.to(new int[]{K}), Pointer.to(dPartial));
        }
        launch(fn("k_cross_finish"), 1, 1024, Pointer.to(dPartial), Pointer.to(dUfirst), Pointer.to(dUcount),
                Pointer.to(dWs), Pointer.to(dNq), Pointer.to(new int[]{np}), Pointer.to(new int[]{K}), Pointer.to(dAlpha),
                Pointer.to(off(dXall, 8L * K * K * v)));
        res.secKernel += (System.nanoTime() - tq) * 1e-9;
        return true;
    }

    /** labelled_counts._prop: Lt = Re(ifft2(a c with the reset mass added to mode 0)) / h2. */
    private void prop(CUdeviceptr c, double t, CUdeviceptr Lt) {
        k("k_prop1", nb, dXw, c, dLam, dMu, t, (long) N, dPart);
        k("k_add0", 1, dXw, dPart, nb);
        fft(dXw, JCufft.CUFFT_INVERSE);
        k("k_c2r_scale_div", nb, Lt, dXw, 1.0 / N, h2, (long) N);
    }

    /** sums (or max) of nv per-block partials of the last kernel, into dScal[slot..slot+nv) */
    private void reduce(int nblk, int nv, boolean max, int slot) {
        k("k_reduce_parts", 1, dPart, nblk, nv, max ? 1 : 0, off(dScal, 8L * slot));
    }

    /** the propagated message of v: stored, or a grid of ones for a tip missing at the current residue */
    private CUdeviceptr lt(int v) {
        if (isTip[v] && tips[cur][v] == null) return dOnes;
        return off(dLt, 8L * N * v);
    }

    private CUdeviceptr cOf(int v) { return off(dC, 16L * N * cslot[v]); }

    private CUdeviceptr outBuf(int i) { return off(dOut, 8L * N * i); }

    // ---- numerics shared with numpy

    private int mode(int i) { return i < n / 2 ? i : i - n; }

    /** labelled_counts._quad: 0.5 (S00 m1^2 + 2 S01 m1 m2 + S11 m2^2), numpy's operation order */
    private static double quad(double[][] s, double m1, double m2) {
        return 0.5 * (s[0][0] * (m1 * m1) + 2 * s[0][1] * m1 * m2 + s[1][1] * (m2 * m2));
    }

    /** ceil(log2(x)) for x > 1, exact (np.ceil(np.log2(x)) except where np.log2 rounds across an integer) */
    static int ceilLog2(double x) {
        final int e = Math.getExponent(x);
        return x == Math.scalb(1.0, e) ? e : e + 1;
    }

    /** numpy.argsort(x, kind="stable") for positive finite doubles: LSD radix sort on the bit patterns. */
    static int[] stableArgsortPositive(double[] x) {
        final int m = x.length;
        int[] idx = new int[m], tmp = new int[m];
        long[] key = new long[m], ktmp = new long[m];
        for (int i = 0; i < m; i++) { idx[i] = i; key[i] = Double.doubleToRawLongBits(x[i]); }
        final int[] count = new int[65537];
        for (int shift = 0; shift < 64; shift += 16) {
            Arrays.fill(count, 0);
            for (int i = 0; i < m; i++) count[(int) ((key[i] >>> shift) & 0xffffL) + 1]++;
            for (int b = 0; b < 65536; b++) count[b + 1] += count[b];
            for (int i = 0; i < m; i++) {
                final int b = (int) ((key[i] >>> shift) & 0xffffL);
                final int p = count[b]++;
                tmp[p] = idx[i];
                ktmp[p] = key[i];
            }
            int[] ti = idx; idx = tmp; tmp = ti;
            long[] tk = key; key = ktmp; ktmp = tk;
        }
        return idx;
    }

    // ---- CUDA plumbing

    private void fft(CUdeviceptr x, int dir) { JCufft.cufftExecZ2Z(plan, x, x, dir); }

    private CUfunction fn(String name) {
        CUfunction f = fn.get(name);
        if (f == null) {
            f = new CUfunction();
            cuModuleGetFunction(f, module, name);
            fn.put(name, f);
        }
        return f;
    }

    /** launch kernel name on blocks x BLOCK threads; args: CUdeviceptr, Integer, Long, Double */
    private void k(String name, int blocks, Object... args) {
        final Pointer[] p = new Pointer[args.length];
        for (int i = 0; i < args.length; i++) {
            final Object a = args[i];
            if (a instanceof CUdeviceptr) p[i] = Pointer.to((CUdeviceptr) a);
            else if (a instanceof Integer) p[i] = Pointer.to(new int[]{(Integer) a});
            else if (a instanceof Long) p[i] = Pointer.to(new long[]{(Long) a});
            else if (a instanceof Double) p[i] = Pointer.to(new double[]{(Double) a});
            else throw new IllegalArgumentException("kernel argument " + a);
        }
        launch(fn(name), blocks, BLOCK, p);
    }

    private void launch(CUfunction f, int blocks, int threads, Pointer... args) {
        cuLaunchKernel(f, blocks, 1, 1, threads, 1, 1, 0, null, Pointer.to(args), null);
    }

    private static CUdeviceptr off(CUdeviceptr p, long bytes) { return (CUdeviceptr) p.withByteOffset(bytes); }

    private CUdeviceptr alloc(long bytes) {
        final CUdeviceptr p = new CUdeviceptr();
        cuMemAlloc(p, Math.max(8L, bytes));
        allocated.add(p);
        return p;
    }

    private CUdeviceptr upload(double[] a) {
        final CUdeviceptr p = alloc(8L * a.length);
        cuMemcpyHtoD(p, Pointer.to(a), 8L * a.length);
        return p;
    }

    private static byte[] loadPtx() {
        final String override = System.getProperty("torus.count.ptx");
        try (InputStream in = override != null ? new java.io.FileInputStream(override)
                : TorusLandingCountEngine.class.getResourceAsStream(PTX_RESOURCE)) {
            if (in == null) throw new IllegalStateException("PTX resource " + PTX_RESOURCE + " not found");
            final ByteArrayOutputStream out = new ByteArrayOutputStream();
            final byte[] buffer = new byte[8192];
            int r;
            while ((r = in.read(buffer)) > 0) out.write(buffer, 0, r);
            out.write(0);
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
        JCufft.cufftDestroy(plan);
        for (CUdeviceptr p : allocated) cuMemFree(p);
        allocated.clear();
        cuModuleUnload(module);
        cuCtxDestroy(context);
    }
}
