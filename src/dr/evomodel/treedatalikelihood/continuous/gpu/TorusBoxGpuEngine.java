/*
 * TorusBoxGpuEngine.java
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static jcuda.driver.JCudaDriver.*;

/**
 * The landing likelihood in BOX arithmetic on a GPU, batched over residues ("sites"): a port of the numpy reference
 * gpu_build/box_engine/box_landing.py (BoxLanding with domain "boxes", prec_check False, on_flag "warn"; md5
 * 74f54bd8c642, SPEC.md sections 3 and 5) to CUDA (torus_box_kernels.cu).
 *
 * <p>Every message of a site is a constant C plus, per box of the site, a local part on an n1 x n2 grid over the box.
 * A box of a site is a SLOT; one call evaluates any list of sites in chunks of whole sites holding at most
 * {@code maxBatch} slots, with one batched D2Z / Z2D cuFFT plan of fixed batch size (padding). The arithmetic is
 * box_landing.py's _box_pass operation for operation (tips from closed-form coefficients, propagation on the half
 * spectrum, the landing constant by Parseval against the per-slot table gam_b, products in Python's schedule and
 * rounding order with the per-site rescale, the root against the stationary law on the box grid). README.md in
 * gpu_build/box_gpu maps every Python line to the code here and lists every deliberate difference.
 *
 * <p>Guards are returned as VALUES per site, exactly the quantities Python's _Guards tracks with prec_check False
 * (whose round-off allowances are then zero): the LEAK value (with the landing truncation term), the RESOLUTION
 * ratio, rho_min (the worst product overlap) and NONPOSITIVE. The LEAK and RESOLUTION thresholds are applied here
 * ({@link #setTolerances}); the PRECISION shadow, refinement, verification and the torus fallback are not ported.
 *
 * <p>Transforms: batched cuFFT D2Z / Z2D plans of batch maxBatch plus single-slot plans, sharing one work area.
 * A construction self-test checks that the batched plans treat slots independently and that the single-slot plans
 * give bitwise the same output (cuFFT couples slots at some sizes, e.g. a last dimension of 75 or 375); the engine
 * then transforms batched, slot by slot, or either per transform ({@link #describeTransforms}). Before every batched
 * transform the slots not read afterwards are zeroed. Values are therefore bitwise reproducible across calls,
 * chunkings, call orders and maxBatch values.
 *
 * <p>Not thread safe: one thread at a time (the CUDA context is made current on entry to every call).
 *
 * @author Samir Bhatt
 */
public final class TorusBoxGpuEngine implements AutoCloseable {

    public static final int STATUS_OK = 0;
    public static final int STATUS_LEAK = 1;
    public static final int STATUS_RESOLUTION = 2;
    public static final int STATUS_NONPOSITIVE = 4;

    static final double TWO_PI = 2.0 * Math.PI;
    static final double DROP = 1e-280;
    static final double U = Math.scalb(1.0, -53);
    static final double C_FFT = 4.0;
    static final double G_REL = 1e-17;
    static final double IMG_EXP = 50.0;
    static final double TOUCH = 1e-12;
    private static final double TWO_PI_LO = 2.4492935982947064e-16;    // 2 pi - TWO_PI
    private static final double SPLIT = 134217729.0;                   // 2^27 + 1 (Veltkamp)
    private static final int BLOCK = 256;
    private static final int NPART = 12;
    private static final int MAX_BOXES = 16;                           // torus_box_kernels.cu MAX_BOXES
    private static final String PTX_RESOURCE = "torus_box_kernels.ptx";

    /** One box of one site: lower-left corner (radians, in [0, 2 pi]), widths (radians, 2 pi on a full-circle axis). */
    public static final class Box {
        public final double phi0, psi0, widthPhi, widthPsi;
        public final int nPhi, nPsi;
        public final boolean fullPhi, fullPsi;

        public Box(double phi0, double psi0, double widthPhi, double widthPsi, int nPhi, int nPsi, boolean fullPhi,
                   boolean fullPsi) {
            if (!(Double.isFinite(phi0) && Double.isFinite(psi0) && phi0 >= 0 && phi0 <= TWO_PI && psi0 >= 0
                    && psi0 <= TWO_PI)) {
                throw new IllegalArgumentException("box origin must be finite radians in [0, 2 pi], got " + phi0 + ", "
                        + psi0);
            }
            if (!(widthPhi > 0 && widthPhi <= TWO_PI && widthPsi > 0 && widthPsi <= TWO_PI)) {
                throw new IllegalArgumentException("box widths must be in (0, 2 pi] radians, got " + widthPhi + ", "
                        + widthPsi);
            }
            if ((fullPhi && widthPhi != TWO_PI) || (fullPsi && widthPsi != TWO_PI)) {
                throw new IllegalArgumentException("a full-circle box axis must have width 2 pi");
            }
            if (nPhi < 4 || nPsi < 4) throw new IllegalArgumentException("box points per axis must be >= 4");
            this.phi0 = phi0;
            this.psi0 = psi0;
            this.widthPhi = widthPhi;
            this.widthPsi = widthPsi;
            this.nPhi = nPhi;
            this.nPsi = nPsi;
            this.fullPhi = fullPhi;
            this.fullPsi = fullPsi;
        }

        /** box_landing.py _parse_record (L896-926): a box record in degrees, as BoxLanding(domain="boxes") reads it. */
        public static Box fromDegrees(double phi0Deg, double psi0Deg, double widthPhiDeg, double widthPsiDeg, int nPhi,
                                      int nPsi) {
            final double d2r = Math.PI / 180.0;                 // numpy radians / math.radians: x * (pi / 180)
            final double[] w = {widthPhiDeg, widthPsiDeg};
            final boolean[] full = new boolean[2];
            final double[] L = new double[2];
            for (int ax = 0; ax < 2; ax++) {
                if (!(w[ax] > 0 && w[ax] <= 360.0 + 1e-9)) {
                    throw new IllegalArgumentException("box width must be in (0, 360] degrees, not " + w[ax]);
                }
                full[ax] = w[ax] >= 360.0 - 1e-9;
                L[ax] = full[ax] ? TWO_PI : w[ax] * d2r;
            }
            return new Box(npmod(phi0Deg * d2r, TWO_PI), npmod(psi0Deg * d2r, TWO_PI), L[0], L[1], nPhi, nPsi, full[0],
                    full[1]);
        }
    }

    /** Tip assignment of one site: box index per node (-1 = missing or not a tip) and the offset d from the box origin. */
    public static final class TipAssignment {
        public final int[] box;
        public final double[][] offset;

        TipAssignment(int[] box, double[][] offset) {
            this.box = box;
            this.offset = offset;
        }
    }

    // ---- tree and schedule ----------------------------------------------------------------------------
    private final int nodeCount, root;
    private final int[][] children;
    private final boolean[] isTip;
    private final double[] branch;
    private final int[] order;                         // box_landing.py _schedule post-order
    private final int[] opChild, opParent, opBufChild, opBufParent;
    private final boolean[] opMulti;
    private final int nBuffers, rootBuffer;
    private final int[] tipOrdinal;
    private final int nTips;

    // ---- sites and slots ------------------------------------------------------------------------------
    private final int nSites, nSlotsTotal;
    private final int[] siteSlot0, siteBoxes;
    private final boolean[] siteHasRoot;
    private final int n1, n2, n2h, M, Mcut, nblk, maxBatch;
    private final int r0, r1, c0;
    private final double gsup, lconst, fct;
    private final double pi00h, pi01, pi11h;
    private final double[] tipLeak, tipRes;            // per site (parameter-free, box_landing.py L353-369, L341-346)

    // ---- device ---------------------------------------------------------------------------------------
    private final int deviceId;
    private CUcontext context;                         // set once retained (released by close / a failed constructor)
    private CUmodule module;                           // set once loaded
    private final CUfunction fGEval, fGpScale, fETable, fSiteInit, fMusub, fTipAxis, fTipBuild, fSlotPrep, fPropagate,
            fSiteLanding, fHermitianize, fCombine, fSiteProduct, fScaleParts, fRootStats, fRootRes, fRootT, fRootDot,
            fRootFinal, fZeroGrid, fZeroSpec, fFillNoise;
    // cuFFT: batched plans (batch maxBatch) and single-slot plans (batch 1), any of them null when not used; all
    // created without auto-allocation and sharing ONE work area (they run one after another on the null stream)
    private cufftHandle planD2Z, planZ2D, plan1D2Z, plan1Z2D;
    private final List<cufftHandle> plans = new ArrayList<cufftHandle>();
    private CUdeviceptr dFftWork;
    private long fftWorkBytes;
    // one slot of grid values on an aligned allocation: cuFFT refuses a real grid that does not start on a 16-byte
    // boundary (slot s > 0 of an odd n1 n2), so a single-slot transform of such a slot goes through this copy
    private CUdeviceptr dAlignR;
    /** FFT_BATCHED: every transform batched (padding, idle slots zeroed); FFT_SLOTWISE: every transform slot by slot
     *  (cuFFT's batched plan couples slots at this n); FFT_HYBRID: both verified bitwise equal, chosen per transform. */
    public static final int FFT_BATCHED = 0, FFT_HYBRID = 1, FFT_SLOTWISE = 2;
    /** Constructor fftPolicy: choose by the construction self-test (the default; see setupTransforms). */
    public static final int FFT_AUTO = -1;
    private final int fftPolicy;                       // FFT_AUTO, or FFT_BATCHED / FFT_SLOTWISE forced (tests)
    private int fftMode;
    private boolean fftIndependent, fftEquivalent, fftTested;
    private boolean fftKeepsInput;                     // the D2Z plans left their input bitwise unchanged (d2zInputTest)
    private final List<CUdeviceptr> allocations = new ArrayList<CUdeviceptr>();
    private final byte[] codesHost;                    // [op][site] product code (0 skip, 1 first, 2 multiply)
    private final int[] tipBoxHost;                    // [tip ordinal][site] box of the tip's observation, -1 = missing
    private final CUdeviceptr dOM1, dOM2, dGP, dE1, dE2, dHH, dNB, dGsub, dPhase, dTipBox, dPhaseOff, dCodes, dHasRoot,
            dTipLeak, dTipRes;
    private final CUdeviceptr[] dBuf;
    private final CUdeviceptr dX, dR, dT, dTipAx, dMU, dPart, dC, dLS, dPres, dPresTmp, dOPres, dZeros, dCp, dScaleS,
            dX00, dX0, dLand, dMode, dSlotInfo, dSiteInfo, dSiteParam, dGLeak, dGRes, dGRho, dGStat, dGDead, dGNode,
            dOutLogL;
    private boolean closed = false;

    // ---- options and counters -------------------------------------------------------------------------
    private double leakTol = 1e-12, resTol = 1e-12;
    private long evaluations = 0L, leakFlags = 0L, resFlags = 0L, nonpositive = 0L;
    private double lastSeconds = 0.0;

    /**
     * @param children     children of every node, in the order BoxLanding receives them (empty for tips)
     * @param root         root node (must be internal)
     * @param branchLength length of the branch above every node, finite and >= 0 (root entry ignored)
     * @param boxes        [site][box] the site's boxes, as BoxLanding(domain="boxes") builds them (radians); every
     *                     box of every site must have the same nPhi and the same nPsi; a site without observed tips
     *                     may have no boxes
     * @param tipBox       [site][node] box index (within the site) of the node's observation, or -1 if the tip is
     *                     missing at that site; entries of internal nodes are ignored
     * @param tipOffset    [site][node][2] the observation's offset d = (y - o) mod 2 pi from its box's origin
     *                     (radians), as box_landing.py computes it (L943); ignored where tipBox is -1
     * @param sigmaEps     tip observation-error covariance Pi (2 x 2, symmetric positive definite, rad^2)
     * @param w            landing mixture weights (sum 1)
     * @param mu           landing mixture means (radians)
     * @param s            landing mixture covariances (rad^2)
     * @param deviceId     CUDA device ordinal
     * @param maxBatch     slots per chunk and the batch of the batched cuFFT plans (>= the most boxes of any site);
     *                     clamped to the engine's slot count. Every work buffer holds maxBatch slots, and a BATCHED
     *                     transform costs maxBatch slots whatever the chunk holds; in FFT_HYBRID mode (the usual
     *                     one, see setupTransforms) a transform that uses few slots runs slot by slot instead, so
     *                     one-site calls do not pay for padding. A per-device engine of a multi-GPU split needs a
     *                     maxBatch no larger than its own slot count (the clamp ensures it).
     * @throws IllegalArgumentException on invalid inputs; any CUDA / cuFFT failure propagates after everything the
     *                     constructor acquired (plans, buffers, module, primary context) has been released
     */
    public TorusBoxGpuEngine(int[][] children, int root, double[] branchLength, Box[][] boxes, int[][] tipBox,
                             double[][][] tipOffset, double[][] sigmaEps, double[] w, double[][] mu, double[][][] s,
                             int deviceId, int maxBatch) {
        this(children, root, branchLength, boxes, tipBox, tipOffset, sigmaEps, w, mu, s, deviceId, maxBatch, FFT_AUTO);
    }

    /**
     * As the 12-argument constructor, with the transform policy given explicitly (an argument of this engine, not
     * a JVM-global setting, so engines built side by side in one JVM never affect each other).
     *
     * @param fftPolicy FFT_AUTO (production: the construction self-test chooses hybrid, batched or slotwise), or, for
     *                  tests only, FFT_BATCHED (refused with IllegalStateException where the self-test finds that
     *                  cuFFT's batched plans couple slots) or FFT_SLOTWISE (every transform slot by slot)
     */
    public TorusBoxGpuEngine(int[][] children, int root, double[] branchLength, Box[][] boxes, int[][] tipBox,
                             double[][][] tipOffset, double[][] sigmaEps, double[] w, double[][] mu, double[][][] s,
                             int deviceId, int maxBatch, int fftPolicy) {
        if (fftPolicy != FFT_AUTO && fftPolicy != FFT_BATCHED && fftPolicy != FFT_SLOTWISE) {
            throw new IllegalArgumentException("fftPolicy must be FFT_AUTO, FFT_BATCHED or FFT_SLOTWISE, not " + fftPolicy);
        }
        this.fftPolicy = fftPolicy;
        // ---- tree (box_landing.py L584-622) and schedule (L254-281)
        this.nodeCount = children.length;
        if (nodeCount < 2) throw new IllegalArgumentException("the tree needs a root and at least one tip");
        this.root = root;
        if (root < 0 || root >= nodeCount) throw new IllegalArgumentException("root is not a node id");
        this.children = new int[nodeCount][];
        this.isTip = new boolean[nodeCount];
        final int[] parent = new int[nodeCount];
        Arrays.fill(parent, -1);
        for (int v = 0; v < nodeCount; v++) {
            this.children[v] = children[v].clone();
            this.isTip[v] = children[v].length == 0;
            for (int c : children[v]) {
                if (c < 0 || c >= nodeCount || c == v) throw new IllegalArgumentException("node " + v + " lists an invalid child " + c);
                if (parent[c] >= 0) throw new IllegalArgumentException("node " + c + " has two parents");
                parent[c] = v;
            }
        }
        if (parent[root] != -1) throw new IllegalArgumentException("the root has a parent");
        if (isTip[root]) throw new IllegalArgumentException("the root must be an internal node");
        if (branchLength.length != nodeCount) throw new IllegalArgumentException("branchLength needs one entry per node");
        this.branch = branchLength.clone();
        branch[root] = 0.0;
        for (int v = 0; v < nodeCount; v++) {
            if (!(Double.isFinite(branch[v]) && branch[v] >= 0.0)) {
                throw new IllegalArgumentException("branch length above node " + v + " must be finite and >= 0, got " + branch[v]);
            }
        }
        this.order = schedule(this.children, root);
        if (order.length != nodeCount) throw new IllegalArgumentException("the tree is not connected");
        this.tipOrdinal = new int[nodeCount];
        Arrays.fill(tipOrdinal, -1);
        int nt = 0;
        for (int v = 0; v < nodeCount; v++) if (isTip[v]) tipOrdinal[v] = nt++;
        this.nTips = nt;
        final int nOps = nodeCount - 1;
        this.opChild = new int[nOps];
        this.opParent = new int[nOps];
        this.opBufChild = new int[nOps];
        this.opBufParent = new int[nOps];
        this.opMulti = new boolean[nOps];
        final int[] buf = new int[nodeCount];
        Arrays.fill(buf, -1);
        final List<Integer> free = new ArrayList<Integer>();
        int nb = 0, o = 0;
        for (int v : order) {
            if (v == root) continue;
            final int p = parent[v];
            opChild[o] = v;
            opParent[o] = p;
            opMulti[o] = !isTip[v] && children[v].length >= 3;
            opBufChild[o] = isTip[v] ? -1 : buf[v];
            if (!isTip[v]) free.add(buf[v]);                 // its grid is in the transform buffer from here on
            if (buf[p] < 0) buf[p] = free.isEmpty() ? nb++ : free.remove(free.size() - 1);
            opBufParent[o] = buf[p];
            o++;
        }
        this.nBuffers = nb;
        this.rootBuffer = buf[root];

        // ---- tip error (L644-651) and the landing mixture (L652-690)
        requireSymmetricPd(sigmaEps, "sigmaEps");
        this.pi00h = 0.5 * sigmaEps[0][0];
        this.pi01 = sigmaEps[0][1];
        this.pi11h = 0.5 * sigmaEps[1][1];
        final Mixture mix = new Mixture(w, mu, s);
        this.Mcut = mix.mcut;
        this.M = 2 * Mcut + 1;

        // ---- sites, slots and their common grid
        this.nSites = boxes.length;
        if (tipBox.length != nSites || tipOffset.length != nSites) {
            throw new IllegalArgumentException("boxes, tipBox and tipOffset need one entry per site");
        }
        this.siteSlot0 = new int[nSites];
        this.siteBoxes = new int[nSites];
        int ns = 0, gn1 = -1, gn2 = -1, mostBoxes = 0;
        for (int site = 0; site < nSites; site++) {
            siteSlot0[site] = ns;
            siteBoxes[site] = boxes[site].length;
            mostBoxes = Math.max(mostBoxes, boxes[site].length);
            for (Box bx : boxes[site]) {
                if (gn1 < 0) { gn1 = bx.nPhi; gn2 = bx.nPsi; }
                if (bx.nPhi != gn1 || bx.nPsi != gn2) {
                    throw new IllegalArgumentException("every box must have the same points per axis (" + gn1 + " x "
                            + gn2 + "), site " + site + " has " + bx.nPhi + " x " + bx.nPsi);
                }
            }
            ns += boxes[site].length;
        }
        if (ns == 0) throw new IllegalArgumentException("no site has a box");
        if (mostBoxes > MAX_BOXES) throw new IllegalArgumentException("at most " + MAX_BOXES + " boxes per site");
        this.nSlotsTotal = ns;
        this.n1 = gn1;
        this.n2 = gn2;
        this.n2h = n2 / 2 + 1;
        if (maxBatch < mostBoxes) {
            throw new IllegalArgumentException("maxBatch (" + maxBatch + ") must hold the largest site (" + mostBoxes + " boxes)");
        }
        // every work buffer and the batched cuFFT plans hold maxBatch slots, and a batched transform costs maxBatch
        // slots whatever the chunk holds: never more than the engine has (mostBoxes <= ns, so this keeps the check)
        this.maxBatch = Math.min(maxBatch, ns);
        final long NG = (long) n1 * n2, N2 = (long) n1 * n2h;
        this.nblk = (int) Math.max(1L, Math.min(128L, NG / (BLOCK * 4L)));
        this.fct = 1.0 / (double) NG;
        this.lconst = ((10.0 * C_FFT) * U) * Math.sqrt(log2(NG));
        // RESOLUTION modes (L335-340): rows |p| >= 7 n1 / 16 (one contiguous block), columns q >= 7 n2 / 16
        final double[] pf = fftfreq(n1), qf = rfftfreq(n2);
        int rlo = -1, rhi = -2;
        for (int i = 0; i < n1; i++) {
            if (Math.abs(pf[i]) >= 7.0 * n1 / 16.0) {
                if (rlo < 0) rlo = i;
                if (rhi >= 0 && i != rhi + 1) throw new IllegalStateException("RESOLUTION rows are not contiguous");
                rhi = i;
            }
        }
        this.r0 = rlo < 0 ? n1 : rlo;
        this.r1 = rlo < 0 ? -1 : rhi;
        int cl = n2h;
        for (int j = 0; j < n2h; j++) if (qf[j] >= 7.0 * n2 / 16.0) { cl = j; break; }
        this.c0 = cl;

        // ---- per slot geometry (L311-334, L347-369), per site tip guards and the zero-jump structure
        final double[] om1All = new double[ns * n1], om2All = new double[ns * n2h], hhAll = new double[ns];
        final int[] nbAll = new int[2 * ns];
        final double[][] x1All = new double[ns][], x2All = new double[ns][];
        this.tipLeak = new double[nSites];
        this.tipRes = new double[nSites];
        this.siteHasRoot = new boolean[nSites];
        final byte[] codes = new byte[nOps * nSites];
        final int[] tipBoxDev = new int[nTips * nSites];
        Arrays.fill(tipBoxDev, -1);
        final long[] phaseOff = new long[nTips * nSites];
        Arrays.fill(phaseOff, -1L);
        long phaseLen = 0L;
        for (int site = 0; site < nSites; site++) {
            final Box[] bxs = boxes[site];
            final int[] tb = tipBox[site];
            if (tb == null || tb.length != nodeCount) throw new IllegalArgumentException("tipBox[" + site + "] needs one entry per node");
            final List<List<double[]>> members = new ArrayList<List<double[]>>();
            for (int b = 0; b < bxs.length; b++) members.add(new ArrayList<double[]>());
            for (int v = 0; v < nodeCount; v++) {
                if (!isTip[v] || tb[v] < 0) continue;
                if (tb[v] >= bxs.length) throw new IllegalArgumentException("site " + site + " tip " + v + ": box " + tb[v] + " does not exist");
                final double[] d = tipOffset[site][v];
                final Box bx = bxs[tb[v]];
                if (d == null || d.length != 2) throw new IllegalArgumentException("site " + site + " tip " + v + ": offset must have 2 values");
                final double[] L = {bx.widthPhi, bx.widthPsi};
                final boolean[] full = {bx.fullPhi, bx.fullPsi};
                for (int ax = 0; ax < 2; ax++) {
                    if (!(d[ax] >= 0.0 && d[ax] <= TWO_PI && (full[ax] || d[ax] < L[ax]))) {
                        throw new IllegalArgumentException("site " + site + " tip " + v + ": offset " + d[ax]
                                + " is not inside box " + tb[v] + " on axis " + ax);
                    }
                }
                members.get(tb[v]).add(new double[]{v, d[0], d[1]});
                tipBoxDev[tipOrdinal[v] * nSites + site] = tb[v];
                phaseOff[tipOrdinal[v] * nSites + site] = phaseLen;
                phaseLen += n1 + n2h;
            }
            double leakSite = 0.0, resSite = 0.0;
            for (int b = 0; b < bxs.length; b++) {
                final int gsl = siteSlot0[site] + b;
                final Box bx = bxs[b];
                final double[] o0 = {bx.phi0, bx.psi0}, L = {bx.widthPhi, bx.widthPsi};
                final boolean[] full = {bx.fullPhi, bx.fullPsi};
                final int[] nn = {n1, n2};
                final double[] h = {L[0] / (double) n1, L[1] / (double) n2};
                hhAll[gsl] = h[0] * h[1];
                for (int i = 0; i < n1; i++) om1All[gsl * n1 + i] = (TWO_PI * pf[i]) / L[0];
                for (int j = 0; j < n2h; j++) om2All[gsl * n2h + j] = (TWO_PI * qf[j]) / L[1];
                x1All[gsl] = new double[n1];
                x2All[gsl] = new double[n2];
                for (int i = 0; i < n1; i++) x1All[gsl][i] = o0[0] + (double) i * h[0];
                for (int j = 0; j < n2; j++) x2All[gsl][j] = o0[1] + (double) j * h[1];
                final List<double[]> tips = members.get(b);
                // margins and LEAK band (L347-358)
                final double[] margin = {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY};
                for (double[] t : tips) {
                    for (int ax = 0; ax < 2; ax++) {
                        if (!full[ax]) margin[ax] = Math.min(Math.min(margin[ax], t[1 + ax]), L[ax] - t[1 + ax]);
                    }
                }
                for (int ax = 0; ax < 2; ax++) {
                    if (full[ax]) { nbAll[2 * gsl + ax] = 0; continue; }
                    final double band = !Double.isFinite(margin[ax]) ? 2.0 * h[ax] : Math.max(2.0 * h[ax], 0.25 * margin[ax]);
                    nbAll[2 * gsl + ax] = (int) Math.min(nn[ax] / 2, Math.max(2L, (long) Math.ceil(band / h[ax] - 1e-9)));
                }
                // tip LEAK (L359-369): Gaussian tail at the innermost band cell
                for (double[] t : tips) {
                    double lk = 0.0;
                    for (int ax = 0; ax < 2; ax++) {
                        if (full[ax]) continue;
                        final int nbx = nbAll[2 * gsl + ax];
                        final double hx = h[ax], d = t[1 + ax];
                        final double delta = Math.min(d - (nbx - 1) * hx, (nn[ax] - nbx) * hx - d);
                        lk = Math.max(lk, delta <= 0 ? 1.0 : Math.exp(-0.5 * delta * delta / sigmaEps[ax][ax]));
                    }
                    if (lk > leakSite) leakSite = lk;
                }
                // tip RESOLUTION (L332-346): exp(-min P) over the outer modes, P = 1/2 omega' Pi omega
                if (!tips.isEmpty()) {
                    double pmin = Double.POSITIVE_INFINITY;
                    for (int i = 0; i < n1; i++) {
                        final double w1 = om1All[gsl * n1 + i];
                        for (int j = 0; j < n2h; j++) {
                            if (!((i >= r0 && i <= r1) || j >= c0)) continue;
                            final double w2 = om2All[gsl * n2h + j];
                            final double P = (pi00h * (w1 * w1) + (pi01 * w1) * w2) + pi11h * (w2 * w2);
                            if (P < pmin) pmin = P;
                        }
                    }
                    final double tr = Math.exp(-pmin);
                    if (tr > resSite) resSite = tr;
                }
            }
            tipLeak[site] = leakSite;
            tipRes[site] = resSite;
            // which nodes carry a message at this site, and the per-op product code (0 skip, 1 first, 2 multiply)
            final boolean[] has = new boolean[nodeCount];
            final boolean[] seen = new boolean[nodeCount];
            int op = 0;
            for (int v : order) {
                if (isTip[v]) has[v] = tb[v] >= 0;
                else for (int c : this.children[v]) has[v] |= has[c];
                if (v == root) continue;
                final int p = parent[v];
                codes[op * nSites + site] = (byte) (!has[v] ? 0 : (seen[p] ? 2 : 1));
                if (has[v]) seen[p] = true;
                op++;
            }
            siteHasRoot[site] = has[root];
            if (has[root] && bxs.length == 0) throw new IllegalArgumentException("site " + site + " has observed tips but no box");
        }

        // ---- device: context, module, plans. Exception safe: if anything below throws, every plan, buffer, the
        // module and the retained primary context acquired so far are released before the exception propagates
        this.deviceId = deviceId;
        this.codesHost = codes;
        this.tipBoxHost = tipBoxDev;
        boolean built = false;
        try {
            JCudaDriver.setExceptionsEnabled(true);
            JCufft.setExceptionsEnabled(true);
            cuInit(0);
            final CUdevice device = new CUdevice();
            cuDeviceGet(device, deviceId);
            final CUcontext ctx = new CUcontext();
            cuDevicePrimaryCtxRetain(ctx, device);
            this.context = ctx;
            cuCtxSetCurrent(context);
            final CUmodule mod = new CUmodule();
            cuModuleLoadData(mod, loadPtx());
            this.module = mod;
            this.fGEval = function("g_eval");
            this.fGpScale = function("gp_scale");
            this.fETable = function("e_table");
            this.fSiteInit = function("site_init");
            this.fMusub = function("musub_build");
            this.fTipAxis = function("tip_axis");
            this.fTipBuild = function("tip_build");
            this.fSlotPrep = function("slot_prep");
            this.fPropagate = function("propagate");
            this.fSiteLanding = function("site_landing");
            this.fHermitianize = function("hermitianize");
            this.fCombine = function("combine");
            this.fSiteProduct = function("site_product");
            this.fScaleParts = function("scale_parts");
            this.fRootStats = function("root_stats");
            this.fRootRes = function("root_res");
            this.fRootT = function("root_T");
            this.fRootDot = function("root_dot");
            this.fRootFinal = function("root_final");
            this.fZeroGrid = function("zero_idle_grid");
            this.fZeroSpec = function("zero_idle_spec");
            this.fFillNoise = function("fill_noise");

            // ---- work buffers (maxBatch slots), zeroed so that padding slots hold finite values
            final long B = this.maxBatch;
            this.dX = zeros(B * N2 * 16L);
            this.dR = zeros(B * NG * 8L);
            this.dBuf = new CUdeviceptr[nBuffers];
            for (int i = 0; i < nBuffers; i++) dBuf[i] = zeros(B * NG * 8L);
            this.dT = zeros(B * M * (long) n2 * 16L);
            this.dTipAx = zeros(B * (long) (n1 + n2h) * 16L);
            this.dMU = zeros(B * (long) M * M * 16L);
            this.dPart = zeros(B * nblk * NPART * 8L);
            this.dC = zeros((long) nodeCount * B * 8L);
            this.dLS = zeros((long) nodeCount * B * 8L);
            this.dPres = zeros((long) nodeCount * B * 4L);
            this.dPresTmp = zeros(B * 4L);
            this.dOPres = zeros(B * 4L);
            this.dZeros = zeros(B * 8L);
            this.dCp = zeros(B * 8L);
            this.dScaleS = zeros(B * 8L);
            this.dX00 = zeros(B * 16L);
            this.dX0 = zeros(B * 8L);
            this.dLand = zeros(B * 8L);
            this.dMode = zeros(B * 4L);
            this.dSlotInfo = zeros(B * 4L * 4L);
            this.dSiteInfo = zeros(B * 4L * 4L);
            this.dSiteParam = zeros(B * 8L * 8L);
            this.dGLeak = zeros(B * 8L);
            this.dGRes = zeros(B * 8L);
            this.dGRho = zeros(B * 8L);
            this.dGStat = zeros(B * 4L);
            this.dGDead = zeros(B * 4L);
            this.dGNode = zeros(B * 4L);
            this.dOutLogL = zeros(B * 8L);

            // ---- cuFFT plans and their construction self-test (see setupTransforms)
            setupTransforms();
            fftKeepsInput = d2zInputTest();

            // ---- persistent tables
            this.dOM1 = upload(om1All);
            this.dOM2 = upload(om2All);
            this.dHH = upload(hhAll);
            this.dNB = upload(nbAll);
            this.dTipLeak = upload(tipLeak);
            this.dTipRes = upload(tipRes);
            this.dCodes = uploadBytes(codes);
            final byte[] hasRootB = new byte[nSites];
            for (int site = 0; site < nSites; site++) hasRootB[site] = (byte) (siteHasRoot[site] ? 1 : 0);
            this.dHasRoot = uploadBytes(hasRootB);
            this.dTipBox = upload(tipBoxDev.length == 0 ? new int[1] : tipBoxDev);
            this.dPhaseOff = upload(phaseOff.length == 0 ? new long[1] : phaseOff);
            this.dGsub = upload(mix.gsub);
            this.dGP = alloc((long) ns * N2 * 16L);
            this.dE1 = alloc((long) ns * n1 * M * 16L);
            this.dE2 = alloc((long) ns * n2 * M * 16L);
            this.dPhase = alloc(Math.max(1L, phaseLen) * 8L);

            // g_sup (L787-800): 1.01 x max(g on a 0.25 deg torus grid, g at the component means)
            {
                final int ng = 1440;
                final double[] xg = new double[ng];
                for (int i = 0; i < ng; i++) xg[i] = (double) i * (TWO_PI / 1440);
                final CUdeviceptr tmp = alloc((long) ng * ng * 8L);
                gEval(mix, xg, xg, tmp);
                final double[] gv = new double[ng * ng];
                cuMemcpyDtoH(Pointer.to(gv), tmp, 8L * gv.length);
                double gmax = Double.NEGATIVE_INFINITY;
                for (double v : gv) gmax = Math.max(gmax, v);
                for (int k = 0; k < mix.K; k++) {
                    gEval(mix, new double[]{mu[k][0]}, new double[]{mu[k][1]}, tmp);
                    final double[] one = new double[1];
                    cuMemcpyDtoH(Pointer.to(one), tmp, 8L);
                    gmax = Math.max(gmax, one[0]);
                }
                freeNow(tmp);
                this.gsup = 1.01 * gmax;
            }
            // gam_b (L370-377): g on every box grid, rfft2, times wq h1 h2 / (n1 n2); groups of maxBatch slots
            // through the engine's own D2Z (dR -> dX), batched or slot by slot as every other transform
            final double[] wq = new double[n2h];
            Arrays.fill(wq, 2.0);
            wq[0] = 1.0;
            if (n2 % 2 == 0) wq[n2h - 1] = 1.0;
            for (int g0 = 0; g0 < ns; g0 += this.maxBatch) {
                final int g1 = Math.min(ns, g0 + this.maxBatch);
                for (int gsl = g0; gsl < g1; gsl++) {
                    gEval(mix, x1All[gsl], x2All[gsl], (CUdeviceptr) dR.withByteOffset((gsl - g0) * NG * 8L));
                }
                if (slotwiseFor(g1 - g0)) {
                    for (int k = 0; k < g1 - g0; k++) d2zSlot(dR, k, dX);
                } else {
                    if (g1 - g0 < this.maxBatch) {                 // padding slots: zero, not the previous group's g
                        cuMemsetD8(dR.withByteOffset((g1 - g0) * NG * 8L), (byte) 0, (this.maxBatch - (g1 - g0)) * NG * 8L);
                    }
                    JCufft.cufftExecD2Z(planD2Z, dR, dX);
                }
                for (int gsl = g0; gsl < g1; gsl++) {
                    final double hfac = hhAll[gsl] / (double) NG;
                    final double[] colfac = new double[n2h];
                    for (int j = 0; j < n2h; j++) colfac[j] = wq[j] * hfac;
                    final CUdeviceptr dcf = upload(colfac);
                    launchGrid(fGpScale, gridFor(N2), 1, Pointer.to((CUdeviceptr) dGP.withByteOffset(gsl * N2 * 16L)),
                            Pointer.to((CUdeviceptr) dX.withByteOffset((gsl - g0) * N2 * 16L)), Pointer.to(dcf),
                            Pointer.to(new int[]{n2h}), Pointer.to(new long[]{N2}));
                    cuCtxSynchronize();
                    freeNow(dcf);
                }
            }
            // E1, E2 (L379-382): exp(-1j x m) on every box grid, |m_i| <= Mcut
            for (int gsl = 0; gsl < ns; gsl++) {
                eTable(x1All[gsl], (CUdeviceptr) dE1.withByteOffset((long) gsl * n1 * M * 16L));
                eTable(x2All[gsl], (CUdeviceptr) dE2.withByteOffset((long) gsl * n2 * M * 16L));
            }
            // tip phase tables (L384-391): th = 2 pi frac(p d / L) per axis, reduced error-free (double-double)
            for (int site = 0; site < nSites; site++) {
                for (int v = 0; v < nodeCount; v++) {
                    if (!isTip[v] || tipBox[site][v] < 0) continue;
                    final Box bx = boxes[site][tipBox[site][v]];
                    final double[] d = tipOffset[site][v];
                    final double[] th = new double[n1 + n2h];
                    for (int i = 0; i < n1; i++) th[i] = tipPhase(pf[i], d[0], bx.widthPhi);
                    for (int j = 0; j < n2h; j++) th[n1 + j] = tipPhase(qf[j], d[1], bx.widthPsi);
                    final long off = phaseOff[tipOrdinal[v] * nSites + site];
                    cuMemcpyHtoD(dPhase.withByteOffset(off * 8L), Pointer.to(th), 8L * th.length);
                }
            }
            // the transform buffers start every call from zero, not from the constructor's g and Gp
            cuMemsetD8(dR, (byte) 0, B * NG * 8L);
            cuMemsetD8(dX, (byte) 0, B * N2 * 16L);
            cuCtxSynchronize();
            built = true;
        } finally {
            if (!built) {
                closed = true;
                releaseAll();                                  // best effort; the original exception propagates
            }
        }
    }

    // ---- cuFFT plans, the slot-independence self-test and the transform policy ------------------------------

    /**
     * Creates the cuFFT plans and decides how transforms run. cuFFT's batched D2Z / Z2D is not slot-independent at
     * every size: at some n (seen at a last dimension of 75, 375 and 1125 with cuFFT 11.3 on an RTX A6000) a slot's
     * transform picks up round-off proportional to the OTHER batch elements' magnitude, so a site's value would
     * depend on its other boxes, on the other sites of its chunk and on stale padding. This is tested here, on this
     * device and library, before any table is built:
     *
     * <ol>
     *   <li>independence: probe slots 0, 1 and maxBatch - 1 hold a fixed field (its spectrum for the Z2D); the
     *   batched D2Z and Z2D must give the probe slot bitwise the same output with every other slot zero and with
     *   every other slot holding noise of size 1e6;</li>
     *   <li>equivalence: the single-slot plans must give the probe slot bitwise the batched output.</li>
     * </ol>
     *
     * Independent and equivalent: FFT_HYBRID, each transform runs batched or slot by slot, whichever is cheaper
     * (the values are bitwise the same). Independent only: FFT_BATCHED. Not independent: FFT_SLOTWISE, every
     * transform slot by slot (only the slots a step uses). The constructor argument fftPolicy = FFT_BATCHED |
     * FFT_SLOTWISE forces a policy (tests only; batched refuses a coupled n). Before every BATCHED transform the slots
     * whose content is not read afterwards (padding and idle slots) are zeroed, so its input depends on the current
     * call alone.
     */
    private void setupTransforms() {
        final String force = fftPolicy == FFT_BATCHED ? "batched" : fftPolicy == FFT_SLOTWISE ? "slotwise" : "auto";
        final boolean wantBatched = !force.equals("slotwise") && maxBatch > 1;
        final boolean wantSingle = !force.equals("batched") || maxBatch == 1;
        if (wantBatched) {
            planD2Z = makePlan(cufftType.CUFFT_D2Z, maxBatch);
            planZ2D = makePlan(cufftType.CUFFT_Z2D, maxBatch);
        }
        if (wantSingle) {
            plan1D2Z = makePlan(cufftType.CUFFT_D2Z, 1);
            plan1Z2D = makePlan(cufftType.CUFFT_Z2D, 1);
        }
        attachWorkArea();
        if (maxBatch == 1) {                                   // one slot: the single plans ARE the batched ones
            planD2Z = plan1D2Z;
            planZ2D = plan1Z2D;
            fftIndependent = fftEquivalent = true;
            fftMode = FFT_BATCHED;
            return;
        }
        if (force.equals("slotwise")) {
            fftMode = FFT_SLOTWISE;
            if (((long) n1 * n2) % 2 == 1) alignBuffer();
            return;
        }
        final boolean[] t = fftSelfTest();
        fftTested = true;
        fftIndependent = t[0];
        fftEquivalent = t[1];
        if (!fftIndependent) {
            if (force.equals("batched")) {
                throw new IllegalStateException("fftPolicy FFT_BATCHED, but cuFFT's batched " + n1 + " x " + n2
                        + " transforms are not slot-independent on device " + deviceId);
            }
            fftMode = FFT_SLOTWISE;
            destroyPlan(planD2Z);
            destroyPlan(planZ2D);
            planD2Z = planZ2D = null;
        } else if (fftEquivalent && !force.equals("batched")) {
            fftMode = FFT_HYBRID;
        } else {
            fftMode = FFT_BATCHED;
            destroyPlan(plan1D2Z);
            destroyPlan(plan1Z2D);
            plan1D2Z = plan1Z2D = null;
        }
        attachWorkArea();                                      // shrink the shared work area to the plans kept
        if (plan1D2Z != null && ((long) n1 * n2) % 2 == 1) alignBuffer();     // allocated here, not mid-call
    }

    private cufftHandle makePlan(int type, int batch) {
        final cufftHandle h = new cufftHandle();
        JCufft.cufftCreate(h);
        plans.add(h);
        JCufft.cufftSetAutoAllocation(h, 0);
        final long[] ws = new long[1];
        final int NG = n1 * n2, N2 = n1 * n2h;
        if (type == cufftType.CUFFT_D2Z) {
            JCufft.cufftMakePlanMany(h, 2, new int[]{n1, n2}, null, 1, NG, null, 1, N2, type, batch, ws);
        } else {
            JCufft.cufftMakePlanMany(h, 2, new int[]{n1, n2}, null, 1, N2, null, 1, NG, type, batch, ws);
        }
        planWork.put(h, ws[0]);
        return h;
    }

    private final java.util.Map<cufftHandle, Long> planWork = new java.util.IdentityHashMap<cufftHandle, Long>();

    /** One work area of the largest size any live plan needs, given to every live plan. */
    private void attachWorkArea() {
        long need = 8L;
        for (cufftHandle h : plans) need = Math.max(need, planWork.get(h));
        if (dFftWork == null || need != fftWorkBytes) {
            if (dFftWork != null) freeNow(dFftWork);
            dFftWork = null;
            dFftWork = alloc(need);
            fftWorkBytes = need;
        }
        for (cufftHandle h : plans) JCufft.cufftSetWorkArea(h, dFftWork);
    }

    private void destroyPlan(cufftHandle h) {
        if (h == null || !plans.remove(h)) return;
        planWork.remove(h);
        JCufft.cufftDestroy(h);
    }

    /** {independent, equivalent}: see setupTransforms. Uses dR and dX and leaves them zero. */
    private boolean[] fftSelfTest() {
        final int B = maxBatch;
        final long NG = (long) n1 * n2, N2 = (long) n1 * n2h;
        final long seed = 0x7B0C5EEDL;
        boolean indep = true, equiv = true;
        final int[] probes = B > 2 ? new int[]{0, 1, B - 1} : new int[]{0, 1};
        for (int s : probes) {
            final CUdeviceptr rS = (CUdeviceptr) dR.withByteOffset(s * NG * 8L);
            final CUdeviceptr xS = (CUdeviceptr) dX.withByteOffset(s * N2 * 16L);
            // the probe field alone, every other slot zero
            cuMemsetD8(dR, (byte) 0, B * NG * 8L);
            fillNoise(rS, NG, seed, 1.0);
            JCufft.cufftExecD2Z(planD2Z, dR, dX);
            final double[] specA = fetch(xS, 2 * N2);
            cuMemsetD8(dX, (byte) 0, B * N2 * 16L);
            cuMemcpyHtoD(xS, Pointer.to(specA), 16L * N2);
            JCufft.cufftExecZ2D(planZ2D, dX, dR);                 // (a C2R may overwrite its input)
            final double[] gridA = fetch(rS, NG);
            // the same field, every other slot holding noise 1e6 times larger
            for (int o = 0; o < B; o++) {
                fillNoise((CUdeviceptr) dR.withByteOffset(o * NG * 8L), NG, o == s ? seed : seed + 1 + o, o == s ? 1.0 : 1e6);
            }
            JCufft.cufftExecD2Z(planD2Z, dR, dX);
            indep &= Arrays.equals(fetch(xS, 2 * N2), specA);    // Arrays.equals(double[]) compares bit patterns
            cuMemcpyHtoD(xS, Pointer.to(specA), 16L * N2);       // the other slots keep the noise spectra
            JCufft.cufftExecZ2D(planZ2D, dX, dR);
            indep &= Arrays.equals(fetch(rS, NG), gridA);
            // the single-slot plans on this slot
            if (plan1D2Z != null) {
                fillNoise(rS, NG, seed, 1.0);
                d2zSlot(dR, s, dX);
                equiv &= Arrays.equals(fetch(xS, 2 * N2), specA);
                cuMemcpyHtoD(xS, Pointer.to(specA), 16L * N2);
                z2dSlot(dX, s, dR);
                equiv &= Arrays.equals(fetch(rS, NG), gridA);
            } else {
                equiv = false;
            }
        }
        cuMemsetD8(dR, (byte) 0, B * NG * 8L);
        cuMemsetD8(dX, (byte) 0, B * N2 * 16L);
        cuCtxSynchronize();
        return new boolean[]{indep, equiv};
    }

    /**
     * Whether every D2Z plan this engine keeps (batched and single-slot) leaves its input grid bitwise unchanged on
     * this device and library (slot 0 holding noise, the batched plan's other slots zero). cuFFT documents only that
     * a C2R (Z2D) may overwrite its input. The engine does not depend on the answer: no step reads the input of a
     * D2Z after the transform (the root's root_stats and root_dot run before its D2Z). The result is reported by
     * describeTransforms, so a device whose R2C kernels use their input as scratch is visible in the log. Uses dR and
     * dX and leaves them zero.
     */
    private boolean d2zInputTest() {
        final long NG = (long) n1 * n2, N2 = (long) n1 * n2h;
        final long seed = 0x1A9E7B0CL;
        boolean kept = true;
        if (planD2Z != null) {
            cuMemsetD8(dR, (byte) 0, maxBatch * NG * 8L);
            fillNoise(dR, NG, seed, 1.0);
            final double[] before = fetch(dR, NG);
            JCufft.cufftExecD2Z(planD2Z, dR, dX);
            kept &= Arrays.equals(fetch(dR, NG), before);
        }
        if (plan1D2Z != null && plan1D2Z != planD2Z) {
            fillNoise(dR, NG, seed + 1, 1.0);
            final double[] before = fetch(dR, NG);
            d2zSlot(dR, 0, dX);
            kept &= Arrays.equals(fetch(dR, NG), before);
        }
        cuMemsetD8(dR, (byte) 0, maxBatch * NG * 8L);
        cuMemsetD8(dX, (byte) 0, maxBatch * N2 * 16L);
        cuCtxSynchronize();
        return kept;
    }

    private void fillNoise(CUdeviceptr out, long n, long seed, double scale) {
        launchGrid(fFillNoise, gridFor(n), 1, Pointer.to(out), Pointer.to(new long[]{n}), Pointer.to(new long[]{seed}),
                Pointer.to(new double[]{scale}));
    }

    private double[] fetch(CUdeviceptr src, long len) {
        final double[] a = new double[(int) len];
        cuCtxSynchronize();
        cuMemcpyDtoH(Pointer.to(a), src, 8L * len);
        return a;
    }

    /**
     * Whether a transform over k used slots runs slot by slot: always in FFT_SLOTWISE, never in FFT_BATCHED. In
     * FFT_HYBRID (values bitwise identical either way) when it is cheaper. Measured on an RTX A6000 (623-node tree,
     * 42 slots): slot by slot costs about 1.1 x a batched slot at n 1024 and 1.7 x at n 256 (launch-bound), while a
     * batched transform always costs maxBatch slots. So: on grids of 2^18 points and more when 1.1 k < maxBatch,
     * on smaller grids when 2 k <= maxBatch.
     */
    private boolean slotwiseFor(int k) {
        if (fftMode == FFT_SLOTWISE) return true;
        if (fftMode == FFT_BATCHED) return false;
        return ((long) n1 * n2 >= (1L << 18)) ? 11L * k < 10L * maxBatch : 2 * k <= maxBatch;
    }

    /** Single-slot D2Z of slot s: grid values at real + s n1 n2 doubles into the half spectrum at spec + s n1 n2h. */
    private void d2zSlot(CUdeviceptr real, int s, CUdeviceptr spec) {
        final long NG = (long) n1 * n2, ro = s * NG * 8L;
        final Pointer out = spec.withByteOffset(s * (long) n1 * n2h * 16L);
        if ((ro & 15L) == 0) {
            JCufft.cufftExecD2Z(plan1D2Z, real.withByteOffset(ro), out);
        } else {
            cuMemcpyDtoD(alignBuffer(), (CUdeviceptr) real.withByteOffset(ro), NG * 8L);
            JCufft.cufftExecD2Z(plan1D2Z, dAlignR, out);
        }
    }

    /** Single-slot Z2D of slot s (the inverse of d2zSlot; a C2R may overwrite its input). */
    private void z2dSlot(CUdeviceptr spec, int s, CUdeviceptr real) {
        final long NG = (long) n1 * n2, ro = s * NG * 8L;
        final Pointer in = spec.withByteOffset(s * (long) n1 * n2h * 16L);
        if ((ro & 15L) == 0) {
            JCufft.cufftExecZ2D(plan1Z2D, in, real.withByteOffset(ro));
        } else {
            JCufft.cufftExecZ2D(plan1Z2D, in, alignBuffer());
            cuMemcpyDtoD((CUdeviceptr) real.withByteOffset(ro), dAlignR, NG * 8L);
        }
    }

    private CUdeviceptr alignBuffer() {
        if (dAlignR == null) dAlignR = alloc((long) n1 * n2 * 8L);
        return dAlignR;
    }

    // ---- public accessors -----------------------------------------------------------------------------------

    /** LEAK and RESOLUTION thresholds applied to the returned guard values (Python leak_tol, res_tol; default 1e-12). */
    public void setTolerances(double leakTol, double resTol) {
        if (!(leakTol > 0 && resTol > 0)) throw new IllegalArgumentException("tolerances must be positive");
        this.leakTol = leakTol;
        this.resTol = resTol;
    }

    public int getSiteCount() { return nSites; }
    public int getSlotCount() { return nSlotsTotal; }
    public int getGridPhi() { return n1; }
    public int getGridPsi() { return n2; }
    public int getMcut() { return Mcut; }
    public double getGSup() { return gsup; }
    public int getBufferCount() { return nBuffers; }
    public int getMaxBatch() { return maxBatch; }
    public int[] getOrder() { return order.clone(); }
    public double getTipLeak(int site) { return tipLeak[site]; }
    public double getTipResolution(int site) { return tipRes[site]; }
    public long getEvaluationCount() { return evaluations; }
    public long getLeakFlagCount() { return leakFlags; }
    public long getResolutionFlagCount() { return resFlags; }
    public long getNonpositiveCount() { return nonpositive; }
    public double getLastSeconds() { return lastSeconds; }
    /** FFT_BATCHED, FFT_HYBRID or FFT_SLOTWISE: how this engine's transforms run (see setupTransforms). */
    public int getFftMode() { return fftMode; }
    /** Whether the construction self-test ran (maxBatch > 1 and no forced slotwise policy). */
    public boolean isFftTested() { return fftTested; }
    /** Self-test: the batched plans' output for a slot does not depend on the other slots (bitwise). */
    public boolean isFftSlotIndependent() { return fftIndependent; }
    /** Self-test: the single-slot plans give bitwise the batched plans' output. */
    public boolean isFftSingleEquivalent() { return fftEquivalent; }
    /** Whether the D2Z plans left their input bitwise unchanged at construction (informational; see d2zInputTest). */
    public boolean isFftD2ZInputKept() { return fftKeepsInput; }
    /** The transform policy given to the constructor: FFT_AUTO, FFT_BATCHED or FFT_SLOTWISE. */
    public int getFftPolicy() { return fftPolicy; }
    /** Bytes of the one cuFFT work area all plans share. */
    public long getFftWorkBytes() { return fftWorkBytes; }

    public String describeTransforms() {
        final String mode = fftMode == FFT_SLOTWISE ? "slotwise" : fftMode == FFT_HYBRID ? "hybrid" : "batched";
        return mode + " (n " + n1 + " x " + n2 + ", maxBatch " + maxBatch + (fftTested ? ", self-test: batched plans "
                + (fftIndependent ? "slot-independent" : "COUPLE slots") + ", single-slot plans "
                + (fftEquivalent ? "bitwise equal" : "differ or unused") : ", not tested")
                + (fftPolicy == FFT_AUTO ? "" : ", policy FORCED by the constructor")
                + (fftKeepsInput ? ", D2Z input kept" : ", D2Z OVERWRITES its input (harmless: no step reads it)")
                + ", work area " + fftWorkBytes + " B)";
    }

    /**
     * The one guard policy callers should apply: a value is usable only with status 0. LEAK, RESOLUTION and
     * NONPOSITIVE must be treated alike, because where RESOLUTION is far above its tolerance (an under-resolved,
     * aliased grid) the sign of the root integral or of a node product is decided by round-off, so whether such a
     * state comes back NONPOSITIVE (NaN) or finite with LEAK / RESOLUTION differs between this engine and
     * box_landing.py (seen at a RESOLUTION ratio of 8e11). Neither value is a likelihood there.
     */
    public static boolean usable(int status) { return status == STATUS_OK; }

    /**
     * Log-likelihoods of the listed sites, each at its own diffusion covariance and landing rate.
     *
     * @param sites     site indices (any order, repeats allowed)
     * @param sigma4    [4 k .. 4 k + 3] = Sigma00, Sigma01, Sigma10, Sigma11 of sites[k] (rad^2 per year; symmetric to
     *                  1e-12 relative and positive semi-definite, as box_landing.py _params requires)
     * @param lambda    [k] landing rate of sites[k] (per year, finite, >= 0)
     * @param outLogL   out: [k] log-likelihood; NaN at a NONPOSITIVE node scale, NaN or -infinity at a
     *                  NONPOSITIVE root (as box_landing.py returns them); 0 for a site with no observed tip
     * @param outStatus out (nullable): [k] STATUS_* bits: LEAK and RESOLUTION from the tolerances, NONPOSITIVE
     * @param outLeak   out (nullable): [k] the LEAK value (max over the evaluation, tip LEAK included)
     * @param outRes    out (nullable): [k] the RESOLUTION ratio (max over the evaluation, tip RESOLUTION included)
     * @param outRhoMin out (nullable): [k] the worst product overlap s / sc (1 when no product was formed)
     */
    public void logLikelihood(int[] sites, double[] sigma4, double[] lambda, double[] outLogL, int[] outStatus,
                              double[] outLeak, double[] outRes, double[] outRhoMin) {
        if (closed) throw new IllegalStateException("engine closed");
        final int k = sites.length;
        if (sigma4.length < 4 * k || lambda.length < k || outLogL.length < k) {
            throw new IllegalArgumentException("sites, sigma4 (4 per site), lambda and outLogL must have matching lengths");
        }
        final long t0 = System.nanoTime();
        final double[] sym = new double[3 * k];
        for (int i = 0; i < k; i++) {
            if (sites[i] < 0 || sites[i] >= nSites) throw new IllegalArgumentException("site " + sites[i] + " out of range");
            params(sigma4, 4 * i, lambda[i], sym, 3 * i);
        }
        cuCtxSetCurrent(context);
        final int[] chunk = new int[maxBatch];
        int nc = 0, slots = 0;
        for (int i = 0; i < k; i++) {
            final int site = sites[i];
            if (siteBoxes[site] == 0 || !siteHasRoot[site]) {          // no observed tip: L = 1 (L1077-1080, L1587-1588)
                outLogL[i] = 0.0;
                if (outStatus != null) outStatus[i] = guardStatus(tipLeak[site], tipRes[site], 0);
                if (outLeak != null) outLeak[i] = tipLeak[site];
                if (outRes != null) outRes[i] = tipRes[site];
                if (outRhoMin != null) outRhoMin[i] = 1.0;
                continue;
            }
            if (slots + siteBoxes[site] > maxBatch) {
                evaluateChunk(sites, sym, lambda, chunk, nc, slots, outLogL, outStatus, outLeak, outRes, outRhoMin);
                nc = 0;
                slots = 0;
            }
            chunk[nc++] = i;
            slots += siteBoxes[site];
        }
        if (nc > 0) evaluateChunk(sites, sym, lambda, chunk, nc, slots, outLogL, outStatus, outLeak, outRes, outRhoMin);
        evaluations += k;
        lastSeconds = (System.nanoTime() - t0) * 1e-9;
    }

    private int guardStatus(double leak, double res, int stat) {
        int st = stat;
        if (!(leak <= leakTol)) st |= STATUS_LEAK;
        if (!(res <= resTol)) st |= STATUS_RESOLUTION;
        return st;
    }

    // ---- one chunk ------------------------------------------------------------------------------------------

    private void evaluateChunk(int[] sites, double[] sym, double[] lambda, int[] chunk, int nc, int nSl,
                               double[] outLogL, int[] outStatus, double[] outLeak, double[] outRes,
                               double[] outRhoMin) {
        final int[] slotInfo = new int[4 * nSl], siteInfo = new int[4 * nc];
        final int[] slotSite = new int[nSl], slotBox = new int[nSl];        // host copies for the transform policy
        final boolean[] idle = new boolean[nSl];
        final double[] siteParam = new double[8 * nc];
        boolean anyLam = false;
        int sl = 0;
        for (int j = 0; j < nc; j++) {
            final int i = chunk[j], site = sites[i];
            siteInfo[4 * j] = site;
            siteInfo[4 * j + 1] = sl;
            siteInfo[4 * j + 2] = siteBoxes[site];
            for (int b = 0; b < siteBoxes[site]; b++) {
                slotInfo[4 * sl] = siteSlot0[site] + b;
                slotInfo[4 * sl + 1] = j;
                slotInfo[4 * sl + 2] = b;
                slotSite[sl] = site;
                slotBox[sl] = b;
                sl++;
            }
            final double S00 = sym[3 * i], S01 = sym[3 * i + 1], S11 = sym[3 * i + 2];
            siteParam[8 * j] = S00;
            siteParam[8 * j + 1] = S01;
            siteParam[8 * j + 2] = S11;
            siteParam[8 * j + 3] = 0.5 * S00;
            siteParam[8 * j + 4] = 0.5 * S11;
            siteParam[8 * j + 5] = lambda[i];
            anyLam |= lambda[i] > 0.0;
        }
        cuMemcpyHtoD(dSlotInfo, Pointer.to(slotInfo), 4L * slotInfo.length);
        cuMemcpyHtoD(dSiteInfo, Pointer.to(siteInfo), 4L * siteInfo.length);
        cuMemcpyHtoD(dSiteParam, Pointer.to(siteParam), 8L * siteParam.length);
        final long NG = (long) n1 * n2, N2 = (long) n1 * n2h;
        final Pointer pSlot = Pointer.to(dSlotInfo), pSite = Pointer.to(dSiteInfo), pParam = Pointer.to(dSiteParam);
        final Pointer pDead = Pointer.to(dGDead), pPart = Pointer.to(dPart);
        final Pointer iN1 = Pointer.to(new int[]{n1}), iN2 = Pointer.to(new int[]{n2}), iN2h = Pointer.to(new int[]{n2h});
        final Pointer iNblk = Pointer.to(new int[]{nblk});

        launchGrid(fSiteInit, (nc + BLOCK - 1) / BLOCK, 1, Pointer.to(new int[]{nc}), pSite, Pointer.to(dTipLeak),
                Pointer.to(dTipRes), Pointer.to(dGLeak), Pointer.to(dGRes), Pointer.to(dGRho), Pointer.to(dGStat), pDead,
                Pointer.to(dGNode));
        if (anyLam) {
            launchGrid(fMusub, (int) Math.min(1024L, ((long) M * M + BLOCK - 1) / BLOCK), nc, Pointer.to(dMU),
                    Pointer.to(dGsub), pParam, Pointer.to(new int[]{M}), Pointer.to(new int[]{Mcut}));
        }

        for (int o = 0; o < opChild.length; o++) {
            final int v = opChild[o], p = opParent[o];
            final double t = branch[v];
            final Pointer pCode = Pointer.to((CUdeviceptr) dCodes.withByteOffset((long) o * nSites));
            final Pointer pT = Pointer.to(new double[]{t});
            final CUdeviceptr presChild, cChild, lsChild;
            if (isTip[v]) {
                final int to = tipOrdinal[v];
                final Pointer pTipBox = Pointer.to((CUdeviceptr) dTipBox.withByteOffset((long) to * nSites * 4L));
                launchGrid(fTipAxis, (n1 + n2h + BLOCK - 1) / BLOCK, nSl, Pointer.to(dTipAx), pSlot, pSite, pCode, pDead,
                        pTipBox, Pointer.to((CUdeviceptr) dPhaseOff.withByteOffset((long) to * nSites * 8L)),
                        Pointer.to(dPhase), iN1, iN2h);
                launchGrid(fTipBuild, nblk, nSl, Pointer.to(dX), Pointer.to(dPresTmp), Pointer.to(dTipAx), pSlot, pSite,
                        pCode, pDead, pTipBox, Pointer.to(dOM1), Pointer.to(dOM2), Pointer.to(dHH),
                        Pointer.to(new double[]{pi00h}), Pointer.to(new double[]{pi01}), Pointer.to(new double[]{pi11h}),
                        iN1, iN2h);
                presChild = dPresTmp;
                cChild = dZeros;
                lsChild = dZeros;
            } else {
                // a slot is idle for this D2Z when its site is skipped at this op (known here); slots without a part
                // or of a dead site are known on the device only (zeroed there before a batched transform)
                int used = 0;
                for (int s = 0; s < nSl; s++) {
                    idle[s] = codesHost[o * nSites + slotSite[s]] == 0;
                    if (!idle[s]) used++;
                }
                final CUdeviceptr in = dBuf[opBufChild[o]];
                if (slotwiseFor(used)) {
                    for (int s = 0; s < nSl; s++) {
                        if (!idle[s]) d2zSlot(in, s, dX);
                    }
                } else {
                    launchGrid(fZeroGrid, nblk, maxBatch, Pointer.to(in), Pointer.to(presOf(v)), pSlot, pSite, pCode, pDead,
                            Pointer.to(new int[]{nSl}), Pointer.to(new long[]{NG}));
                    JCufft.cufftExecD2Z(planD2Z, in, dX);
                }
                presChild = presOf(v);
                cChild = cOf(v);
                lsChild = lsOf(v);
            }
            final int multi = opMulti[o] ? 1 : 0;
            launchGrid(fSlotPrep, (nSl + BLOCK - 1) / BLOCK, 1, Pointer.to(dX), Pointer.to(presChild), pSlot, pSite, pCode,
                    pDead, pParam, Pointer.to(dHH), pT, Pointer.to(new int[]{nSl}), Pointer.to(new long[]{N2}),
                    Pointer.to(dX00), Pointer.to(dX0), Pointer.to(dLand), Pointer.to(dMode));
            launchGrid(fPropagate, nblk, nSl, Pointer.to(dX), Pointer.to(dMode), pSlot, pParam, Pointer.to(dOM1),
                    Pointer.to(dOM2), Pointer.to(dGP), pT, Pointer.to(new int[]{multi}), iN1, iN2h,
                    Pointer.to(new int[]{r0}), Pointer.to(new int[]{r1}), Pointer.to(new int[]{c0}), pPart);
            launchGrid(fSiteLanding, nc, 1, pSite, pCode, pDead, Pointer.to(dMode), pParam, Pointer.to(cChild), pPart,
                    iNblk, Pointer.to(new int[]{multi}), pT, Pointer.to(dX00), Pointer.to(dX0), Pointer.to(dCp),
                    Pointer.to(dGRes));
            launchGrid(fHermitianize, (2 * (n1 / 2 + 1) + BLOCK - 1) / BLOCK, nSl, Pointer.to(dX), Pointer.to(dMode),
                    iN1, iN2, iN2h);
            // the Z2D: a slot is idle when its site is skipped at this op or (a tip) the observation is in another
            // box (known here), or its mode is 0 or 3 (known on the device only)
            {
                int used = 0;
                final int tipOrd = isTip[v] ? tipOrdinal[v] : -1;
                for (int s = 0; s < nSl; s++) {
                    idle[s] = codesHost[o * nSites + slotSite[s]] == 0
                            || (tipOrd >= 0 && tipBoxHost[tipOrd * nSites + slotSite[s]] != slotBox[s]);
                    if (!idle[s]) used++;
                }
                if (slotwiseFor(used)) {
                    for (int s = 0; s < nSl; s++) {
                        if (!idle[s]) z2dSlot(dX, s, dR);
                    }
                } else {
                    launchGrid(fZeroSpec, nblk, maxBatch, Pointer.to(dX), Pointer.to(dMode), Pointer.to(new int[]{nSl}),
                            Pointer.to(new long[]{N2}));
                    JCufft.cufftExecZ2D(planZ2D, dX, dR);
                }
            }
            final CUdeviceptr P = dBuf[opBufParent[o]];
            launchGrid(fCombine, nblk, nSl, Pointer.to(P), Pointer.to(dR), Pointer.to(dMode), Pointer.to(presOf(p)),
                    Pointer.to(dOPres), pSlot, pSite, pCode, pDead, Pointer.to(cOf(p)), Pointer.to(dCp), Pointer.to(dNB),
                    Pointer.to(new double[]{fct}), iN1, iN2, pPart);
            launchGrid(fSiteProduct, nc, 1, pSite, pCode, pDead, Pointer.to(dMode), Pointer.to(dOPres),
                    Pointer.to(presOf(p)), pPart, iNblk, Pointer.to(dLand), pT, pParam, Pointer.to(cOf(p)),
                    Pointer.to(lsOf(p)), Pointer.to(lsChild), Pointer.to(dCp), Pointer.to(dScaleS), Pointer.to(dGLeak),
                    Pointer.to(dGRho), Pointer.to(dGStat), Pointer.to(dGNode), Pointer.to(new int[]{p}),
                    Pointer.to(new double[]{gsup}), Pointer.to(new double[]{lconst}),
                    Pointer.to(new int[]{STATUS_NONPOSITIVE}));
            launchGrid(fScaleParts, nblk, nSl, Pointer.to(P), Pointer.to(presOf(p)), pSlot, pSite, pCode, pDead,
                    Pointer.to(dScaleS), Pointer.to(new long[]{NG}));
        }

        // the root (L1575-1638). Every kernel that reads the root grid Proot (root_stats, root_dot) runs BEFORE its
        // D2Z: no step reads the input of a D2Z after the transform, so nothing relies on cuFFT leaving the input of
        // an out-of-place R2C untouched (it is documented only for C2R that the input is overwritten; the self-test
        // records what the running device does, describeTransforms). root_T needs only musub and E2. The kernels
        // write disjoint fields of part (FHI / FLO / FBAND, RS / RE, POST), so the order does not change any value.
        final CUdeviceptr Proot = dBuf[rootBuffer];
        final Pointer pHasRoot = Pointer.to(dHasRoot), pPresRoot = Pointer.to(presOf(root));
        final Pointer iR0 = Pointer.to(new int[]{r0}), iR1 = Pointer.to(new int[]{r1}), iC0 = Pointer.to(new int[]{c0});
        launchGrid(fRootStats, nblk, nSl, Pointer.to(Proot), pPresRoot, pSlot, pSite, pHasRoot, pDead, Pointer.to(dNB),
                iN1, iN2, pPart);
        if (anyLam) {
            launchGrid(fRootT, gridFor((long) M * n2), nSl, Pointer.to(dT), Pointer.to(dMU), Pointer.to(dE2), pPresRoot,
                    pSlot, pSite, pHasRoot, pDead, pParam, iN2, Pointer.to(new int[]{M}));
        }
        launchGrid(fRootDot, nblk, nSl, Pointer.to(Proot), Pointer.to(dT), Pointer.to(dE1), pPresRoot, pSlot, pSite,
                pHasRoot, pDead, pParam, iN1, iN2, Pointer.to(new int[]{M}), pPart);
        if (slotwiseFor(nSl)) {                         // every site of a chunk has a root message
            for (int s = 0; s < nSl; s++) {
                d2zSlot(Proot, s, dX);
            }
        } else {
            launchGrid(fZeroGrid, nblk, maxBatch, Pointer.to(Proot), pPresRoot, pSlot, pSite, pHasRoot, pDead,
                    Pointer.to(new int[]{nSl}), Pointer.to(new long[]{NG}));
            JCufft.cufftExecD2Z(planD2Z, Proot, dX);
        }
        launchGrid(fRootRes, nblk, nSl, Pointer.to(dX), pPresRoot, pSlot, pSite, pHasRoot, pDead, iN1, iN2h, iR0, iR1, iC0,
                pPart);
        launchGrid(fRootFinal, nc, 1, pSite, pSlot, pHasRoot, pDead, pPresRoot, pPart, iNblk, Pointer.to(dX),
                Pointer.to(new long[]{N2}), Pointer.to(cOf(root)), Pointer.to(lsOf(root)), Pointer.to(dHH),
                Pointer.to(new double[]{TWO_PI * TWO_PI}), Pointer.to(dGLeak), Pointer.to(dGRes), Pointer.to(dGStat),
                Pointer.to(dGNode), Pointer.to(dOutLogL), Pointer.to(new int[]{STATUS_NONPOSITIVE}));

        final double[] ll = new double[nc], leak = new double[nc], res = new double[nc], rho = new double[nc];
        final int[] stat = new int[nc];
        cuCtxSynchronize();
        cuMemcpyDtoH(Pointer.to(ll), dOutLogL, 8L * nc);
        cuMemcpyDtoH(Pointer.to(leak), dGLeak, 8L * nc);
        cuMemcpyDtoH(Pointer.to(res), dGRes, 8L * nc);
        cuMemcpyDtoH(Pointer.to(rho), dGRho, 8L * nc);
        cuMemcpyDtoH(Pointer.to(stat), dGStat, 4L * nc);
        for (int j = 0; j < nc; j++) {
            final int i = chunk[j];
            final int st = guardStatus(leak[j], res[j], stat[j]);
            if ((st & STATUS_LEAK) != 0) leakFlags++;
            if ((st & STATUS_RESOLUTION) != 0) resFlags++;
            if ((st & STATUS_NONPOSITIVE) != 0) nonpositive++;
            outLogL[i] = ll[j];
            if (outStatus != null) outStatus[i] = st;
            if (outLeak != null) outLeak[i] = leak[j];
            if (outRes != null) outRes[i] = res[j];
            if (outRhoMin != null) outRhoMin[i] = rho[j];
        }
    }

    private CUdeviceptr cOf(int node) { return (CUdeviceptr) dC.withByteOffset((long) node * maxBatch * 8L); }
    private CUdeviceptr lsOf(int node) { return (CUdeviceptr) dLS.withByteOffset((long) node * maxBatch * 8L); }
    private CUdeviceptr presOf(int node) { return (CUdeviceptr) dPres.withByteOffset((long) node * maxBatch * 4L); }

    /** box_landing.py _params (L1028-1046): Sigma finite, symmetric to 1e-12, positive semi-definite; lambda >= 0. */
    private static void params(double[] s4, int off, double lam, double[] out, int o) {
        final double S00 = s4[off], S01 = s4[off + 1], S10 = s4[off + 2], S11 = s4[off + 3];
        if (!(Double.isFinite(S00) && Double.isFinite(S01) && Double.isFinite(S10) && Double.isFinite(S11))) {
            throw new IllegalArgumentException("Sigma must be finite");
        }
        final double amax = Math.max(Math.max(Math.abs(S00), Math.abs(S01)), Math.max(Math.abs(S10), Math.abs(S11)));
        if (Math.abs(S01 - S10) > 1e-12 * Math.max(amax, 1e-300)) {
            throw new IllegalArgumentException("Sigma must be symmetric (" + S01 + " vs " + S10 + ")");
        }
        final double s01 = (S01 != S10) ? 0.5 * (S01 + S10) : S01;
        final double[] ev = eig2(S00, s01, S11);
        if (ev[0] < -1e-12 * Math.max(ev[1], 0.0) || (ev[0] < -1e-300 && ev[1] <= 0)) {
            throw new IllegalArgumentException("Sigma must be positive semi-definite (eigenvalues " + ev[0] + ", " + ev[1] + ")");
        }
        if (!(Double.isFinite(lam) && lam >= 0.0)) throw new IllegalArgumentException("lambda must be finite and >= 0");
        out[o] = S00;
        out[o + 1] = s01;
        out[o + 2] = S11;
    }

    // ---- the landing mixture (box_landing.py L652-690) ----------------------------------------------------------

    private static final class Mixture {
        final int K;
        final double[] comp;          // [4 k] = c0, 0.5 Sinv00, Sinv01, 0.5 Sinv11
        final double[] lminInv;       // smallest eigenvalue of Sinv (lower triangle, as eigvalsh)
        final int[] J;                // images per axis
        final double[][] mu;
        final int mcut;
        final double[] gsub;          // [M][M] complex (interleaved)

        Mixture(double[] w, double[][] mu, double[][][] s) {
            K = w.length;
            if (K == 0 || mu.length != K || s.length != K) throw new IllegalArgumentException("mixture shapes do not match");
            double wsum = 0.0;
            for (int k = 0; k < K; k++) {
                if (!(Double.isFinite(w[k]) && w[k] >= 0)) throw new IllegalArgumentException("mixture weights must be finite and >= 0");
                wsum += w[k];
                requireSymmetricPd(s[k], "landing component " + k);
                if (mu[k].length != 2 || !Double.isFinite(mu[k][0]) || !Double.isFinite(mu[k][1])) {
                    throw new IllegalArgumentException("landing mean " + k + " must be 2 finite values");
                }
            }
            if (Math.abs(wsum - 1.0) > 1e-6) throw new IllegalArgumentException("the landing weights must sum to 1 (" + wsum + ")");
            this.mu = new double[K][];
            for (int k = 0; k < K; k++) this.mu[k] = mu[k].clone();
            // M_cut (L669-673): the omitted |ghat| sum to < G_REL, decay from the smallest eigenvalue of each S_k
            final double[] lmin = new double[K];
            for (int k = 0; k < K; k++) lmin[k] = eig2(s[k][0][0], s[k][1][0], s[k][1][1])[0];
            int mc = 1;
            while (true) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) sum += w[k] * (1 + TWO_PI / lmin[k]) * Math.exp(-0.5 * lmin[k] * (double) (mc * mc));
                if (!(sum >= G_REL)) break;
                mc++;
            }
            this.mcut = mc;
            // gsub (L674-678): sum_k w_k exp(i m.mu_k - 1/2 m' S_k m) on |m_i| <= Mcut
            final int Mm = 2 * mc + 1;
            gsub = new double[2 * Mm * Mm];
            for (int a = 0; a < Mm; a++) {
                final double m1 = a - mc;
                for (int b = 0; b < Mm; b++) {
                    final double m2 = b - mc;
                    double re = 0.0, im = 0.0;
                    for (int k = 0; k < K; k++) {
                        final double ph = m1 * mu[k][0] + m2 * mu[k][1];
                        final double q = 0.5 * ((s[k][0][0] * (m1 * m1) + ((2.0 * s[k][0][1]) * m1) * m2) + s[k][1][1] * (m2 * m2));
                        final double e = Math.exp(-q);
                        re += w[k] * (e * Math.cos(ph));
                        im += w[k] * (e * Math.sin(ph));
                    }
                    gsub[2 * (a * Mm + b)] = re;
                    gsub[2 * (a * Mm + b) + 1] = im;
                }
            }
            // image sums (L679-688): Sinv, c0 = w / (2 pi sqrt(det S)), J from the largest eigenvalue of S
            comp = new double[4 * K];
            lminInv = new double[K];
            J = new int[K];
            for (int k = 0; k < K; k++) {
                final double a = s[k][0][0], b = s[k][0][1], c = s[k][1][0], d = s[k][1][1];
                final double det = a * d - b * c;
                final double i00 = d / det, i01 = -b / det, i10 = -c / det, i11 = a / det;
                comp[4 * k] = w[k] / (TWO_PI * Math.sqrt(det));
                comp[4 * k + 1] = 0.5 * i00;
                comp[4 * k + 2] = i01;
                comp[4 * k + 3] = 0.5 * i11;
                lminInv[k] = eig2(i00, i10, i11)[0];
                final double lmax = eig2(a, c, d)[1];
                int jj = 1;
                while (0.5 * Math.pow(Math.PI * (2 * jj + 1), 2) / lmax < IMG_EXP + Math.log(8 * (jj + 1))) jj++;
                J[k] = jj;
            }
        }
    }

    /** box_landing.py _g_images (L764-785) on the device: g on the grid x1 x x2 into out (row-major). */
    private void gEval(Mixture mix, double[] x1, double[] x2, CUdeviceptr out) {
        final int K = mix.K, a = x1.length, b = x2.length;
        final double[] D1 = new double[K * a], D2 = new double[K * b];
        final int[] start = new int[K + 1];
        final List<int[]> imgs = new ArrayList<int[]>();
        for (int k = 0; k < K; k++) {
            for (int i = 0; i < a; i++) D1[k * a + i] = npmod(x1[i] - mix.mu[k][0] + Math.PI, TWO_PI) - Math.PI;
            for (int j = 0; j < b; j++) D2[k * b + j] = npmod(x2[j] - mix.mu[k][1] + Math.PI, TWO_PI) - Math.PI;
            start[k] = imgs.size();
            final int J = mix.J[k];
            for (int j1 = -J; j1 <= J; j1++) {
                final double r1 = imageDistance(D1, k * a, a, TWO_PI * j1);
                for (int j2 = -J; j2 <= J; j2++) {
                    final double r2 = imageDistance(D2, k * b, b, TWO_PI * j2);
                    if (0.5 * mix.lminInv[k] * (r1 + r2) > IMG_EXP) continue;
                    imgs.add(new int[]{j1, j2});
                }
            }
        }
        start[K] = imgs.size();
        final int[] img = new int[Math.max(2, 2 * imgs.size())];
        for (int m = 0; m < imgs.size(); m++) {
            img[2 * m] = imgs.get(m)[0];
            img[2 * m + 1] = imgs.get(m)[1];
        }
        final CUdeviceptr dComp = upload(mix.comp), dD1 = upload(D1), dD2 = upload(D2), dStart = upload(start), dImg = upload(img);
        launchGrid(fGEval, gridFor((long) a * b), 1, Pointer.to(out), Pointer.to(new int[]{a}), Pointer.to(new int[]{b}),
                Pointer.to(new int[]{K}), Pointer.to(dComp), Pointer.to(dD1), Pointer.to(dD2), Pointer.to(dStart),
                Pointer.to(dImg));
        cuCtxSynchronize();
        for (CUdeviceptr p : new CUdeviceptr[]{dComp, dD1, dD2, dStart, dImg}) freeNow(p);
    }

    /** r = 0 if the image's shifted distances straddle 0, else the smaller squared end (L773, L776). */
    private static double imageDistance(double[] d, int off, int len, double shift) {
        double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < len; i++) {
            final double v = d[off + i] + shift;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        return (lo <= 0 && 0 <= hi) ? 0.0 : Math.min(lo * lo, hi * hi);
    }

    private void eTable(double[] x, CUdeviceptr out) {
        final CUdeviceptr dx = upload(x);
        launchGrid(fETable, gridFor((long) x.length * M), 1, Pointer.to(out), Pointer.to(dx), Pointer.to(new int[]{x.length}),
                Pointer.to(new int[]{M}), Pointer.to(new int[]{Mcut}));
        cuCtxSynchronize();
        freeNow(dx);
    }

    // ---- error-free tip phases ------------------------------------------------------------------------------

    /**
     * 2 pi frac(p d / L), the phase of a tip coefficient on one box axis (box_landing.py _Box.tip_coeffs, L384-391,
     * which reduces p d / L mod 1 in long double). Here d / L and p (d / L) are carried as double-doubles with Dekker's
     * exact product (Veltkamp split, no fma), the integer part is removed exactly, and 2 pi is applied as a
     * double-double, so the phase is the correctly rounded value up to ~1e-16 rad whatever the box width.
     */
    static double tipPhase(double p, double d, double L) {
        final double qh = d / L;
        final double ph = qh * L;
        final double pl = twoProdErr(qh, L, ph);
        final double ql = ((d - ph) - pl) / L;
        final double xh = p * qh;
        final double xl = twoProdErr(p, qh, xh) + p * ql;
        final double fh = xh - Math.floor(xh);              // exact
        double s = fh + xl;
        double e = twoSumErr(fh, xl, s);
        if (s < 0.0) {
            final double s2 = s + 1.0;
            e += twoSumErr(s, 1.0, s2);
            s = s2;
        } else if (s >= 1.0) {
            final double s2 = s - 1.0;
            e += twoSumErr(s, -1.0, s2);
            s = s2;
        }
        final double th = TWO_PI * s;
        final double tl = twoProdErr(TWO_PI, s, th) + (TWO_PI * e + TWO_PI_LO * s);
        return th + tl;
    }

    /** The rounding error of p = fl(a b): a b = p + err exactly (Dekker, Veltkamp split; no overflow assumed). */
    static double twoProdErr(double a, double b, double p) {
        double t = SPLIT * a;
        final double ah = t - (t - a), al = a - ah;
        t = SPLIT * b;
        final double bh = t - (t - b), bl = b - bh;
        return ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    }

    /** The rounding error of s = fl(a + b): a + b = s + err exactly (Knuth). */
    static double twoSumErr(double a, double b, double s) {
        final double bp = s - a;
        return (a - (s - bp)) + (b - bp);
    }

    // ---- numpy conventions ------------------------------------------------------------------------------------

    /** numpy.mod for floats: fmod, plus the divisor when the remainder's sign differs from it; +0 for a zero. */
    static double npmod(double a, double b) {
        double r = a % b;
        if (r != 0.0) {
            if ((b < 0) != (r < 0)) r += b;
        } else {
            r = Math.copySign(0.0, b);
        }
        return r;
    }

    /** numpy.fft.fftfreq(n, 1 / n): k * (1 / (n * (1 / n))), k = 0 .. (n - 1) / 2, -(n / 2) .. -1. */
    static double[] fftfreq(int n) {
        final double val = 1.0 / (n * (1.0 / n));
        final double[] out = new double[n];
        final int N = (n - 1) / 2 + 1;
        for (int i = 0; i < N; i++) out[i] = (double) i * val;
        for (int i = N; i < n; i++) out[i] = (double) (i - n) * val;
        return out;
    }

    /** numpy.fft.rfftfreq(n, 1 / n): k * (1 / (n * (1 / n))), k = 0 .. n / 2. */
    static double[] rfftfreq(int n) {
        final double val = 1.0 / (n * (1.0 / n));
        final double[] out = new double[n / 2 + 1];
        for (int i = 0; i < out.length; i++) out[i] = (double) i * val;
        return out;
    }

    /** Eigenvalues (ascending) of the symmetric 2 x 2 [[a, b], [b, c]]. */
    static double[] eig2(double a, double b, double c) {
        final double m = 0.5 * (a + c);
        final double r = Math.hypot(0.5 * (a - c), b);
        final double hi = m + r;
        double lo = m - r;
        final double det = a * c - b * b;
        if (hi > 0 && Math.abs(lo) < 1e-8 * Math.abs(hi)) lo = det / hi;   // the small one without cancellation
        return new double[]{lo, hi};
    }

    private static double log2(long n) {
        if (n > 0 && (n & (n - 1)) == 0) return Long.numberOfTrailingZeros(n);
        return Math.log((double) n) / Math.log(2.0);
    }

    private static void requireSymmetricPd(double[][] a, String name) {
        if (a == null || a.length != 2 || a[0] == null || a[1] == null || a[0].length != 2 || a[1].length != 2) {
            throw new IllegalArgumentException(name + " must be 2 x 2");
        }
        for (double[] row : a) for (double x : row) if (!Double.isFinite(x)) throw new IllegalArgumentException(name + " must be finite");
        final double amax = Math.max(Math.max(Math.abs(a[0][0]), Math.abs(a[0][1])), Math.max(Math.abs(a[1][0]), Math.abs(a[1][1])));
        if (Math.abs(a[0][1] - a[1][0]) > 1e-12 * amax) throw new IllegalArgumentException(name + " must be symmetric");
        if (!(eig2(a[0][0], a[1][0], a[1][1])[0] > 0)) throw new IllegalArgumentException(name + " must be positive definite");
    }

    // ---- box layout helpers (box_landing.py L214-221, L928-972) ---------------------------------------------------

    /** box_landing.py _arc_overlap: do [s1, s1 + l1) and [s2, s2 + l2) share a set of positive length (touching = no)? */
    static boolean arcOverlap(double s1, double l1, double s2, double l2) {
        if (l1 >= TWO_PI || l2 >= TWO_PI) return true;
        return npmod(s2 - s1, TWO_PI) < l1 - TOUCH || npmod(s1 - s2, TWO_PI) < l2 - TOUCH;
    }

    /**
     * box_landing.py _layout_from_records' validation and tip assignment for one site: boxes pairwise disjoint, every
     * observed tip in the first box that contains it with offset d = (y mod 2 pi - o) mod 2 pi, at least
     * marginSd * sd_tip (sd_tip = sqrt of the largest eigenvalue of sigmaEps) from that box's edges.
     *
     * @param tips [node][2] observed angles (radians) or null for a missing tip / internal node
     */
    public static TipAssignment assignTips(Box[] boxes, double[][] tips, double[][] sigmaEps, double marginSd) {
        for (int i = 0; i < boxes.length; i++) {
            for (int j = i + 1; j < boxes.length; j++) {
                if (arcOverlap(boxes[i].phi0, boxes[i].widthPhi, boxes[j].phi0, boxes[j].widthPhi)
                        && arcOverlap(boxes[i].psi0, boxes[i].widthPsi, boxes[j].psi0, boxes[j].widthPsi)) {
                    throw new IllegalArgumentException("boxes " + i + " and " + j + " overlap on the torus");
                }
            }
        }
        requireSymmetricPd(sigmaEps, "sigmaEps");
        final double need = marginSd * Math.sqrt(eig2(sigmaEps[0][0], sigmaEps[1][0], sigmaEps[1][1])[1]);
        final int[] box = new int[tips.length];
        final double[][] off = new double[tips.length][];
        Arrays.fill(box, -1);
        for (int v = 0; v < tips.length; v++) {
            if (tips[v] == null) continue;
            final double[] y = {npmod(tips[v][0], TWO_PI), npmod(tips[v][1], TWO_PI)};
            for (int b = 0; b < boxes.length && box[v] < 0; b++) {
                final Box bx = boxes[b];
                final double[] d = {npmod(y[0] - bx.phi0, TWO_PI), npmod(y[1] - bx.psi0, TWO_PI)};
                if ((bx.fullPhi || d[0] < bx.widthPhi) && (bx.fullPsi || d[1] < bx.widthPsi)) {
                    box[v] = b;
                    off[v] = d;
                }
            }
            if (box[v] < 0) throw new IllegalArgumentException("tip " + v + " lies outside every box");
            final Box bx = boxes[box[v]];
            double mar = Double.POSITIVE_INFINITY;
            if (!bx.fullPhi) mar = Math.min(mar, Math.min(off[v][0], bx.widthPhi - off[v][0]));
            if (!bx.fullPsi) mar = Math.min(mar, Math.min(off[v][1], bx.widthPsi - off[v][1]));
            if (mar < need * (1 - 1e-9)) {
                throw new IllegalArgumentException("tip " + v + " is " + Math.toDegrees(mar) + " deg from an edge of box "
                        + box[v] + "; it must be >= marginSd * sd_tip = " + Math.toDegrees(need) + " deg");
            }
        }
        return new TipAssignment(box, off);
    }

    // ---- schedule (box_landing.py _schedule, L254-281) ----------------------------------------------------------

    /**
     * Post-order in which every node's children come in decreasing register need (internal children first, sorted
     * stably by need, then the tips in input order). The products are formed in this order, as in Python.
     */
    static int[] schedule(int[][] children, int root) {
        final int nn = children.length;
        final int[] bfs = new int[nn];
        int head = 0, tail = 0;
        bfs[tail++] = root;
        while (head < tail) {
            final int v = bfs[head++];
            for (int c : children[v]) {
                if (tail >= nn) throw new IllegalArgumentException("the tree has a cycle");
                bfs[tail++] = c;
            }
        }
        final int[] need = new int[nn];
        final int[][] kids = new int[nn][];
        for (int k = tail - 1; k >= 0; k--) {
            final int v = bfs[k];
            final int[] ch = children[v];
            if (ch.length == 0) {
                kids[v] = new int[0];
                continue;
            }
            final List<Integer> internal = new ArrayList<Integer>();
            for (int c : ch) if (children[c].length > 0) internal.add(c);
            internal.sort((x, y) -> Integer.compare(need[y], need[x]));       // stable: ties keep input order
            final int[] seq = new int[ch.length];
            int m = 0;
            for (int c : internal) seq[m++] = c;
            for (int c : ch) if (children[c].length == 0) seq[m++] = c;
            int nv = 1;
            for (int i = 0; i < seq.length; i++) nv = Math.max(nv, need[seq[i]] + (i > 0 ? 1 : 0));
            need[v] = nv;
            kids[v] = seq;
        }
        final int[] order = new int[tail];
        int no = 0;
        final int[] stackV = new int[2 * nn + 2];
        final boolean[] stackE = new boolean[2 * nn + 2];
        int sp = 0;
        stackV[sp] = root;
        stackE[sp++] = false;
        while (sp > 0) {
            sp--;
            final int v = stackV[sp];
            final boolean expanded = stackE[sp];
            if (expanded || children[v].length == 0) {
                order[no++] = v;
                continue;
            }
            stackV[sp] = v;
            stackE[sp++] = true;
            for (int i = kids[v].length - 1; i >= 0; i--) {
                stackV[sp] = kids[v][i];
                stackE[sp++] = false;
            }
        }
        return no == tail ? order : Arrays.copyOf(order, no);
    }

    // ---- debugging --------------------------------------------------------------------------------------------

    /**
     * A per-slot table as stored on the device, for comparison with box_landing.py: "gp" (Gp, interleaved complex,
     * n1 x n2h), "e1" (n1 x M complex), "e2" (n2 x M complex), "om1" (n1), "om2" (n2h).
     */
    public double[] debugSlotTable(String name, int globalSlot) {
        if (closed) throw new IllegalStateException("engine closed");
        cuCtxSetCurrent(context);
        final long N2 = (long) n1 * n2h;
        final CUdeviceptr src;
        final int len;
        switch (name) {
            case "gp": src = (CUdeviceptr) dGP.withByteOffset(globalSlot * N2 * 16L); len = (int) (2 * N2); break;
            case "e1": src = (CUdeviceptr) dE1.withByteOffset((long) globalSlot * n1 * M * 16L); len = 2 * n1 * M; break;
            case "e2": src = (CUdeviceptr) dE2.withByteOffset((long) globalSlot * n2 * M * 16L); len = 2 * n2 * M; break;
            case "om1": src = (CUdeviceptr) dOM1.withByteOffset((long) globalSlot * n1 * 8L); len = n1; break;
            case "om2": src = (CUdeviceptr) dOM2.withByteOffset((long) globalSlot * n2h * 8L); len = n2h; break;
            default: throw new IllegalArgumentException("unknown table " + name);
        }
        final double[] out = new double[len];
        cuMemcpyDtoH(Pointer.to(out), src, 8L * len);
        return out;
    }

    /** The first global slot of a site (its boxes occupy consecutive slots). */
    public int getSiteFirstSlot(int site) { return siteSlot0[site]; }

    // ---- device helpers ---------------------------------------------------------------------------------------

    private static int gridFor(long n) {
        return (int) Math.max(1L, Math.min(65535L, (n + BLOCK - 1) / BLOCK));
    }

    private void launchGrid(CUfunction f, int gridX, int gridY, Pointer... args) {
        cuLaunchKernel(f, gridX, gridY, 1, BLOCK, 1, 1, 0, null, Pointer.to(args), null);
    }

    private CUfunction function(String name) {
        final CUfunction f = new CUfunction();
        cuModuleGetFunction(f, module, name);
        return f;
    }

    private CUdeviceptr alloc(long bytes) {
        final CUdeviceptr p = new CUdeviceptr();
        cuMemAlloc(p, Math.max(8L, bytes));
        allocations.add(p);
        return p;
    }

    private CUdeviceptr zeros(long bytes) {
        final CUdeviceptr p = alloc(bytes);
        cuMemsetD8(p, (byte) 0, Math.max(8L, bytes));
        return p;
    }

    private void freeNow(CUdeviceptr p) {
        allocations.remove(p);
        cuMemFree(p);
    }

    private CUdeviceptr upload(double[] a) {
        final CUdeviceptr p = alloc(8L * a.length);
        if (a.length > 0) cuMemcpyHtoD(p, Pointer.to(a), 8L * a.length);
        return p;
    }

    private CUdeviceptr upload(int[] a) {
        final CUdeviceptr p = alloc(4L * a.length);
        if (a.length > 0) cuMemcpyHtoD(p, Pointer.to(a), 4L * a.length);
        return p;
    }

    private CUdeviceptr upload(long[] a) {
        final CUdeviceptr p = alloc(8L * a.length);
        if (a.length > 0) cuMemcpyHtoD(p, Pointer.to(a), 8L * a.length);
        return p;
    }

    private CUdeviceptr uploadBytes(byte[] a) {
        final CUdeviceptr p = alloc(Math.max(1, a.length));
        if (a.length > 0) cuMemcpyHtoD(p, Pointer.to(a), a.length);
        return p;
    }

    private static byte[] loadPtx() {
        final String override = System.getProperty("torus.box.ptx");
        try (InputStream in = override != null ? new java.io.FileInputStream(override)
                : TorusBoxGpuEngine.class.getResourceAsStream(PTX_RESOURCE)) {
            if (in == null) throw new IllegalStateException("PTX resource " + PTX_RESOURCE + " not found next to "
                    + TorusBoxGpuEngine.class.getName());
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

    /**
     * Releases every plan, buffer, the module and the retained primary context. Each step runs even when an earlier
     * one fails (e.g. a sticky context error); the first failure is rethrown at the end.
     */
    @Override
    public void close() {
        if (closed) return;
        closed = true;
        final RuntimeException first = releaseAll();
        if (first != null) throw first;
    }

    /** Best-effort release of everything acquired so far (close and a failed constructor); returns the first failure. */
    private RuntimeException releaseAll() {
        RuntimeException first = null;
        if (context != null) {
            try {
                cuCtxSetCurrent(context);
            } catch (RuntimeException e) {
                first = e;
            }
        }
        for (cufftHandle h : new ArrayList<cufftHandle>(plans)) {
            try {
                JCufft.cufftDestroy(h);
            } catch (RuntimeException e) {
                if (first == null) first = e;
            }
        }
        plans.clear();
        planWork.clear();
        planD2Z = planZ2D = plan1D2Z = plan1Z2D = null;
        for (CUdeviceptr p : allocations) {
            try {
                cuMemFree(p);
            } catch (RuntimeException e) {
                if (first == null) first = e;
            }
        }
        allocations.clear();
        dFftWork = null;
        dAlignR = null;
        if (module != null) {
            try {
                cuModuleUnload(module);
            } catch (RuntimeException e) {
                if (first == null) first = e;
            }
            module = null;
        }
        if (context != null) {
            try {
                final CUdevice device = new CUdevice();
                cuDeviceGet(device, deviceId);
                cuDevicePrimaryCtxRelease(device);
            } catch (RuntimeException e) {
                if (first == null) first = e;
            }
            context = null;
        }
        return first;
    }
}
