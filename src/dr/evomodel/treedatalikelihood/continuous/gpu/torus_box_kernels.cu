// torus_box_kernels.cu
//
// CUDA kernels for TorusBoxGpuEngine: the BOX arithmetic of the landing likelihood, a port of the numpy reference
// gpu_build/box_engine/box_landing.py (BoxLanding._box_pass, md5 74f54bd8c642) with prec_check = False, batched over
// (site, box) SLOTS. Every kernel names the Python lines it implements; README.md (gpu_build/box_gpu) has the full map.
//
// Representation (box_landing.py docstring): a message is a per-SITE constant C plus, per box (slot), a local part f
// held as grid values (n1 x n2, row-major, x_jk = o + (j h1, k h2)) or as its raw half spectrum X = rfft2(f)
// (n1 x n2h, n2h = n2 / 2 + 1, rows p = fftfreq(n1) * n1, columns q = rfftfreq(n2) * n2 including +n2/2).
//
// Launch layout: per-slot kernels use blockIdx.y = slot and grid-stride over blockIdx.x * blockDim.x; per-site
// kernels use one block per site. Block size is fixed at BLOCK (256). Reductions are deterministic (no atomics):
// per-block partials (part[slot][block][NPART]) are combined by one block per site in a fixed order, and the boxes
// of a site are always visited in increasing box index (Python's set / dict order for small ints), so a
// recomputation reproduces a value bitwise.
//
// Arithmetic: compiled with -fmad=false so that a*b+c is two roundings, as numpy computes it. The complex products
// numpy 2.2 forms with its SIMD loop (re = fma(ar, br, -(ai*bi)), im = fma(ar, bi, ai*br)) are written with
// explicit fma() where Python forms them (tip coefficients). Complex-by-real products are (re*r, im*r); numpy's
// complex-by-real DIVISION is a multiplication by the reciprocal (Smith's method) and is written that way.
// Transcendentals (exp, expm1, sincos, log, hypot) are CUDA's, not glibc's / numpy's: differences of an ulp or two.
//
// Mode flags carried per slot through one branch (sMode):
//   0  inactive (site skipped at this op: flat child, dead site) or no local part in this box
//   1  local part present and kept after the propagation
//   2  local part decayed to nothing (|X(0)| <= DROP after damping) with a landing constant: the slot then carries
//      K_t * f_b (diffusion without the survival factor) for LEAK's landing term (box_landing.py L1457-1477)
//   3  local part decayed to nothing without a landing constant (lambda = 0 or t = 0): dropped

#define BLOCK 256
#define NPART 12
#define P_LS 0      // landing constant, compensated sum
#define P_LE 1      //   and its error term
#define P_PRE 2     // max |X| over the outer (RESOLUTION) modes before the propagation (nodes with >= 3 children)
#define P_POST 3    // max |X| over the outer modes after the damping, or of rfft2(f) at the root
#define P_FHI 4     // max of the propagated grid values (or of K_t f_b), or of the root part
#define P_FLO 5     // min of the same
#define P_FBAND 6   // max |.| over the LEAK band of the same
#define P_OHI 7     // max of the product written into the parent
#define P_OLO 8     // min of the same
#define P_RS 9      // root integral of the part, compensated sum
#define P_RE 10     //   and its error term
#define DROP 1e-280
#define TWO_PI (2.0 * 3.141592653589793)
#define MAX_BOXES 16

// ---- helpers -----------------------------------------------------------------------------------------

// error-free transformation a + b = s + e
__device__ __forceinline__ void two_sum(double a, double b, double* s, double* e) {
    const double sum = a + b;
    const double bp = sum - a;
    *e = (a - (sum - bp)) + (b - bp);
    *s = sum;
}

// NaN-propagating max / min (a NaN anywhere makes the result NaN, so no guard value can hide one)
__device__ __forceinline__ double nanmax(double a, double b) { return (a != a || b != b) ? a + b : (a > b ? a : b); }
__device__ __forceinline__ double nanmin(double a, double b) { return (a != a || b != b) ? a + b : (a < b ? a : b); }

// compensated block reduction of (sum, err) pairs; result in every thread
__device__ void block_reduce_pair(double* s, double* e) {
    __shared__ double ss[BLOCK];
    __shared__ double se[BLOCK];
    const int tid = threadIdx.x;
    __syncthreads();
    ss[tid] = *s;
    se[tid] = *e;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            double sum, err;
            two_sum(ss[tid], ss[tid + stride], &sum, &err);
            ss[tid] = sum;
            se[tid] = se[tid] + se[tid + stride] + err;
        }
        __syncthreads();
    }
    *s = ss[0];
    *e = se[0];
}

// NaN-propagating max / min block reductions; result in every thread
__device__ double block_reduce_max(double v) {
    __shared__ double sm[BLOCK];
    const int tid = threadIdx.x;
    __syncthreads();
    sm[tid] = v;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] = nanmax(sm[tid], sm[tid + stride]);
        __syncthreads();
    }
    return sm[0];
}

__device__ double block_reduce_min(double v) {
    __shared__ double sm[BLOCK];
    const int tid = threadIdx.x;
    __syncthreads();
    sm[tid] = v;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sm[tid] = nanmin(sm[tid], sm[tid + stride]);
        __syncthreads();
    }
    return sm[0];
}

// slotInfo[4 k] = {global slot, local site, box index within the site, unused}
// siteInfo[4 j] = {global site, first local slot, number of slots (boxes), unused}
// siteParam[8 j] = {S00, S01 (symmetrised), S11, 0.5 S00, 0.5 S11, lambda, unused, unused}

// ---- construction: the landing density g on a grid ------------------------------------------------------

// box_landing.py _g_images (L764-785): g(x1[i], x2[j]) = sum_k sum_images c0_k exp(-qf), with
//   a1 = d1_k[i] + 2 pi j1, a2 = d2_k[j] + 2 pi j2 (d = (x - mu_k + pi) mod 2 pi - pi, from the host),
//   qf = (0.5 Sinv00 a1^2 + (Sinv01 a1) a2) + 0.5 Sinv11 a2^2,
// summed in Python's order (components, then j1, then j2), skipped images as on the host (IMG_EXP rule).
// comp[4 k] = {c0, 0.5 Sinv00, Sinv01, 0.5 Sinv11}; img[2 m] = (j1, j2) for m in imgStart[k] .. imgStart[k + 1].
extern "C" __global__ void g_eval(double* out, int n1, int n2, int K, const double* comp, const double* D1,
                                  const double* D2, const int* imgStart, const int* img) {
    const long long N = (long long) n1 * n2;
    for (long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x; g < N; g += (long long) gridDim.x * blockDim.x) {
        const int i = (int) (g / n2);
        const int j = (int) (g - (long long) i * n2);
        double acc = 0.0;
        for (int k = 0; k < K; k++) {
            const double c0 = comp[4 * k], s00h = comp[4 * k + 1], s01 = comp[4 * k + 2], s11h = comp[4 * k + 3];
            const double d1 = D1[(long long) k * n1 + i];
            const double d2 = D2[(long long) k * n2 + j];
            for (int m = imgStart[k]; m < imgStart[k + 1]; m++) {
                const double a1 = d1 + TWO_PI * (double) img[2 * m];
                const double a2 = d2 + TWO_PI * (double) img[2 * m + 1];
                const double qf = (s00h * (a1 * a1) + (s01 * a1) * a2) + s11h * (a2 * a2);
                acc = acc + exp(-qf) * c0;
            }
        }
        out[g] = acc;
    }
}

// box_landing.py L377: Gp = rfft2(g) * (wq * (h1 h2 / (n1 n2))), one column factor colfac[q] per half-spectrum column
extern "C" __global__ void gp_scale(double2* GP, const double2* G, const double* colfac, int n2h, long long N) {
    for (long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += (long long) gridDim.x * blockDim.x) {
        const double f = colfac[idx % n2h];
        const double2 v = G[idx];
        double2 o;
        o.x = v.x * f;
        o.y = v.y * f;
        GP[idx] = o;
    }
}

// box_landing.py L380-382: E[i][k] = exp(-1j * x[i] * m[k]), m[k] = k - Mcut, i.e. (cos(-th), sin(-th)), th = x[i] m[k]
extern "C" __global__ void e_table(double2* E, const double* x, int nx, int M, int Mcut) {
    const long long N = (long long) nx * M;
    for (long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += (long long) gridDim.x * blockDim.x) {
        const int i = (int) (idx / M);
        const int k = (int) (idx - (long long) i * M);
        const double th = x[i] * (double) (k - Mcut);
        double s, c;
        sincos(-th, &s, &c);
        double2 e;
        e.x = c;
        e.y = s;
        E[idx] = e;
    }
}

// ---- one evaluation: setup ----------------------------------------------------------------------------

// box_landing.py _Guards.__init__ (L458-466): leak = tip LEAK, res = tip RESOLUTION, rho_min = 1, no NONPOSITIVE
extern "C" __global__ void site_init(int nSites, const int* siteInfo, const double* tipLeak, const double* tipRes,
                                     double* gLeak, double* gRes, double* gRho, int* gStat, int* gDead, int* gNode) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nSites) return;
    const int gs = siteInfo[4 * j];
    gLeak[j] = tipLeak[gs];
    gRes[j] = tipRes[gs];
    gRho[j] = 1.0;
    gStat[j] = 0;
    gDead[j] = 0;
    gNode[j] = -2;
}

// box_landing.py L1604-1606: musub = lam * gsub / (0.5 (S00 m1^2 + 2 S01 m1 m2 + S11 m2^2) + lam) for lambda > 0.
// numpy divides complex by real as a multiplication by the reciprocal (Smith's method).
extern "C" __global__ void musub_build(double2* MU, const double2* gsub, const double* siteParam, int M, int Mcut) {
    const int j = blockIdx.y;
    const double lam = siteParam[8 * j + 5];
    if (!(lam > 0.0)) return;
    const double S00 = siteParam[8 * j], S01 = siteParam[8 * j + 1], S11 = siteParam[8 * j + 2];
    const long long MM = (long long) M * M;
    double2* mu = MU + (long long) j * MM;
    for (long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x; idx < MM; idx += (long long) gridDim.x * blockDim.x) {
        const int a = (int) (idx / M);
        const int b = (int) (idx - (long long) a * M);
        const double m1 = (double) (a - Mcut), m2 = (double) (b - Mcut);
        const double den = 0.5 * ((S00 * (m1 * m1) + ((2.0 * S01) * m1) * m2) + S11 * (m2 * m2)) + lam;
        const double scl = 1.0 / den;
        const double2 g = gsub[idx];
        double2 o;
        o.x = (lam * g.x) * scl;
        o.y = (lam * g.y) * scl;
        mu[idx] = o;
    }
}

// ---- tips ---------------------------------------------------------------------------------------------

// box_landing.py _Box.tip_coeffs (L384-395), per axis: exp(-1j * th) = (cos(-th), sin(-th)) from the host phase
// tables th = 2 pi frac(p d / L) (reduced error-free on the host). T[slot][0 .. n1) axis 1, [n1 .. n1 + n2h) axis 2.
extern "C" __global__ void tip_axis(double2* T, const int* slotInfo, const int* siteInfo, const unsigned char* code,
                                    const int* gDead, const int* tipBox, const long long* phaseOff, const double* phase,
                                    int n1, int n2h) {
    const int slot = blockIdx.y;
    const int site = slotInfo[4 * slot + 1];
    const int gs = siteInfo[4 * site];
    if (code[gs] == 0 || gDead[site] || tipBox[gs] != slotInfo[4 * slot + 2]) return;
    const int len = n1 + n2h;
    const double* th = phase + phaseOff[gs];
    double2* t = T + (long long) slot * len;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < len; k += gridDim.x * blockDim.x) {
        double s, c;
        sincos(-th[k], &s, &c);
        double2 e;
        e.x = c;
        e.y = s;
        t[k] = e;
    }
}

// box_landing.py _Box.tip_coeffs (L392-394) and tipamp (L332-334): X = (E1[:, None] * E2[None, :]) * tipamp with
// tipamp = exp(-P) / (h1 h2), P = (0.5 Pi00 om1^2 + (Pi01 om1) om2) + 0.5 Pi11 om2^2. A tip lives in one box: the
// other boxes of its site get no local part (presTmp = 0), and its constant is 0 (box_landing.py L1416-1419).
extern "C" __global__ void tip_build(double2* X, int* presTmp, const double2* T, const int* slotInfo,
                                     const int* siteInfo, const unsigned char* code, const int* gDead, const int* tipBox,
                                     const double* OM1, const double* OM2, const double* slotHH,
                                     double pi00h, double pi01, double pi11h, int n1, int n2h) {
    const int slot = blockIdx.y;
    const int site = slotInfo[4 * slot + 1];
    const int gs = siteInfo[4 * site];
    const int present = (code[gs] != 0 && !gDead[site] && tipBox[gs] == slotInfo[4 * slot + 2]) ? 1 : 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) presTmp[slot] = present;
    if (!present) return;
    const long long gslot = slotInfo[4 * slot];
    const double hh = slotHH[gslot];
    const double* om1 = OM1 + gslot * n1;
    const double* om2 = OM2 + gslot * n2h;
    const double2* e1 = T + (long long) slot * (n1 + n2h);
    const double2* e2 = e1 + n1;
    const long long N2 = (long long) n1 * n2h;
    double2* x = X + (long long) slot * N2;
    for (long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x; idx < N2; idx += (long long) gridDim.x * blockDim.x) {
        const int i = (int) (idx / n2h);
        const int j = (int) (idx - (long long) i * n2h);
        const double2 a = e1[i], b = e2[j];
        const double re = fma(a.x, b.x, -(a.y * b.y));
        const double im = fma(a.x, b.y, a.y * b.x);
        const double w1 = om1[i], w2 = om2[j];
        const double P = (pi00h * (w1 * w1) + (pi01 * w1) * w2) + pi11h * (w2 * w2);
        const double ta = exp(-P) / hh;
        double2 c;
        c.x = re * ta;
        c.y = im * ta;
        x[idx] = c;
    }
}

// ---- one branch: propagation --------------------------------------------------------------------------

// Per slot, before the propagation (one thread per slot): box_landing.py L1446-1467, the scalars the rest of the
// branch needs from X(0, 0): the damped |X(0, 0)| (x0, L1466: the part is dropped when it is <= DROP), and LEAK's
// landing mass m_b = (1 - e^{-lambda t}) h1 h2 |X(0, 0)| (L1456). Mode codes: see the file header.
extern "C" __global__ void slot_prep(const double2* X, const int* presChild, const int* slotInfo, const int* siteInfo,
                                     const unsigned char* code, const int* gDead, const double* siteParam,
                                     const double* slotHH, double t, int nSlots, long long N2,
                                     double2* sX00, double* sX0, double* sLand, int* sMode) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= nSlots) return;
    const int site = slotInfo[4 * slot + 1];
    const int gs = siteInfo[4 * site];
    if (code[gs] == 0 || gDead[site] || !presChild[slot]) {
        sMode[slot] = 0;
        return;
    }
    const double2 x = X[(long long) slot * N2];
    const double lam = siteParam[8 * site + 5];
    double x0;
    if (t > 0.0) {
        const double d = exp(lam * (-t));               // mode (0, 0): LQ = lam, nx = lam * (-t)
        x0 = hypot(x.x * d, x.y * d);
    } else {
        x0 = hypot(x.x, x.y);
    }
    const int land = (t > 0.0 && lam > 0.0);
    sX00[slot] = x;
    sX0[slot] = x0;
    sLand[slot] = land ? ((-expm1((-lam) * t)) * slotHH[slotInfo[4 * slot]]) * hypot(x.x, x.y) : 0.0;
    sMode[slot] = (x0 > DROP) ? 1 : (land ? 2 : 3);
}

// box_landing.py L1394-1400 and L1426-1465, per mode of every active slot:
//   Q = ((0.5 S00 om1^2 + (S01 om1) om2) + 0.5 S11 om2^2) + lam,  nx = Q (-t)
//   landing constant (lam > 0, t > 0): Cp -= Re vdot(Gc expm1(nx), X), Gc = Gp (lam / Q)   [per-block partials]
//   damping: X *= exp(nx)   (mode 2: X *= exp((Q - lam) (-t)), the K_t * f_b of L1460, for LEAK only)
//   RESOLUTION: max |X| over the outer modes before (multi: >= 3 children, L1432-1436) and after (mode 1, L1484)
extern "C" __global__ void propagate(double2* X, const int* sMode, const int* slotInfo, const double* siteParam,
                                     const double* OM1, const double* OM2, const double2* GP, double t, int multi,
                                     int n1, int n2h, int r0, int r1, int c0, double* part) {
    const int slot = blockIdx.y;
    const int mode = sMode[slot];
    double acc = 0.0, err = 0.0, pre = 0.0, post = 0.0;
    if (mode != 0) {
        const int site = slotInfo[4 * slot + 1];
        const long long gslot = slotInfo[4 * slot];
        const double* sp = siteParam + 8 * site;
        const double s01 = sp[1], a00 = sp[3], a11 = sp[4], lam = sp[5];
        const double* om1 = OM1 + gslot * n1;
        const double* om2 = OM2 + gslot * n2h;
        const long long N2 = (long long) n1 * n2h;
        const double2* gp = GP + gslot * N2;
        double2* x = X + (long long) slot * N2;
        for (long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x; idx < N2; idx += (long long) gridDim.x * blockDim.x) {
            const int i = (int) (idx / n2h);
            const int j = (int) (idx - (long long) i * n2h);
            double2 c = x[idx];
            const int outer = (i >= r0 && i <= r1) || (j >= c0);
            if (multi && outer) pre = nanmax(pre, hypot(c.x, c.y));
            if (t > 0.0) {
                const double w1 = om1[i], w2 = om2[j];
                const double Q = ((a00 * (w1 * w1) + (s01 * w1) * w2) + a11 * (w2 * w2)) + lam;
                const double nx = Q * (-t);
                if (lam > 0.0) {
                    const double em = expm1(nx);
                    const double r = lam / Q;
                    const double2 g = gp[idx];
                    const double are = (g.x * r) * em, aim = (g.y * r) * em;
                    const double term = are * c.x + aim * c.y;
                    double e2;
                    two_sum(acc, term, &acc, &e2);
                    err += e2;
                }
                const double damp = (mode == 2) ? exp((Q - lam) * (-t)) : exp(nx);
                c.x *= damp;
                c.y *= damp;
                x[idx] = c;
            }
            if (mode == 1 && outer) post = nanmax(post, hypot(c.x, c.y));
        }
    }
    block_reduce_pair(&acc, &err);
    pre = block_reduce_max(pre);
    post = block_reduce_max(post);
    if (threadIdx.x == 0) {
        double* p = part + ((long long) slot * gridDim.x + blockIdx.x) * NPART;
        p[P_LS] = acc;
        p[P_LE] = err;
        p[P_PRE] = pre;
        p[P_POST] = post;
    }
}

// One block per site (box_landing.py L1440-1452 and L1484, the scalar part): the propagated constant
//   Cp = C - sum_b landing_b   (boxes in increasing order, each landing_b = compensated sum of its block partials)
// and the RESOLUTION ratios (before the propagation at nodes with >= 3 children, after it for kept parts).
extern "C" __global__ void site_landing(const int* siteInfo, const unsigned char* code, const int* gDead,
                                        const int* sMode, const double* siteParam, const double* Cchild,
                                        const double* part, int nblk, int multi, double t, const double2* sX00,
                                        const double* sX0, double* CpArr, double* gRes) {
    const int j = blockIdx.x;
    const int gs = siteInfo[4 * j];
    if (code[gs] == 0 || gDead[j]) return;
    const int first = siteInfo[4 * j + 1], cnt = siteInfo[4 * j + 2];
    const double lam = siteParam[8 * j + 5];
    const int land = (t > 0.0 && lam > 0.0);
    double Cp = Cchild[j];
    double res = gRes[j];
    for (int b = 0; b < cnt; b++) {
        const int slot = first + b;
        const int mode = sMode[slot];
        if (mode == 0) continue;
        double s = 0.0, e = 0.0, pre = 0.0, post = 0.0;
        for (int k = threadIdx.x; k < nblk; k += blockDim.x) {
            const double* p = part + ((long long) slot * nblk + k) * NPART;
            double e2;
            two_sum(s, p[P_LS], &s, &e2);
            e += e2 + p[P_LE];
            pre = nanmax(pre, p[P_PRE]);
            post = nanmax(post, p[P_POST]);
        }
        block_reduce_pair(&s, &e);
        pre = block_reduce_max(pre);
        post = block_reduce_max(post);
        if (threadIdx.x == 0) {
            if (land && mode != 3) Cp = Cp - (s + e);
            if (multi) {
                const double2 x00 = sX00[slot];
                const double ax = hypot(x00.x, x00.y);
                if (ax > 0.0) res = nanmax(res, pre / ax);
            }
            if (mode == 1) res = nanmax(res, post / sX0[slot]);
        }
    }
    if (threadIdx.x == 0) {
        CpArr[j] = Cp;
        gRes[j] = res;
    }
}

// Before every inverse transform: the half-spectrum columns q = 0 and q = n2 / 2 (n2 even) are replaced by their
// Hermitian part in p, H(p) = (X(p) + conj X(-p)) / 2. scipy's irfft2 (pocketfft c2r) uses only the real part of
// those columns after the inverse transform along p, which is exactly the transform of H; cuFFT's Z2D is defined
// only for Hermitian input, so this makes the two compute the same function of X (box_landing.py L1479, L1460).
extern "C" __global__ void hermitianize(double2* X, const int* sMode, int n1, int n2, int n2h) {
    const int slot = blockIdx.y;
    const int mode = sMode[slot];
    if (mode != 1 && mode != 2) return;
    const int ncol = (n2 % 2 == 0) ? 2 : 1;
    const int half = n1 / 2 + 1;
    double2* x = X + (long long) slot * n1 * n2h;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ncol * half; idx += gridDim.x * blockDim.x) {
        const int col = idx / half;
        const int i = idx - col * half;
        const int q = (col == 0) ? 0 : n2h - 1;
        const int i2 = (n1 - i) % n1;
        if (i > i2) continue;
        if (i == i2) {
            x[(long long) i * n2h + q].y = 0.0;
        } else {
            const double2 a = x[(long long) i * n2h + q], b = x[(long long) i2 * n2h + q];
            double2 u, w;
            u.x = (a.x + b.x) * 0.5;
            u.y = (a.y - b.y) * 0.5;
            w.x = (b.x + a.x) * 0.5;
            w.y = (b.y - a.y) * 0.5;
            x[(long long) i * n2h + q] = u;
            x[(long long) i2 * n2h + q] = w;
        }
    }
}

// ---- one branch: product into the parent ------------------------------------------------------------------

// box_landing.py L1479-1494 (grid values F = irfft2(X), their max / min / LEAK-band max) and L1509-1554 (the
// product into the parent's accumulator), per grid point of every active slot. R is the unnormalised Z2D output,
// F = R * fct (fct = 1 / (n1 n2), pocketfft's backward factor). code = 1: the parent's first factor at this site
// (F is copied); code = 2: f_v = f_a f_c + f_a Cp + f_c Ca, each term only where Python forms it (Cp != 0,
// Ca != 0), in Python's order of roundings. oPres = product present in this slot (before the DROP rule).
extern "C" __global__ void combine(double* P, const double* R, const int* sMode, const int* presPar, int* oPres,
                                   const int* slotInfo, const int* siteInfo, const unsigned char* code,
                                   const int* gDead, const double* Cpar, const double* CpArr, const int* slotNB,
                                   double fct, int n1, int n2, double* part) {
    const int slot = blockIdx.y;
    const int site = slotInfo[4 * slot + 1];
    const int gs = siteInfo[4 * site];
    const int c = code[gs];
    const int active = (c != 0 && !gDead[site]);
    const int mode = active ? sMode[slot] : 0;
    const int pc = (mode == 1), pk = (mode == 2);
    const int pa = (active && c == 2) ? presPar[slot] : 0;
    const double Ca = (active && c == 2) ? Cpar[site] : 0.0;
    const double Cp = active ? CpArr[site] : 0.0;
    int present = 0;
    if (active) present = (c == 1) ? pc : ((pa && pc) || (pa && Cp != 0.0) || (pc && Ca != 0.0));
    const long long gslot = slotInfo[4 * slot];
    const int nb1 = slotNB[2 * gslot], nb2 = slotNB[2 * gslot + 1];
    double fhi = -INFINITY, flo = INFINITY, fband = 0.0, ohi = -INFINITY, olo = INFINITY;
    if (pc || pk || present) {
        const long long NG = (long long) n1 * n2;
        const double* r = R + (long long) slot * NG;
        double* pp = P + (long long) slot * NG;
        for (long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x; g < NG; g += (long long) gridDim.x * blockDim.x) {
            double v = 0.0;
            if (pc || pk) {
                const int i = (int) (g / n2);
                const int jj = (int) (g - (long long) i * n2);
                v = r[g] * fct;
                fhi = nanmax(fhi, v);
                flo = nanmin(flo, v);
                const int band = (nb1 > 0 && (i < nb1 || i >= n1 - nb1)) || (nb2 > 0 && (jj < nb2 || jj >= n2 - nb2));
                if (band) fband = nanmax(fband, fabs(v));
            }
            if (present) {
                double o;
                if (c == 1) {
                    o = v;
                } else if (pa && pc) {
                    const double fa = pp[g];
                    o = fa * v;
                    if (Cp != 0.0) o = o + fa * Cp;
                    if (Ca != 0.0) o = o + v * Ca;
                } else if (pa) {
                    o = pp[g] * Cp;
                } else {
                    o = v * Ca;
                }
                pp[g] = o;
                ohi = nanmax(ohi, o);
                olo = nanmin(olo, o);
            }
        }
    }
    fhi = block_reduce_max(fhi);
    flo = block_reduce_min(flo);
    fband = block_reduce_max(fband);
    ohi = block_reduce_max(ohi);
    olo = block_reduce_min(olo);
    if (threadIdx.x == 0) {
        double* p = part + ((long long) slot * gridDim.x + blockIdx.x) * NPART;
        p[P_FHI] = fhi;
        p[P_FLO] = flo;
        p[P_FBAND] = fband;
        p[P_OHI] = ohi;
        p[P_OLO] = olo;
        if (blockIdx.x == 0) oPres[slot] = present;
    }
}

// One block per site, after combine: box_landing.py L1496-1506 (LEAK of the propagated child, with the landing
// truncation term and the dropped-part term, no round-off allowance: prec_check = False) and L1555-1574 (DROP rule,
// the per-site rescale s = max(|Cn|, max_b max |Cn + f_b|), NONPOSITIVE, the overlap rho = s / sc, Cn / s and the
// log scale). lconst = 10 C_FFT U sqrt(log2(n1 n2)) (the band ratio's round-off floor, L1488-1489).
extern "C" __global__ void site_product(const int* siteInfo, const unsigned char* code, int* gDead, const int* sMode,
                                        int* oPres, int* presPar, const double* part, int nblk, const double* sLand,
                                        double t, const double* siteParam, double* Cpar, double* LSpar,
                                        const double* LSchild, const double* CpArr, double* sScale, double* gLeak,
                                        double* gRho, int* gStat, int* gNode, int parentNode, double gsup,
                                        double lconst, int nonposBit) {
    __shared__ double shi[MAX_BOXES], slo[MAX_BOXES], sband[MAX_BOXES], ssc;
    const int j = blockIdx.x;
    const int gs = siteInfo[4 * j];
    const int c = code[gs];
    if (c == 0 || gDead[j]) return;
    const int first = siteInfo[4 * j + 1], cnt = siteInfo[4 * j + 2];
    const double lam = siteParam[8 * j + 5];
    const int land = (t > 0.0 && lam > 0.0);
    const double Cp = CpArr[j];
    // LEAK of the child's propagated message
    for (int b = 0; b < cnt; b++) {
        const int slot = first + b;
        const int mode = sMode[slot];
        if (mode != 1 && mode != 2) continue;
        double hi = -INFINITY, lo = INFINITY, band = 0.0;
        for (int k = threadIdx.x; k < nblk; k += blockDim.x) {
            const double* p = part + ((long long) slot * nblk + k) * NPART;
            hi = nanmax(hi, p[P_FHI]);
            lo = nanmin(lo, p[P_FLO]);
            band = nanmax(band, p[P_FBAND]);
        }
        hi = block_reduce_max(hi);
        lo = block_reduce_min(lo);
        band = block_reduce_max(band);
        if (threadIdx.x == 0) {
            shi[b] = hi;
            slo[b] = lo;
            sband[b] = band;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double fmaxAll = 0.0;
        for (int b = 0; b < cnt; b++) {
            if (sMode[first + b] == 1) fmaxAll = nanmax(fmaxAll, nanmax(shi[b], -slo[b]));
        }
        const double sc = fabs(Cp) + fmaxAll;
        double leak = gLeak[j];
        for (int b = 0; b < cnt; b++) {
            const int slot = first + b;
            const int mode = sMode[slot];
            if (mode == 1) {
                const double fm = nanmax(shi[b], -slo[b]);
                double lk = (sc > 0.0) ? sband[b] / sc : INFINITY;
                if (land && fm > 0.0 && sc > 0.0) {
                    const double d = sband[b] - lconst * fm;
                    const double q = (d > 0.0 ? d : 0.0) / fm;
                    lk = lk + (((q * 2.0) * gsup) * sLand[slot]) / sc;
                }
                leak = nanmax(leak, lk);
            } else if (mode == 2) {
                const double fm = nanmax(shi[b], -slo[b]);
                double q = 0.0;
                if (fm > 0.0) {
                    const double d = sband[b] - lconst * fm;
                    q = (d > 0.0 ? d : 0.0) / fm;
                }
                const double lk = (sc > 0.0) ? (((q * 2.0) * gsup) * sLand[slot]) / sc
                                             : ((q * sLand[slot] > 0.0) ? INFINITY : 0.0);
                leak = nanmax(leak, lk);
            }
        }
        gLeak[j] = leak;
        ssc = sc;                                       // kept for the overlap below
    }
    __syncthreads();
    const double sc = ssc;
    // the product's max / min per box
    for (int b = 0; b < cnt; b++) {
        const int slot = first + b;
        if (!oPres[slot]) continue;
        double hi = -INFINITY, lo = INFINITY;
        for (int k = threadIdx.x; k < nblk; k += blockDim.x) {
            const double* p = part + ((long long) slot * nblk + k) * NPART;
            hi = nanmax(hi, p[P_OHI]);
            lo = nanmin(lo, p[P_OLO]);
        }
        hi = block_reduce_max(hi);
        lo = block_reduce_min(lo);
        if (threadIdx.x == 0) {
            shi[b] = hi;
            slo[b] = lo;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const double Cn = (c == 1) ? Cp : Cpar[j] * Cp;
        double s = fabs(Cn);
        for (int b = 0; b < cnt; b++) {
            const int slot = first + b;
            if (!oPres[slot]) {
                presPar[slot] = 0;
                continue;
            }
            const double hi = shi[b], lo = slo[b];
            if (nanmax(hi, -lo) <= DROP) {             // underflowed: identically zero (L1560-1562)
                oPres[slot] = 0;
                presPar[slot] = 0;
                continue;
            }
            s = nanmax(s, nanmax(fabs(Cn + hi), fabs(Cn + lo)));
            presPar[slot] = 1;
        }
        if (!(isfinite(s) && s > 0.0)) {
            gStat[j] |= nonposBit;
            gDead[j] = 1;
            gNode[j] = parentNode;
        } else {
            if (c == 2) {
                const double rho = (sc > 0.0) ? s / sc : 0.0;
                if (rho < gRho[j]) gRho[j] = rho;
            }
            Cpar[j] = Cn / s;
            LSpar[j] = ((c == 1) ? LSchild[j] : (LSpar[j] + LSchild[j])) + log(s);
            sScale[j] = s;
        }
    }
}

// box_landing.py L1569-1573: f /= s for the parts kept in the parent (per grid point)
extern "C" __global__ void scale_parts(double* P, const int* presPar, const int* slotInfo, const int* siteInfo,
                                       const unsigned char* code, const int* gDead, const double* sScale,
                                       long long NG) {
    const int slot = blockIdx.y;
    const int site = slotInfo[4 * slot + 1];
    const int gs = siteInfo[4 * site];
    if (code[gs] == 0 || gDead[site] || !presPar[slot]) return;
    const double s = sScale[site];
    double* pp = P + (long long) slot * NG;
    for (long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x; g < NG; g += (long long) gridDim.x * blockDim.x) {
        pp[g] = pp[g] / s;
    }
}

// ---- the root -----------------------------------------------------------------------------------------

// box_landing.py L1590-1597: max / min / LEAK-band max of each root part (grid values)
extern "C" __global__ void root_stats(const double* P, const int* presRoot, const int* slotInfo, const int* siteInfo,
                                      const unsigned char* hasRoot, const int* gDead, const int* slotNB, int n1, int n2,
                                      double* part) {
    const int slot = blockIdx.y;
    const int site = slotInfo[4 * slot + 1];
    const int gs = siteInfo[4 * site];
    const int active = hasRoot[gs] && !gDead[site] && presRoot[slot];
    double hi = -INFINITY, lo = INFINITY, band = 0.0;
    if (active) {
        const long long gslot = slotInfo[4 * slot];
        const int nb1 = slotNB[2 * gslot], nb2 = slotNB[2 * gslot + 1];
        const long long NG = (long long) n1 * n2;
        const double* f = P + (long long) slot * NG;
        for (long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x; g < NG; g += (long long) gridDim.x * blockDim.x) {
            const int i = (int) (g / n2);
            const int jj = (int) (g - (long long) i * n2);
            const double v = f[g];
            hi = nanmax(hi, v);
            lo = nanmin(lo, v);
            const int bnd = (nb1 > 0 && (i < nb1 || i >= n1 - nb1)) || (nb2 > 0 && (jj < nb2 || jj >= n2 - nb2));
            if (bnd) band = nanmax(band, fabs(v));
        }
    }
    hi = block_reduce_max(hi);
    lo = block_reduce_min(lo);
    band = block_reduce_max(band);
    if (threadIdx.x == 0) {
        double* p = part + ((long long) slot * gridDim.x + blockIdx.x) * NPART;
        p[P_FHI] = hi;
        p[P_FLO] = lo;
        p[P_FBAND] = band;
    }
}

// box_landing.py L1598-1602: RESOLUTION of each root part, max |rfft2(f)| over the outer modes
extern "C" __global__ void root_res(const double2* X, const int* presRoot, const int* slotInfo, const int* siteInfo,
                                    const unsigned char* hasRoot, const int* gDead, int n1, int n2h, int r0, int r1,
                                    int c0, double* part) {
    const int slot = blockIdx.y;
    const int site = slotInfo[4 * slot + 1];
    const int gs = siteInfo[4 * site];
    const int active = hasRoot[gs] && !gDead[site] && presRoot[slot];
    double post = 0.0;
    if (active) {
        const long long N2 = (long long) n1 * n2h;
        const double2* x = X + (long long) slot * N2;
        for (long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x; idx < N2; idx += (long long) gridDim.x * blockDim.x) {
            const int i = (int) (idx / n2h);
            const int j = (int) (idx - (long long) i * n2h);
            if ((i >= r0 && i <= r1) || (j >= c0)) {
                const double2 c = x[idx];
                post = nanmax(post, hypot(c.x, c.y));
            }
        }
    }
    post = block_reduce_max(post);
    if (threadIdx.x == 0) part[((long long) slot * gridDim.x + blockIdx.x) * NPART + P_POST] = post;
}

// box_landing.py L1612 (inner product): T[slot][m1][j] = sum_m2 musub[m1][m2] E2[j][m2]   (musub @ E2.T), lam > 0
extern "C" __global__ void root_T(double2* T, const double2* MU, const double2* E2, const int* presRoot,
                                  const int* slotInfo, const int* siteInfo, const unsigned char* hasRoot,
                                  const int* gDead, const double* siteParam, int n2, int M) {
    const int slot = blockIdx.y;
    const int site = slotInfo[4 * slot + 1];
    const int gs = siteInfo[4 * site];
    if (!(hasRoot[gs] && !gDead[site] && presRoot[slot] && siteParam[8 * site + 5] > 0.0)) return;
    const long long gslot = slotInfo[4 * slot];
    const double2* mu = MU + (long long) site * M * M;
    const double2* e2 = E2 + gslot * n2 * M;
    double2* tt = T + (long long) slot * M * n2;
    const long long N = (long long) M * n2;
    for (long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += (long long) gridDim.x * blockDim.x) {
        const int m1 = (int) (idx / n2);
        const int j = (int) (idx - (long long) m1 * n2);
        const double2* mrow = mu + (long long) m1 * M;
        const double2* erow = e2 + (long long) j * M;
        double re = 0.0, im = 0.0;
        for (int m2 = 0; m2 < M; m2++) {
            const double2 a = mrow[m2], b = erow[m2];
            re = re + (a.x * b.x - a.y * b.y);
            im = im + (a.x * b.y + a.y * b.x);
        }
        double2 o;
        o.x = re;
        o.y = im;
        tt[idx] = o;
    }
}

// box_landing.py L1608-1619: the root integral of each part, sum_grid pig f with pig = Re(E1 @ T) (lam > 0) or
// sum_grid f (lam = 0), as compensated per-block partials (h1 h2 / (2 pi)^2 is applied per site in root_final)
extern "C" __global__ void root_dot(const double* P, const double2* T, const double2* E1, const int* presRoot,
                                    const int* slotInfo, const int* siteInfo, const unsigned char* hasRoot,
                                    const int* gDead, const double* siteParam, int n1, int n2, int M, double* part) {
    const int slot = blockIdx.y;
    const int site = slotInfo[4 * slot + 1];
    const int gs = siteInfo[4 * site];
    const int active = hasRoot[gs] && !gDead[site] && presRoot[slot];
    double acc = 0.0, err = 0.0;
    if (active) {
        const double lam = siteParam[8 * site + 5];
        const long long gslot = slotInfo[4 * slot];
        const long long NG = (long long) n1 * n2;
        const double* f = P + (long long) slot * NG;
        const double2* e1 = E1 + gslot * n1 * M;
        const double2* tt = T + (long long) slot * M * n2;
        for (long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x; g < NG; g += (long long) gridDim.x * blockDim.x) {
            double term;
            if (lam > 0.0) {
                const int i = (int) (g / n2);
                const int j = (int) (g - (long long) i * n2);
                const double2* erow = e1 + (long long) i * M;
                double pig = 0.0;
                for (int m1 = 0; m1 < M; m1++) {
                    const double2 a = erow[m1], b = tt[(long long) m1 * n2 + j];
                    pig = pig + (a.x * b.x - a.y * b.y);
                }
                term = pig * f[g];
            } else {
                term = f[g];
            }
            double e2;
            two_sum(acc, term, &acc, &e2);
            err += e2;
        }
    }
    block_reduce_pair(&acc, &err);
    if (threadIdx.x == 0) {
        double* p = part + ((long long) slot * gridDim.x + blockIdx.x) * NPART;
        p[P_RS] = acc;
        p[P_RE] = err;
    }
}

// One block per site: box_landing.py L1576-1638 without the shadow. LEAK and RESOLUTION of the root parts,
// L = C + sum_b (h1 h2 dot_b) / (2 pi)^2 in increasing box order, NONPOSITIVE, logL = log L + log scale.
// A site whose every tip is missing (no root message) has L = 1 (L1587-1588).
extern "C" __global__ void root_final(const int* siteInfo, const int* slotInfo, const unsigned char* hasRoot,
                                      const int* gDead, const int* presRoot, const double* part, int nblk,
                                      const double2* X, long long N2, const double* Croot, const double* LSroot,
                                      const double* slotHH, double twoPiSq, double* gLeak, double* gRes, int* gStat,
                                      int* gNode, double* outLogL, int nonposBit) {
    __shared__ double shi[MAX_BOXES], slo[MAX_BOXES], sband[MAX_BOXES], spost[MAX_BOXES], sdot[MAX_BOXES];
    const int j = blockIdx.x;
    const int gs = siteInfo[4 * j];
    if (!hasRoot[gs]) {
        if (threadIdx.x == 0) outLogL[j] = 0.0;
        return;
    }
    if (gDead[j]) {
        if (threadIdx.x == 0) outLogL[j] = nan("");
        return;
    }
    const int first = siteInfo[4 * j + 1], cnt = siteInfo[4 * j + 2];
    for (int b = 0; b < cnt; b++) {
        const int slot = first + b;
        if (!presRoot[slot]) continue;
        double hi = -INFINITY, lo = INFINITY, band = 0.0, post = 0.0, s = 0.0, e = 0.0;
        for (int k = threadIdx.x; k < nblk; k += blockDim.x) {
            const double* p = part + ((long long) slot * nblk + k) * NPART;
            hi = nanmax(hi, p[P_FHI]);
            lo = nanmin(lo, p[P_FLO]);
            band = nanmax(band, p[P_FBAND]);
            post = nanmax(post, p[P_POST]);
            double e2;
            two_sum(s, p[P_RS], &s, &e2);
            e += e2 + p[P_RE];
        }
        hi = block_reduce_max(hi);
        lo = block_reduce_min(lo);
        band = block_reduce_max(band);
        post = block_reduce_max(post);
        block_reduce_pair(&s, &e);
        if (threadIdx.x == 0) {
            shi[b] = hi;
            slo[b] = lo;
            sband[b] = band;
            spost[b] = post;
            sdot[b] = s + e;
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    const double C = Croot[j];
    double fm = 0.0;
    for (int b = 0; b < cnt; b++) {
        if (presRoot[first + b]) fm = nanmax(fm, nanmax(shi[b], -slo[b]));
    }
    const double sc = fabs(C) + fm;
    double leak = gLeak[j], res = gRes[j];
    double L = C;
    for (int b = 0; b < cnt; b++) {
        const int slot = first + b;
        if (!presRoot[slot]) continue;
        leak = nanmax(leak, (sc > 0.0) ? sband[b] / sc : INFINITY);
        const double2 x00 = X[(long long) slot * N2];
        const double x0 = hypot(x00.x, x00.y);
        const double r = spost[b];
        res = nanmax(res, (x0 > 0.0) ? r / x0 : ((r == 0.0) ? 0.0 : INFINITY));
        L = L + (slotHH[slotInfo[4 * slot]] * sdot[b]) / twoPiSq;
    }
    gLeak[j] = leak;
    gRes[j] = res;
    if (!(isfinite(L) && L > 0.0)) {
        gStat[j] |= nonposBit;
        gNode[j] = -1;
        outLogL[j] = (L == 0.0) ? -INFINITY : nan("");
    } else {
        outLogL[j] = log(L) + LSroot[j];
    }
}
