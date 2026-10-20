// torus_count_kernels.cu
//
// Kernels of TorusLandingCountEngine: the exact expected labelled landing counts of
// results_pipeline/labelled_counts.py (LabelledCounts.counts), step for step as its JAX port
// results_pipeline/labelled_counts_gpu.py does them: inside (pruning) pass, pre-order outside pass,
// series coefficients AR, BR, kept-mode compaction, the candidate counts the host plan needs, and the
// Gauss-Legendre panel sums of the double sum over mode pairs. FP64 and complex128 only. Compiled with
// -fmad=false so each expression rounds as numpy evaluates it (same operation order).
//
// Conventions (numpy's, which the jar shares): flat index idx = i0 * n + i1, axis 0 = phi; mode
// m = i for i < n/2, else i - n; FORWARD = numpy fft2 (sign -), INVERSE = n^2 x numpy ifft2 (sign +).
// Reductions: per-block partials in a fixed order, then one block combines them in block order, so a
// rerun is bitwise identical. No float atomics.
//
// @author Samir Bhatt

#define BLOCK 256
#define KMAX 3          // coarse regions (helical, extended, alpha_L)
#define NQMAX 32        // Gauss-Legendre nodes per panel at most (16 x refine 2)
#define TILE 1024       // compaction tile (4 elements per thread)

__device__ __forceinline__ int mode_of(long i, int n) { return (int)(i < n / 2 ? i : i - n); }

__device__ __forceinline__ double2 cmul(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// NaN-propagating max, as numpy's max
__device__ __forceinline__ double nmax(double a, double b) {
    if (isnan(a) || isnan(b)) return nan("");
    return a > b ? a : b;
}

// fixed-order block sum (tree), result valid in thread 0; sh has BLOCK doubles
__device__ double block_sum(double v, double* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = sh[threadIdx.x] + sh[threadIdx.x + s];
        __syncthreads();
    }
    const double r = sh[0];
    __syncthreads();
    return r;
}

__device__ double block_max(double v, double* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] = nmax(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const double r = sh[0];
    __syncthreads();
    return r;
}

extern "C" {

// ---------------------------------------------------------------------------------- generic

// part[nv][nblk] -> out[nv] (sum, or max if isMax), in block order
__global__ void k_reduce_parts(const double* part, int nblk, int nv, int isMax, double* out) {
    __shared__ double sh[BLOCK];
    for (int j = 0; j < nv; j++) {
        double acc = isMax ? 0.0 : 0.0;
        bool init = false;
        for (int b = threadIdx.x; b < nblk; b += BLOCK) {
            const double v = part[(long)j * nblk + b];
            if (isMax) { acc = init ? nmax(acc, v) : v; init = true; } else acc += v;
        }
        if (isMax && !init) acc = 0.0;
        const double r = isMax ? block_max(acc, sh) : block_sum(acc, sh);
        if (threadIdx.x == 0) out[j] = r;
    }
}

__global__ void k_r2c(double2* X, const double* x, long N) {
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK)
        X[i] = make_double2(x[i], 0.0);
}

__global__ void k_copy_c(double2* Y, const double2* X, long N) {
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) Y[i] = X[i];
}

// X = X * s1 * s2 (left to right)
__global__ void k_scale_c(double2* X, double s1, double s2, long N) {
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) {
        const double2 v = X[i];
        X[i] = make_double2((v.x * s1) * s2, (v.y * s1) * s2);
    }
}

// out = Re(X) / d
__global__ void k_c2r_div(double* out, const double2* X, double d, long N) {
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) out[i] = X[i].x / d;
}

// out = (Re(X) * s) / d      (numpy: ifft2(ct).real / h2 = (Re(IFFT_u ct) / N) / h2)
__global__ void k_c2r_scale_div(double* out, const double2* X, double s, double d, long N) {
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) out[i] = (X[i].x * s) / d;
}

// part[0][blk] += Re(a * b), part[1][blk] += Im(a * b)
__global__ void k_sum_cmul(const double2* a, const double2* b, long N, double* part) {
    __shared__ double sh[BLOCK];
    double sx = 0.0, sy = 0.0;
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) {
        const double2 p = cmul(a[i], b[i]);
        sx += p.x; sy += p.y;
    }
    sx = block_sum(sx, sh); sy = block_sum(sy, sh);
    if (threadIdx.x == 0) { part[blockIdx.x] = sx; part[gridDim.x + blockIdx.x] = sy; }
}

// ---------------------------------------------------------------------------------- per-draw model

// mu = (lam * ghat) / Lam
__global__ void k_mu(double2* mu, const double2* ghat, const double* Lam, double lam, long N) {
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) {
        const double2 g = ghat[i];
        const double L = Lam[i];
        mu[i] = make_double2((lam * g.x) / L, (lam * g.y) / L);
    }
}

// mus = mu[idx]; rho_s[a] = rho[a][idx]; gamG[a] = gam[a][idx] - mus * G[a]
__global__ void k_gather(double2* mus, double2* rho_s, double2* gamG, const double2* mu, const double2* rho,
                         const double2* gam, const double* G, const int* idx, int S, int K, long N) {
    for (long j = blockIdx.x * (long)BLOCK + threadIdx.x; j < S; j += (long)gridDim.x * BLOCK) {
        const long ix = idx[j];
        const double2 m = mu[ix];
        mus[j] = m;
        for (int a = 0; a < K; a++) {
            rho_s[(long)a * S + j] = rho[(long)a * N + ix];
            const double2 g = gam[(long)a * N + ix];
            const double2 mg = make_double2(m.x * G[a], m.y * G[a]);
            gamG[(long)a * S + j] = make_double2(g.x - mg.x, g.y - mg.y);
        }
    }
}

// ---------------------------------------------------------------------------------- inside pass

// c = exp(-i (m1 y0 + m2 y1)) * damp      (LabelledCounts._tip)
__global__ void k_tip_c(double2* c, const double* damp, double y0, double y1, int n) {
    const long N = (long)n * n;
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) {
        const double m1 = mode_of(i / n, n), m2 = mode_of(i % n, n);
        const double arg = m1 * y0 + m2 * y1;
        double s, co;
        sincos(arg, &s, &co);
        const double d = damp[i];
        c[i] = make_double2(co * d, -s * d);
    }
}

// ct = a c, a = exp(-Lam t); part += (mu (1 - a)) c      (LabelledCounts._prop, before the mode-0 correction)
__global__ void k_prop1(double2* ct, const double2* c, const double* Lam, const double2* mu, double t, long N,
                        double* part) {
    __shared__ double sh[BLOCK];
    double sx = 0.0, sy = 0.0;
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) {
        const double a = exp(-Lam[i] * t);
        const double2 cc = c[i];
        ct[i] = make_double2(a * cc.x, a * cc.y);
        const double oma = 1.0 - a;
        const double2 m = mu[i];
        const double2 p = cmul(make_double2(m.x * oma, m.y * oma), cc);
        sx += p.x; sy += p.y;
    }
    sx = block_sum(sx, sh); sy = block_sum(sy, sh);
    if (threadIdx.x == 0) { part[blockIdx.x] = sx; part[gridDim.x + blockIdx.x] = sy; }
}

// ct[0] += sum of the partials (one block)
__global__ void k_add0(double2* ct, const double* part, int nblk) {
    __shared__ double sh[BLOCK];
    double sx = 0.0, sy = 0.0;
    for (int b = threadIdx.x; b < nblk; b += BLOCK) { sx += part[b]; sy += part[nblk + b]; }
    sx = block_sum(sx, sh); sy = block_sum(sy, sh);
    if (threadIdx.x == 0) { const double2 v = ct[0]; ct[0] = make_double2(v.x + sx, v.y + sy); }
}

// prod = first ? Lt : prod * Lt; part = block max |prod|
__global__ void k_mul_max(double* prod, const double* Lt, int first, long N, double* part) {
    __shared__ double sh[BLOCK];
    double mx = 0.0;
    bool init = false;
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) {
        const double v = first ? Lt[i] : prod[i] * Lt[i];
        prod[i] = v;
        const double av = fabs(v);
        mx = init ? nmax(mx, av) : av; init = true;
    }
    if (!init) mx = 0.0;
    mx = block_max(mx, sh);
    if (threadIdx.x == 0) part[blockIdx.x] = mx;
}

// W = A * B (A may be W); part = block max |W| if wantMax
__global__ void k_mul2_max(double* W, const double* A, const double* B, long N, int wantMax, double* part) {
    __shared__ double sh[BLOCK];
    double mx = 0.0;
    bool init = false;
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) {
        const double v = A[i] * B[i];
        W[i] = v;
        if (wantMax) { const double av = fabs(v); mx = init ? nmax(mx, av) : av; init = true; }
    }
    if (wantMax) {
        if (!init) mx = 0.0;
        mx = block_max(mx, sh);
        if (threadIdx.x == 0) part[blockIdx.x] = mx;
    }
}

// W = A (copy); part = block max |W|
__global__ void k_copy_max(double* W, const double* A, long N, double* part) {
    __shared__ double sh[BLOCK];
    double mx = 0.0;
    bool init = false;
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) {
        const double v = A[i];
        W[i] = v;
        const double av = fabs(v);
        mx = init ? nmax(mx, av) : av; init = true;
    }
    if (!init) mx = 0.0;
    mx = block_max(mx, sh);
    if (threadIdx.x == 0) part[blockIdx.x] = mx;
}

// x = x / s[0]
__global__ void k_div_scalar(double* x, const double* s, long N) {
    const double d = s[0];
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) x[i] = x[i] / d;
}

// W = W / s[0]; part += W * Ltv      (W normalised, then the branch integral <W, P_t L_v> / h2)
__global__ void k_div_dot(double* W, const double* s, const double* Ltv, long N, double* part) {
    __shared__ double sh[BLOCK];
    const double d = s[0];
    double acc = 0.0;
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) {
        const double w = W[i] / d;
        W[i] = w;
        acc += w * Ltv[i];
    }
    acc = block_sum(acc, sh);
    if (threadIdx.x == 0) part[blockIdx.x] = acc;
}

// ---------------------------------------------------------------------------------- outside pass, per branch

// AR[a][j] = wts Re(rho_s (Wh[idx] - W1 mus)) / TP2;  BR[b][j] = wts Re(cv[idx] gamG) / TP2
// part (4K values x nblk): sum |AR_a|, sum |BR_b|, sum AR_a Ft, sum BR_b Ft, Ft = -expm1(-L t) / L
__global__ void k_series(double* AR, double* BR, const double2* Wh, const double2* cv, const int* idx,
                         const double* wts, const double* Ls, const double2* mus, const double2* rho_s,
                         const double2* gamG, double t, double TP2, int S, int K, double* part) {
    __shared__ double sh[BLOCK];
    double acc[4 * KMAX];
    for (int q = 0; q < 4 * KMAX; q++) acc[q] = 0.0;
    const double W1 = Wh[0].x;
    for (long j = blockIdx.x * (long)BLOCK + threadIdx.x; j < S; j += (long)gridDim.x * BLOCK) {
        const long ix = idx[j];
        const double2 wh = Wh[ix];
        const double2 m = mus[j];
        const double2 d = make_double2(wh.x - W1 * m.x, wh.y - W1 * m.y);
        const double w = wts[j];
        const double2 cj = cv[ix];
        const double L = Ls[j];
        const double Ft = -expm1(-L * t) / L;
        for (int a = 0; a < K; a++) {
            const double2 r = rho_s[(long)a * S + j];
            const double ar = (w * (r.x * d.x - r.y * d.y)) / TP2;
            AR[(long)a * S + j] = ar;
            acc[a] += fabs(ar);
            acc[2 * K + a] += ar * Ft;
            const double2 g = gamG[(long)a * S + j];
            const double br = (w * (cj.x * g.x - cj.y * g.y)) / TP2;
            BR[(long)a * S + j] = br;
            acc[K + a] += fabs(br);
            acc[3 * K + a] += br * Ft;
        }
    }
    for (int q = 0; q < 4 * K; q++) {
        const double r = block_sum(acc[q], sh);
        if (threadIdx.x == 0) part[(long)q * gridDim.x + blockIdx.x] = r;
    }
}

// keep thresholds from the reduced sums: tau[0] (A) = 1e-15 max_a SA_a / S, tau[1] (B) likewise (LabelledCounts._keep)
__global__ void k_tau(const double* sums, int K, int S, double* tau) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double mA = sums[0], mB = sums[K];
        for (int a = 1; a < K; a++) { mA = nmax(mA, sums[a]); mB = nmax(mB, sums[K + a]); }
        tau[0] = 1e-15 * mA / (double)S;
        tau[1] = 1e-15 * mB / (double)S;
    }
}

// keep flag per mode (max_a |R_a| > tau, strict) and per-tile counts
__global__ void k_keep(const double* R, int K, int S, const double* tau, int side, int* flags, int* tileCnt) {
    __shared__ int cnt[BLOCK];
    const double th = tau[side];
    int c = 0;
    for (int k = 0; k < TILE / BLOCK; k++) {
        const long j = (long)blockIdx.x * TILE + k * BLOCK + threadIdx.x;
        if (j < S) {
            double mag = fabs(R[j]);
            for (int a = 1; a < K; a++) mag = nmax(mag, fabs(R[(long)a * S + j]));
            const int f = mag > th ? 1 : 0;
            flags[j] = f;
            c += f;
        }
    }
    cnt[threadIdx.x] = c;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) cnt[threadIdx.x] += cnt[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) tileCnt[blockIdx.x] = cnt[0];
}

// exclusive prefix of the tile counts (one thread; a few thousand tiles at most); total -> total[0]
__global__ void k_scan_tiles(const int* tileCnt, int ntile, int* tileOff, int* total) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int s = 0;
        for (int b = 0; b < ntile; b++) { tileOff[b] = s; s += tileCnt[b]; }
        total[0] = s;
    }
}

// kept modes packed in front, Lambda order kept: Lc[pos] = Ls[j], Rc[a][pos] = R[a][j]
__global__ void k_compact(const double* R, const double* Ls, const int* flags, const int* tileOff, int K, int S,
                          long SP, double* Lc, double* Rc) {
    __shared__ int off[TILE];
    const long base = (long)blockIdx.x * TILE;
    for (int k = 0; k < TILE / BLOCK; k++) {
        const int u = k * BLOCK + threadIdx.x;
        off[u] = (base + u < S) ? flags[base + u] : 0;
    }
    __syncthreads();
    if (threadIdx.x == 0) {                      // exclusive scan of the tile, in order
        int s = tileOff[blockIdx.x];
        for (int u = 0; u < TILE; u++) { const int f = off[u]; off[u] = f ? s : -1; s += f; }
    }
    __syncthreads();
    for (int k = 0; k < TILE / BLOCK; k++) {
        const int u = k * BLOCK + threadIdx.x;
        const long j = base + u;
        if (j < S && off[u] >= 0) {
            const long p = off[u];
            Lc[p] = Ls[j];
            for (int a = 0; a < K; a++) Rc[(long)a * SP + p] = R[(long)a * S + j];
        }
    }
}

// for each candidate threshold: pc = searchsorted(Lc[:nk], thr, 'left'), Lpc = Lc[max(pc - 1, 0)];
// plan[0] = nk, plan[1] = Lc[nk - 1] (Lmax), plan[2 + c] = pc, plan[2 + NC + c] = Lpc
__global__ void k_plan(const double* Lc, const int* total, const double* thr, int NC, double* plan) {
    const int nk = total[0];
    const double INF = 1.0 / 0.0;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        plan[0] = (double)nk;
        plan[1] = nk > 0 ? Lc[nk - 1] : INF;
    }
    for (int c = blockIdx.x * BLOCK + threadIdx.x; c < NC; c += gridDim.x * BLOCK) {
        const double th = thr[c];
        int lo = 0, hi = nk;                     // first index with Lc >= th
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (Lc[mid] < th) lo = mid + 1; else hi = mid;
        }
        plan[2 + c] = (double)lo;
        plan[2 + NC + c] = nk > 0 ? Lc[lo > 0 ? lo - 1 : 0] : INF;
    }
}

// ---------------------------------------------------------------------------------- cross term

}  // extern "C" (the cross kernel is a template, instantiated per Gauss-Legendre order below)

// one block per work unit (panel p, side, modes [start, start + count)), all units of one launch with NQ nodes:
// partial[uoff + b][q][a] = sum_m Rc[a][m] exp(-(x_pq Lc[m])),  x = s (side A) or t - s (side B)
// units: int4 (p, side, start, count); nodes: [p][side][NQMAX]
template <int NQ>
__device__ void cross_body(const int4* units, int uoff, const double* nodes, const double* LcA, const double* RcA,
                           const double* LcB, const double* RcB, long SP, int K, double* partial) {
    __shared__ double xs[NQ];
    __shared__ double red[BLOCK / 32][NQ * KMAX];
    const int4 u = units[uoff + blockIdx.x];
    const int p = u.x, side = u.y, start = u.z, count = u.w;
    if (threadIdx.x < NQ) xs[threadIdx.x] = nodes[((long)p * 2 + side) * NQMAX + threadIdx.x];
    __syncthreads();
    const double* Lc = side == 0 ? LcA : LcB;
    const double* Rc = side == 0 ? RcA : RcB;
    double acc[NQ * KMAX];
#pragma unroll
    for (int q = 0; q < NQ * KMAX; q++) acc[q] = 0.0;
    for (int m = start + threadIdx.x; m < start + count; m += BLOCK) {
        const double L = Lc[m];
        double r[KMAX];
#pragma unroll
        for (int a = 0; a < KMAX; a++) r[a] = a < K ? Rc[(long)a * SP + m] : 0.0;
#pragma unroll
        for (int q = 0; q < NQ; q++) {
            const double e = exp(-(xs[q] * L));
#pragma unroll
            for (int a = 0; a < KMAX; a++) acc[q * KMAX + a] += e * r[a];
        }
    }
    // deterministic reduction: warp shuffles (fixed tree), then the warps in order
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int q = 0; q < NQ * KMAX; q++) {
        double v = acc[q];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) red[warp][q] = v;
    }
    __syncthreads();
    for (int q = threadIdx.x; q < NQ * KMAX; q += BLOCK) {
        double s = 0.0;
        for (int w = 0; w < BLOCK / 32; w++) s += red[w][q];
        partial[(long)(uoff + blockIdx.x) * NQMAX * KMAX + q] = s;
    }
}

extern "C" {

#define CROSS(NQ) __global__ void k_cross_##NQ(const int4* units, int uoff, const double* nodes, const double* LcA, \
        const double* RcA, const double* LcB, const double* RcB, long SP, int K, double* partial) { \
    cross_body<NQ>(units, uoff, nodes, LcA, RcA, LcB, RcB, SP, K, partial); }
CROSS(6) CROSS(8) CROSS(12) CROSS(16) CROSS(24) CROSS(32)

// one block: al/be per (panel, side, node, region) from the unit partials (units of a (p, side) are contiguous,
// first unit ufirst[p * 2 + side], ucount units), then X[a][b] = sum_p sum_q (ws_pq al_pqa) be_pqb (panel order)
__global__ void k_cross_finish(const double* partial, const int* ufirst, const int* ucount, const double* ws,
                               const int* nqs, int npan, int K, double* alpha, double* X) {
    const int tot = npan * 2 * NQMAX * KMAX;
    for (int i = threadIdx.x; i < tot; i += blockDim.x) {
        const int a = i % KMAX, q = (i / KMAX) % NQMAX, ps = i / (KMAX * NQMAX);   // ps = p * 2 + side
        double s = 0.0;
        if (q < nqs[ps / 2] && a < K) {
            const int f = ufirst[ps], c = ucount[ps];
            for (int k = 0; k < c; k++) s += partial[(long)(f + k) * NQMAX * KMAX + q * KMAX + a];
        }
        alpha[i] = s;
    }
    __syncthreads();
    if (threadIdx.x < K * K) {
        const int a = threadIdx.x / K, b = threadIdx.x % K;
        double x = 0.0;
        for (int p = 0; p < npan; p++) {
            const double* al = alpha + (long)(p * 2) * NQMAX * KMAX;
            const double* be = alpha + (long)(p * 2 + 1) * NQMAX * KMAX;
            double xp = 0.0;
            for (int q = 0; q < nqs[p]; q++) xp += (ws[(long)p * NQMAX + q] * al[q * KMAX + a]) * be[q * KMAX + b];
            x += xp;
        }
        X[a * K + b] = x;
    }
}

// ---------------------------------------------------------------------------------- outside message

// Xo = a Wh + (mu (1 - a)) Wh[0], a = exp(-Lam t)      (LabelledCounts._fwd before its forward transform)
__global__ void k_fwd_prep(double2* Xo, const double2* Wh, const double* Lam, const double2* mu, double t, long N) {
    const double2 w0 = Wh[0];
    for (long i = blockIdx.x * (long)BLOCK + threadIdx.x; i < N; i += (long)gridDim.x * BLOCK) {
        const double a = exp(-Lam[i] * t);
        const double2 wh = Wh[i];
        const double oma = 1.0 - a;
        const double2 m = mu[i];
        const double2 r = cmul(make_double2(m.x * oma, m.y * oma), w0);
        Xo[i] = make_double2(a * wh.x + r.x, a * wh.y + r.y);
    }
}

}  // extern "C"
