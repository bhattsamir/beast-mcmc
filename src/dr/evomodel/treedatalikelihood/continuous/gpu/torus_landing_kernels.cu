// torus_landing_kernels.cu
//
// CUDA kernels for TorusLandingGpuEngine: the production landing likelihood of
// TorusFourierTreeLikelihood / TorusFourierKernel (as run in HX2 jobs 846999 and 981654, commit 06e5709ad),
// batched over residues. Every kernel reproduces one production method; the comment above each names it.
// dim = 2 only. Grid index idx = i0 * n + i1 (row-major, the production stride), mode m_ax = fftfreq * n.
//
// Layout: one residue per blockIdx.y, n*n modes per residue strided over blockIdx.x * blockDim.x threads.
// Block size is fixed at BLOCK (256). Reductions are deterministic (no atomics): per-block partial results,
// then one block per residue combines them in a fixed order, so a recomputation reproduces the cached value.
//
// Compile with -fmad=false: the production arithmetic is plain IEEE multiply and add (Java never contracts),
// and FMA contraction would change rounding in the per-mode exponents.

#define BLOCK 256

// ---- helpers -----------------------------------------------------------------------------------------

// production TorusFourierKernel.modeOf for dim 2: m[ax] = freq[(idx / stride[ax]) % n], stride = {n, 1}
__device__ __forceinline__ void mode_of(int idx, int n, double* m0, double* m1) {
    const int i0 = idx / n;
    const int i1 = idx - i0 * n;
    *m0 = (i0 < n / 2) ? (double) i0 : (double) (i0 - n);
    *m1 = (i1 < n / 2) ? (double) i1 : (double) (i1 - n);
}

// production quadForm: q = sum_i sum_j s[i][j] * m[i] * m[j], evaluated left to right in this order
__device__ __forceinline__ double quad_form(const double* s, double m0, double m1) {
    double q = 0.0;
    q += s[0] * m0 * m0;
    q += s[1] * m0 * m1;
    q += s[2] * m1 * m0;
    q += s[3] * m1 * m1;
    return q;
}

// error-free transformation a + b = s + e
__device__ __forceinline__ void two_sum(double a, double b, double* s, double* e) {
    const double sum = a + b;
    const double bp = sum - a;
    *e = (a - (sum - bp)) + (b - bp);
    *s = sum;
}

// compensated block reduction of (sum, err) pairs; result valid in thread 0
__device__ void block_reduce_pair(double* s, double* e) {
    __shared__ double ss[BLOCK];
    __shared__ double se[BLOCK];
    const int tid = threadIdx.x;
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

// max block reduction; NaN propagates (production Math.max(NaN, x) is NaN), result valid in thread 0
__device__ double block_reduce_max(double v) {
    __shared__ double sm[BLOCK];
    const int tid = threadIdx.x;
    sm[tid] = v;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const double a = sm[tid], b = sm[tid + stride];
            sm[tid] = (a != a || b != b) ? (a + b) : (a > b ? a : b);
        }
        __syncthreads();
    }
    return sm[0];
}

// ---- tips --------------------------------------------------------------------------------------------

// production TorusFourierKernel.landingTipCoefficients: c(m) = scale exp(-1/2 m'Se m) (cos phase, -sin phase),
// phase = m . theta_wrapped. tipMag (= scale exp(-1/2 m'Se m)) is precomputed on the host with the
// production expression. theta is [R][2] for this tip (wrapped on the host by IEEEremainder), rmap maps the
// batch slot to the residue.
extern "C" __global__ void tip_coeffs(double2* X, const double* tipMag, const double* theta, const int* rmap,
                                      int n, int N) {
    const int b = blockIdx.y;
    const int r = rmap[b];
    const double t0 = theta[2 * r], t1 = theta[2 * r + 1];
    double2* x = X + (size_t) b * N;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x) {
        double m0, m1;
        mode_of(idx, n, &m0, &m1);
        double phase = 0.0;
        phase += m0 * t0;
        phase += m1 * t1;
        const double mag = tipMag[idx];
        double2 c;
        c.x = mag * cos(phase);
        c.y = -mag * sin(phase);
        x[idx] = c;
    }
}

// ---- branch operator ---------------------------------------------------------------------------------

// The landing operator for one mode, production TorusFourierKernel.applyLanding:
//   damping = exp(x), reset mass = (lambda / (lambda + a)) * -expm1(x), x = -(lambda + a) t.
// One transcendental instead of production's two: expm1 when |x| < 1/2 (damping = 1 + expm1(x), accurate to
// an ulp because damping is near 1) and exp otherwise (-expm1(x) = 1 - exp(x) has no cancellation there,
// relative error below 3 ulp). Below x = -746 exp is exactly 0 in double and 1 - 0 = 1 = -expm1(x) exactly, as
// in production, so no call is made. Only rounding differs from production.
__device__ __forceinline__ void landing_mode(double x, double* damping, double* oneMinus) {
    if (x > -0.5) {
        const double em = expm1(x);
        *damping = 1.0 + em;
        *oneMinus = -em;
    } else {
        const double d = (x < -746.0) ? 0.0 : exp(x);
        *damping = d;
        *oneMinus = 1.0 - d;
    }
}

// production applyLanding, first half: per mode, accumulate
//   landingResetMass(a, t, lambda) * Re(ghat * c)   (from the UNDAMPED coefficients)
// and damp c by exp(-(lambda + a) t). sig is [B][4] covariance (row-major), lam [B].
// Writes per-block compensated partial sums to part[b][blockIdx.x][2].
extern "C" __global__ void apply_landing(double2* X, const double2* ghat, const double* sig, const double* lam,
                                         double t, int n, int N, double* part) {
    const int b = blockIdx.y;
    const double* s = sig + 4 * b;
    const double L = lam[b];
    double2* x = X + (size_t) b * N;
    double acc = 0.0, err = 0.0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x) {
        double m0, m1;
        mode_of(idx, n, &m0, &m1);
        const double a = 0.5 * quad_form(s, m0, m1);
        double damping, oneMinus;
        landing_mode(-(L + a) * t, &damping, &oneMinus);
        double2 c = x[idx];
        if (L > 0.0) {
            const double mass = (L / (L + a)) * oneMinus;
            const double2 g = ghat[idx];
            const double term = mass * (g.x * c.x - g.y * c.y);
            double e2;
            two_sum(acc, term, &acc, &e2);
            err += e2;
        }
        c.x *= damping;
        c.y *= damping;
        x[idx] = c;
    }
    block_reduce_pair(&acc, &err);
    if (threadIdx.x == 0) {
        double* p = part + ((size_t) b * gridDim.x + blockIdx.x) * 2;
        p[0] = acc;
        p[1] = err;
    }
}

// Tip phase tables: T[b][0][k] = exp(-i m(k) theta0), T[b][1][k] = exp(-i m(k) theta1) for the tip's angles of
// the residue in batch slot b. exp(-i m.theta) = T[b][0][i0] T[b][1][i1] replaces production's cos/sin of the
// summed phase (rounding differs by a few ulp).
extern "C" __global__ void tip_phases(double2* T, const double* theta, const int* rmap, int n) {
    const int b = blockIdx.y;
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int r = rmap[b];
    const double m = (k < n / 2) ? (double) k : (double) (k - n);
    double s0, c0, s1, c1;
    sincos(m * theta[2 * r], &s0, &c0);
    sincos(m * theta[2 * r + 1], &s1, &c1);
    double2 v0, v1;
    v0.x = c0; v0.y = -s0;
    v1.x = c1; v1.y = -s1;
    T[((size_t) b * 2) * n + k] = v0;
    T[((size_t) b * 2 + 1) * n + k] = v1;
}

// Tip branch in one pass: the tip's Fourier coefficients (production landingTipCoefficients:
// magnitude exp(-i m.theta) with magnitude = (nTotal / (2 pi)^2) exp(-1/2 m'Se m)), then the landing operator
// exactly as apply_landing. Writes the damped coefficients to X and the reset partial sums to part.
extern "C" __global__ void tip_apply(double2* X, const double2* T, const double* tipMag, const double2* ghat,
                                     const double* sig, const double* lam, double t, int n, int N, double* part) {
    const int b = blockIdx.y;
    const double* s = sig + 4 * b;
    const double L = lam[b];
    const double2* T0 = T + ((size_t) b * 2) * n;
    const double2* T1 = T + ((size_t) b * 2 + 1) * n;
    double2* x = X + (size_t) b * N;
    double acc = 0.0, err = 0.0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x) {
        const int i0 = idx / n;
        const int i1 = idx - i0 * n;
        const double m0 = (i0 < n / 2) ? (double) i0 : (double) (i0 - n);
        const double m1 = (i1 < n / 2) ? (double) i1 : (double) (i1 - n);
        const double2 p0 = T0[i0], p1 = T1[i1];
        const double mag = tipMag[idx];
        double2 c;
        c.x = mag * (p0.x * p1.x - p0.y * p1.y);
        c.y = mag * (p0.x * p1.y + p0.y * p1.x);
        const double a = 0.5 * quad_form(s, m0, m1);
        double damping, oneMinus;
        landing_mode(-(L + a) * t, &damping, &oneMinus);
        if (L > 0.0) {
            const double mass = (L / (L + a)) * oneMinus;
            const double2 g = ghat[idx];
            const double term = mass * (g.x * c.x - g.y * c.y);
            double e2;
            two_sum(acc, term, &acc, &e2);
            err += e2;
        }
        c.x *= damping;
        c.y *= damping;
        x[idx] = c;
    }
    block_reduce_pair(&acc, &err);
    if (threadIdx.x == 0) {
        double* p = part + ((size_t) b * gridDim.x + blockIdx.x) * 2;
        p[0] = acc;
        p[1] = err;
    }
}

// production applyLanding, second half: re[0] += correction. One block per residue.
extern "C" __global__ void add_correction(double2* X, const double* part, int nblk, int N) {
    const int b = blockIdx.x;
    double acc = 0.0, err = 0.0;
    for (int k = threadIdx.x; k < nblk; k += blockDim.x) {
        const double* p = part + ((size_t) b * nblk + k) * 2;
        double e2;
        two_sum(acc, p[0], &acc, &e2);
        err += e2 + p[1];
    }
    block_reduce_pair(&acc, &err);
    if (threadIdx.x == 0) X[(size_t) b * N].x += acc + err;
}

// complex buffer from a real partial (production: re = partial.clone(), im = 0)
extern "C" __global__ void load_real(double2* X, const double* P, int N) {
    const int b = blockIdx.y;
    const double* p = P + (size_t) b * N;
    double2* x = X + (size_t) b * N;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x) {
        double2 c;
        c.x = p[idx];
        c.y = 0.0;
        x[idx] = c;
    }
}

// ---- combine children at a node ----------------------------------------------------------------------

// production updateLandingNodePartial, multiply loop for one child:
//   message = Re(inverse FFT) * (1 / nTotal)   [fftN inverse scaling]; non-finite message -> bad |= 1
//   values = first ? 1.0 * message : values * message   (a missing tip is the flat message 1)
//   max = max |values|                                   (per-block partial)
// present is [R] for a tip child (1 = observed), NULL for an internal child.
extern "C" __global__ void combine(double* P, const double2* X, const unsigned char* present, const int* rmap,
                                   int first, double invn, int N, double* blkmax, int* bad) {
    const int b = blockIdx.y;
    const int observed = (present == 0) ? 1 : (int) present[rmap[b]];
    double* p = P + (size_t) b * N;
    const double2* x = X + (size_t) b * N;
    double mx = 0.0;
    int nonfinite = 0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x) {
        double v;
        if (observed) {
            const double msg = x[idx].x * invn;
            if (!isfinite(msg)) nonfinite = 1;
            v = first ? 1.0 * msg : p[idx] * msg;
        } else {
            v = first ? 1.0 : p[idx];
        }
        p[idx] = v;
        const double av = fabs(v);
        mx = (mx != mx || av != av) ? (mx + av) : (av > mx ? av : mx);
    }
    if (nonfinite) atomicOr(&bad[b], 1);
    mx = block_reduce_max(mx);
    if (threadIdx.x == 0) blkmax[(size_t) b * gridDim.x + blockIdx.x] = mx;
}

// production updateLandingNodePartial, after the multiply loop:
//   if (!(max > 0) || !finite(max)) -> invalid partial (bad |= 2), values left as they are
//   values[k] /= max;  scale = (first ? 0 : scale) + childScale + log(max)
// childScale is NULL for a tip child. Every block recomputes max from the same partials in the same order.
extern "C" __global__ void normalize(double* P, const double* blkmax, int nblk, double* scaleNode,
                                     const double* scaleChild, int first, int N, int* bad) {
    const int b = blockIdx.y;
    __shared__ double mxs;
    if (threadIdx.x == 0) {
        double mx = 0.0;
        for (int k = 0; k < nblk; k++) {
            const double v = blkmax[(size_t) b * nblk + k];
            mx = (mx != mx || v != v) ? (mx + v) : (v > mx ? v : mx);
        }
        mxs = mx;
    }
    __syncthreads();
    const double mx = mxs;
    if (!(mx > 0.0) || !isfinite(mx)) {
        if (blockIdx.x == 0 && threadIdx.x == 0) atomicOr(&bad[b], 2);
        return;
    }
    double* p = P + (size_t) b * N;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x) {
        p[idx] /= mx;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double sc = first ? 0.0 : scaleNode[b];
        if (scaleChild != 0) sc += scaleChild[b];
        sc += log(mx);
        scaleNode[b] = sc;
    }
}

// ---- root --------------------------------------------------------------------------------------------

// production rootIntegralLanding (lambda > 0): sum over modes of
//   lambda / (lambda + 1/2 m'Sigma m) * Re(ghat * c)   from the forward FFT c of the root partial.
// Residues with lambda == 0 contribute 0 here (their root integral is the grid mean, see sum_real).
extern "C" __global__ void root_sum(const double2* X, const double2* ghat, const double* sig, const double* lam,
                                   int n, int N, double* part) {
    const int b = blockIdx.y;
    const double* s = sig + 4 * b;
    const double L = lam[b];
    const double2* x = X + (size_t) b * N;
    double acc = 0.0, err = 0.0;
    if (L > 0.0) {
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x) {
            double m0, m1;
            mode_of(idx, n, &m0, &m1);
            const double weight = L / (L + 0.5 * quad_form(s, m0, m1));
            const double2 g = ghat[idx];
            const double2 c = x[idx];
            const double term = weight * (g.x * c.x - g.y * c.y);
            double e2;
            two_sum(acc, term, &acc, &e2);
            err += e2;
        }
    }
    block_reduce_pair(&acc, &err);
    if (threadIdx.x == 0) {
        double* p = part + ((size_t) b * gridDim.x + blockIdx.x) * 2;
        p[0] = acc;
        p[1] = err;
    }
}

// production rootIntegral (lambda == 0): sum of the root partial's grid values
extern "C" __global__ void sum_real(const double* P, int N, double* part) {
    const int b = blockIdx.y;
    const double* p = P + (size_t) b * N;
    double acc = 0.0, err = 0.0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x) {
        double e2;
        two_sum(acc, p[idx], &acc, &e2);
        err += e2;
    }
    block_reduce_pair(&acc, &err);
    if (threadIdx.x == 0) {
        double* q = part + ((size_t) b * gridDim.x + blockIdx.x) * 2;
        q[0] = acc;
        q[1] = err;
    }
}

// combine per-block partial sums into one value per residue: out[b] = sum + err. One block per residue.
extern "C" __global__ void finalize_sum(const double* part, int nblk, double* out) {
    const int b = blockIdx.x;
    double acc = 0.0, err = 0.0;
    for (int k = threadIdx.x; k < nblk; k += blockDim.x) {
        const double* p = part + ((size_t) b * nblk + k) * 2;
        double e2;
        two_sum(acc, p[0], &acc, &e2);
        err += e2 + p[1];
    }
    block_reduce_pair(&acc, &err);
    if (threadIdx.x == 0) out[b] = acc + err;
}
