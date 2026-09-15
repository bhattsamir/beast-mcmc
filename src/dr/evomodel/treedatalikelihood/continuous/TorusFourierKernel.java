/*
 * TorusFourierKernel.java
 *
 * @author Samir Bhatt
 */

package dr.evomodel.treedatalikelihood.continuous;

import dr.math.FastFourierTransform;

/**
 * FFT heat-kernel operations for Brownian diffusion on a flat d-dimensional torus
 * (-pi,pi]^d, discretised on a grid with nGrid points per axis (nGrid a power of
 * two). Java port of the validated prototype torus_fourier_pruning.py. Grids are
 * row-major double[nGrid^dim] real functions; complex fields are kept as parallel
 * real/imag arrays and transformed with the in-house four1 FFT applied along each
 * axis.
 *
 * These low-level FFT operations are unvalidated candidates: Fourier truncation
 * and floating-point cancellation can corrupt narrow-kernel tails. The public
 * tree likelihood checks them against TorusLogSpacePruning before using a result.
 *
 * Convention matches numpy: the forward transform is sum_x f(x) exp(-i m.x) and the
 * inverse divides by nGrid^dim.
 *
 * @author Samir Bhatt
 */
public class TorusFourierKernel {

    // Keep the candidate FFT arrays finite for legacy callers. This floor is
    // NOT a numerical repair: it changes tail probabilities. The tree likelihood
    // accepts a candidate only after checking the positive log-space reference.
    private static final double MIN_POS = Double.MIN_NORMAL;
    private static final double TWO_PI = 2.0 * Math.PI;

    private final int dim;
    private final int nGrid;
    private final int nTotal;             // nGrid^dim
    private final double volume;          // (2*pi)^dim
    private final double[] freq;          // per-axis mode index: numpy fftfreq(nGrid)*nGrid
    private final int[] stride;           // row-major stride per axis = nGrid^(dim-1-ax)

    public TorusFourierKernel(int dim, int nGrid) {
        if (dim < 1) {
            throw new IllegalArgumentException("dim must be >= 1");
        }
        if (nGrid < 4 || (nGrid & (nGrid - 1)) != 0) {
            throw new IllegalArgumentException("nGrid must be a power of two >= 4");
        }
        this.dim = dim;
        this.nGrid = nGrid;
        int nt = 1;
        for (int i = 0; i < dim; i++) nt *= nGrid;
        this.nTotal = nt;
        this.volume = Math.pow(2.0 * Math.PI, dim);
        this.freq = new double[nGrid];
        for (int i = 0; i < nGrid; i++) {
            // numpy fftfreq(nGrid)*nGrid: the Nyquist index nGrid/2 maps to -nGrid/2 (not +nGrid/2).
            // The sign is irrelevant to the Brownian m^2 damping but flips the directed-jump phase
            // exp(i m.d) at the Nyquist row, so match numpy exactly for the complex multiplier.
            freq[i] = (i < nGrid / 2) ? i : i - nGrid;
        }
        this.stride = new int[dim];
        for (int ax = 0; ax < dim; ax++) {
            int s = 1;
            for (int k = ax + 1; k < dim; k++) s *= nGrid;
            stride[ax] = s;
        }
    }

    public int getDimension() {
        return dim;
    }

    public int getGridCount() {
        return nTotal;
    }

    /** Fill m[dim] with the mode vector at flat grid index idx. */
    private void modeOf(int idx, double[] m) {
        for (int ax = 0; ax < dim; ax++) {
            m[ax] = freq[(idx / stride[ax]) % nGrid];
        }
    }

    /** Per-mode Brownian damping exp(-1/2 t m^T Sigma m), Sigma a dim x dim covariance. */
    public double[] dampingFourier(double[][] sigma, double t) {
        return transferMultiplier(sigma, t, 0.0, null);
    }

    /** Quadratic form m^T S m. Reads the full matrix (both triangles), so it is the exact
     * quadratic form for any S, not only symmetric S -- a non-symmetric input no longer
     * silently yields a different value. (Identical to the upper-triangle form when S is
     * symmetric, as covariances/precisions are by construction here.) */
    private double quadForm(double[][] s, double[] m) {
        double q = 0.0;
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) q += s[i][j] * m[i] * m[j];
        }
        return q;
    }

    /**
     * Per-mode transfer multiplier exp(t * psi(m)) of a wrapped Levy diffusion, with
     * Levy-Khintchine characteristic exponent
     *     psi(m) = -1/2 m^T Sigma m + jumpRate (exp(-1/2 m^T Sigma_J m) - 1)
     * (Brownian + compound-Poisson jumps with wrapped-normal jump covariance Sigma_J).
     * jumpRate <= 0 or sigmaJump == null gives pure Brownian motion.
     */
    private double[] transferMultiplier(double[][] sigma, double t, double jumpRate, double[][] sigmaJump) {
        final boolean hasJump = jumpRate > 0.0 && sigmaJump != null;
        final double[] out = new double[nTotal];
        final double[] m = new double[dim];
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            double psi = -0.5 * quadForm(sigma, m);
            if (hasJump) psi += jumpRate * (Math.exp(-0.5 * quadForm(sigmaJump, m)) - 1.0);
            out[idx] = Math.exp(t * psi);
        }
        return out;
    }

    /** Propagate a non-negative grid partial across a branch of length t (Brownian). */
    public double[] propagate(double[] partial, double[][] sigma, double t) {
        return propagate(partial, sigma, t, 0.0, null);
    }

    /** Propagate across a branch under Brownian motion + compound-Poisson jumps. */
    public double[] propagate(double[] partial, double[][] sigma, double t,
                              double jumpRate, double[][] sigmaJump) {
        final double[] re = partial.clone();
        final double[] im = new double[nTotal];
        fftN(re, im, false);
        final double[] mult = transferMultiplier(sigma, t, jumpRate, sigmaJump);
        for (int k = 0; k < nTotal; k++) {
            re[k] *= mult[k];
            im[k] *= mult[k];
        }
        fftN(re, im, true);                               // == numpy ifftn
        for (int k = 0; k < nTotal; k++) {
            if (re[k] < MIN_POS) re[k] = MIN_POS;   // strictly positive: see MIN_POS note
        }
        return re;
    }

    /** Heat kernel K_t(grid - y) for a tip observation y (dim angles, radians; Brownian). */
    public double[] leafKernel(double[] y, double[][] sigma, double t) {
        return leafKernel(y, sigma, t, 0.0, null);
    }


    /** Grid angle along axis ax for flat index idx (x_j = 2*pi*j/nGrid, the FFT grid). */
    private double angleOf(int idx, int ax) {
        return TWO_PI * (double) ((idx / stride[ax]) % nGrid) / (double) nGrid;
    }

    /** Cholesky inverse of a small SPD matrix; returns log|a|. */
    private double symmetricInverse(double[][] a, double[][] inv) {
        final int n = a.length;
        final double[][] L = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = a[i][j];
                for (int k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
                if (i == j) {
                    if (!(sum > 0.0)) {
                        throw new IllegalArgumentException("t*Sigma must be positive definite");
                    }
                    L[i][i] = Math.sqrt(sum);
                } else {
                    L[i][j] = sum / L[j][j];
                }
            }
        }
        double logDet = 0.0;
        for (int i = 0; i < n; i++) logDet += 2.0 * Math.log(L[i][i]);
        for (int col = 0; col < n; col++) {
            final double[] z = new double[n];
            for (int i = 0; i < n; i++) {
                double sum = (i == col) ? 1.0 : 0.0;
                for (int k = 0; k < i; k++) sum -= L[i][k] * z[k];
                z[i] = sum / L[i][i];
            }
            for (int i = n - 1; i >= 0; i--) {
                double sum = z[i];
                for (int k = i + 1; k < n; k++) sum -= L[k][i] * inv[k][col];
                inv[i][col] = sum / L[i][i];
            }
        }
        return logDet;
    }

    /**
     * Add the zero-jump term, summed analytically over winding images.
     *
     * A tip is a point observation, so its Fourier coefficients have unit modulus at
     * every mode and only its own terminal branch damps them. When t*Sigma is small the
     * series needs |m| ~ 6/sqrt(sigma^2 t) modes, far past the grid; truncating it rings,
     * and clipping the negative lobes leaves a floor that fattens the tails so distant
     * tips score as compatible. The image sum is the same function (Poisson summation)
     * and converges in the opposite regime -- typically one term here.
     */
    private void addLatticeAtom(double[] out, double[] y, double[][] sigma, double t,
                                double logWeight) {
        final double[][] cov = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) cov[i][j] = t * sigma[i][j];
        }
        final double[][] inv = new double[dim][dim];
        final double logDet = symmetricInverse(cov, inv);
        double bound = 0.0;                          // Gershgorin bound on the top eigenvalue
        for (int i = 0; i < dim; i++) {
            double row = 0.0;
            for (int j = 0; j < dim; j++) row += Math.abs(cov[i][j]);
            if (row > bound) bound = row;
        }
        final int kmax = (int) Math.ceil(6.0 * Math.sqrt(bound) / TWO_PI) + 1;
        final int span = 2 * kmax + 1;
        int combos = 1;
        for (int i = 0; i < dim; i++) combos *= span;
        final double logNorm = logWeight - 0.5 * (dim * Math.log(TWO_PI) + logDet);
        final double[] d = new double[dim];
        final double[] q = new double[combos];
        for (int idx = 0; idx < nTotal; idx++) {
            double qmin = Double.POSITIVE_INFINITY;
            for (int c = 0; c < combos; c++) {
                int rem = c;
                for (int ax = 0; ax < dim; ax++) {
                    final int k = rem % span - kmax;
                    rem /= span;
                    d[ax] = angleOf(idx, ax) - y[ax] + TWO_PI * k;
                }
                double form = 0.0;
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) form += inv[i][j] * d[i] * d[j];
                }
                q[c] = form;
                if (form < qmin) qmin = form;
            }
            double acc = 0.0;
            for (int c = 0; c < combos; c++) acc += Math.exp(-0.5 * (q[c] - qmin));
            out[idx] += Math.exp(logNorm + Math.log(acc) - 0.5 * qmin);
        }
    }

    /**
     * Production leaf kernel: zero-jump term analytic, remaining jump mass spectral.
     *
     * e^{t psi(m)} = e^{-lam t} e^{-1/2 m'(t Sigma) m} + D(m) exactly, so the sharp half
     * is summed over winding images and D -- which carries phi(m)^n with n >= 1 and so
     * inherits the (broad) jump covariances -- stays on the FFT. multIm may be null for
     * a real multiplier.
     */
    private double[] leafKernelSplit(double[] y, double[][] sigma, double t,
                                     double jumpRate, double[] multRe, double[] multIm) {
        final double lamT = jumpRate * t;
        final double[] bm = transferMultiplier(sigma, t, 0.0, null);
        final double weight = Math.exp(-lamT);
        final double[] re = new double[nTotal];
        final double[] im = new double[nTotal];
        final double[] m = new double[dim];
        final double scale = (double) nTotal / volume;
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            double phase = 0.0;
            for (int ax = 0; ax < dim; ax++) phase += m[ax] * y[ax];
            final double cp = Math.cos(phase), sp = Math.sin(phase);
            final double dRe = multRe[idx] - weight * bm[idx];
            final double dIm = (multIm == null) ? 0.0 : multIm[idx];
            re[idx] = scale * (dRe * cp + dIm * sp);
            im[idx] = scale * (dIm * cp - dRe * sp);
        }
        fftN(re, im, true);
        // D is a Poisson mixture of densities, hence non-negative; negatives are truncation.
        for (int k = 0; k < nTotal; k++) if (re[k] < 0.0) re[k] = 0.0;
        addLatticeAtom(re, y, sigma, t, -lamT);
        for (int k = 0; k < nTotal; k++) if (re[k] < MIN_POS) re[k] = MIN_POS;
        return re;
    }

    /** Leaf heat kernel under Brownian motion + compound-Poisson jumps. */
    public double[] leafKernel(double[] y, double[][] sigma, double t,
                               double jumpRate, double[][] sigmaJump) {
        return leafKernelSplit(y, sigma, t, jumpRate,
                               transferMultiplier(sigma, t, jumpRate, sigmaJump), null);
    }

    /**
     * Complex per-mode multiplier exp(t * psi(m)) for BM + a DIRECTED jump mixture:
     *   psi(m) = -1/2 m^T Sigma m + jumpRate (sum_k w_k exp(i m.d_k) exp(-1/2 m^T S_k m) - 1).
     * The displacement phase exp(i m.d_k) makes psi complex; real/imag parts go into outRe/outIm.
     * A single zero-displacement component reduces to the real single-Gaussian jump.
     */
    private void transferMultiplierComplex(double[][] sigma, double t, double jumpRate,
                                           double[] w, double[][] d, double[][][] s,
                                           double[] outRe, double[] outIm) {
        final boolean hasJump = jumpRate > 0.0 && w != null && d != null && s != null;
        final double[] m = new double[dim];
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            double psiRe = -0.5 * quadForm(sigma, m);
            double psiIm = 0.0;
            if (hasJump) {
                double phiRe = 0.0, phiIm = 0.0;
                for (int k = 0; k < w.length; k++) {
                    final double g = w[k] * Math.exp(-0.5 * quadForm(s[k], m));
                    double md = 0.0;
                    for (int ax = 0; ax < dim; ax++) md += m[ax] * d[k][ax];
                    phiRe += g * Math.cos(md);
                    phiIm += g * Math.sin(md);
                }
                psiRe += jumpRate * (phiRe - 1.0);
                psiIm += jumpRate * phiIm;
            }
            final double mag = Math.exp(t * psiRe);
            final double ang = t * psiIm;
            outRe[idx] = mag * Math.cos(ang);
            outIm[idx] = mag * Math.sin(ang);
        }
    }

    /** Propagate across a branch under BM + a directed jump mixture (complex multiplier). */
    public double[] propagate(double[] partial, double[][] sigma, double t, double jumpRate,
                              double[] w, double[][] d, double[][][] s) {
        final double[] re = partial.clone();
        final double[] im = new double[nTotal];
        fftN(re, im, false);
        final double[] mRe = new double[nTotal];
        final double[] mIm = new double[nTotal];
        transferMultiplierComplex(sigma, t, jumpRate, w, d, s, mRe, mIm);
        for (int k = 0; k < nTotal; k++) {
            final double a = re[k], b = im[k];
            re[k] = a * mRe[k] - b * mIm[k];
            im[k] = a * mIm[k] + b * mRe[k];
        }
        fftN(re, im, true);
        for (int k = 0; k < nTotal; k++) {
            if (re[k] < MIN_POS) re[k] = MIN_POS;
        }
        return re;
    }

    /** Leaf heat kernel under BM + a directed jump mixture (complex multiplier). */
    public double[] leafKernel(double[] y, double[][] sigma, double t, double jumpRate,
                               double[] w, double[][] d, double[][][] s) {
        final double[] mRe = new double[nTotal];
        final double[] mIm = new double[nTotal];
        transferMultiplierComplex(sigma, t, jumpRate, w, d, s, mRe, mIm);
        return leafKernelSplit(y, sigma, t, jumpRate, mRe, mIm);
    }

    // ---- psi-cached fast path: the Levy-Khintchine exponent psi(m) is branch-length-independent,
    // so it is computed ONCE per residue (computePsiComplex) and reused on every branch via
    // multiplierFromPsi -> exp(t*psi). This hoists the mixture's per-frequency cos/sin (the
    // dominant cost) out of the per-branch loop. Results are identical to transferMultiplierComplex.

    /** Precompute psi(m) = psiRe + i psiIm for all frequencies (shared across a residue's branches):
     *  psiRe = -0.5 m^T sigma m + jumpRate (phiRe - 1), psiIm = jumpRate phiIm, with phi the mixture
     *  characteristic function sum_k w_k exp(-0.5 m^T s_k m) [cos(m.d_k) + i sin(m.d_k)]. */
    public void computePsiComplex(double[][] sigma, double jumpRate,
                                  double[] w, double[][] d, double[][][] s,
                                  double[] psiRe, double[] psiIm) {
        final boolean hasJump = jumpRate > 0.0 && w != null && d != null && s != null;
        final double[] m = new double[dim];
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            double pr = -0.5 * quadForm(sigma, m);
            double pi = 0.0;
            if (hasJump) {
                double phiRe = 0.0, phiIm = 0.0;
                for (int k = 0; k < w.length; k++) {
                    final double g = w[k] * Math.exp(-0.5 * quadForm(s[k], m));
                    double md = 0.0;
                    for (int ax = 0; ax < dim; ax++) md += m[ax] * d[k][ax];
                    phiRe += g * Math.cos(md);
                    phiIm += g * Math.sin(md);
                }
                pr += jumpRate * (phiRe - 1.0);
                pi += jumpRate * phiIm;
            }
            psiRe[idx] = pr;
            psiIm[idx] = pi;
        }
    }

    /** Branch multiplier a_t(m) = exp(t * psi(m)) from a precomputed psi. */
    private void multiplierFromPsi(double t, double[] psiRe, double[] psiIm,
                                   double[] outRe, double[] outIm) {
        for (int idx = 0; idx < nTotal; idx++) {
            final double mag = Math.exp(t * psiRe[idx]);
            final double ang = t * psiIm[idx];
            outRe[idx] = mag * Math.cos(ang);
            outIm[idx] = mag * Math.sin(ang);
        }
    }

    /** propagate() using a precomputed psi (see computePsiComplex). */
    public double[] propagateFromPsi(double[] partial, double t, double[] psiRe, double[] psiIm) {
        final double[] re = partial.clone();
        final double[] im = new double[nTotal];
        fftN(re, im, false);
        final double[] mRe = new double[nTotal];
        final double[] mIm = new double[nTotal];
        multiplierFromPsi(t, psiRe, psiIm, mRe, mIm);
        for (int k = 0; k < nTotal; k++) {
            final double a = re[k], b = im[k];
            re[k] = a * mRe[k] - b * mIm[k];
            im[k] = a * mIm[k] + b * mRe[k];
        }
        fftN(re, im, true);
        for (int k = 0; k < nTotal; k++) {
            if (re[k] < MIN_POS) re[k] = MIN_POS;
        }
        return re;
    }

    /** leafKernel() using a precomputed psi (see computePsiComplex). */
    public double[] leafKernelFromPsi(double[] y, double t, double[] psiRe, double[] psiIm,
                                      double[][] sigma, double jumpRate) {
        // sigma and jumpRate are required: the zero-jump term e^{-lam t} W_{t Sigma} is
        // evaluated analytically and cannot be recovered from psi alone.
        final double[] mRe = new double[nTotal];
        final double[] mIm = new double[nTotal];
        multiplierFromPsi(t, psiRe, psiIm, mRe, mIm);
        return leafKernelSplit(y, sigma, t, jumpRate, mRe, mIm);
    }

    // ---- ancestral-reconstruction support: real-space transition kernel + grid-cell samplers ----
    // These let a downward pass joint-sample internal-node states from the cached inward partials:
    // root ~ partial_root; then child ~ K_t(child-parent) * partial_child. The model otherwise
    // marginalises internal states, so this is what makes ASR possible under the spectral likelihood.

    public int getGridSize() {
        return nGrid;
    }

    /** Real-space transition kernel K_t(delta) on the grid = inverse FFT of the branch multiplier
     *  exp(t*psi). K_t[j] is the (unnormalised) transition density for a displacement of grid offset
     *  j; used as relative weights when sampling a child cell given its parent. */
    public double[] transitionKernel(double t, double[] psiRe, double[] psiIm) {
        final double[] re = new double[nTotal];
        final double[] im = new double[nTotal];
        multiplierFromPsi(t, psiRe, psiIm, re, im);
        fftN(re, im, true);
        return re;   // real part; tiny round-off negatives are clamped at use
    }

    /** Decode grid cell idx -> angle per axis, wrapped to (-pi, pi]. */
    public void cellToAngles(int idx, double[] out) {
        for (int ax = 0; ax < dim; ax++) {
            final int k = (idx / stride[ax]) % nGrid;
            final double a = (2.0 * Math.PI) * k / nGrid;   // real-space grid spans [0, 2pi)
            out[ax] = a > Math.PI ? a - 2.0 * Math.PI : a;
        }
    }

    /** Sample a grid cell with probability proportional to max(partial, 0); u in [0,1). */
    public int sampleCell(double[] partial, double u) {
        double tot = 0.0;
        for (double v : partial) if (v > 0.0) tot += v;
        if (!(tot > 0.0)) return 0;
        final double target = u * tot;     // u in [0,1) -> target in [0,tot); strict '>' skips zero-weight cells
        double c = 0.0;
        int last = 0;
        for (int i = 0; i < nTotal; i++) {
            if (partial[i] > 0.0) { last = i; c += partial[i]; if (c > target) return i; }
        }
        return last;   // last positive cell (never a zero-weight trailing cell at u->1)
    }

    /** Sample a child cell ~ K_t(child - parent) * childPartial(child); u in [0,1). Falls back to
     *  childPartial alone if the joint weight underflows everywhere. */
    public int sampleConditionalCell(double[] kernelGrid, int parentIdx, double[] childPartial, double u) {
        final int[] kp = new int[dim];
        for (int ax = 0; ax < dim; ax++) kp[ax] = (parentIdx / stride[ax]) % nGrid;
        final double[] w = new double[nTotal];
        double tot = 0.0;
        for (int idx = 0; idx < nTotal; idx++) {
            int disp = 0;
            for (int ax = 0; ax < dim; ax++) {
                final int kc = (idx / stride[ax]) % nGrid;
                int d = kc - kp[ax]; if (d < 0) d += nGrid;
                disp += d * stride[ax];
            }
            double kv = kernelGrid[disp];
            if (kv > 0.0 && childPartial[idx] > 0.0) { w[idx] = kv * childPartial[idx]; tot += w[idx]; }
        }
        if (!(tot > 0.0)) return sampleCell(childPartial, u);
        final double target = u * tot;     // strict '>' so a zero-weight leading cell is never chosen
        double c = 0.0;
        int last = 0;
        for (int idx = 0; idx < nTotal; idx++) {
            if (w[idx] > 0.0) last = idx;
            c += w[idx]; if (c > target) return idx;
        }
        return last;   // last positive-weight cell
    }

    /** Uniform/Haar-prior root integral of a grid partial = grid mean. */
    public double rootIntegral(double[] partial) {
        double s = 0.0;
        for (double v : partial) s += v;
        return s / nTotal;
    }

    /* N-D FFT: in-house 1-D four1 applied along each axis (strided lines), with
     * the inverse divided by nTotal (four1 itself is unnormalised). */
    private void fftN(double[] re, double[] im, boolean inverse) {
        final double[] buf = new double[2 * nGrid];
        for (int ax = 0; ax < dim; ax++) {
            final int s = stride[ax];
            final int outer = nTotal / (nGrid * s);
            for (int o = 0; o < outer; o++) {
                for (int innr = 0; innr < s; innr++) {
                    final int base = o * nGrid * s + innr;
                    for (int j = 0; j < nGrid; j++) {
                        final int p = base + j * s;
                        buf[2 * j] = re[p];
                        buf[2 * j + 1] = im[p];
                    }
                    FastFourierTransform.fft(buf, nGrid, inverse);
                    for (int j = 0; j < nGrid; j++) {
                        final int p = base + j * s;
                        re[p] = buf[2 * j];
                        im[p] = buf[2 * j + 1];
                    }
                }
            }
        }
        if (inverse) {
            final double invn = 1.0 / nTotal;
            for (int k = 0; k < nTotal; k++) {
                re[k] *= invn;
                im[k] *= invn;
            }
        }
    }
}
