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
    // Grid-resolution margin used by requiredGrid(): cells per standard deviation, i.e.
    // n >= 2*pi*SHARPNESS_MARGIN/sd. Diagnostic only.
    private static final double SHARPNESS_MARGIN = 1.05;

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

    /**
     * Heat kernel for a tip observed with wrapped-Gaussian error (Brownian).
     *
     * sigmaEps is the tip observation-error COVARIANCE (radians^2). null -- or, from the
     * likelihood, an exactly-zero Sigma_eps -- is the historical Dirac tip and takes
     * bit-for-bit the original arithmetic.
     */
    public double[] leafKernel(double[] y, double[][] sigma, double t, double[][] sigmaEps) {
        return leafKernelSplit(y, sigma, t, 0.0,
                               transferMultiplier(sigma, t, 0.0, null), null, sigmaEps);
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
     * A tip observed with wrapped-Gaussian error Sigma_eps has Fourier coefficients
     * exp(-i m.y) * exp(-1/2 m' Sigma_eps m): the unit-modulus phase of a point
     * observation, damped by a real Gaussian factor. Only its own terminal branch damps
     * it further. Sigma_eps == null recovers the Dirac tip (unit modulus at every mode),
     * which is what this kernel assumed before tip error existed.
     *
     * The zero-jump atom is therefore the wrapped normal of covariance
     * (Sigma_eps + t*Sigma). When that covariance is small the Fourier series needs
     * |m| ~ 6/sqrt(lambda_min) modes, far past the grid; truncating it rings, and
     * clipping the negative lobes leaves a floor that fattens the tails so distant tips
     * score as compatible. The image sum is the same function (Poisson summation) and
     * converges in the opposite regime -- typically one term here. A non-zero Sigma_eps
     * puts a floor under that covariance, so the atom stays well resolved however short
     * the terminal branch is.
     */
    private void addLatticeAtom(double[] out, double[] y, double[][] sigma, double t,
                                double logWeight, double[][] sigmaEps) {
        final double[][] cov = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) cov[i][j] = t * sigma[i][j];
        }
        if (sigmaEps != null) {                      // tip error broadens the atom: Sigma_eps + t*Sigma
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) cov[i][j] += sigmaEps[i][j];
            }
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
     *
     * TIP OBSERVATION ERROR. A tip observed with wrapped-Gaussian error Sigma_eps has
     * leaf coefficients
     *     exp(-i m.y) * exp(-1/2 m' Sigma_eps m) * a_t(m),   a_t(m) = e^{t psi(m)}.
     * The split is taken on the UNDAMPED multiplier a_t, and the real damping factor
     * exp(-1/2 m' Sigma_eps m) then multiplies BOTH halves: the atom absorbs it exactly,
     * as covariance (Sigma_eps + t*Sigma) in real space, and the spectral remainder D(m)
     * carries it as a per-mode real factor. Folding Sigma_eps into the SUBTRACTED
     * Gaussian instead would leave multRe - e^{-lam t} e^{-1/2 m'(t Sigma + Sigma_eps) m},
     * which is not D(m) times anything; the analytic and spectral halves would then
     * disagree with no symptom but a wrong number. sigmaEps == null skips the damping
     * entirely, so the Dirac-tip path is unchanged to the last bit.
     */
    private double[] leafKernelSplit(double[] y, double[][] sigma, double t,
                                     double jumpRate, double[] multRe, double[] multIm,
                                     double[][] sigmaEps) {
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
            double dRe = multRe[idx] - weight * bm[idx];
            double dIm = (multIm == null) ? 0.0 : multIm[idx];
            if (sigmaEps != null) {
                final double damp = Math.exp(-0.5 * quadForm(sigmaEps, m));
                dRe *= damp;
                dIm *= damp;
            }
            re[idx] = scale * (dRe * cp + dIm * sp);
            im[idx] = scale * (dIm * cp - dRe * sp);
        }
        fftN(re, im, true);
        // D is a Poisson mixture of densities, hence non-negative; negatives are truncation.
        // Convolving with the tip-error Gaussian preserves non-negativity, so this still holds.
        for (int k = 0; k < nTotal; k++) if (re[k] < 0.0) re[k] = 0.0;
        addLatticeAtom(re, y, sigma, t, -lamT, sigmaEps);
        for (int k = 0; k < nTotal; k++) if (re[k] < MIN_POS) re[k] = MIN_POS;
        return re;
    }

    /** Leaf heat kernel under Brownian motion + compound-Poisson jumps. */
    public double[] leafKernel(double[] y, double[][] sigma, double t,
                               double jumpRate, double[][] sigmaJump) {
        return leafKernel(y, sigma, t, jumpRate, sigmaJump, null);
    }

    /** Leaf heat kernel under BM + compound-Poisson jumps, tip observed with error sigmaEps
     *  (covariance, radians^2; null = Dirac tip). */
    public double[] leafKernel(double[] y, double[][] sigma, double t,
                               double jumpRate, double[][] sigmaJump, double[][] sigmaEps) {
        return leafKernelSplit(y, sigma, t, jumpRate,
                               transferMultiplier(sigma, t, jumpRate, sigmaJump), null, sigmaEps);
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
        return leafKernel(y, sigma, t, jumpRate, w, d, s, null);
    }

    /** Leaf heat kernel under BM + a directed jump mixture, tip observed with error sigmaEps
     *  (covariance, radians^2; null = Dirac tip). */
    public double[] leafKernel(double[] y, double[][] sigma, double t, double jumpRate,
                               double[] w, double[][] d, double[][][] s, double[][] sigmaEps) {
        final double[] mRe = new double[nTotal];
        final double[] mIm = new double[nTotal];
        transferMultiplierComplex(sigma, t, jumpRate, w, d, s, mRe, mIm);
        return leafKernelSplit(y, sigma, t, jumpRate, mRe, mIm, sigmaEps);
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
        return leafKernelFromPsi(y, t, psiRe, psiIm, sigma, jumpRate, null);
    }

    /** leafKernelFromPsi() for a tip observed with error sigmaEps (covariance, radians^2;
     *  null = Dirac tip). Sigma_eps damps the leaf coefficients and is branch-independent,
     *  so it does NOT belong in psi: psi is exponentiated by t, the tip factor is not. */
    public double[] leafKernelFromPsi(double[] y, double t, double[] psiRe, double[] psiIm,
                                      double[][] sigma, double jumpRate, double[][] sigmaEps) {
        // sigma and jumpRate are required: the zero-jump term e^{-lam t} W_{t Sigma} is
        // evaluated analytically and cannot be recovered from psi alone.
        final double[] mRe = new double[nTotal];
        final double[] mIm = new double[nTotal];
        multiplierFromPsi(t, psiRe, psiIm, mRe, mIm);
        return leafKernelSplit(y, sigma, t, jumpRate, mRe, mIm, sigmaEps);
    }

    // ---- ancestral-reconstruction support: real-space transition kernel + grid-cell samplers ----
    // These let a downward pass joint-sample internal-node states from the cached inward partials:
    // root ~ partial_root; then child ~ K_t(child-parent) * partial_child. The model otherwise
    // marginalises internal states, so this is what makes ASR possible under the spectral likelihood.

    public int getGridSize() {
        return nGrid;
    }

    /**
     * Smallest power-of-two grid that resolves a wrapped normal of covariance cov:
     * n >= 2*pi*SHARPNESS_MARGIN / sqrt(lambda_min(cov)). With a Dirac tip the sharpest
     * object in the tree is the shortest terminal branch (t_min*Sigma), so the requirement
     * is unbounded as t_min -> 0. With tip error the sharpest object anywhere is Sigma_eps
     * itself -- pruning messages only broaden going up -- so the requirement is bounded and
     * tree-independent. Diagnostic only; nothing in the kernel calls it.
     */
    public static int requiredGrid(double[][] cov) {
        final int d = cov.length;
        double lamMin;
        if (d == 1) {
            lamMin = cov[0][0];
        } else if (d == 2) {
            final double tr = cov[0][0] + cov[1][1];
            final double det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
            lamMin = 0.5 * tr - Math.sqrt(Math.max(0.25 * tr * tr - det, 0.0));
        } else {
            lamMin = Double.POSITIVE_INFINITY;   // Gershgorin lower bound (conservative)
            for (int i = 0; i < d; i++) {
                double off = 0.0;
                for (int j = 0; j < d; j++) if (j != i) off += Math.abs(cov[i][j]);
                lamMin = Math.min(lamMin, cov[i][i] - off);
            }
        }
        if (!(lamMin > 0.0)) return Integer.MAX_VALUE;   // a delta needs an unbounded grid
        final double need = TWO_PI * SHARPNESS_MARGIN / Math.sqrt(lamMin);
        int n = 4;
        while (n < need && n < (1 << 20)) n <<= 1;
        return n;
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

    // ============== landing ancestral sampling: FFT downward pass ==============
    // The dense positive reference (TorusLandingPruning) builds a cells x cells transition
    // matrix per edge, i.e. O(g^4) per edge. On the 225-taxon HA tree at nGrid 128 that is
    // 6.0e10 operations per residue per logged state and the cost guard rejects it; at the
    // grids where it does fit (<= 64 here) its quadrature does not converge. It is a
    // small-tree validator, not a production sampler.
    //
    // This is the same conditional law evaluated with the FFT already used by the upward
    // pass, O(g^2 log g) per edge -- about 1170x cheaper at nGrid 128. The maths is
    // unchanged. For a landing branch of length t, given the PARENT angle x,
    //
    //     p_t(y | x) = e^{-lambda t} K_{t Sigma}(y - x)  +  (1 - e^{-lambda t}) r_t(y),
    //
    // where K is the wrapped diffusion kernel and r_t -- the law of a point that jumped at
    // some time in (0, t] and then diffused for the remaining time -- does NOT depend on x.
    // So a draw is: flip a coin with P(jump) = 1 - e^{-lambda t}; on heads draw from
    // r_t * subtreePartial, on tails from K_{t Sigma}(. - x) * subtreePartial. Both targets
    // are a pointwise product of a grid density with the child's cached partial, so each is
    // one inverse FFT plus a categorical draw over the grid.
    //
    // NEGATIVE TAILS. Fourier truncation can leave small negative cells. The likelihood
    // integrates them harmlessly, but a sampler must normalise to a probability vector.
    // Clipping silently changes the conditional, so instead the negative mass is measured
    // and reported through the existing landing audit counters; the caller decides whether
    // the grid is adequate. Cells are floored at zero ONLY after that accounting, and the
    // floored mass is recorded so it can never pass unnoticed.

    /** Grid density of the no-jump branch kernel K_{t Sigma}, centred at the ORIGIN.
     *  Shift to a parent cell with sampleConditionalCell, which indexes by integer cell
     *  displacement -- the same convention the dense reference uses. Centring by a continuous
     *  phase factor instead introduces a sub-cell offset that biases every sampled node. */
    public double[] landingDiffusionKernel(double[][] sigma, double t) {
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingTime(t);
        final double[] re = new double[nTotal];
        final double[] im = new double[nTotal];
        final double[] m = new double[dim];
        final double scale = (double) nTotal / volume;
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            re[idx] = scale * Math.exp(-quadForm(sigma, m) * 0.5 * t);
        }
        fftN(re, im, true);
        return re;
    }

    /**
     * Grid density of the post-jump law r_t: a draw from g diffused for the time since the
     * LAST jump in (0, t], conditioned on at least one jump having occurred. In Fourier space
     * the unconditional reset mass at mode m is ghat(m) * lambda (1 - e^{-(lambda+a_m)t})
     * / (lambda + a_m); dividing by its mode-0 value (1 - e^{-lambda t}) conditions on a jump.
     */
    public double[] landingResetDensity(double[][] sigma, double t, double lambda,
                                        double[] ghatRe, double[] ghatIm) {
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingTime(t);
        requireLandingRate(lambda);
        requireLandingGhat(lambda, ghatRe, ghatIm);
        final double jumpProbability = -Math.expm1(-lambda * t);
        if (!(jumpProbability > 0.0)) {
            throw new IllegalArgumentException("landing reset density requires lambda*t > 0");
        }
        final double[] re = new double[nTotal];
        final double[] im = new double[nTotal];
        final double[] m = new double[dim];
        final double scale = (double) nTotal / volume;
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            final double a = 0.5 * quadForm(sigma, m);
            final double mass = landingResetMass(a, t, lambda) / jumpProbability;
            // CONJUGATE. ghat(m) = E_g[exp(+i m.theta)] is the characteristic function: it pairs
            // with FORWARD coefficients in the likelihood (applyLanding, rootIntegralLanding),
            // which is why the likelihood is right. As a DENSITY in the inverse-FFT basis
            // exp(+i m.y), the coefficient is conj(ghat(m)). Using ghat directly reflects every
            // basin through the origin -- found 2026-10-01 when the FFT sampler disagreed with
            // the dense reference by 0.52 rad; a basin at (1,-0.5) peaked at (-1,+0.5).
            re[idx] = scale * mass * ghatRe[idx];
            im[idx] = -scale * mass * ghatIm[idx];
        }
        fftN(re, im, true);
        return re;
    }

    /**
     * Sample one grid cell from (density * partial), returning its flat index, or -1 when the
     * product carries no positive mass. Negative cells are accounted through the landing audit
     * counters and then floored; `uniform` is the caller's U(0,1) draw.
     */
    public int sampleFromProduct(double[] density, double[] partial, double uniform) {
        requireLandingArray(density, nTotal, "density");
        requireLandingArray(partial, nTotal, "partial");
        double total = 0.0, negativeMass = 0.0;
        long nonpositive = 0, negative = 0;
        final double[] weights = new double[nTotal];
        for (int k = 0; k < nTotal; k++) {
            final double v = density[k] * partial[k];
            if (!(v > 0.0)) {
                nonpositive++;
                if (v < 0.0) { negative++; negativeMass -= v; }
                weights[k] = 0.0;
            } else {
                weights[k] = v;
                total += v;
            }
        }
        landingDensityNonpositiveCount.addAndGet(nonpositive);
        landingDensityNegativeCount.addAndGet(negative);
        if (negativeMass > 0.0) landingNegativeMass.add(negativeMass);
        if (!(total > 0.0) || !isFinite(total)) return -1;
        double target = uniform * total;
        for (int k = 0; k < nTotal; k++) {
            target -= weights[k];
            if (target <= 0.0) return k;
        }
        return nTotal - 1;
    }

    /**
     * Sample a child cell from the landing conditional given the parent CELL.
     *
     *   p_t(child | parent) propto [ e^{-lambda t} K_{t Sigma}(child - parent)
     *                                + (1 - e^{-lambda t}) r_t(child) ] * childPartial(child)
     *
     * The no-jump term depends on the integer cell DISPLACEMENT (child - parent), exactly as
     * the dense reference indexes it; the reset term is absolute, independent of the parent.
     * Mixing them in displacement space is what makes this agree with the dense sampler.
     *
     * Negative Fourier tail cells are counted through the landing audit before being floored,
     * so truncation is recorded rather than silently absorbed. Returns -1 when no positive
     * mass remains.
     */
    public int sampleLandingChildCell(double[] diffusionKernel, double[] resetDensity,
                                      double noJumpProbability, int parentIdx,
                                      double[] childPartial, double u) {
        requireLandingArray(diffusionKernel, nTotal, "diffusionKernel");
        requireLandingArray(childPartial, nTotal, "childPartial");
        if (resetDensity != null) requireLandingArray(resetDensity, nTotal, "resetDensity");
        final double jumpProbability = 1.0 - noJumpProbability;
        final int[] kp = new int[dim];
        for (int ax = 0; ax < dim; ax++) kp[ax] = (parentIdx / stride[ax]) % nGrid;

        final double[] w = new double[nTotal];
        double total = 0.0, negativeMass = 0.0;
        long nonpositive = 0, negative = 0;
        for (int idx = 0; idx < nTotal; idx++) {
            int disp = 0;
            for (int ax = 0; ax < dim; ax++) {
                final int kc = (idx / stride[ax]) % nGrid;
                int d = kc - kp[ax];
                if (d < 0) d += nGrid;
                disp += d * stride[ax];
            }
            double density = noJumpProbability * diffusionKernel[disp];
            if (resetDensity != null) density += jumpProbability * resetDensity[idx];
            final double v = density * childPartial[idx];
            if (!(v > 0.0)) {
                nonpositive++;
                if (v < 0.0) { negative++; negativeMass -= v; }
            } else {
                w[idx] = v;
                total += v;
            }
        }
        landingDensityNonpositiveCount.addAndGet(nonpositive);
        landingDensityNegativeCount.addAndGet(negative);
        if (negativeMass > 0.0) landingNegativeMass.add(negativeMass);
        if (!(total > 0.0) || Double.isNaN(total) || Double.isInfinite(total)) return -1;

        final double target = u * total;
        double c = 0.0;
        int last = 0;
        for (int idx = 0; idx < nTotal; idx++) {
            if (w[idx] > 0.0) { last = idx; c += w[idx]; if (c > target) return idx; }
        }
        return last;
    }

    /** Angles at grid cell `idx`. Delegates to cellToAngles so there is ONE convention:
     *  a duplicate here used floor((a+pi)/2pi), which maps +pi to -pi instead of keeping it,
     *  i.e. a different half-open interval from the rest of the engine. */
    public double[] anglesAt(int idx) {
        final double[] out = new double[dim];
        cellToAngles(idx, out);
        return out;
    }

    private static boolean isFinite(double v) { return !Double.isNaN(v) && !Double.isInfinite(v); }

    /** Stationary landing law as a grid density: ghat(m) * lambda / (lambda + a_m). */
    public double[] landingStationaryDensity(double[][] sigma, double lambda,
                                             double[] ghatRe, double[] ghatIm) {
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingRate(lambda);
        requireLandingGhat(lambda, ghatRe, ghatIm);
        final double[] re = new double[nTotal];
        final double[] im = new double[nTotal];
        final double[] m = new double[dim];
        final double scale = (double) nTotal / volume;
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            final double a = 0.5 * quadForm(sigma, m);
            final double w = lambda > 0.0 ? lambda / (lambda + a) : (a == 0.0 ? 1.0 : 0.0);
            // conj(ghat): see landingResetDensity.
            re[idx] = scale * w * ghatRe[idx];
            im[idx] = -scale * w * ghatIm[idx];
        }
        fftN(re, im, true);
        return re;
    }

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

    // ---- Landing jumps: a reset to one fixed mixture, rather than an additive shift. ----
    // These methods deliberately leave the historical shift implementation untouched.
    private final java.util.concurrent.atomic.AtomicLong landingNonpositiveCount =
            new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong landingNegativeCount =
            new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong landingDensityNonpositiveCount =
            new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.AtomicLong landingDensityNegativeCount =
            new java.util.concurrent.atomic.AtomicLong();
    private final java.util.concurrent.atomic.DoubleAccumulator landingMinimumValue =
            new java.util.concurrent.atomic.DoubleAccumulator(Math::min, Double.POSITIVE_INFINITY);
    private final java.util.concurrent.atomic.DoubleAdder landingNegativeMass =
            new java.util.concurrent.atomic.DoubleAdder();

    /** Number of nonpositive cells returned by landing leaf/branch operations. No cells are clipped. */
    public long getLandingNonpositiveCount() { return landingNonpositiveCount.get(); }

    /** Strictly negative subset of getLandingNonpositiveCount(). */
    public long getLandingNegativeCount() { return landingNegativeCount.get(); }

    /** Minimum returned leaf/branch cell value; positive infinity before any branch audit. */
    public double getLandingMinimumValue() { return landingMinimumValue.get(); }

    /** Unweighted sum of absolute negative cell values returned by leaf/branch operations. */
    public double getLandingNegativeMass() { return landingNegativeMass.sum(); }

    /** Separate audit for density grids used in ancestral reconstruction. */
    public long getLandingDensityNonpositiveCount() { return landingDensityNonpositiveCount.get(); }

    public long getLandingDensityNegativeCount() { return landingDensityNegativeCount.get(); }

    /** Landing never applies MIN_POS or any other clipping. */
    public long getLandingClampCount() { return 0L; }

    public void resetLandingNonpositiveCount() {
        landingNonpositiveCount.set(0L);
        landingNegativeCount.set(0L);
        landingDensityNonpositiveCount.set(0L);
        landingDensityNegativeCount.set(0L);
        landingMinimumValue.reset();
        landingNegativeMass.reset();
    }

    private double[] auditLandingGrid(double[] grid, boolean density) {
        long nonpositive = 0L, negative = 0L;
        double minimum = Double.POSITIVE_INFINITY, negativeMass = 0.0;
        for (double value : grid) {
            if (!Double.isFinite(value)) {
                throw new ArithmeticException("Non-finite landing Fourier grid value");
            }
            if (value <= 0.0) nonpositive++;
            if (value < 0.0) {
                negative++;
                negativeMass -= value;
            }
            minimum = Math.min(minimum, value);
        }
        (density ? landingDensityNonpositiveCount : landingNonpositiveCount).addAndGet(nonpositive);
        (density ? landingDensityNegativeCount : landingNegativeCount).addAndGet(negative);
        if (!density) {
            landingMinimumValue.accumulate(minimum);
            landingNegativeMass.add(negativeMass);
        }
        return grid;
    }

    private void requireLandingArray(double[] a, int length, String name) {
        if (a == null || a.length != length) {
            throw new IllegalArgumentException(name + " must have length " + length);
        }
    }

    private void requireLandingAngles(double[] angles, String name) {
        requireLandingArray(angles, dim, name);
        for (double angle : angles) {
            if (!Double.isFinite(angle)) throw new IllegalArgumentException(name + " must be finite");
        }
    }

    /** Validate symmetry as well as positive definiteness, without silently reading one triangle. */
    private void requireLandingCovariance(double[][] covariance, String name) {
        if (covariance == null || covariance.length != dim) {
            throw new IllegalArgumentException(name + " must be a " + dim + " by " + dim + " covariance");
        }
        for (int i = 0; i < dim; i++) {
            requireLandingArray(covariance[i], dim, name + " row");
            for (double entry : covariance[i]) {
                if (!Double.isFinite(entry)) throw new IllegalArgumentException(name + " must be finite");
            }
        }
        final double[][] lower = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
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

    private void requireLandingRate(double lambda) {
        if (!(lambda >= 0.0) || !Double.isFinite(lambda)) {
            throw new IllegalArgumentException("Landing jump rate must be finite and nonnegative");
        }
    }

    private void requireLandingTime(double t) {
        if (!(t >= 0.0) || !Double.isFinite(t)) {
            throw new IllegalArgumentException("Landing branch length must be finite and nonnegative");
        }
    }

    private void requireLandingGhat(double lambda, double[] ghatRe, double[] ghatIm) {
        if (lambda > 0.0) {
            requireLandingArray(ghatRe, nTotal, "ghatRe");
            requireLandingArray(ghatIm, nTotal, "ghatIm");
        }
    }

    /**
     * Cache the characteristic function of the fixed landing law:
     * ghat(m) = sum_b w_b exp(i m.mu_b - m' Omega_b m / 2).
     * Means are absolute landing angles. Weights must be nonnegative and sum to one;
     * covariances must be finite, symmetric and strictly positive definite.
     */
    public void computeGhat(double[] w, double[][] mu, double[][][] omega,
                            double[] outRe, double[] outIm) {
        if (w == null || w.length == 0 || mu == null || omega == null
                || mu.length != w.length || omega.length != w.length) {
            throw new IllegalArgumentException("Landing law requires matching nonempty weights, means and covariances");
        }
        requireLandingArray(outRe, nTotal, "outRe");
        requireLandingArray(outIm, nTotal, "outIm");
        if (outRe == outIm) throw new IllegalArgumentException("ghat real and imaginary outputs must be distinct");
        double weightSum = 0.0;
        final double[][] wrappedMu = new double[w.length][dim];
        for (int b = 0; b < w.length; b++) {
            if (!(w[b] >= 0.0) || !Double.isFinite(w[b])) {
                throw new IllegalArgumentException("Landing mixture weights must be finite and nonnegative");
            }
            weightSum += w[b];
            requireLandingAngles(mu[b], "Landing component mean");
            requireLandingCovariance(omega[b], "Landing component covariance");
            for (int ax = 0; ax < dim; ax++) wrappedMu[b][ax] = Math.IEEEremainder(mu[b][ax], TWO_PI);
        }
        if (!Double.isFinite(weightSum) || Math.abs(weightSum - 1.0) > 1e-10) {
            throw new IllegalArgumentException("Landing mixture weights must sum to one");
        }
        final double[] m = new double[dim];
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            double re = 0.0, im = 0.0;
            for (int b = 0; b < w.length; b++) {
                double phase = 0.0;
                for (int ax = 0; ax < dim; ax++) phase += m[ax] * wrappedMu[b][ax];
                final double magnitude = (w[b] / weightSum) * Math.exp(-0.5 * quadForm(omega[b], m));
                re += magnitude * Math.cos(phase);
                im += magnitude * Math.sin(phase);
            }
            outRe[idx] = re;
            outIm[idx] = im;
        }
        outRe[0] = 1.0;
        outIm[0] = 0.0;
    }

    /** Convenience allocation; result[0] is real and result[1] is imaginary. */
    public double[][] computeGhat(double[] w, double[][] mu, double[][][] omega) {
        final double[][] out = new double[2][nTotal];
        computeGhat(w, mu, omega, out[0], out[1]);
        return out;
    }

    /** Stable integral lambda * integral_0^t exp(-(lambda + a) u) du. */
    private double landingResetMass(double a, double t, double lambda) {
        return lambda == 0.0 ? 0.0 : (lambda / (lambda + a)) * -Math.expm1(-(lambda + a) * t);
    }

    /**
     * Apply the backward reset operator to FFT coefficients. The reset contribution
     * is a scalar in real space, hence is accumulated at mode zero before the inverse FFT.
     * The input coefficients use FFT normalization, so no extra nTotal factor belongs here.
     */
    private void applyLanding(double[] re, double[] im, double[][] sigma, double t,
                              double lambda, double[] ghatRe, double[] ghatIm) {
        final double[] m = new double[dim];
        double correction = 0.0, compensation = 0.0;
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            final double a = 0.5 * quadForm(sigma, m);
            if (lambda > 0.0) {
                final double term = landingResetMass(a, t, lambda)
                        * (ghatRe[idx] * re[idx] - ghatIm[idx] * im[idx]);
                final double adjusted = term - compensation;
                final double next = correction + adjusted;
                compensation = (next - correction) - adjusted;
                correction = next;
            }
            final double damping = Math.exp(-(lambda + a) * t);
            re[idx] *= damping;
            im[idx] *= damping;
        }
        re[0] += correction;
    }

    /** Backward propagation under diffusion plus resets to the fixed landing law. */
    public double[] propagateLanding(double[] partial, double[][] sigma, double t, double lambda,
                                     double[] ghatRe, double[] ghatIm) {
        requireLandingArray(partial, nTotal, "partial");
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingTime(t);
        requireLandingRate(lambda);
        requireLandingGhat(lambda, ghatRe, ghatIm);
        final double[] re = partial.clone();
        if (t == 0.0) return auditLandingGrid(re, false);
        final double[] im = new double[nTotal];
        fftN(re, im, false);
        applyLanding(re, im, sigma, t, lambda, ghatRe, ghatIm);
        fftN(re, im, true);
        return auditLandingGrid(re, false);
    }

    private void landingTipCoefficients(double[] theta, double[][] sigmaEps,
                                         double[] re, double[] im) {
        requireLandingAngles(theta, "Tip angles");
        requireLandingCovariance(sigmaEps, "Landing tip-error covariance");
        final double[] m = new double[dim];
        final double scale = (double) nTotal / volume;
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            double phase = 0.0;
            for (int ax = 0; ax < dim; ax++) phase += m[ax] * Math.IEEEremainder(theta[ax], TWO_PI);
            final double magnitude = scale * Math.exp(-0.5 * quadForm(sigmaEps, m));
            re[idx] = magnitude * Math.cos(phase);
            im[idx] = -magnitude * Math.sin(phase);
        }
    }

    /** Pure Fourier tip with strictly positive-definite observation-error covariance. */
    public double[] leafKernelLanding(double[] theta, double[][] sigmaEps) {
        final double[] re = new double[nTotal];
        final double[] im = new double[nTotal];
        landingTipCoefficients(theta, sigmaEps, re, im);
        fftN(re, im, true);
        return auditLandingGrid(re, false);
    }

    /**
     * Pure Fourier tip and terminal branch in one inverse FFT. Retaining the complex
     * tip coefficients until propagation avoids an extra transform and premature
     * projection to real values at the Nyquist modes.
     */
    public double[] leafKernelLanding(double[] theta, double[][] sigmaEps, double[][] sigma,
                                      double t, double lambda, double[] ghatRe, double[] ghatIm) {
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingTime(t);
        requireLandingRate(lambda);
        requireLandingGhat(lambda, ghatRe, ghatIm);
        final double[] re = new double[nTotal];
        final double[] im = new double[nTotal];
        landingTipCoefficients(theta, sigmaEps, re, im);
        applyLanding(re, im, sigma, t, lambda, ghatRe, ghatIm);
        fftN(re, im, true);
        return auditLandingGrid(re, false);
    }

    /**
     * Stationary landing-root integral: Re sum_m ghat(m) lambda/(lambda+a_m) c(m).
     * FFT coefficients are nTotal times Fourier-series coefficients. At zero rate
     * the stationary law is uniform and the historical grid mean is returned exactly.
     */
    public double rootIntegralLanding(double[] partial, double[][] sigma, double lambda,
                                      double[] ghatRe, double[] ghatIm) {
        requireLandingArray(partial, nTotal, "partial");
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingRate(lambda);
        requireLandingGhat(lambda, ghatRe, ghatIm);
        if (lambda == 0.0) return rootIntegral(partial);
        final double[] re = partial.clone();
        final double[] im = new double[nTotal];
        fftN(re, im, false);
        final double[] m = new double[dim];
        double sum = 0.0, compensation = 0.0;
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            final double weight = lambda / (lambda + 0.5 * quadForm(sigma, m));
            final double term = weight * (ghatRe[idx] * re[idx] - ghatIm[idx] * im[idx]);
            final double adjusted = term - compensation;
            final double next = sum + adjusted;
            compensation = (next - sum) - adjusted;
            sum = next;
        }
        return sum / nTotal;
    }

    /** Stationary density for root sampling, with the uniform limit at lambda == 0. */
    public double[] rootDensityGridLanding(double[][] sigma, double lambda,
                                           double[] ghatRe, double[] ghatIm) {
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingRate(lambda);
        requireLandingGhat(lambda, ghatRe, ghatIm);
        final double[] re = new double[nTotal];
        if (lambda == 0.0) {
            java.util.Arrays.fill(re, 1.0 / volume);
            return auditLandingGrid(re, true);
        }
        final double[] im = new double[nTotal];
        final double[] m = new double[dim];
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            final double weight = ((double) nTotal / volume)
                    * (lambda / (lambda + 0.5 * quadForm(sigma, m)));
            // A density uses the conjugate of the characteristic function.
            re[idx] = weight * ghatRe[idx];
            im[idx] = -weight * ghatIm[idx];
        }
        fftN(re, im, true);
        return auditLandingGrid(re, true);
    }

    /**
     * Forward transition density p_t(y|x) at the grid angles y for a fixed parent x.
     * Resets depend on the absolute child angle, so a shift/displacement kernel cannot
     * be used here. The no-reset Gaussian is summed over winding images; the broad
     * reset contribution is spectral. Values are continuous densities without grid
     * renormalization. A caller doing grid quadrature must ensure the Gaussian is resolved.
     * At t == 0 the transition is a Dirac measure and has no density.
     */
    public double[] transitionDensityGridLanding(double[] parentAngle, double[][] sigma,
                                                 double t, double lambda,
                                                 double[] ghatRe, double[] ghatIm) {
        requireLandingAngles(parentAngle, "Parent angles");
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingTime(t);
        requireLandingRate(lambda);
        requireLandingGhat(lambda, ghatRe, ghatIm);
        if (t == 0.0) throw new IllegalArgumentException("A landing transition density requires positive branch length");
        final double[] re = new double[nTotal];
        if (lambda > 0.0) {
            final double[] im = new double[nTotal];
            final double[] m = new double[dim];
            for (int idx = 0; idx < nTotal; idx++) {
                modeOf(idx, m);
                final double weight = ((double) nTotal / volume)
                        * landingResetMass(0.5 * quadForm(sigma, m), t, lambda);
                re[idx] = weight * ghatRe[idx];
                im[idx] = -weight * ghatIm[idx];
            }
            fftN(re, im, true);
        }
        final double[] wrappedParent = new double[dim];
        for (int ax = 0; ax < dim; ax++) wrappedParent[ax] = Math.IEEEremainder(parentAngle[ax], TWO_PI);
        addLatticeAtom(re, wrappedParent, sigma, t, -lambda * t, null);
        return auditLandingGrid(re, true);
    }
}
