/*
 * TorusFourierKernel.java
 *
 * @author Samir Bhatt
 */

package dr.evomodel.treedatalikelihood.continuous;

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
        // Bit-reversal transpositions and twiddle factors of FastFourierTransform.fft, tabulated once.
        final java.util.List<int[]> swaps = new java.util.ArrayList<int[]>();
        final int n = nGrid << 1;
        int j = 1;
        for (int i = 1; i < n; i += 2) {
            if (j > i) swaps.add(new int[]{(i - 1) >> 1, (j - 1) >> 1});
            int m = nGrid;
            while (m >= 2 && j > m) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }
        this.swapLow = new int[swaps.size()];
        this.swapHigh = new int[swaps.size()];
        for (int i = 0; i < swaps.size(); i++) {
            swapLow[i] = swaps.get(i)[0];
            swapHigh[i] = swaps.get(i)[1];
        }
        this.twiddleRe = new double[2][nGrid];
        this.twiddleIm = new double[2][nGrid];
        for (int direction = 0; direction < 2; direction++) {
            fillTwiddles(direction == 1, twiddleRe[direction], twiddleIm[direction]);
        }
        // Mode vectors as modeOf computes them, grid angles as angleOf computes them, and the index
        // of -m (see mirrorIndex).
        this.modeValue = new double[dim][nTotal];
        this.angleValue = new double[dim][nTotal];
        this.mirrorIndex = new int[nTotal];
        for (int idx = 0; idx < nTotal; idx++) {
            int mirror = 0;
            for (int ax = 0; ax < dim; ax++) {
                final int i = (idx / stride[ax]) % nGrid;
                modeValue[ax][idx] = freq[i];
                angleValue[ax][idx] = angleOf(idx, ax);
                if (mirror >= 0) mirror = i == nGrid / 2 ? -1 : mirror + ((nGrid - i) % nGrid) * stride[ax];
            }
            mirrorIndex[idx] = mirror;
        }
    }

    // ---- per-mode tables ----
    // modeValue[ax][idx] is the mode vector modeOf(idx) fills in, read from a table instead of an
    // integer division and remainder per axis. mirrorIndex[idx] is the index of the mode -m when no
    // axis sits at the Nyquist index (whose negation is not on the grid), and -1 otherwise.
    // quadForm(S, -m) equals quadForm(S, m) bit for bit -- every product s_ij m_i m_j has both signs
    // flipped -- so any per-mode quantity built from it is the same at m and -m, and the landing
    // branch operator evaluates its exponentials once per mirror pair. LandingTables verifies the
    // equality on its own values before relying on it.
    private final double[][] modeValue;
    private final double[][] angleValue;        // [ax][idx] = angleOf(idx, ax)
    private final int[] mirrorIndex;

    // ---- planned FFT ----
    // fftN applies FastFourierTransform.fft (Numerical Recipes four1) along every axis. Its plan
    // reproduces that routine exactly: the same bit-reversal transpositions, and the twiddle factors
    // four1 generates by its own trigonometric recurrence, run once per direction and tabulated
    // (stage with half-span h at offsets h-1 .. 2h-2). Every butterfly is the same arithmetic, in
    // the same order, on the same operands, so the output is identical to the last bit. What
    // changes is the traversal: contiguous lines are transformed in place rather than copied
    // through an interleaved buffer, and lines along a strided axis are gathered a few at a time
    // into contiguous scratch, transformed there, and scattered back.
    private final int[] swapLow, swapHigh;      // four1's bit-reversal transpositions, complex indices
    private final double[][] twiddleRe;         // [0 forward, 1 inverse][h - 1 + q]
    private final double[][] twiddleIm;
    private static final int FFT_GATHER = 8;    // strided lines gathered per block (fastest at nGrid 256)

    /** four1's twiddle recurrence, tabulated: exactly the (wr, wi) it uses at every butterfly. */
    private void fillTwiddles(boolean inverse, double[] outRe, double[] outIm) {
        final double radians = inverse ? 2.0 * Math.PI : -2.0 * Math.PI;
        final int n = nGrid << 1;
        int mmax = 2;
        while (n > mmax) {
            final int istep = mmax << 1;
            final double theta = radians / mmax;
            double wtemp = Math.sin(0.5 * theta);
            final double wpr = -2.0 * wtemp * wtemp;
            final double wpi = Math.sin(theta);
            double wr = 1.0;
            double wi = 0.0;
            final int half = mmax >> 1;
            for (int m = 1; m < mmax; m += 2) {
                outRe[half - 1 + ((m - 1) >> 1)] = wr;
                outIm[half - 1 + ((m - 1) >> 1)] = wi;
                wtemp = wr;
                wr += wr * wpr - wi * wpi;
                wi += wi * wpr + wtemp * wpi;
            }
            mmax = istep;
        }
    }

    /**
     * four1 on the contiguous line re/im[base .. base + nGrid). Stages are taken two at a time:
     * the stages of half-span h and 2h map each set {q, q+h, q+2h, q+3h} (q < h, within a group of
     * 4h) onto itself, so both are applied to a set before moving on, with each element receiving
     * the same butterflies, in the same order, as in the stage-by-stage loop. A last unpaired stage
     * runs on its own.
     */
    private void fftLine(double[] re, double[] im, int base, double[] wRe, double[] wIm) {
        for (int k = 0; k < swapLow.length; k++) {
            final int a = base + swapLow[k], b = base + swapHigh[k];
            final double tr = re[a];
            re[a] = re[b];
            re[b] = tr;
            final double ti = im[a];
            im[a] = im[b];
            im[b] = ti;
        }
        final int end = base + nGrid;
        int half = 1;
        while (half < nGrid) {
            if ((half << 1) < nGrid) {
                final int h = half, h2 = half << 1, group = half << 2;
                for (int q = 0; q < h; q++) {
                    final double w1r = wRe[h - 1 + q], w1i = wIm[h - 1 + q];        // stage h, both pairs
                    final double w2r = wRe[h2 - 1 + q], w2i = wIm[h2 - 1 + q];      // stage 2h, pair (q, q+2h)
                    final double w3r = wRe[h2 - 1 + q + h], w3i = wIm[h2 - 1 + q + h];  // stage 2h, (q+h, q+3h)
                    for (int x0 = base + q; x0 < end; x0 += group) {
                        final int x1 = x0 + h, x2 = x0 + h2, x3 = x2 + h;
                        double r0 = re[x0], i0 = im[x0], r1 = re[x1], i1 = im[x1];
                        double r2 = re[x2], i2 = im[x2], r3 = re[x3], i3 = im[x3];
                        double tempr = w1r * r1 - w1i * i1, tempi = w1r * i1 + w1i * r1;
                        r1 = r0 - tempr;
                        i1 = i0 - tempi;
                        r0 += tempr;
                        i0 += tempi;
                        tempr = w1r * r3 - w1i * i3;
                        tempi = w1r * i3 + w1i * r3;
                        r3 = r2 - tempr;
                        i3 = i2 - tempi;
                        r2 += tempr;
                        i2 += tempi;
                        tempr = w2r * r2 - w2i * i2;
                        tempi = w2r * i2 + w2i * r2;
                        r2 = r0 - tempr;
                        i2 = i0 - tempi;
                        r0 += tempr;
                        i0 += tempi;
                        tempr = w3r * r3 - w3i * i3;
                        tempi = w3r * i3 + w3i * r3;
                        r3 = r1 - tempr;
                        i3 = i1 - tempi;
                        r1 += tempr;
                        i1 += tempi;
                        re[x0] = r0;
                        im[x0] = i0;
                        re[x1] = r1;
                        im[x1] = i1;
                        re[x2] = r2;
                        im[x2] = i2;
                        re[x3] = r3;
                        im[x3] = i3;
                    }
                }
                half <<= 2;
            } else {
                final int step = half << 1;
                for (int q = 0; q < half; q++) {
                    final double wr = wRe[half - 1 + q], wi = wIm[half - 1 + q];
                    for (int p = base + q; p < end; p += step) {
                        final int p2 = p + half;
                        final double tempr = wr * re[p2] - wi * im[p2];
                        final double tempi = wr * im[p2] + wi * re[p2];
                        re[p2] = re[p] - tempr;
                        im[p2] = im[p] - tempi;
                        re[p] += tempr;
                        im[p] += tempi;
                    }
                }
                half <<= 1;
            }
        }
    }

    /** four1 on the `width` adjacent strided lines base + c + j*s (c < width, j < nGrid): gather
     *  them into contiguous scratch, transform each as a contiguous line, scatter them back. */
    private void fftStrided(double[] re, double[] im, int base, int s, int width,
                            double[] scratchRe, double[] scratchIm, double[] wRe, double[] wIm) {
        for (int j = 0; j < nGrid; j++) {
            final int src = base + j * s;
            for (int c = 0; c < width; c++) {
                scratchRe[c * nGrid + j] = re[src + c];
                scratchIm[c * nGrid + j] = im[src + c];
            }
        }
        for (int c = 0; c < width; c++) fftLine(scratchRe, scratchIm, c * nGrid, wRe, wIm);
        for (int j = 0; j < nGrid; j++) {
            final int dst = base + j * s;
            for (int c = 0; c < width; c++) {
                re[dst + c] = scratchRe[c * nGrid + j];
                im[dst + c] = scratchIm[c * nGrid + j];
            }
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
            for (int ax = 0; ax < dim; ax++) m[ax] = modeValue[ax][idx];
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
        final double[] bm = transferMultiplier(sigma, t, 0.0, null);
        if (!brownianSpectralHalfVanishes(y, t, bm, sigmaEps)) {
            return leafKernelSplit(y, sigma, t, 0.0, bm, null, sigmaEps);
        }
        // Without jumps the spectral half of leafKernelSplit is identically zero: its multiplier is
        // the Brownian one minus itself (weight e^{-0 t} = 1), so the inverse FFT returns signed
        // zeros, which the clamp keeps and the atom then overwrites exactly as it would a plain zero.
        // Start from zeros and add the atom directly; the result is the same to the last bit.
        final double lamT = 0.0 * t;
        final double[] re = new double[nTotal];
        addLatticeAtom(re, y, sigma, t, -lamT, sigmaEps);
        for (int k = 0; k < nTotal; k++) if (re[k] < MIN_POS) re[k] = MIN_POS;
        return re;
    }

    /**
     * Whether leafKernelSplit's spectral half is exactly signed zeros for a Brownian tip: true when
     * t and y are finite, every Brownian multiplier is finite (x - 1.0*x = +0.0) and so is every
     * tip-error damping factor (0 * finite = +-0). A non-positive-definite Sigma or Sigma_eps can
     * overflow them, and the spectral half is then NaN, which the shortcut must not hide.
     */
    private boolean brownianSpectralHalfVanishes(double[] y, double t, double[] bm, double[][] sigmaEps) {
        if (!Double.isFinite(t)) return false;
        for (double v : y) if (!Double.isFinite(v)) return false;
        for (double v : bm) if (!Double.isFinite(v)) return false;
        if (sigmaEps != null) {
            final double[] m = new double[dim];
            for (int idx = 0; idx < nTotal; idx++) {
                for (int ax = 0; ax < dim; ax++) m[ax] = modeValue[ax][idx];
                if (!Double.isFinite(Math.exp(-0.5 * quadForm(sigmaEps, m)))) return false;
            }
        }
        return true;
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
        // SCREENING. Every image's term is exp(-(q_c - q_min)/2), and an image with q_c - q_min > 1500
        // contributes exp(< -750) = 0.0 exactly, so adding it changes nothing. Such images are found
        // without forming their quadratic forms, from the expansion
        //     q(b + s) = q(b) + s'(A + A')b + s'As,     b = grid angle - y, s = 2 pi k, A = inv,
        // whose first term is the same for every image: an image is skipped when its screened value
        // exceeds the smallest by more than SCREEN_GAP plus a bound on the rounding of both the screen
        // and the direct form (a relative 1e-12 of sum|A_ij| * reach^2, thousands of times the
        // worst-case rounding). Every other image is evaluated exactly as before, in the same order,
        // so the result is unchanged to the last bit. The screen is used only where none of its terms
        // can overflow: each is bounded by about 3 * sum|A_ij| * reach^2, so above 1e300 (covariances
        // near 1e-305 rad^2), or for any non-finite input, gap is infinite and every image is evaluated.
        final double[][] shift = new double[combos][dim];
        final double[] shiftForm = new double[combos];
        double maxAbsY = 0.0;
        for (int ax = 0; ax < dim; ax++) maxAbsY = Math.max(maxAbsY, Math.abs(y[ax]));
        double invSum = 0.0;
        final double[][] sym = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                invSum += Math.abs(inv[i][j]);
                sym[i][j] = inv[i][j] + inv[j][i];
            }
        }
        for (int c = 0; c < combos; c++) {
            int rem = c;
            for (int ax = 0; ax < dim; ax++) {
                final int k = rem % span - kmax;
                rem /= span;
                shift[c][ax] = TWO_PI * k;
            }
            double form = 0.0;
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) form += inv[i][j] * shift[c][i] * shift[c][j];
            }
            shiftForm[c] = form;
        }
        final double reach = TWO_PI + maxAbsY + TWO_PI * kmax;
        final double magnitude = invSum * reach * reach;
        final double gap = magnitude < 1.0e300 ? SCREEN_GAP + 1.0e-12 * magnitude : Double.POSITIVE_INFINITY;
        final double[] b = new double[dim];
        final double[] ab = new double[dim];
        final double[] d = new double[dim];
        final double[] screen = new double[combos];
        final double[] q = new double[combos];
        for (int idx = 0; idx < nTotal; idx++) {
            for (int ax = 0; ax < dim; ax++) b[ax] = angleValue[ax][idx] - y[ax];
            for (int i = 0; i < dim; i++) {
                double v = 0.0;
                for (int j = 0; j < dim; j++) v += sym[i][j] * b[j];
                ab[i] = v;
            }
            double smin = Double.POSITIVE_INFINITY;
            for (int c = 0; c < combos; c++) {
                double v = shiftForm[c];
                for (int ax = 0; ax < dim; ax++) v += shift[c][ax] * ab[ax];
                screen[c] = v;
                if (v < smin) smin = v;
            }
            double qmin = Double.POSITIVE_INFINITY;
            for (int c = 0; c < combos; c++) {
                if (screen[c] - smin > gap) continue;               // exactly zero term, see above
                for (int ax = 0; ax < dim; ax++) d[ax] = b[ax] + shift[c][ax];
                double form = 0.0;
                for (int i = 0; i < dim; i++) {
                    for (int j = 0; j < dim; j++) form += inv[i][j] * d[i] * d[j];
                }
                q[c] = form;
                if (form < qmin) qmin = form;
            }
            double acc = 0.0;
            for (int c = 0; c < combos; c++) {
                if (screen[c] - smin > gap) continue;
                final double diff = q[c] - qmin;
                acc += diff == 0.0 ? 1.0 : Math.exp(-0.5 * diff);  // exp(-0.0) is exactly 1.0
            }
            out[idx] += Math.exp(logNorm + Math.log(acc) - 0.5 * qmin);
        }
    }

    /** An image whose exponent trails the leading one by more than this (in q, so 760 in the exponent)
     *  underflows to exactly 0.0 (exp(x) is 0.0 for x <= -745.2). */
    private static final double SCREEN_GAP = 1520.0;

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
            for (int ax = 0; ax < dim; ax++) m[ax] = modeValue[ax][idx];
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

    /**
     * The jump mixture's characteristic function phi(m) = sum_k w_k exp(-1/2 m'S_k m) exp(i m.d_k),
     * accumulated exactly as computePsiComplex accumulates it. The mixture is fixed for a run, so a
     * likelihood computes this once instead of once per residue per evaluation.
     */
    public void computeJumpCharacteristic(double[] w, double[][] d, double[][][] s,
                                          double[] phiRe, double[] phiIm) {
        final double[] m = new double[dim];
        for (int idx = 0; idx < nTotal; idx++) {
            for (int ax = 0; ax < dim; ax++) m[ax] = modeValue[ax][idx];
            double re = 0.0, im = 0.0;
            for (int k = 0; k < w.length; k++) {
                final double g = w[k] * Math.exp(-0.5 * quadForm(s[k], m));
                double md = 0.0;
                for (int ax = 0; ax < dim; ax++) md += m[ax] * d[k][ax];
                re += g * Math.cos(md);
                im += g * Math.sin(md);
            }
            phiRe[idx] = re;
            phiIm[idx] = im;
        }
    }

    /**
     * computePsiComplex(sigma, jumpRate, w, d, s, ...) for jumpRate > 0 and a complete mixture, from
     * its characteristic function (computeJumpCharacteristic). Same arithmetic, same bits.
     */
    public void computePsiFromCharacteristic(double[][] sigma, double jumpRate, double[] phiRe, double[] phiIm,
                                             double[] psiRe, double[] psiIm) {
        final double[] m = new double[dim];
        for (int idx = 0; idx < nTotal; idx++) {
            for (int ax = 0; ax < dim; ax++) m[ax] = modeValue[ax][idx];
            double pr = -0.5 * quadForm(sigma, m);
            double pi = 0.0;
            pr += jumpRate * (phiRe[idx] - 1.0);
            pi += jumpRate * phiIm[idx];
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

    /** Sample a child cell ~ K_t(child - parent) * childPartial(child); u in [0,1). Returns -1
     *  when the joint weight has no positive mass: substituting the unconditioned child partial
     *  would draw from the wrong law (see testSmallSigmaInternalPropagation). */
    public int sampleConditionalCell(double[] kernelGrid, int parentIdx, double[] childPartial, double u) {
        return sampleChildCell(kernelGrid, null, parentIdx, childPartial, u);
    }

    /** Uniform/Haar-prior root integral of a grid partial = grid mean. */

    // ===================== ancestral sampling: split downward pass =====================
    // Joint sampling of internal states from the cached upward partials, for BOTH jump laws.
    // A child given its sampled parent CELL is drawn from p_t(child | parent) * childPartial.
    // The transition is indexed by integer cell DISPLACEMENT (child - parent), the convention
    // of the dense references, plus -- for landing only -- an absolute, parent-independent term:
    //
    //   shift  : p_t = e^{-lambda t} WN(child - parent; t Sigma) + D_t(child - parent)
    //   landing: p_t = e^{-lambda t} WN(child - parent; t Sigma) + (1 - e^{-lambda t}) r_t(child)
    //
    // WHY A SPLIT. The zero-jump wrapped normal is narrower than a grid cell on short branches,
    // where its TRUNCATED Fourier series rings: at nGrid 128 a pointwise FFT kernel put 1.5-3%
    // of a child's mass two or more cells from the parent for t*Sigma in ~[1e-4, 1e-3] (exact:
    // 1e-21..1e-7). The upward pass tolerates this because it INTEGRATES the kernel against a
    // smooth partial; a sampler reads it POINTWISE. So, as for tips (leafKernelSplit), the
    // zero-jump half is the exact wrapped normal in real space (branchDiffusionKernel, which also
    // renormalises it to unit mass on the grid), and only the >=1-jump half -- which inherits the
    // broad jump or basin covariances and is resolved by the grid -- stays on the FFT. That
    // assumption is the one the tip split makes; its failure shows up as clipped negative mass
    // (getAncestralClippedMass).
    //
    // ORIENTATION. ghat(m) and e^{t psi(m)} are CHARACTERISTIC FUNCTIONS, E[exp(+i m.Z)]. As
    // multipliers of FORWARD coefficients (upward pass) they are used as-is; as the
    // coefficients of a DENSITY in the inverse-FFT basis exp(+i m.z) they are CONJUGATED.
    // Omitting the conjugate reflects every directed component through the origin.
    //
    // DYNAMIC RANGE. Partials carry ~16 decades. A draw whose conditional mass sits on cells
    // whose partial is below that floor is not determined by the available numerics (the J2
    // regime, NUMERICS.md); such draws are counted (getAncestralUnresolvedDraws), never
    // silently trusted.

    private final java.util.concurrent.atomic.DoubleAdder ancestralClippedMass =
            new java.util.concurrent.atomic.DoubleAdder();
    private final java.util.concurrent.atomic.AtomicLong ancestralUnresolvedDraws =
            new java.util.concurrent.atomic.AtomicLong();

    /** Total negative mass floored to zero while forming ancestral conditionals. */
    public double getAncestralClippedMass() { return ancestralClippedMass.sum(); }

    /** Ancestral draws whose conditional relied on partial values below the FFT's resolution
     *  floor (more than 1e-6 of the mass on cells with |partial| < 1e-13 max|partial|). */
    public long getAncestralUnresolvedDraws() { return ancestralUnresolvedDraws.get(); }

    public void resetAncestralDiagnostics() {
        ancestralClippedMass.reset();
        ancestralUnresolvedDraws.set(0L);
    }

    /**
     * Zero-jump branch kernel WN(z; t Sigma) on the grid, centred at the ORIGIN (index =
     * displacement): the EXACT wrapped normal at the grid points (winding sum, never a truncated
     * Fourier series), RENORMALISED to unit mass on the grid.
     *
     * Why renormalise: when t Sigma is narrower than a cell, point values do not conserve
     * probability -- their Riemann sum is 1 + 2 sum_n exp(-v (n nGrid)^2 / 2) per axis, 1.17 at
     * t Sigma = 3e-4, nGrid 128 -- which would over-weight "no jump" against the jump half by up
     * to 17%. The FFT upward pass conserves mass exactly, so the sampler must too. For resolved
     * kernels the factor is 1 to machine precision; for sub-cell kernels the child falls in the
     * parent's cell, which is the correct limit at grid resolution. Validated against the exact
     * continuous posterior quantised to cells (ancestral_continuous_truth.py): P(child within two
     * cells) to 1e-4, where unnormalised point values miss by 6e-3 and a truncated series by 2e-2.
     */
    public double[] branchDiffusionKernel(double[][] sigma, double t) {
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingTime(t);
        final double[] out = new double[nTotal];
        addLatticeAtom(out, new double[dim], sigma, t, 0.0, null);
        double mass = 0.0;
        for (double v : out) mass += v;
        mass *= volume / nTotal;                           // cell volume (2 pi / nGrid)^dim
        if (!(mass > 0.0) || Double.isInfinite(mass)) {
            throw new ArithmeticException("zero-jump branch kernel has no representable mass at nGrid="
                    + nGrid + " for t=" + t);
        }
        for (int k = 0; k < nTotal; k++) out[k] /= mass;
        return out;
    }

    /**
     * SHIFT >=1-jump kernel D_t(z), centred at the origin: the inverse FFT of
     * conj(e^{t psi(m)} - e^{-lambda t} e^{-t m'Sigma m/2}), floored at zero with the floored
     * mass recorded. Returns null for pure Brownian motion (psiRe == null or jumpRate <= 0).
     */
    public double[] shiftJumpKernel(double[][] sigma, double t, double jumpRate,
                                    double[] psiRe, double[] psiIm) {
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingTime(t);
        if (psiRe == null || !(jumpRate > 0.0)) return null;
        final double[] mRe = new double[nTotal], mIm = new double[nTotal];
        multiplierFromPsi(t, psiRe, psiIm, mRe, mIm);
        final double[] bm = transferMultiplier(sigma, t, 0.0, null);
        final double weight = Math.exp(-jumpRate * t);
        final double scale = (double) nTotal / volume;
        final double[] re = new double[nTotal], im = new double[nTotal];
        for (int k = 0; k < nTotal; k++) {
            re[k] = scale * (mRe[k] - weight * bm[k]);   // conj: real part unchanged
            im[k] = -scale * mIm[k];                      //       imaginary part negated
        }
        fftN(re, im, true);
        double negative = 0.0;
        for (int k = 0; k < nTotal; k++) if (re[k] < 0.0) { negative -= re[k]; re[k] = 0.0; }
        if (negative > 0.0) ancestralClippedMass.add(negative);
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
     * product carries no positive mass. Used for the ROOT, whose density is absolute.
     */
    public int sampleFromProduct(double[] density, double[] partial, double uniform) {
        requireLandingArray(density, nTotal, "density");
        return sampleChildCell(null, density, 0, partial, uniform);
    }

    /**
     * Sample a child cell given its parent CELL from
     *     [ displacementKernel[child - parent] + absoluteDensity[child] ] * childPartial[child],
     * either term may be null. Callers pre-weight the terms (shift: e^{-lambda t} WN + D_t as
     * one displacement kernel; landing: e^{-lambda t} WN as displacement, (1 - e^{-lambda t})
     * r_t as absolute). Negative products are floored with their mass recorded; a draw whose
     * mass relies on unresolved partial cells is counted. Returns -1 when no positive mass
     * remains.
     */
    public int sampleChildCell(double[] displacementKernel, double[] absoluteDensity,
                               int parentIdx, double[] childPartial, double u) {
        if (displacementKernel != null) requireLandingArray(displacementKernel, nTotal, "displacementKernel");
        if (absoluteDensity != null) requireLandingArray(absoluteDensity, nTotal, "absoluteDensity");
        requireLandingArray(childPartial, nTotal, "childPartial");
        final int[] kp = new int[dim];
        for (int ax = 0; ax < dim; ax++) kp[ax] = (parentIdx / stride[ax]) % nGrid;
        double maxAbs = 0.0;
        for (int k = 0; k < nTotal; k++) maxAbs = Math.max(maxAbs, Math.abs(childPartial[k]));
        final double floor = 1.0e-13 * maxAbs;

        final double[] w = new double[nTotal];
        double total = 0.0, negativeMass = 0.0, unresolved = 0.0;
        for (int idx = 0; idx < nTotal; idx++) {
            double density = absoluteDensity == null ? 0.0 : absoluteDensity[idx];
            if (displacementKernel != null) {
                int disp = 0;
                for (int ax = 0; ax < dim; ax++) {
                    int d = (idx / stride[ax]) % nGrid - kp[ax];
                    if (d < 0) d += nGrid;
                    disp += d * stride[ax];
                }
                density += displacementKernel[disp];
            }
            final double v = density * childPartial[idx];
            if (v > 0.0) {
                w[idx] = v;
                total += v;
                if (Math.abs(childPartial[idx]) < floor) unresolved += v;
            } else if (v < 0.0) {
                negativeMass -= v;
            }
        }
        if (negativeMass > 0.0) ancestralClippedMass.add(negativeMass);
        if (!(total > 0.0) || Double.isNaN(total) || Double.isInfinite(total)) return -1;
        if (unresolved > 1.0e-6 * total) ancestralUnresolvedDraws.incrementAndGet();

        final double target = u * total;
        double c = 0.0;
        int last = -1;
        for (int idx = 0; idx < nTotal; idx++) {
            if (w[idx] > 0.0) { last = idx; c += w[idx]; if (c > target) return idx; }
        }
        return last;                                       // u at 1 - ulp: last positive cell
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

    /* N-D FFT: the 1-D four1 of FastFourierTransform.fft applied along each axis, with the
     * inverse divided by nTotal (four1 itself is unnormalised). Planned and blocked, see
     * fillTwiddles; the result is bit-identical to calling FastFourierTransform.fft line by line. */
    private void fftN(double[] re, double[] im, boolean inverse) {
        final double[] wRe = twiddleRe[inverse ? 1 : 0];
        final double[] wIm = twiddleIm[inverse ? 1 : 0];
        double[] scratchRe = null, scratchIm = null;
        for (int ax = 0; ax < dim; ax++) {
            final int s = stride[ax];
            final int outer = nTotal / (nGrid * s);
            if (s > 1 && scratchRe == null) {
                scratchRe = new double[FFT_GATHER * nGrid];
                scratchIm = new double[FFT_GATHER * nGrid];
            }
            for (int o = 0; o < outer; o++) {
                final int base = o * nGrid * s;
                if (s == 1) {
                    fftLine(re, im, base, wRe, wIm);
                } else {
                    for (int c = 0; c < s; c += FFT_GATHER) {
                        fftStrided(re, im, base + c, s, Math.min(FFT_GATHER, s - c), scratchRe, scratchIm, wRe, wIm);
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
     * The branch-independent part of the landing operator for one (Sigma, lambda):
     * rate[idx] = lambda + a_m and, for lambda > 0, weight[idx] = lambda / (lambda + a_m), with
     * a_m = 0.5 m' Sigma m. These are exactly the values applyLanding and rootIntegralLanding
     * used to recompute per mode on every branch. A residue keeps one while its Sigma and lambda
     * are unchanged, which the key (their raw bits) records.
     */
    static final class LandingTables {
        private final long[] key;
        final double lambda;
        final double[] rate;
        final double[] weight;
        /** rate and weight agree bit for bit at every mirror pair, so one exponential serves both. */
        final boolean mirrorExact;

        private LandingTables(long[] key, double lambda, double[] rate, double[] weight, boolean mirrorExact) {
            this.key = key;
            this.lambda = lambda;
            this.rate = rate;
            this.weight = weight;
            this.mirrorExact = mirrorExact;
        }
    }

    private long[] landingKey(double[][] sigma, double lambda) {
        final long[] key = new long[dim * dim + 1];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) key[i * dim + j] = Double.doubleToRawLongBits(sigma[i][j]);
        }
        key[dim * dim] = Double.doubleToRawLongBits(lambda);
        return key;
    }

    /** Tables for (sigma, lambda): `previous` itself when it was built for the same bits. */
    LandingTables landingTables(LandingTables previous, double[][] sigma, double lambda) {
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingRate(lambda);
        final long[] key = landingKey(sigma, lambda);
        if (previous != null && java.util.Arrays.equals(previous.key, key)) return previous;
        final double[] rate = new double[nTotal];
        final double[] weight = lambda > 0.0 ? new double[nTotal] : null;
        final double[] m = new double[dim];
        for (int idx = 0; idx < nTotal; idx++) {
            for (int ax = 0; ax < dim; ax++) m[ax] = modeValue[ax][idx];
            final double a = 0.5 * quadForm(sigma, m);
            rate[idx] = lambda + a;
            if (weight != null) weight[idx] = lambda / (lambda + a);
        }
        boolean mirrorExact = true;
        for (int idx = 0; idx < nTotal && mirrorExact; idx++) {
            final int mi = mirrorIndex[idx];
            if (mi < 0) continue;
            mirrorExact = Double.doubleToRawLongBits(rate[idx]) == Double.doubleToRawLongBits(rate[mi])
                    && (weight == null || Double.doubleToRawLongBits(weight[idx]) == Double.doubleToRawLongBits(weight[mi]));
        }
        return new LandingTables(key, lambda, rate, weight, mirrorExact);
    }

    private void requireTables(LandingTables tables, double[][] sigma, double lambda) {
        if (!java.util.Arrays.equals(tables.key, landingKey(sigma, lambda))) {
            throw new IllegalStateException("landing tables were built for a different Sigma or lambda");
        }
    }

    /**
     * Apply the backward reset operator to FFT coefficients. The reset contribution
     * is a scalar in real space, hence is accumulated at mode zero before the inverse FFT.
     * The input coefficients use FFT normalization, so no extra nTotal factor belongs here.
     *
     * Per mode this is the arithmetic the operator always used: x = -(lambda + a_m) t, damping
     * exp(x), reset mass (lambda / (lambda + a_m)) * -expm1(x), compensated sum in index order.
     * The exponentials are evaluated first, once per mirror pair, into `damping` and `reset`.
     */
    private void applyLanding(double[] re, double[] im, LandingTables tables, double t,
                              double[] ghatRe, double[] ghatIm, double[] damping, double[] reset) {
        final double[] rate = tables.rate, weight = tables.weight;
        final boolean jumps = tables.lambda > 0.0;
        for (int idx = 0; idx < nTotal; idx++) {
            final int mi = tables.mirrorExact ? mirrorIndex[idx] : -1;
            if (mi >= 0 && mi < idx) continue;            // already filled from its mirror
            final double x = -rate[idx] * t;
            final double e = Math.exp(x);
            damping[idx] = e;
            if (jumps) reset[idx] = weight[idx] * -Math.expm1(x);
            if (mi > idx) {
                damping[mi] = e;
                if (jumps) reset[mi] = reset[idx];
            }
        }
        double correction = 0.0, compensation = 0.0;
        for (int idx = 0; idx < nTotal; idx++) {
            if (jumps) {
                final double term = reset[idx] * (ghatRe[idx] * re[idx] - ghatIm[idx] * im[idx]);
                final double adjusted = term - compensation;
                final double next = correction + adjusted;
                compensation = (next - correction) - adjusted;
                correction = next;
            }
            final double d = damping[idx];
            re[idx] *= d;
            im[idx] *= d;
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
        if (t == 0.0) return auditLandingGrid(partial.clone(), false);
        return propagateLanding(partial, landingTables(null, sigma, lambda), sigma, t, ghatRe, ghatIm,
                new double[nTotal], new double[nTotal], new double[nTotal], new double[nTotal]);
    }

    /**
     * propagateLanding with tables for (sigma, lambda) and caller-owned arrays: the message is
     * written to `out` and returned; im, damping and reset are scratch.
     */
    double[] propagateLanding(double[] partial, LandingTables tables, double[][] sigma, double t,
                              double[] ghatRe, double[] ghatIm,
                              double[] out, double[] im, double[] damping, double[] reset) {
        requireLandingArray(partial, nTotal, "partial");
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingTime(t);
        requireLandingRate(tables.lambda);
        requireLandingGhat(tables.lambda, ghatRe, ghatIm);
        requireTables(tables, sigma, tables.lambda);
        System.arraycopy(partial, 0, out, 0, nTotal);
        if (t == 0.0) return auditLandingGrid(out, false);
        java.util.Arrays.fill(im, 0.0);
        fftN(out, im, false);
        applyLanding(out, im, tables, t, ghatRe, ghatIm, damping, reset);
        fftN(out, im, true);
        return auditLandingGrid(out, false);
    }

    /**
     * Fourier coefficients of a landing tip: exp(-i m.theta) * exp(-1/2 m' Sigma_eps m), scaled to
     * FFT normalization. Depend on nothing but the tip angles and Sigma_eps, so the likelihood
     * computes them once and keeps them (see leafKernelLandingFromCoefficients).
     */
    void landingTipCoefficients(double[] theta, double[][] sigmaEps, double[] re, double[] im) {
        requireLandingAngles(theta, "Tip angles");
        requireLandingCovariance(sigmaEps, "Landing tip-error covariance");
        final double[] magnitude = tipMagnitude(sigmaEps);
        final double[] wrapped = new double[dim];
        for (int ax = 0; ax < dim; ax++) wrapped[ax] = Math.IEEEremainder(theta[ax], TWO_PI);
        for (int idx = 0; idx < nTotal; idx++) {
            double phase = 0.0;
            for (int ax = 0; ax < dim; ax++) phase += modeValue[ax][idx] * wrapped[ax];
            re[idx] = magnitude[idx] * Math.cos(phase);
            im[idx] = -magnitude[idx] * Math.sin(phase);
        }
    }

    /** scale * exp(-1/2 m' Sigma_eps m) at every mode, for the most recent Sigma_eps (shared, immutable). */
    private volatile Object[] tipMagnitudeMemo;          // {long[] key, double[] magnitude}

    private double[] tipMagnitude(double[][] sigmaEps) {
        final long[] key = landingKey(sigmaEps, 0.0);
        final Object[] memo = tipMagnitudeMemo;
        if (memo != null && java.util.Arrays.equals((long[]) memo[0], key)) return (double[]) memo[1];
        final double scale = (double) nTotal / volume;
        final double[] magnitude = new double[nTotal];
        final double[] m = new double[dim];
        for (int idx = 0; idx < nTotal; idx++) {
            for (int ax = 0; ax < dim; ax++) m[ax] = modeValue[ax][idx];
            magnitude[idx] = scale * Math.exp(-0.5 * quadForm(sigmaEps, m));
        }
        tipMagnitudeMemo = new Object[]{key, magnitude};
        return magnitude;
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
        return leafKernelLandingFromCoefficients(re, im, landingTables(null, sigma, lambda), sigma, t,
                ghatRe, ghatIm, re, im, new double[nTotal], new double[nTotal]);
    }

    /**
     * leafKernelLanding from a tip's coefficients (landingTipCoefficients), with tables for
     * (sigma, lambda) and caller-owned arrays. The coefficients are copied into out/im first, so
     * they survive unless the caller passes them as out/im.
     */
    double[] leafKernelLandingFromCoefficients(double[] tipRe, double[] tipIm, LandingTables tables,
                                               double[][] sigma, double t, double[] ghatRe, double[] ghatIm,
                                               double[] out, double[] im, double[] damping, double[] reset) {
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingTime(t);
        requireLandingRate(tables.lambda);
        requireLandingGhat(tables.lambda, ghatRe, ghatIm);
        requireTables(tables, sigma, tables.lambda);
        if (out != tipRe) System.arraycopy(tipRe, 0, out, 0, nTotal);
        if (im != tipIm) System.arraycopy(tipIm, 0, im, 0, nTotal);
        applyLanding(out, im, tables, t, ghatRe, ghatIm, damping, reset);
        fftN(out, im, true);
        return auditLandingGrid(out, false);
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
        return rootIntegralLanding(partial, landingTables(null, sigma, lambda), sigma, ghatRe, ghatIm,
                new double[nTotal], new double[nTotal]);
    }

    /** rootIntegralLanding with tables for (sigma, lambda); re and im are scratch. */
    double rootIntegralLanding(double[] partial, LandingTables tables, double[][] sigma,
                               double[] ghatRe, double[] ghatIm, double[] re, double[] im) {
        requireLandingArray(partial, nTotal, "partial");
        requireLandingCovariance(sigma, "Diffusion covariance");
        requireLandingRate(tables.lambda);
        requireLandingGhat(tables.lambda, ghatRe, ghatIm);
        requireTables(tables, sigma, tables.lambda);
        if (tables.lambda == 0.0) return rootIntegral(partial);
        System.arraycopy(partial, 0, re, 0, nTotal);
        java.util.Arrays.fill(im, 0.0);
        fftN(re, im, false);
        final double[] weight = tables.weight;
        double sum = 0.0, compensation = 0.0;
        for (int idx = 0; idx < nTotal; idx++) {
            final double term = weight[idx] * (ghatRe[idx] * re[idx] - ghatIm[idx] * im[idx]);
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
