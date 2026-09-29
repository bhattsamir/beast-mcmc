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

    /**
     * Centre the analytic winding-image window on the nearest image instead of on k = 0.
     * OFF by default, so the shipped numerics are unchanged unless this is asked for;
     * enable with -Dtorus.centredImages=true.
     *
     * <p>{@link #addLatticeAtom} sums Gaussian images at displacements
     * d = x - y + 2*pi*k over a window of k. The window half-width is chosen from a
     * Gershgorin bound so that everything within 6 standard deviations is included, but
     * the window is centred on k = 0, i.e. on d = x - y, which is NOT where the mass is:
     * the grid runs x in [0, 2*pi) while a tip angle y is conventionally in (-pi, pi], so
     * x - y reaches 3*pi and the nearest image sits at k = -1 over most of the grid. The
     * shipped half-width absorbs that with a constant +1, which is why it is correct.
     *
     * <p>Centring on the nearest image instead -- fold x - y into (-pi, pi] first -- makes
     * the +1 unnecessary: a half-width of ceil(6*sigma/2pi) then covers 6 standard
     * deviations measured from the mass, which is the criterion that actually matters.
     * In dim 2 that is 9 images per grid cell where the shipped window needs 25, over the
     * whole realistic parameter range (sigma^2 in [0.005, 1], t in [0.005, 2]).
     *
     * <p>Applied only when six standard deviations fit inside one period; see the gate in
     * addLatticeAtom. This is NOT bit-identical there: a different set of images is summed,
     * in a different order. It was measured to be no less accurate over 144 (Sigma, t, tip)
     * combinations spanning sigma^2 in [0.02, 5], t in [0.01, 3] and tips inside and
     * outside [0, 2*pi): against an over-converged reference the worst error was 3.7e-15
     * centred versus 1.8e-15 shipped, both at rounding level, with no case where centring
     * was worse.
     *
     * <p>Where it applies it is also strictly more robust: the shipped window assumes
     * |x - y| stays within the window's reach, so a tip angle supplied outside (-pi, pi]
     * by more than a period or two misses the nearest image entirely. Folding removes
     * that assumption.
     */
    private static final boolean CENTRED_IMAGES = Boolean.getBoolean("torus.centredImages");

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
        return leafKernel(y, sigma, t, 0.0, null, sigmaEps);
    }


    /** Grid angle along axis ax for flat index idx (x_j = 2*pi*j/nGrid, the FFT grid). */
    private double angleOf(int idx, int ax) {
        return TWO_PI * (double) ((idx / stride[ax]) % nGrid) / (double) nGrid;
    }

    /** NaN denotes an unobserved coordinate. A null vector has no observations. */
    private int[] observedAxes(double[] y) {
        if (y == null) return new int[0];
        if (y.length != dim) throw new IllegalArgumentException("Tip dimension mismatch");
        int count = 0;
        for (double value : y) {
            if (Double.isInfinite(value)) throw new IllegalArgumentException("Observed tip angles must be finite");
            if (!Double.isNaN(value)) count++;
        }
        int[] axes = new int[count];
        int next = 0;
        for (int ax = 0; ax < dim; ax++) if (!Double.isNaN(y[ax])) axes[next++] = ax;
        return axes;
    }

    private double[] neutralLeaf() {
        double[] values = new double[nTotal];
        java.util.Arrays.fill(values, 1.0);
        return values;
    }

    private double[] projectPoint(double[] y, int[] axes) {
        double[] point = new double[axes.length];
        for (int ax = 0; ax < axes.length; ax++) point[ax] = y[axes[ax]];
        return point;
    }

    private double[][] projectMatrix(double[][] matrix, int[] axes) {
        double[][] projected = new double[axes.length][axes.length];
        for (int i = 0; i < axes.length; i++) for (int j = 0; j < axes.length; j++)
            projected[i][j] = matrix[axes[i]][axes[j]];
        return projected;
    }

    /** Embed a reduced Fourier mode with frequency zero in every missing axis.
     * Grid indices use the same ordering, so this also embeds grid coordinates. */
    private int embeddedIndex(int index, TorusFourierKernel reduced, int[] axes) {
        int full = 0;
        for (int ax = 0; ax < axes.length; ax++)
            full += ((index / reduced.stride[ax]) % nGrid) * stride[axes[ax]];
        return full;
    }

    private double[] projectModes(double[] full, TorusFourierKernel reduced, int[] axes) {
        if (full == null) return null;
        double[] projected = new double[reduced.nTotal];
        for (int i = 0; i < projected.length; i++) projected[i] = full[embeddedIndex(i, reduced, axes)];
        return projected;
    }

    /** A marginalized tip is constant along every unobserved parent coordinate. */
    private double[] broadcastLeaf(double[] reduced, int[] axes) {
        double[] full = new double[nTotal];
        for (int index = 0; index < nTotal; index++) {
            int projected = 0;
            for (int axis : axes) projected = projected * nGrid + (index / stride[axis]) % nGrid;
            full[index] = reduced[projected];
        }
        return full;
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
        // Half-width of the image window, in periods needed to reach 6 standard deviations.
        final int shells = (int) Math.ceil(6.0 * Math.sqrt(bound) / TWO_PI);
        // Centre the window on the nearest image only where that actually pays, which is
        // also the only place it is unambiguously safe: when six standard deviations fit
        // inside one period (shells == 1, since bound > 0 makes ceil >= 1). There, folding
        // lets one image each side replace the shipped two -- 9 per cell instead of 25 in
        // dim 2 -- and that band covers the whole realistic operating range. Past it the
        // shipped window is kept verbatim: centring would drop one shell in thirty, worth
        // nothing, and at t*Sigma of 100 to 1000 it measured further from the log-space
        // quadrature reference than the shipped window (3.9e-9 against 1.0e-9).
        final boolean centre = CENTRED_IMAGES && shells <= 1;
        final int kmax = centre ? 1 : shells + 1;
        final int span = 2 * kmax + 1;
        int combos = 1;
        for (int i = 0; i < dim; i++) combos *= span;
        final double logNorm = logWeight - 0.5 * (dim * Math.log(TWO_PI) + logDet);

        // ---- loop-invariant work hoisted out of the per-cell / per-image loops ----
        // This is pure code motion: every value below is computed from the same
        // expression, in the same association order, as the version it replaces, so
        // the result is bit-for-bit identical. It is worth doing because this method
        // dominates the likelihood (measured ~83% of wall time): the inner loop ran
        // dim * combos * nTotal times and repeated two integer divisions and a
        // multiply per iteration that do not depend on the innermost index.

        // (1) The winding offset 2*pi*k depends only on the image index, not the cell.
        final double[] off = new double[combos * dim];          // off[c * dim + ax]
        for (int c = 0; c < combos; c++) {
            int rem = c;
            for (int ax = 0; ax < dim; ax++) {
                off[c * dim + ax] = TWO_PI * (rem % span - kmax);
                rem /= span;
            }
        }
        // (2) Flatten the inverse covariance: one array read per term instead of a
        //     row dereference plus an element read.
        final double[] invFlat = new double[dim * dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) invFlat[i * dim + j] = inv[i][j];
        }
        // (3) Tabulate the grid angle, so angleOf()'s division per axis per cell
        //     becomes an odometer step. gridAngle[k] is exactly angleOf's expression.
        final double[] gridAngle = new double[nGrid];
        for (int k = 0; k < nGrid; k++) gridAngle[k] = TWO_PI * (double) k / (double) nGrid;

        final int[] cell = new int[dim];                        // cell[ax] == (idx / stride[ax]) % nGrid
        final double[] base = new double[dim];
        final double[] d = new double[dim];
        final double[] q = new double[combos];
        for (int idx = 0; idx < nTotal; idx++) {
            // (4) The cell's displacement from the tip is invariant across images.
            if (centre) {
                for (int ax = 0; ax < dim; ax++) {
                    final double b = gridAngle[cell[ax]] - y[ax];
                    base[ax] = b + TWO_PI * Math.rint(-b / TWO_PI);   // fold onto the nearest image
                }
            } else {
                for (int ax = 0; ax < dim; ax++) base[ax] = gridAngle[cell[ax]] - y[ax];
            }
            double qmin = Double.POSITIVE_INFINITY;
            for (int c = 0; c < combos; c++) {
                final int co = c * dim;
                for (int ax = 0; ax < dim; ax++) d[ax] = base[ax] + off[co + ax];
                double form = 0.0;
                for (int i = 0; i < dim; i++) {
                    final double di = d[i];
                    final int io = i * dim;
                    for (int j = 0; j < dim; j++) form += invFlat[io + j] * di * d[j];
                }
                q[c] = form;
                if (form < qmin) qmin = form;
            }
            double acc = 0.0;
            for (int c = 0; c < combos; c++) acc += Math.exp(-0.5 * (q[c] - qmin));
            out[idx] += Math.exp(logNorm + Math.log(acc) - 0.5 * qmin);
            // advance the row-major odometer (last axis fastest, stride[dim-1] == 1)
            for (int ax = dim - 1; ax >= 0; ax--) {
                if (++cell[ax] < nGrid) break;
                cell[ax] = 0;
            }
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
     *
     * PARTIAL OBSERVATION. Sigma_eps and missing coordinates are NOT orthogonal, and the
     * composition is forced by what "missing" means here. The error model is
     * y = x + eps with eps ~ N(0, Sigma_eps) on all dim axes; an unobserved axis is one
     * whose y-component was never recorded, so the tip's contribution is the MARGINAL
     * of that joint over the observed axes. The marginal of a Gaussian on a sub-vector is
     * the principal SUBMATRIX of its covariance, so Sigma_eps is projected exactly as
     * sigma and the jump covariances already are -- projectMatrix, not a precision
     * submatrix (that would be the conditional) and not a diagonal read-off (that would
     * drop error correlation between two observed axes). Concretely: a tip with phi
     * observed and psi missing carries observation error Sigma_eps[phi][phi] only, and a
     * fully missing tip carries none because it contributes no message at all.
     */
    private double[] leafKernelSplit(double[] y, double[][] sigma, double t,
                                     double jumpRate, double[] multRe, double[] multIm,
                                     double[][] sigmaEps) {
        final int[] axes = observedAxes(y);
        if (axes.length == 0) return neutralLeaf();
        if (axes.length < dim) {
            TorusFourierKernel reduced = new TorusFourierKernel(axes.length, nGrid);
            double[] marginal = reduced.leafKernelSplit(projectPoint(y, axes), projectMatrix(sigma, axes),
                    t, jumpRate, projectModes(multRe, reduced, axes), projectModes(multIm, reduced, axes),
                    sigmaEps == null ? null : projectMatrix(sigmaEps, axes));
            return broadcastLeaf(marginal, axes);
        }
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
        if (observedAxes(y).length == 0) return neutralLeaf();
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
        if (observedAxes(y).length == 0) return neutralLeaf();
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
        if (observedAxes(y).length == 0) return neutralLeaf();
        // sigma and jumpRate are required: the zero-jump term e^{-lam t} W_{t Sigma} is
        // evaluated analytically and cannot be recovered from psi alone.
        final double[] mRe = new double[nTotal];
        final double[] mIm = new double[nTotal];
        multiplierFromPsi(t, psiRe, psiIm, mRe, mIm);
        return leafKernelSplit(y, sigma, t, jumpRate, mRe, mIm, sigmaEps);
    }

    // ---- analytic directional derivatives for the production FFT likelihood ----

    /** Residue-specific spectral data, reused across all branches in a gradient sweep.
     * Directions are derivatives of covariance and jump rate with respect to the
     * caller's parameters; fixed mixture weights, means and covariances do not vary. */
    public static final class GradientContext {
        private final TorusFourierKernel owner;
        private final double[][] sigma;
        private final double[][][] sigmaDirections;
        private final double[] rateDirections;
        private final double rate;
        private final boolean hasLaw;
        private final double[] psiRe, psiIm, phiRe, phiIm;
        private final double[][] brownianGradient, psiGradientRe, psiGradientIm;
        // Tip observation error. sigmaEps is the leaf covariance (null = Dirac tip);
        // epsDirections[p] is d(Sigma_eps)/d(parameter p) (null = no parameter moves it).
        // damping[m] = exp(-1/2 m' Sigma_eps m) and epsQuad[p][m] = m' epsDirections[p] m
        // are precomputed here because they are branch-independent, exactly like psi.
        // Both stay null when neither is supplied, which is what keeps the Dirac-tip
        // gradient arithmetic bit-for-bit what it was.
        private final double[][] sigmaEps;
        private final double[][][] epsDirections;
        private final double[] damping;
        private final double[][] epsQuad;

        private GradientContext(TorusFourierKernel owner, double[][] sigma,
                                double[][][] sigmaDirections, double[] rateDirections,
                                double rate, boolean hasLaw) {
            this(owner, sigma, sigmaDirections, rateDirections, rate, hasLaw, null, null);
        }

        private GradientContext(TorusFourierKernel owner, double[][] sigma,
                                double[][][] sigmaDirections, double[] rateDirections,
                                double rate, boolean hasLaw,
                                double[][] sigmaEps, double[][][] epsDirections) {
            this.owner = owner;
            this.sigma = sigma;
            this.sigmaDirections = sigmaDirections;
            this.rateDirections = rateDirections;
            this.rate = rate;
            this.hasLaw = hasLaw;
            this.sigmaEps = sigmaEps;
            this.epsDirections = epsDirections;
            int size = owner.nTotal, parameters = rateDirections.length;
            psiRe = new double[size]; psiIm = new double[size];
            phiRe = new double[size]; phiIm = new double[size];
            brownianGradient = new double[parameters][size];
            psiGradientRe = new double[parameters][size];
            psiGradientIm = new double[parameters][size];
            if (sigmaEps == null && epsDirections == null) {
                damping = null;
                epsQuad = null;
            } else {
                damping = new double[size];
                epsQuad = epsDirections == null ? null : new double[parameters][size];
                final double[] m = new double[owner.dim];
                for (int idx = 0; idx < size; idx++) {
                    owner.modeOf(idx, m);
                    damping[idx] = sigmaEps == null ? 1.0
                            : Math.exp(-0.5 * owner.quadForm(sigmaEps, m));
                    if (epsDirections != null) {
                        for (int p = 0; p < parameters; p++) {
                            epsQuad[p][idx] = owner.quadForm(epsDirections[p], m);
                        }
                    }
                }
            }
        }

        public int getParameterCount() { return rateDirections.length; }

        /** True when a tip observation error is present or differentiated. */
        public boolean hasTipError() { return damping != null; }
    }

    /** Unscaled values and directional derivatives, gradients[parameter][grid cell]. */
    public static final class ValueGradient {
        public final double[] values;
        public final double[][] gradients;

        private ValueGradient(double[] values, double[][] gradients) {
            this.values = values;
            this.gradients = gradients;
        }
    }

    private double[][] copyGradientMatrix(double[][] matrix, String name) {
        if (matrix == null || matrix.length != dim)
            throw new IllegalArgumentException(name + " has the wrong dimension");
        double[][] copy = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            if (matrix[i] == null || matrix[i].length != dim)
                throw new IllegalArgumentException(name + " has the wrong dimension");
            for (int j = 0; j < dim; j++) {
                if (!Double.isFinite(matrix[i][j]))
                    throw new IllegalArgumentException(name + " must be finite");
                copy[i][j] = matrix[i][j];
            }
        }
        return copy;
    }

    /** Cache psi and its analytic derivatives once per residue. An absent jump law
     * disables the rate, as in the tree likelihood. At rate zero a present law
     * supplies the right derivative with respect to the nonnegative rate. */
    public GradientContext prepareGradientContext(double[][] sigma, double jumpRate,
                                                  double[] w, double[][] d, double[][][] s,
                                                  double[][][] dSigma, double[] dRate) {
        return prepareGradientContext(sigma, jumpRate, w, d, s, dSigma, dRate, null, null);
    }

    /**
     * prepareGradientContext() for a tip observed with wrapped-Gaussian error.
     *
     * sigmaEps is the tip observation-error covariance (radians^2, null = Dirac tip);
     * dSigmaEps[p] is its derivative with respect to parameter p (null = no parameter
     * touches it). Sigma_eps is branch-independent, so -- like psi -- its per-mode damping
     * exp(-1/2 m' Sigma_eps m) is computed once here and reused on every terminal branch.
     * It does NOT go into psi: psi is exponentiated by t, the tip factor is not.
     *
     * Both null reproduces the Dirac-tip context and gradient exactly.
     */
    public GradientContext prepareGradientContext(double[][] sigma, double jumpRate,
                                                  double[] w, double[][] d, double[][][] s,
                                                  double[][][] dSigma, double[] dRate,
                                                  double[][] sigmaEps, double[][][] dSigmaEps) {
        if (dSigma == null || dRate == null || dSigma.length != dRate.length)
            throw new IllegalArgumentException("Covariance and rate directions must have equal length");
        if (!Double.isFinite(jumpRate) || jumpRate < 0.0)
            throw new IllegalArgumentException("Gradient jump rate must be finite and nonnegative");
        final boolean hasLaw = w != null && d != null && s != null;
        if (hasLaw && (w.length != d.length || w.length != s.length))
            throw new IllegalArgumentException("Jump mixture dimensions disagree");
        final double[][] cov = copyGradientMatrix(sigma, "Sigma");
        final double[][][] directions = new double[dRate.length][][];
        final double[] rateDirections = new double[dRate.length];
        for (int p = 0; p < dRate.length; p++) {
            directions[p] = copyGradientMatrix(dSigma[p], "Covariance direction");
            if (!Double.isFinite(dRate[p]))
                throw new IllegalArgumentException("Rate directions must be finite");
            rateDirections[p] = hasLaw ? dRate[p] : 0.0;
        }
        if (hasLaw) {
            for (int k = 0; k < w.length; k++) {
                if (!Double.isFinite(w[k]) || d[k] == null || d[k].length != dim)
                    throw new IllegalArgumentException("Invalid jump mixture component");
                copyGradientMatrix(s[k], "Jump covariance");
                for (double value : d[k]) if (!Double.isFinite(value))
                    throw new IllegalArgumentException("Jump means must be finite");
            }
        }
        double[][] eps = null;
        double[][][] epsDirections = null;
        if (sigmaEps != null) eps = copyGradientMatrix(sigmaEps, "Sigma_eps");
        if (dSigmaEps != null) {
            if (dSigmaEps.length != dRate.length)
                throw new IllegalArgumentException("Tip-error directions must match the parameter count");
            epsDirections = new double[dRate.length][][];
            for (int p = 0; p < dRate.length; p++)
                epsDirections[p] = copyGradientMatrix(dSigmaEps[p], "Tip-error direction");
        }
        final GradientContext context = new GradientContext(this, cov, directions,
                rateDirections, hasLaw ? jumpRate : 0.0, hasLaw, eps, epsDirections);
        final double[] m = new double[dim];
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            double phiRe = 0.0, phiIm = 0.0;
            if (hasLaw) {
                for (int k = 0; k < w.length; k++) {
                    double g = w[k] * Math.exp(-0.5 * quadForm(s[k], m));
                    double phase = 0.0;
                    for (int ax = 0; ax < dim; ax++) phase += m[ax] * d[k][ax];
                    phiRe += g * Math.cos(phase);
                    phiIm += g * Math.sin(phase);
                }
            }
            context.phiRe[idx] = phiRe;
            context.phiIm[idx] = phiIm;
            context.psiRe[idx] = -0.5 * quadForm(cov, m);
            if (hasLaw && jumpRate > 0.0) {
                context.psiRe[idx] += jumpRate * (phiRe - 1.0);
                context.psiIm[idx] = jumpRate * phiIm;
            }
            for (int p = 0; p < dRate.length; p++) {
                double brownian = -0.5 * quadForm(directions[p], m);
                context.brownianGradient[p][idx] = brownian;
                context.psiGradientRe[p][idx] = brownian + rateDirections[p] * (phiRe - 1.0);
                context.psiGradientIm[p][idx] = rateDirections[p] * phiIm;
            }
        }
        return context;
    }

    private void checkGradientBranch(double t, GradientContext context) {
        if (context == null || context.owner != this)
            throw new IllegalArgumentException("Gradient context belongs to a different grid");
        if (!(t > 0.0) || !Double.isFinite(t))
            throw new IllegalArgumentException("Gradient branch length must be finite and positive");
    }

    private void checkGradientResult(double[] values, double[][] gradients) {
        for (int idx = 0; idx < nTotal; idx++) {
            if (!Double.isFinite(values[idx]))
                throw new ArithmeticException("Non-finite kernel value in analytic gradient");
            if (values[idx] <= MIN_POS) {
                values[idx] = MIN_POS;
                for (double[] gradient : gradients) gradient[idx] = 0.0;
            }
            for (double[] gradient : gradients) if (!Double.isFinite(gradient[idx]))
                throw new ArithmeticException("Non-finite analytic kernel gradient");
        }
    }

    /** Forward derivative of branch propagation, including input-partial derivatives.
     * Floored output cells have zero derivatives. The caller handles any rescaling. */
    public ValueGradient propagateWithGradient(double[] partial, double[][] partialGradients,
                                                double t, GradientContext context) {
        checkGradientBranch(t, context);
        final int parameters = context.getParameterCount();
        if (partial == null || partial.length != nTotal || partialGradients == null
                || partialGradients.length != parameters)
            throw new IllegalArgumentException("Partial gradient dimensions disagree");
        final double[] re = partial.clone(), im = new double[nTotal];
        for (double value : re) if (!Double.isFinite(value))
            throw new IllegalArgumentException("Partial values must be finite");
        fftN(re, im, false);
        final double[][] gradientRe = new double[parameters][], gradientIm = new double[parameters][nTotal];
        for (int p = 0; p < parameters; p++) {
            if (partialGradients[p] == null || partialGradients[p].length != nTotal)
                throw new IllegalArgumentException("Partial gradient dimensions disagree");
            gradientRe[p] = partialGradients[p].clone();
            for (double value : gradientRe[p]) if (!Double.isFinite(value))
                throw new IllegalArgumentException("Partial derivatives must be finite");
            fftN(gradientRe[p], gradientIm[p], false);
        }
        final double[] mRe = new double[nTotal], mIm = new double[nTotal];
        multiplierFromPsi(t, context.psiRe, context.psiIm, mRe, mIm);
        for (int idx = 0; idx < nTotal; idx++) {
            double ar = re[idx], ai = im[idx];
            re[idx] = ar * mRe[idx] - ai * mIm[idx];
            im[idx] = ar * mIm[idx] + ai * mRe[idx];
            for (int p = 0; p < parameters; p++) {
                double gr = gradientRe[p][idx], gi = gradientIm[p][idx];
                double pr = context.psiGradientRe[p][idx], pi = context.psiGradientIm[p][idx];
                gradientRe[p][idx] = gr * mRe[idx] - gi * mIm[idx]
                        + t * (pr * re[idx] - pi * im[idx]);
                gradientIm[p][idx] = gr * mIm[idx] + gi * mRe[idx]
                        + t * (pr * im[idx] + pi * re[idx]);
            }
        }
        fftN(re, im, true);
        for (int p = 0; p < parameters; p++) fftN(gradientRe[p], gradientIm[p], true);
        checkGradientResult(re, gradientRe);
        return new ValueGradient(re, gradientRe);
    }

    /** Analytic derivative of the production split tip, with the same winding
     * radius, observation coordinates and clipping as leafKernelSplit(). */
    public ValueGradient leafKernelWithGradient(double[] y, double t, GradientContext context) {
        checkGradientBranch(t, context);
        final int parameters = context.getParameterCount();
        final int[] axes = observedAxes(y);
        if (axes.length == 0) return new ValueGradient(neutralLeaf(), new double[parameters][nTotal]);
        if (axes.length < dim) {
            TorusFourierKernel reduced = new TorusFourierKernel(axes.length, nGrid);
            double[][][] directions = new double[parameters][][];
            for (int p = 0; p < parameters; p++) directions[p] = projectMatrix(context.sigmaDirections[p], axes);
            // Sigma_eps marginalises with the tip, exactly as sigma does: the observed
            // sub-vector of y = x + eps has error covariance Sigma_eps[axes][axes]. Its
            // DERIVATIVE directions project the same way (differentiation commutes with
            // taking a submatrix), so a tip with psi missing contributes d/dSigma_eps only
            // through its phi block.
            double[][] marginalEps = context.sigmaEps == null ? null
                    : projectMatrix(context.sigmaEps, axes);
            double[][][] marginalEpsDirections = null;
            if (context.epsDirections != null) {
                marginalEpsDirections = new double[parameters][][];
                for (int p = 0; p < parameters; p++)
                    marginalEpsDirections[p] = projectMatrix(context.epsDirections[p], axes);
            }
            GradientContext marginal = new GradientContext(reduced, projectMatrix(context.sigma, axes),
                    directions, context.rateDirections.clone(), context.rate, context.hasLaw,
                    marginalEps, marginalEpsDirections);
            for (int index = 0; index < reduced.nTotal; index++) {
                int full = embeddedIndex(index, reduced, axes);
                marginal.psiRe[index] = context.psiRe[full];
                marginal.psiIm[index] = context.psiIm[full];
                marginal.phiRe[index] = context.phiRe[full];
                marginal.phiIm[index] = context.phiIm[full];
                for (int p = 0; p < parameters; p++) {
                    marginal.brownianGradient[p][index] = context.brownianGradient[p][full];
                    marginal.psiGradientRe[p][index] = context.psiGradientRe[p][full];
                    marginal.psiGradientIm[p][index] = context.psiGradientIm[p][full];
                }
            }
            ValueGradient result = reduced.leafKernelWithGradient(projectPoint(y, axes), t, marginal);
            double[][] gradients = new double[parameters][];
            for (int p = 0; p < parameters; p++) gradients[p] = broadcastLeaf(result.gradients[p], axes);
            return new ValueGradient(broadcastLeaf(result.values, axes), gradients);
        }
        final double[] mRe = new double[nTotal], mIm = new double[nTotal];
        multiplierFromPsi(t, context.psiRe, context.psiIm, mRe, mIm);
        final double[] bm = transferMultiplier(context.sigma, t, 0.0, null);
        final double weight = Math.exp(-context.rate * t);
        final double[] re = new double[nTotal], im = new double[nTotal], m = new double[dim];
        final double[][] gradientRe = new double[parameters][nTotal], gradientIm = new double[parameters][nTotal];
        final double scale = (double) nTotal / volume;
        int rateDirection = -1;
        for (int p = 0; p < parameters; p++) if (context.rateDirections[p] != 0.0) {
            rateDirection = p;
            break;
        }
        // Scratch for the per-mode remainder derivative. With no tip error these hold the
        // same values the old in-line locals held, so the writes below are bit-identical.
        final double[] gradR = new double[parameters], gradI = new double[parameters];
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            double phase = 0.0;
            for (int ax = 0; ax < dim; ax++) phase += m[ax] * y[ax];
            final double cp = Math.cos(phase), sp = Math.sin(phase);
            double dr = mRe[idx] - weight * bm[idx], di = mIm[idx];
            for (int p = 0; p < parameters; p++) {
                // dD/dSigma = t*dpsi_B*D; dD/dlambda = t*(phi*A-D).
                // This form retains the nonzero right derivative at lambda=0.
                double br = context.brownianGradient[p][idx], lr = context.rateDirections[p];
                gradR[p] = t * (br * dr + lr * (context.phiRe[idx] * mRe[idx]
                        - context.phiIm[idx] * mIm[idx] - dr));
                gradI[p] = t * (br * di + lr * (context.phiRe[idx] * mIm[idx]
                        + context.phiIm[idx] * mRe[idx] - di));
            }
            if (context.damping != null) {
                // Dtilde(m) = damp(m) * D(m). damp is CONSTANT in Sigma and in lambda, so it
                // is a plain multiplier on their derivatives; it is not constant in Sigma_eps,
                // and d(damp)/dE = -1/2 (m' E m) damp gives the extra term.
                final double damp = context.damping[idx];
                for (int p = 0; p < parameters; p++) {
                    if (context.epsQuad != null) {
                        final double half = 0.5 * context.epsQuad[p][idx];
                        gradR[p] -= half * dr;
                        gradI[p] -= half * di;
                    }
                    gradR[p] *= damp;
                    gradI[p] *= damp;
                }
                dr *= damp;
                di *= damp;
            }
            re[idx] = scale * (dr * cp + di * sp);
            im[idx] = scale * (di * cp - dr * sp);
            for (int p = 0; p < parameters; p++) {
                gradientRe[p][idx] = scale * (gradR[p] * cp + gradI[p] * sp);
                gradientIm[p][idx] = scale * (gradI[p] * cp - gradR[p] * sp);
            }
        }
        fftN(re, im, true);
        for (int p = 0; p < parameters; p++) fftN(gradientRe[p], gradientIm[p], true);
        for (int idx = 0; idx < nTotal; idx++) {
            boolean active = re[idx] > 0.0;
            if (context.hasLaw && context.rate == 0.0 && rateDirection >= 0) {
                active = gradientRe[rateDirection][idx] / context.rateDirections[rateDirection] > 0.0;
            }
            if (!active) for (int p = 0; p < parameters; p++) gradientRe[p][idx] = 0.0;
            if (re[idx] < 0.0) re[idx] = 0.0;
        }
        addLatticeAtomWithGradient(re, gradientRe, y, t, context);
        checkGradientResult(re, gradientRe);
        return new ValueGradient(re, gradientRe);
    }

    private void addLatticeAtomWithGradient(double[] out, double[][] gradients,
                                            double[] y, double t, GradientContext context) {
        final int parameters = context.getParameterCount();
        // THE ATOM'S COVARIANCE IS (Sigma_eps + t*Sigma), NOT t*Sigma. This is the half of
        // the tip-error change that genuinely alters the derivative: inv and logDet below are
        // those of the broadened covariance, so every score term built from them changes even
        // though the direction dC/dSigma is still t*Sigma'.
        final double[][] cov = new double[dim][dim];
        for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) cov[i][j] = t * context.sigma[i][j];
        if (context.sigmaEps != null) {
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++) cov[i][j] += context.sigmaEps[i][j];
        }
        final double[][] inv = new double[dim][dim];
        final double logDet = symmetricInverse(cov, inv);
        double bound = 0.0;
        for (int i = 0; i < dim; i++) {
            double row = 0.0;
            for (int j = 0; j < dim; j++) row += Math.abs(cov[i][j]);
            if (row > bound) bound = row;
        }
        final int kmax = (int) Math.ceil(6.0 * Math.sqrt(bound) / TWO_PI) + 1;
        final int span = 2 * kmax + 1;
        int combos = 1;
        for (int i = 0; i < dim; i++) {
            if (span <= 0 || combos > Integer.MAX_VALUE / span)
                throw new ArithmeticException("Winding image count exceeds integer range");
            combos *= span;
        }
        final double logNorm = -context.rate * t - 0.5 * (dim * Math.log(TWO_PI) + logDet);
        // trace[p] = tr(C^{-1} dC/dtheta_p) with dC/dtheta_p = t*Sigma'_p + E_p. The two
        // pieces are accumulated separately so that, with no tip-error direction, this is
        // character-for-character the expression it replaces.
        final double[] trace = new double[parameters];
        for (int p = 0; p < parameters; p++) for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++) trace[p] += t * inv[i][j] * context.sigmaDirections[p][j][i];
        if (context.epsDirections != null) {
            for (int p = 0; p < parameters; p++) for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++) trace[p] += inv[i][j] * context.epsDirections[p][j][i];
        }
        final double[] delta = new double[dim], solved = new double[dim], q = new double[combos];
        final double[][] score = new double[parameters][combos];
        for (int idx = 0; idx < nTotal; idx++) {
            double qmin = Double.POSITIVE_INFINITY;
            for (int c = 0; c < combos; c++) {
                int rem = c;
                for (int ax = 0; ax < dim; ax++) {
                    final int k = rem % span - kmax;
                    rem /= span;
                    delta[ax] = angleOf(idx, ax) - y[ax] + TWO_PI * k;
                }
                double form = 0.0;
                for (int i = 0; i < dim; i++) {
                    solved[i] = 0.0;
                    for (int j = 0; j < dim; j++) {
                        form += inv[i][j] * delta[i] * delta[j];
                        solved[i] += inv[i][j] * delta[j];
                    }
                }
                q[c] = form;
                if (form < qmin) qmin = form;
                for (int p = 0; p < parameters; p++)
                    score[p][c] = 0.5 * (t * quadForm(context.sigmaDirections[p], solved) - trace[p])
                            - t * context.rateDirections[p];
                // d/dtheta of -1/2 delta' C^{-1} delta contributes +1/2 (C^{-1}delta)' dC (C^{-1}delta).
                // dC/dSigma carries the factor t; dC/dSigma_eps does NOT.
                if (context.epsDirections != null) {
                    for (int p = 0; p < parameters; p++)
                        score[p][c] += 0.5 * quadForm(context.epsDirections[p], solved);
                }
            }
            double acc = 0.0;
            final double[] derivative = new double[parameters];
            for (int c = 0; c < combos; c++) {
                double term = Math.exp(-0.5 * (q[c] - qmin));
                acc += term;
                if (term != 0.0) for (int p = 0; p < parameters; p++) derivative[p] += term * score[p][c];
            }
            final double atom = Math.exp(logNorm + Math.log(acc) - 0.5 * qmin);
            out[idx] += atom;
            if (atom != 0.0) for (int p = 0; p < parameters; p++) gradients[p][idx] += atom * (derivative[p] / acc);
        }
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
