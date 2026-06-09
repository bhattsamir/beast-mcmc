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
 * Convention matches numpy: the forward transform is sum_x f(x) exp(-i m.x) and the
 * inverse divides by nGrid^dim.
 *
 * @author Samir Bhatt
 */
public class TorusFourierKernel {

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
            freq[i] = (i <= nGrid / 2) ? i : i - nGrid;
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

    /** Quadratic form m^T S m for a symmetric dim x dim matrix S. */
    private double quadForm(double[][] s, double[] m) {
        double q = 0.0;
        for (int i = 0; i < dim; i++) {
            q += s[i][i] * m[i] * m[i];
            for (int j = i + 1; j < dim; j++) q += 2.0 * s[i][j] * m[i] * m[j];
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
            if (re[k] < 0.0) re[k] = 0.0;
        }
        return re;
    }

    /** Heat kernel K_t(grid - y) for a tip observation y (dim angles, radians; Brownian). */
    public double[] leafKernel(double[] y, double[][] sigma, double t) {
        return leafKernel(y, sigma, t, 0.0, null);
    }

    /** Leaf heat kernel under Brownian motion + compound-Poisson jumps. */
    public double[] leafKernel(double[] y, double[][] sigma, double t,
                               double jumpRate, double[][] sigmaJump) {
        final double[] mult = transferMultiplier(sigma, t, jumpRate, sigmaJump);
        final double[] re = new double[nTotal];
        final double[] im = new double[nTotal];
        final double[] m = new double[dim];
        final double scale = (double) nTotal / volume;
        for (int idx = 0; idx < nTotal; idx++) {
            modeOf(idx, m);
            double phase = 0.0;
            for (int ax = 0; ax < dim; ax++) phase += m[ax] * y[ax];
            final double c = scale * mult[idx];
            re[idx] = c * Math.cos(phase);                // exp(-i phase)
            im[idx] = -c * Math.sin(phase);
        }
        fftN(re, im, true);                               // == numpy ifftn
        for (int k = 0; k < nTotal; k++) {
            if (re[k] < 0.0) re[k] = 0.0;
        }
        return re;
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
