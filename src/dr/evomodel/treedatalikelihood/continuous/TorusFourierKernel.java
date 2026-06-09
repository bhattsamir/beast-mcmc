/*
 * TorusFourierKernel.java
 *
 * @author Samir Bhatt
 */

package dr.evomodel.treedatalikelihood.continuous;

import dr.math.FastFourierTransform;

/**
 * FFT heat-kernel operations for Brownian diffusion on a flat 2D torus (-pi,pi]^2,
 * discretised on an nGrid x nGrid grid (nGrid a power of two). Java port of the
 * validated prototype torus_fourier_pruning.py. Grids are row-major
 * double[nGrid*nGrid] real functions; complex fields are kept as parallel
 * real/imag arrays and transformed with the in-house four1 FFT (rows then cols).
 *
 * Convention matches numpy: the forward transform is sum_x f(x) exp(-i m.x) and
 * the inverse divides by n^2.
 *
 * @author Samir Bhatt
 */
public class TorusFourierKernel {

    private final int nGrid;
    private final int n2;
    private final double volume;          // (2*pi)^2
    private final double[] freq;          // mode index per axis: numpy fftfreq(nGrid)*nGrid

    public TorusFourierKernel(int nGrid) {
        if (nGrid < 4 || (nGrid & (nGrid - 1)) != 0) {
            throw new IllegalArgumentException("nGrid must be a power of two >= 4");
        }
        this.nGrid = nGrid;
        this.n2 = nGrid * nGrid;
        this.volume = 4.0 * Math.PI * Math.PI;
        this.freq = new double[nGrid];
        for (int i = 0; i < nGrid; i++) {
            freq[i] = (i <= nGrid / 2) ? i : i - nGrid;
        }
    }

    public int getGridSize() {
        return nGrid;
    }

    public int getGridCount() {
        return n2;
    }

    /**
     * Per-mode Brownian damping exp(-1/2 t m^T Sigma m), with
     * Sigma = [[s00, s01], [s01, s11]]. Indexed [a * nGrid + b].
     */
    public double[] dampingFourier(double s00, double s01, double s11, double t) {
        final double[] d = new double[n2];
        for (int a = 0; a < nGrid; a++) {
            final double ma = freq[a];
            for (int b = 0; b < nGrid; b++) {
                final double mb = freq[b];
                final double quad = s00 * ma * ma + 2.0 * s01 * ma * mb + s11 * mb * mb;
                d[a * nGrid + b] = Math.exp(-0.5 * t * quad);
            }
        }
        return d;
    }

    /**
     * Propagate a non-negative grid partial likelihood across a branch of length t
     * by torus heat convolution (FFT -> diagonal damping -> inverse FFT).
     */
    public double[] propagate(double[] partial, double s00, double s01, double s11, double t) {
        final double[] re = partial.clone();
        final double[] im = new double[n2];
        fft2(re, im, false);
        final double[] damp = dampingFourier(s00, s01, s11, t);
        for (int k = 0; k < n2; k++) {
            re[k] *= damp[k];
            im[k] *= damp[k];
        }
        fft2(re, im, true);                               // == numpy ifftn
        for (int k = 0; k < n2; k++) {
            if (re[k] < 0.0) re[k] = 0.0;
        }
        return re;
    }

    /**
     * Heat kernel K_t(grid - y) for a tip observation y = (y0, y1) in radians,
     * built in Fourier space and inverse-transformed. Because K_t is symmetric in
     * its argument for symmetric Sigma, this grid is also the message K_t(y - theta)
     * sent from a leaf to its parent.
     */
    public double[] leafKernel(double y0, double y1, double s00, double s01, double s11, double t) {
        final double[] damp = dampingFourier(s00, s01, s11, t);
        final double[] re = new double[n2];
        final double[] im = new double[n2];
        final double scale = n2 / volume;
        for (int a = 0; a < nGrid; a++) {
            final double ma = freq[a];
            for (int b = 0; b < nGrid; b++) {
                final double mb = freq[b];
                final double phase = ma * y0 + mb * y1;
                final int k = a * nGrid + b;
                final double c = scale * damp[k];
                re[k] = c * Math.cos(phase);              // exp(-i phase)
                im[k] = -c * Math.sin(phase);
            }
        }
        fft2(re, im, true);                               // == numpy ifftn
        for (int k = 0; k < n2; k++) {
            if (re[k] < 0.0) re[k] = 0.0;
        }
        return re;
    }

    /**
     * Uniform/Haar-prior root integral of a grid partial: the average over the
     * torus equals the integral divided by the volume.
     */
    public double rootIntegral(double[] partial) {
        double s = 0.0;
        for (double v : partial) {
            s += v;
        }
        return s / n2;
    }

    /* 2D FFT via the in-house 1D four1 (interleaved re,im), rows then columns. */
    private void fft2(double[] re, double[] im, boolean inverse) {
        final double[] buf = new double[2 * nGrid];
        for (int a = 0; a < nGrid; a++) {                 // transform rows
            final int off = a * nGrid;
            for (int b = 0; b < nGrid; b++) {
                buf[2 * b] = re[off + b];
                buf[2 * b + 1] = im[off + b];
            }
            FastFourierTransform.fft(buf, nGrid, inverse);
            for (int b = 0; b < nGrid; b++) {
                re[off + b] = buf[2 * b];
                im[off + b] = buf[2 * b + 1];
            }
        }
        for (int b = 0; b < nGrid; b++) {                 // transform columns
            for (int a = 0; a < nGrid; a++) {
                buf[2 * a] = re[a * nGrid + b];
                buf[2 * a + 1] = im[a * nGrid + b];
            }
            FastFourierTransform.fft(buf, nGrid, inverse);
            for (int a = 0; a < nGrid; a++) {
                re[a * nGrid + b] = buf[2 * a];
                im[a * nGrid + b] = buf[2 * a + 1];
            }
        }
        if (inverse) {                                    // four1 inverse is unnormalised
            final double inv = 1.0 / n2;
            for (int k = 0; k < n2; k++) {
                re[k] *= inv;
                im[k] *= inv;
            }
        }
    }
}
