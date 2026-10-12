package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.math.FastFourierTransform;
import junit.framework.TestCase;

import java.lang.reflect.Method;
import java.util.Random;

/**
 * The kernel's planned N-D FFT must equal, bit for bit, FastFourierTransform.fft applied line by
 * line along each axis through an interleaved buffer, which is how fftN computed it before the plan.
 * Raw bit patterns are compared, so signed zeros count.
 */
public class TorusPlannedFftTest extends TestCase {

    /** The pre-plan fftN, verbatim apart from taking its sizes as arguments. */
    private static void reference(double[] re, double[] im, boolean inverse, int dim, int nGrid) {
        int nTotal = 1;
        for (int i = 0; i < dim; i++) nTotal *= nGrid;
        final int[] stride = new int[dim];
        for (int ax = 0; ax < dim; ax++) {
            int s = 1;
            for (int k = ax + 1; k < dim; k++) s *= nGrid;
            stride[ax] = s;
        }
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

    private static double sample(Random rng, int kind) {
        switch (kind) {
            case 0: return rng.nextGaussian();
            case 1: return rng.nextDouble();                              // nonnegative, like a partial
            case 2: return rng.nextBoolean() ? 0.0 : -0.0;               // signed zeros
            case 3: return rng.nextGaussian() * Math.pow(10.0, 300 * (rng.nextDouble() - 0.5));
            case 4: return rng.nextGaussian() * Double.MIN_NORMAL * rng.nextDouble();   // subnormal range
            default: return rng.nextInt(5) == 0 ? rng.nextGaussian() : 0.0;            // sparse
        }
    }

    private static void compare(int dim, int nGrid, Random rng) throws Exception {
        final TorusFourierKernel kernel = new TorusFourierKernel(dim, nGrid);
        final Method fftN = TorusFourierKernel.class.getDeclaredMethod("fftN", double[].class, double[].class, boolean.class);
        fftN.setAccessible(true);
        final int n = kernel.getGridCount();
        for (int kind = 0; kind < 6; kind++) {
            for (boolean inverse : new boolean[]{false, true}) {
                final double[] re = new double[n], im = new double[n];
                for (int i = 0; i < n; i++) {
                    re[i] = sample(rng, kind);
                    im[i] = kind == 1 ? 0.0 : sample(rng, kind);
                }
                final double[] re2 = re.clone(), im2 = im.clone();
                fftN.invoke(kernel, re, im, inverse);
                reference(re2, im2, inverse, dim, nGrid);
                for (int i = 0; i < n; i++) {
                    if (Double.doubleToRawLongBits(re[i]) != Double.doubleToRawLongBits(re2[i])
                            || Double.doubleToRawLongBits(im[i]) != Double.doubleToRawLongBits(im2[i])) {
                        fail("dim " + dim + " nGrid " + nGrid + " kind " + kind + (inverse ? " inverse" : " forward")
                                + " cell " + i + ": planned (" + re[i] + ", " + im[i] + ") reference (" + re2[i] + ", " + im2[i] + ")");
                    }
                }
            }
        }
    }

    public void testBitIdenticalToLineByLineFourOne() throws Exception {
        final Random rng = new Random(20261012L);
        for (int nGrid = 4; nGrid <= 2048; nGrid <<= 1) compare(1, nGrid, rng);
        for (int nGrid = 4; nGrid <= 256; nGrid <<= 1) compare(2, nGrid, rng);
        for (int nGrid = 4; nGrid <= 32; nGrid <<= 1) compare(3, nGrid, rng);
        for (int nGrid = 4; nGrid <= 8; nGrid <<= 1) compare(4, nGrid, rng);
    }

    /** Forward then inverse returns the input to rounding, a check independent of the reference. */
    public void testRoundTrip() throws Exception {
        final Random rng = new Random(7L);
        final TorusFourierKernel kernel = new TorusFourierKernel(2, 128);
        final Method fftN = TorusFourierKernel.class.getDeclaredMethod("fftN", double[].class, double[].class, boolean.class);
        fftN.setAccessible(true);
        final int n = kernel.getGridCount();
        final double[] re = new double[n], im = new double[n];
        for (int i = 0; i < n; i++) re[i] = rng.nextGaussian();
        final double[] original = re.clone();
        fftN.invoke(kernel, re, im, false);
        fftN.invoke(kernel, re, im, true);
        for (int i = 0; i < n; i++) {
            assertEquals(original[i], re[i], 1.0e-13);
            assertEquals(0.0, im[i], 1.0e-13);
        }
    }
}
