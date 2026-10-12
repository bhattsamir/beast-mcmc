package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import junit.framework.TestCase;

import java.lang.reflect.Method;
import java.util.Random;

/**
 * The screened winding-image sum (addLatticeAtom) must equal, bit for bit, the exhaustive sum it
 * replaced, which is kept here verbatim as the reference. Screening may only drop images whose term
 * underflows to exactly zero, so any difference at all is a failure. Covers 1 to 3 dimensions,
 * strongly correlated and badly conditioned covariances, branch lengths from 1e-8 to 10, with and
 * without tip error, and observations outside (-pi, pi].
 */
public class TorusAtomScreeningTest extends TestCase {

    private static final double TWO_PI = 2.0 * Math.PI;

    /** The pre-screening addLatticeAtom, verbatim except for taking the kernel's sizes as arguments. */
    private static void reference(TorusFourierKernel kernel, Method symmetricInverse, int dim, int nGrid,
                                  double[] out, double[] y, double[][] sigma, double t, double logWeight,
                                  double[][] sigmaEps) throws Exception {
        int nTotal = 1;
        for (int i = 0; i < dim; i++) nTotal *= nGrid;
        final int[] stride = new int[dim];
        for (int ax = 0; ax < dim; ax++) {
            int s = 1;
            for (int k = ax + 1; k < dim; k++) s *= nGrid;
            stride[ax] = s;
        }
        final double[][] cov = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) cov[i][j] = t * sigma[i][j];
        }
        if (sigmaEps != null) {
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j < dim; j++) cov[i][j] += sigmaEps[i][j];
            }
        }
        final double[][] inv = new double[dim][dim];
        final double logDet = (Double) symmetricInverse.invoke(kernel, cov, inv);
        double bound = 0.0;
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
                    d[ax] = TWO_PI * (double) ((idx / stride[ax]) % nGrid) / (double) nGrid - y[ax] + TWO_PI * k;
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

    /** Random SPD matrix with eigenvalues spread over [lo, hi] (log-uniform) and a random rotation. */
    private static double[][] covariance(Random rng, int dim, double lo, double hi, double correlation) {
        final double[][] s = new double[dim][dim];
        if (dim == 2 && !Double.isNaN(correlation)) {
            final double a = Math.exp(Math.log(lo) + rng.nextDouble() * (Math.log(hi) - Math.log(lo)));
            final double b = Math.exp(Math.log(lo) + rng.nextDouble() * (Math.log(hi) - Math.log(lo)));
            s[0][0] = a;
            s[1][1] = b;
            s[0][1] = s[1][0] = correlation * Math.sqrt(a * b);
            return s;
        }
        final double[][] q = new double[dim][dim];
        for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) q[i][j] = rng.nextGaussian();
        for (int j = 0; j < dim; j++) {                         // Gram-Schmidt
            for (int k = 0; k < j; k++) {
                double dot = 0.0;
                for (int i = 0; i < dim; i++) dot += q[i][j] * q[i][k];
                for (int i = 0; i < dim; i++) q[i][j] -= dot * q[i][k];
            }
            double norm = 0.0;
            for (int i = 0; i < dim; i++) norm += q[i][j] * q[i][j];
            norm = Math.sqrt(norm);
            for (int i = 0; i < dim; i++) q[i][j] /= norm;
        }
        final double[] eig = new double[dim];
        for (int i = 0; i < dim; i++) eig[i] = Math.exp(Math.log(lo) + rng.nextDouble() * (Math.log(hi) - Math.log(lo)));
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j <= i; j++) {
                double v = 0.0;
                for (int k = 0; k < dim; k++) v += q[i][k] * eig[k] * q[j][k];
                s[i][j] = v;
                s[j][i] = v;
            }
        }
        return s;
    }

    public void testScreenedImageSumIsBitIdentical() throws Exception {
        final Random rng = new Random(20261012L);
        final Method atom = TorusFourierKernel.class.getDeclaredMethod("addLatticeAtom", double[].class,
                double[].class, double[][].class, double.class, double.class, double[][].class);
        final Method inverse = TorusFourierKernel.class.getDeclaredMethod("symmetricInverse", double[][].class,
                double[][].class);
        atom.setAccessible(true);
        inverse.setAccessible(true);
        final int[][] shapes = {{1, 8}, {1, 64}, {1, 256}, {2, 8}, {2, 16}, {2, 32}, {2, 64}, {3, 8}, {3, 16}};
        final double[] correlations = {Double.NaN, 0.0, 0.9, -0.9, 0.99, -0.99, 0.999};
        final double[] lengths = {1.0e-8, 1.0e-5, 1.0e-3, 0.05, 0.7, 4.0, 10.0};
        int cases = 0, cells = 0, skippedSomewhere = 0;
        for (int[] shape : shapes) {
            final int dim = shape[0], nGrid = shape[1];
            final TorusFourierKernel kernel = new TorusFourierKernel(dim, nGrid);
            final int n = kernel.getGridCount();
            for (double correlation : correlations) {
                if (dim != 2 && !Double.isNaN(correlation)) continue;
                for (double t : lengths) {
                    for (int rep = 0; rep < 3; rep++) {
                        final double lo = Math.pow(10.0, -10 + 6 * rng.nextDouble());
                        final double hi = lo * Math.pow(10.0, 8 * rng.nextDouble());
                        final double[][] sigma = covariance(rng, dim, lo, Math.min(hi, 10.0), correlation);
                        final double[][] eps = rep == 0 ? null : covariance(rng, dim, 1e-6, 0.05, Double.NaN);
                        final double[] y = new double[dim];
                        for (int ax = 0; ax < dim; ax++) {
                            y[ax] = rep == 2 ? 10.0 * (2 * rng.nextDouble() - 1) : Math.PI * (2 * rng.nextDouble() - 1);
                        }
                        final double logWeight = -3.0 * rng.nextDouble();
                        final double[] start = new double[n];
                        for (int i = 0; i < n; i++) start[i] = rng.nextInt(3) == 0 ? 0.0 : rng.nextDouble();
                        final double[] got = start.clone(), want = start.clone();
                        try {
                            atom.invoke(kernel, got, y, sigma, t, logWeight, eps);
                        } catch (java.lang.reflect.InvocationTargetException e) {
                            // the reference must fail the same way (a covariance the inverse rejects)
                            try {
                                reference(kernel, inverse, dim, nGrid, want, y, sigma, t, logWeight, eps);
                                fail("screened atom threw but the reference did not: " + e.getCause());
                            } catch (java.lang.reflect.InvocationTargetException expected) {
                                continue;
                            }
                        }
                        reference(kernel, inverse, dim, nGrid, want, y, sigma, t, logWeight, eps);
                        for (int i = 0; i < n; i++) {
                            if (Double.doubleToRawLongBits(got[i]) != Double.doubleToRawLongBits(want[i])) {
                                fail("dim " + dim + " nGrid " + nGrid + " correlation " + correlation + " t " + t
                                        + " cell " + i + ": screened " + got[i] + " exhaustive " + want[i]);
                            }
                        }
                        cases++;
                        cells += n;
                        if (t * lo < 0.01) skippedSomewhere++;
                    }
                }
            }
        }
        System.out.println("  screened atom: " + cases + " cases, " + cells + " cells, identical bits");
        assertTrue(cases > 300);
        assertTrue(skippedSomewhere > 50);
    }

    /**
     * Covariances near 1e-306 rad^2, where the screen's quadratic forms overflow: screening must switch
     * itself off and the sum must equal the exhaustive one bit for bit, NaN and infinite cells included.
     * The first case is the one the review found (screened: MIN_POS at cell 61, exhaustive: 1.56e305).
     */
    public void testOverflowRegimeIsBitIdentical() throws Exception {
        final Method atom = TorusFourierKernel.class.getDeclaredMethod("addLatticeAtom", double[].class,
                double[].class, double[][].class, double.class, double.class, double[][].class);
        final Method inverse = TorusFourierKernel.class.getDeclaredMethod("symmetricInverse", double[][].class,
                double[][].class);
        atom.setAccessible(true);
        inverse.setAccessible(true);
        final Random rng = new Random(306L);
        int cases = 0;
        for (int rep = 0; rep < 400; rep++) {
            final int dim = rep == 0 ? 2 : 1 + rng.nextInt(2);
            final int nGrid = 8;
            final TorusFourierKernel kernel = new TorusFourierKernel(dim, nGrid);
            final int n = kernel.getGridCount();
            final double[][] sigma;
            final double[] y = new double[dim];
            if (rep == 0) {
                sigma = new double[][]{{1.1842461248098172e-306, -7.007203203054595e-308},
                        {-7.007203203054595e-308, 8.831981323964145e-307}};
                y[0] = -0.7853981633974483;
                y[1] = -2.356194490192345;
            } else {
                final double scale = Math.pow(10.0, -308 + 10 * rng.nextDouble());
                sigma = covariance(rng, dim, scale, scale * 100.0, dim == 2 ? 0.9 * (2 * rng.nextDouble() - 1) : Double.NaN);
                for (int ax = 0; ax < dim; ax++) {   // on a grid point or its winding image, where q = 0 exactly
                    y[ax] = TWO_PI * rng.nextInt(nGrid) / nGrid - (rng.nextBoolean() ? TWO_PI : 0.0);
                }
            }
            final double[] got = new double[n], want = new double[n];
            Throwable gotFailure = null, wantFailure = null;
            try {
                atom.invoke(kernel, got, y, sigma, 1.0, 0.0, null);
            } catch (java.lang.reflect.InvocationTargetException e) {
                gotFailure = e.getCause();
            }
            try {
                reference(kernel, inverse, dim, nGrid, want, y, sigma, 1.0, 0.0, null);
            } catch (java.lang.reflect.InvocationTargetException e) {
                wantFailure = e.getCause();
            }
            assertEquals("case " + rep + " failure", String.valueOf(wantFailure), String.valueOf(gotFailure));
            if (gotFailure != null) continue;
            for (int i = 0; i < n; i++) {
                assertEquals("case " + rep + " cell " + i + ": screened " + got[i] + " exhaustive " + want[i],
                        Double.doubleToRawLongBits(want[i]), Double.doubleToRawLongBits(got[i]));
            }
            cases++;
        }
        System.out.println("  overflow regime: " + cases + " cases identical");
        assertTrue(cases > 100);
    }

    /**
     * The Brownian tip without jumps skips leafKernelSplit's spectral half, which is signed zeros only
     * when every multiplier and damping factor is finite. Compare it with the split path (the jump
     * overload at rate zero) for valid and invalid inputs: non-positive-definite Sigma or Sigma_eps,
     * overflowing forms, and non-finite angles or times. Values and failures must agree exactly.
     */
    public void testBrownianTipShortcutMatchesSplitPath() throws Exception {
        final Random rng = new Random(77L);
        int cases = 0;
        for (int rep = 0; rep < 300; rep++) {
            final int dim = 1 + rng.nextInt(2);
            final TorusFourierKernel kernel = new TorusFourierKernel(dim, dim == 1 ? 256 : 32);
            final double[][] sigma = covariance(rng, dim, 0.01, 2.0, Double.NaN);
            double[][] eps = rng.nextBoolean() ? null : covariance(rng, dim, 0.001, 0.3, Double.NaN);
            final double[] y = new double[dim];
            for (int ax = 0; ax < dim; ax++) y[ax] = Math.PI * (2 * rng.nextDouble() - 1);
            double t = Math.pow(10.0, -3 + 3 * rng.nextDouble());
            switch (rep % 8) {
                case 1: sigma[0][0] = -sigma[0][0]; break;                       // Sigma not positive definite
                case 2: if (eps != null) eps[0][0] = -eps[0][0]; else eps = new double[dim][dim]; break;
                case 3: for (double[] row : sigma) for (int j = 0; j < dim; j++) row[j] *= -1e3; break;
                case 4: y[0] = rng.nextBoolean() ? Double.POSITIVE_INFINITY : Double.NaN; break;
                case 5: t = rng.nextBoolean() ? Double.POSITIVE_INFINITY : Double.NaN; break;
                case 6: for (double[] row : sigma) for (int j = 0; j < dim; j++) row[j] *= 1e300; break;
                case 7: y[0] = (rng.nextBoolean() ? 1.0 : -1.0) * Math.pow(10.0, 305 + 3 * rng.nextDouble()); break;  // phase overflows
                default: break;                                                   // valid inputs (kind 0)
            }
            double[] got = null, want = null;
            Throwable gotFailure = null, wantFailure = null;
            try {
                got = kernel.leafKernel(y, sigma, t, eps);
            } catch (RuntimeException e) {
                gotFailure = e;
            }
            try {
                want = kernel.leafKernel(y, sigma, t, 0.0, (double[][]) null, eps);
            } catch (RuntimeException e) {
                wantFailure = e;
            }
            assertEquals("case " + rep + " failure", String.valueOf(wantFailure), String.valueOf(gotFailure));
            if (want == null) continue;
            for (int i = 0; i < want.length; i++) {
                assertEquals("case " + rep + " (kind " + rep % 8 + ") cell " + i + ": shortcut " + got[i] + " split " + want[i],
                        Double.doubleToRawLongBits(want[i]), Double.doubleToRawLongBits(got[i]));
            }
            cases++;
        }
        System.out.println("  Brownian tip shortcut: " + cases + " cases identical to the split path");
        assertTrue(cases > 150);
    }
}
