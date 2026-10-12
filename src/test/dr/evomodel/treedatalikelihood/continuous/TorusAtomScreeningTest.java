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
}
