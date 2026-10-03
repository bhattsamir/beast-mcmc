package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import junit.framework.TestCase;

/** Marginal transition densities for individual unobserved angular coordinates. */
public class TorusFourierMissingKernelTest extends TestCase {
    private static final double[][] SIGMA = {{0.35, 0.12}, {0.12, 0.24}};
    private static final double[] WEIGHTS = {0.4, 0.6};
    private static final double[][] MEANS = {{0.7, -0.4}, {-0.3, 0.8}};
    private static final double[][][] COVARIANCES = {
            {{0.3, 0.07}, {0.07, 0.25}}, {{0.2, -0.03}, {-0.03, 0.4}}};
    private static final double[][][] DIRECTIONS = {
            {{1, 0}, {0, 0}}, {{0, 0}, {0, 1}}, {{0, 1}, {1, 0}}, {{0, 0}, {0, 0}}};
    private static final double[] RATE_DIRECTIONS = {0, 0, 0, 1};

    public void testPhiOnlyAndPsiOnlyMatchMarginalBrownianAndMixture() {
        int n = 32;
        TorusFourierKernel full = new TorusFourierKernel(2, n);
        TorusFourierKernel marginal = new TorusFourierKernel(1, n);
        double[] psiRe = new double[n * n], psiIm = new double[n * n];
        full.computePsiComplex(SIGMA, 0.6, WEIGHTS, MEANS, COVARIANCES, psiRe, psiIm);
        for (int observed = 0; observed < 2; observed++) {
            double[] y = {Double.NaN, Double.NaN};
            y[observed] = 0.17;
            double[][] covariance = {{SIGMA[observed][observed]}};
            double[][] means = {{MEANS[0][observed]}, {MEANS[1][observed]}};
            double[][][] jumps = {{{COVARIANCES[0][observed][observed]}},
                    {{COVARIANCES[1][observed][observed]}}};
            double[] bm = full.leafKernel(y, SIGMA, 0.8);
            double[] expectedBm = marginal.leafKernel(new double[]{0.17}, covariance, 0.8);
            double[] mixture = full.leafKernel(y, SIGMA, 0.8, 0.6, WEIGHTS, MEANS, COVARIANCES);
            double[] cached = full.leafKernelFromPsi(y, 0.8, psiRe, psiIm, SIGMA, 0.6);
            double[] expectedMixture = marginal.leafKernel(new double[]{0.17}, covariance, 0.8,
                    0.6, WEIGHTS, means, jumps);
            for (int index = 0; index < n * n; index++) {
                int projected = observed == 0 ? index / n : index % n;
                assertEquals(expectedBm[projected], bm[index], 1e-14);
                assertEquals(expectedMixture[projected], mixture[index], 1e-14);
                assertEquals(expectedMixture[projected], cached[index], 1e-14);
            }
        }
    }

    public void testFullyMissingTipsAreNeutralIncludingGradients() {
        TorusFourierKernel grid = new TorusFourierKernel(2, 16);
        TorusFourierKernel.GradientContext context = grid.prepareGradientContext(
                SIGMA, 0.6, WEIGHTS, MEANS, COVARIANCES, DIRECTIONS, RATE_DIRECTIONS);
        for (double[] missing : new double[][]{null, {Double.NaN, Double.NaN}}) {
            double[][] values = {
                    grid.leafKernel(missing, SIGMA, 0.8),
                    grid.leafKernel(missing, SIGMA, 0.8, 0.6, COVARIANCES[0]),
                    grid.leafKernel(missing, SIGMA, 0.8, 0.6, WEIGHTS, MEANS, COVARIANCES),
                    grid.leafKernelFromPsi(missing, 0.8, null, null, SIGMA, 0.6)};
            for (double[] vector : values) for (double value : vector) assertEquals(1.0, value, 0.0);
            TorusFourierKernel.ValueGradient result = grid.leafKernelWithGradient(missing, 0.8, context);
            for (double value : result.values) assertEquals(1.0, value, 0.0);
            for (double[] gradient : result.gradients)
                for (double value : gradient) assertEquals(0.0, value, 0.0);
        }
    }

    public void testPartialTipGradientsMatchProductionFiniteDifferences() {
        TorusFourierKernel grid = new TorusFourierKernel(2, 32);
        double step = 1e-6;
        for (int observed = 0; observed < 2; observed++) {
            double[] y = {Double.NaN, Double.NaN};
            y[observed] = 0.17;
            TorusFourierKernel.ValueGradient result = grid.leafKernelWithGradient(y, 0.8,
                    grid.prepareGradientContext(SIGMA, 0.6, WEIGHTS, MEANS, COVARIANCES,
                            DIRECTIONS, RATE_DIRECTIONS));
            double[] production = grid.leafKernel(y, SIGMA, 0.8, 0.6, WEIGHTS, MEANS, COVARIANCES);
            for (int index = 0; index < production.length; index++)
                assertEquals(production[index], result.values[index], 0.0);
            for (int p = 0; p < DIRECTIONS.length; p++) {
                double[][] plusSigma = new double[2][2], minusSigma = new double[2][2];
                for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
                    plusSigma[i][j] = SIGMA[i][j] + step * DIRECTIONS[p][i][j];
                    minusSigma[i][j] = SIGMA[i][j] - step * DIRECTIONS[p][i][j];
                }
                double[] plus = grid.leafKernel(y, plusSigma, 0.8, 0.6 + step * RATE_DIRECTIONS[p],
                        WEIGHTS, MEANS, COVARIANCES);
                double[] minus = grid.leafKernel(y, minusSigma, 0.8, 0.6 - step * RATE_DIRECTIONS[p],
                        WEIGHTS, MEANS, COVARIANCES);
                for (int index = 0; index < plus.length; index++) {
                    double numeric = (plus[index] - minus[index]) / (2 * step);
                    assertEquals(numeric, result.gradients[p][index], 1e-7 + 1e-5 * Math.abs(numeric));
                    if (p == 2 || p == 1 - observed)
                        assertEquals("marginal density ignores missing variance and cross covariance",
                                0.0, result.gradients[p][index], 0.0);
                }
            }
        }
    }

    public void testNonadjacentObservedAxesUseCorrectModeAndBroadcastIndices() {
        int n = 8;
        TorusFourierKernel full = new TorusFourierKernel(3, n);
        TorusFourierKernel marginal = new TorusFourierKernel(2, n);
        double[][] sigma = {{0.4, 0.02, 0.05}, {0.02, 0.3, -0.01}, {0.05, -0.01, 0.5}};
        double[] result = full.leafKernel(new double[]{0.2, Double.NaN, -0.4}, sigma, 0.7);
        double[] expected = marginal.leafKernel(new double[]{0.2, -0.4},
                new double[][]{{0.4, 0.05}, {0.05, 0.5}}, 0.7);
        for (int i = 0; i < n * n * n; i++) {
            int projected = (i / (n * n)) * n + i % n;
            assertEquals(expected[projected], result[i], 0.0);
        }
    }
}
