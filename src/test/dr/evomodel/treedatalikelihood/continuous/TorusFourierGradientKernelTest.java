package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import junit.framework.TestCase;

/** Independent finite differences of the production values, including input
 * partial derivatives and the one-sided rate derivative at zero. */
public class TorusFourierGradientKernelTest extends TestCase {
    private static final double[][] SIGMA = {{0.35, 0.06}, {0.06, 0.24}};
    private static final double[][][] DIRECTIONS = {
            {{0.3, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.2}},
            {{0.0, 0.1}, {0.1, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}}};
    private static final double[] RATE_DIRECTIONS = {0.0, 0.0, 0.0, 1.0};

    private static double[][] perturb(double[][] sigma, double[][] direction, double step) {
        double[][] out = new double[sigma.length][sigma.length];
        for (int i = 0; i < sigma.length; i++) for (int j = 0; j < sigma.length; j++)
            out[i][j] = sigma[i][j] + step * direction[i][j];
        return out;
    }

    private void assertDerivative(double finiteDifference, double analytic, double absoluteTolerance) {
        assertTrue("Non-finite analytic derivative", Double.isFinite(analytic));
        double tolerance = absoluteTolerance + 2e-5 * Math.abs(finiteDifference);
        assertEquals("Analytic versus production finite difference", finiteDifference, analytic, tolerance);
    }

    public void testBrownianLeafDirectionsAndValues() {
        TorusFourierKernel grid = new TorusFourierKernel(2, 32);
        double[] y = {0.4, -0.7};
        TorusFourierKernel.GradientContext context = grid.prepareGradientContext(
                SIGMA, 0.0, null, null, null, DIRECTIONS, RATE_DIRECTIONS);
        TorusFourierKernel.ValueGradient result = grid.leafKernelWithGradient(y, 0.7, context);
        double[] production = grid.leafKernel(y, SIGMA, 0.7);
        for (int i = 0; i < production.length; i++) assertEquals(production[i], result.values[i], 0.0);
        double step = 1e-6;
        for (int p = 0; p < DIRECTIONS.length; p++) {
            double[] plus = grid.leafKernel(y, perturb(SIGMA, DIRECTIONS[p], step), 0.7);
            double[] minus = grid.leafKernel(y, perturb(SIGMA, DIRECTIONS[p], -step), 0.7);
            for (int i = 0; i < plus.length; i++)
                assertDerivative((plus[i] - minus[i]) / (2 * step), result.gradients[p][i], 2e-8);
        }
    }

    public void testDirectedThirteenComponentLeafAndPropagation() {
        TorusFourierKernel grid = new TorusFourierKernel(2, 32);
        double[] weights = new double[13];
        double[][] means = new double[13][2];
        double[][][] covariances = new double[13][2][2];
        for (int k = 0; k < 13; k++) {
            weights[k] = 1.0 / 13;
            means[k][0] = 0.15 * k; means[k][1] = -0.08 * k;
            covariances[k] = new double[][]{{0.08 + 0.01 * k, 0.006}, {0.006, 0.1 + 0.01 * k}};
        }
        double rate = 0.7, time = 0.8, step = 1e-6;
        double[] y = {0.4, -0.7};
        TorusFourierKernel.GradientContext context = grid.prepareGradientContext(
                SIGMA, rate, weights, means, covariances, DIRECTIONS, RATE_DIRECTIONS);
        TorusFourierKernel.ValueGradient leaf = grid.leafKernelWithGradient(y, time, context);
        double[] production = grid.leafKernel(y, SIGMA, time, rate, weights, means, covariances);
        for (int i = 0; i < production.length; i++) assertEquals(production[i], leaf.values[i], 0.0);
        double[] partial = new double[grid.getGridCount()];
        double[][] partialGradient = new double[DIRECTIONS.length][partial.length];
        for (int i = 0; i < partial.length; i++) {
            partial[i] = 1.0 + 0.2 * Math.cos(i * 0.17) + 0.1 * Math.sin(i * 0.08);
            for (int p = 0; p < DIRECTIONS.length; p++)
                partialGradient[p][i] = 0.04 * Math.sin(i * 0.13 + p);
        }
        TorusFourierKernel.ValueGradient propagated = grid.propagateWithGradient(partial, partialGradient, time, context);
        production = grid.propagate(partial, SIGMA, time, rate, weights, means, covariances);
        for (int i = 0; i < production.length; i++) assertEquals(production[i], propagated.values[i], 0.0);
        for (int p = 0; p < DIRECTIONS.length; p++) {
            double[][] plusSigma = perturb(SIGMA, DIRECTIONS[p], step);
            double[][] minusSigma = perturb(SIGMA, DIRECTIONS[p], -step);
            double plusRate = rate + step * RATE_DIRECTIONS[p];
            double minusRate = rate - step * RATE_DIRECTIONS[p];
            double[] plus = grid.leafKernel(y, plusSigma, time, plusRate, weights, means, covariances);
            double[] minus = grid.leafKernel(y, minusSigma, time, minusRate, weights, means, covariances);
            for (int i = 0; i < plus.length; i++)
                assertDerivative((plus[i] - minus[i]) / (2 * step), leaf.gradients[p][i], 1e-7);
            double[] plusPartial = partial.clone(), minusPartial = partial.clone();
            for (int i = 0; i < partial.length; i++) {
                plusPartial[i] += step * partialGradient[p][i];
                minusPartial[i] -= step * partialGradient[p][i];
            }
            plus = grid.propagate(plusPartial, plusSigma, time, plusRate, weights, means, covariances);
            minus = grid.propagate(minusPartial, minusSigma, time, minusRate, weights, means, covariances);
            for (int i = 0; i < plus.length; i++)
                assertDerivative((plus[i] - minus[i]) / (2 * step), propagated.gradients[p][i], 1e-7);
        }
    }

    public void testZeroRateRightDerivativeAndNoLaw() {
        TorusFourierKernel grid = new TorusFourierKernel(1, 64);
        double[][] sigma = {{0.2}};
        double[] weights = {1.0}, rateDirections = {1.0};
        double[][] means = {{0.45}};
        double[][][] covariances = {{{0.3}}}, directions = {{{0.0}}};
        double[] y = {0.17};
        TorusFourierKernel.ValueGradient result = grid.leafKernelWithGradient(y, 1.0,
                grid.prepareGradientContext(sigma, 0.0, weights, means, covariances, directions, rateDirections));
        double step = 1e-7;
        double[] plus = grid.leafKernel(y, sigma, 1.0, step, weights, means, covariances);
        double[] zero = grid.leafKernel(y, sigma, 1.0, 0.0, weights, means, covariances);
        for (int i = 0; i < plus.length; i++)
            assertDerivative((plus[i] - zero[i]) / step, result.gradients[0][i], 3e-7);
        result = grid.leafKernelWithGradient(y, 1.0,
                grid.prepareGradientContext(sigma, 7.0, null, null, null, directions, rateDirections));
        for (int i = 0; i < zero.length; i++) {
            assertEquals(zero[i], result.values[i], 0.0);
            assertEquals(0.0, result.gradients[0][i], 0.0);
        }
    }

    public void testLargeRateAndFlooredCellsRemainFinite() {
        TorusFourierKernel grid = new TorusFourierKernel(1, 64);
        double[][] sigma = {{0.2}};
        double[][][] directions = {{{0.1}}, {{0.0}}};
        double[] rateDirections = {0.0, 1.0};
        TorusFourierKernel.ValueGradient result = grid.leafKernelWithGradient(new double[]{0.17}, 1.0,
                grid.prepareGradientContext(sigma, 1000.0, new double[]{1.0}, new double[][]{{0.45}},
                        new double[][][]{{{0.5}}}, directions, rateDirections));
        for (int i = 0; i < result.values.length; i++) {
            assertEquals(1.0 / (2 * Math.PI), result.values[i], 1e-12);
            for (double[] gradient : result.gradients) assertEquals(0.0, gradient[i], 1e-12);
        }
        result = grid.leafKernelWithGradient(new double[]{0.03}, 1.0,
                grid.prepareGradientContext(new double[][]{{1e-4}}, 0.0, null, null, null,
                        new double[][][]{{{1.0}}}, new double[]{0.0}));
        int floored = 0;
        for (int i = 0; i < result.values.length; i++) if (result.values[i] == Double.MIN_NORMAL) {
            assertEquals(0.0, result.gradients[0][i], 0.0);
            floored++;
        }
        assertTrue("Test must exercise the production floor", floored > 0);
    }

    public void testClippedRemainderUsesTheProductionDerivative() {
        TorusFourierKernel grid = new TorusFourierKernel(1, 16);
        double[][] sigma = {{0.001}};
        double[][][] directions = {{{0.0002}}, {{0.0}}};
        double[] rateDirections = {0.0, 1.0}, weights = {1.0}, y = {0.173};
        double[][] means = {{0.4}};
        double[][][] covariances = {{{0.003}}};
        double step = 1e-7;
        for (double rate : new double[]{0.0, 0.7}) {
            TorusFourierKernel.ValueGradient result = grid.leafKernelWithGradient(y, 1.0,
                    grid.prepareGradientContext(sigma, rate, weights, means, covariances, directions, rateDirections));
            for (int p = 0; p < directions.length; p++) {
                double[] plus = grid.leafKernel(y, perturb(sigma, directions[p], step), 1.0,
                        rate + step * rateDirections[p], weights, means, covariances);
                double[] minus = rate == 0.0 && p == 1 ? result.values
                        : grid.leafKernel(y, perturb(sigma, directions[p], -step), 1.0,
                        rate - step * rateDirections[p], weights, means, covariances);
                double denominator = rate == 0.0 && p == 1 ? step : 2 * step;
                for (int i = 0; i < plus.length; i++) {
                    // At lambda=0 an underflow-floor cell stays locally flat,
                    // but any practical finite-difference step crosses the tiny
                    // floor threshold. Its derivative is checked directly.
                    if (rate == 0.0 && result.values[i] == Double.MIN_NORMAL) {
                        assertEquals(0.0, result.gradients[p][i], 0.0);
                        continue;
                    }
                    assertDerivative((plus[i] - minus[i]) / denominator, result.gradients[p][i], 5e-7);
                }
            }
        }
    }

    public void testClippedPropagationDerivative() {
        TorusFourierKernel grid = new TorusFourierKernel(1, 16);
        double[] partial = new double[16];
        partial[0] = 1.0;
        double[][] sigma = {{0.001}}, direction = {{0.0002}};
        TorusFourierKernel.ValueGradient result = grid.propagateWithGradient(partial, new double[1][16], 1.0,
                grid.prepareGradientContext(sigma, 0.0, null, null, null,
                        new double[][][]{direction}, new double[]{0.0}));
        double step = 1e-6;
        double[] plus = grid.propagate(partial, perturb(sigma, direction, step), 1.0);
        double[] minus = grid.propagate(partial, perturb(sigma, direction, -step), 1.0);
        int clipped = 0;
        for (int i = 0; i < plus.length; i++) {
            assertDerivative((plus[i] - minus[i]) / (2 * step), result.gradients[0][i], 1e-8);
            if (result.values[i] == Double.MIN_NORMAL) {
                assertEquals(0.0, result.gradients[0][i], 0.0);
                clipped++;
            }
        }
        assertTrue("Test must exercise clipping in internal propagation", clipped > 0);
    }
}
