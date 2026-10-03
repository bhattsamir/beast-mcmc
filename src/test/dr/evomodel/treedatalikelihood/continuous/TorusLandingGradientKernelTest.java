package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import junit.framework.TestCase;

import java.util.Random;

/**
 * Landing-law kernel derivatives (leafKernelLandingWithGradient, propagateLandingWithGradient,
 * rootIntegralLandingWithGradient and the scalar reset-mass pieces) against Richardson-extrapolated
 * central finite differences of the production value primitives, and bit-identity of the values
 * those gradient primitives return.
 *
 * Steps are ~1e-3 relative (then halved, Richardson): landing grids carry ~1e-8 relative FFT
 * noise in tail-dominated cases, so tiny steps would test round-off rather than the gradient.
 */
public class TorusLandingGradientKernelTest extends TestCase {
    private static final int N = 32;
    private static final double[] LAMBDAS = {0.0, 1e-3, 0.08, 0.5};
    private static final double[] TIMES = {1e-8, 0.03, 0.5, 3.0};

    // K6 production landing law (res_49 XML): weights, absolute means, covariances.
    static final double[] W = {0.5091644683155903, 0.15441088436727965, 0.1286541929417431,
            0.1263930827814436, 0.04160342386627065, 0.03977394772767277};
    static final double[][] MU = {{-2.1296294066484154, 2.415474731019504}, {-1.210793676477717, 2.525461627685921},
            {-1.7625362679790866, -0.08396637415928199}, {-1.1438206077490312, -0.5899272002700391},
            {1.2148167504132186, 0.416763633966152}, {2.8318091427922383, -2.8628426621043963}};
    static final double[][][] OMEGA = {
            {{0.18130576280419633, -0.08390673485426386}, {-0.08390673485426386, 0.19258080639207528}},
            {{0.037962215354622485, -0.017106384498147973}, {-0.017106384498147973, 0.07176370988196648}},
            {{0.11374575316901496, -0.08396806245956447}, {-0.08396806245956447, 0.2421231450715892}},
            {{0.022836044180592915, -0.015634390774296}, {-0.015634390774296, 0.046035397006745395}},
            {{0.0818571698733314, -0.07093613547085424}, {-0.07093613547085426, 0.17947664172882224}},
            {{1.7245300282598262, -0.4888028485766212}, {-0.4888028485766211, 0.8254137593370251}}};
    // OMEGA[4] and OMEGA[5] are not exactly symmetric in the XML (last digit); computeGhat accepts that.

    private final TorusFourierKernel kernel = new TorusFourierKernel(2, N);
    private final int n2 = N * N;
    private double[] ghatRe, ghatIm;
    private final double[][] sigma0 = {{0.021, -0.006}, {-0.006, 0.014}};      // ~ 8 degree diffusion, correlated
    private final double[][] eps0 = {{0.0076154354946700, 0.0}, {0.0, 0.0076154354946700}};   // (5 deg)^2
    private final double[] tip = {-1.9, 2.2};

    @Override
    protected void setUp() {
        double[][] g = kernel.computeGhat(W, MU, OMEGA);
        ghatRe = g[0];
        ghatIm = g[1];
    }

    // ---- directions: p = 0..2 covariance, 3 rate, 4 isotropic tip error ----
    private double[][][] dSigma() {
        return new double[][][]{
                {{0.02, 0.0}, {0.0, 0.0}},
                {{0.0, 0.0}, {0.0, 0.015}},
                {{0.0, 0.008}, {0.008, 0.0}},
                {{0.0, 0.0}, {0.0, 0.0}},
                {{0.0, 0.0}, {0.0, 0.0}}};
    }

    private double[] dRate(double lambda) {
        return new double[]{0, 0, 0, lambda > 0 ? lambda : 0.01, 0};
    }

    private double[][][] dEps() {
        double[][][] d = new double[5][2][2];
        d[4][0][0] = d[4][1][1] = 0.004;
        return d;
    }

    private static double[][] axpy(double[][] a, double h, double[][] d) {
        return new double[][]{{a[0][0] + h * d[0][0], a[0][1] + h * d[0][1]},
                {a[1][0] + h * d[1][0], a[1][1] + h * d[1][1]}};
    }

    private static double[] axpy(double[] a, double h, double[] d) {
        double[] out = a.clone();
        if (d != null) for (int k = 0; k < out.length; k++) out[k] += h * d[k];
        return out;
    }

    private static double dot(double[] w, double[] v) {
        double s = 0.0;
        for (int k = 0; k < v.length; k++) s += w[k] * v[k];
        return s;
    }

    private static double absDot(double[] w, double[] v) {
        double s = 0.0;
        for (int k = 0; k < v.length; k++) s += Math.abs(w[k] * v[k]);
        return s;
    }

    private double[] weights(long seed) {
        Random r = new Random(seed);
        double[] w = new double[n2];
        for (int k = 0; k < n2; k++) w[k] = r.nextGaussian();
        return w;
    }

    /** A function of the step h along direction p. */
    private interface Path { double at(double h); }

    /**
     * Richardson-extrapolated derivative at h = 0: central differences, or a second-order
     * one-sided stencil when only h >= 0 is admissible (lambda == 0 in the rate direction).
     */
    private static double derivative(Path f, double h, boolean oneSided) {
        if (oneSided) {
            double f0 = f.at(0.0);
            double d1 = (-3 * f0 + 4 * f.at(h) - f.at(2 * h)) / (2 * h);
            double d2 = (-3 * f0 + 4 * f.at(h / 2) - f.at(h)) / h;
            return (4 * d2 - d1) / 3;
        }
        double d1 = (f.at(h) - f.at(-h)) / (2 * h);
        double d2 = (f.at(h / 2) - f.at(-h / 2)) / h;
        return (4 * d2 - d1) / 3;
    }

    private static void assertClose(String what, double analytic, double fd, double scale) {
        assertClose(what, analytic, fd, scale, 0.0);
    }

    private static void assertClose(String what, double analytic, double fd, double scale, double absFloor) {
        double tol = 2e-7 * Math.max(Math.abs(analytic), Math.abs(fd)) + 1e-9 * scale + absFloor;
        assertTrue(what + ": analytic " + analytic + " vs finite difference " + fd + " (scale " + scale + ")",
                Math.abs(analytic - fd) <= tol);
    }

    // ---------------------------------------------------------------------------------------

    public void testPsiSeriesAndDirectFormAgreeAtTheCrossover() {
        // Exactly at the crossover, both branches must agree to rounding.
        double below = TorusFourierKernel.landingPsi(Math.nextDown(0.1));
        double at = TorusFourierKernel.landingPsi(0.1);
        assertEquals(0.4678840160444466, at, 2e-16);
        assertEquals(at, below, 1e-15);
        // Series and direct form agree over a band where the direct form is accurate.
        for (double x = 0.05; x <= 0.2; x += 0.01) {
            double direct = (-Math.expm1(-x) - x * Math.exp(-x)) / (x * x);
            assertEquals("x=" + x, direct, TorusFourierKernel.landingPsi(x), 1e-12);
        }
        // Small-x limit, where the direct form would have lost ~5e-5 relative.
        assertEquals(0.5 - 1e-12 / 3.0, TorusFourierKernel.landingPsi(1e-12), 1e-16);
        assertEquals(0.5, TorusFourierKernel.landingPsi(0.0), 0.0);
        // Large x: psi -> 1/x^2, finite at infinity.
        assertEquals(1.0 / 1e6, TorusFourierKernel.landingPsi(1e3), 1e-21);
        assertEquals(0.0, TorusFourierKernel.landingPsi(Double.POSITIVE_INFINITY), 0.0);
    }

    private static double resetMass(double s, double t, double lambda) {
        return lambda == 0.0 ? 0.0 : (lambda / (lambda + s)) * -Math.expm1(-(lambda + s) * t);
    }

    public void testResetMassDerivativesMatchFiniteDifferences() {
        double[] out = new double[2];
        for (double lambda : new double[]{0.0, 1e-4, 1e-3, 0.08, 0.5, 3.0}) {
            for (double t : new double[]{1e-8, 1e-3, 0.03, 0.5, 3.0}) {
                // s values chosen so x = (lambda + s) t straddles the 0.1 series/direct crossover
                for (double s : new double[]{0.0, 1e-6, 0.01, 0.1 / t - lambda, 0.1 / t - lambda + 1e-9, 0.5, 40.0}) {
                    if (s < 0.0) continue;
                    TorusFourierKernel.landingResetMassDerivatives(s, t, lambda, out);
                    final double ss = s, ll = lambda;
                    double hl = Math.max(lambda, 1e-3) * 1e-3;
                    double fdl = derivative(h -> resetMass(ss, t, ll + h), hl, lambda == 0.0);
                    double hs = Math.max(s, 1e-3) * 1e-3;
                    double fds = derivative(h -> resetMass(ss + h, t, ll), hs, s == 0.0);
                    // FD round-off floor: rho carries ~1e-16 relative error, divided by the step.
                    double rho = Math.abs(resetMass(s, t, lambda));
                    assertClose("drho/dlambda s=" + s + " t=" + t + " lambda=" + lambda, out[0], fdl,
                            Math.max(t, 1e-12), 1e3 * 1e-16 * rho / hl);
                    assertClose("drho/ds s=" + s + " t=" + t + " lambda=" + lambda, out[1], fds,
                            Math.max(lambda * t * t, 1e-30), 1e3 * 1e-16 * rho / hs);
                }
            }
        }
        // Exact limits.
        TorusFourierKernel.landingResetMassDerivatives(0.0, 0.7, 0.0, out);
        assertEquals(0.7, out[0], 0.0);
        assertEquals(0.0, out[1], 0.0);
        TorusFourierKernel.landingResetMassDerivatives(2.0, 0.7, 0.0, out);
        assertEquals(-Math.expm1(-1.4) / 2.0, out[0], 1e-16);
        assertEquals(0.0, out[1], 0.0);
    }

    public void testLeafDerivativesAndBitIdenticalValues() {
        final double[] w = weights(11);
        for (double lambda : LAMBDAS) {
            for (double t : TIMES) {
                final double[][][] dS = dSigma();
                final double[] dR = dRate(lambda);
                final double[][][] dE = dEps();
                TorusFourierKernel.LandingGradientContext ctx = kernel.prepareLandingGradientContext(
                        sigma0, lambda, ghatRe, ghatIm, dS, dR, eps0, dE);
                long before = kernel.getLandingNonpositiveCount();
                TorusFourierKernel.ValueGradient vg = kernel.leafKernelLandingWithGradient(tip, t, ctx);
                assertEquals("gradient primitives must not touch audit counters",
                        before, kernel.getLandingNonpositiveCount());
                double[] reference = kernel.leafKernelLanding(tip, eps0, sigma0, t, lambda, ghatRe, ghatIm);
                assertBitIdentical("leaf lambda=" + lambda + " t=" + t, reference, vg.values);
                double scale = absDot(w, reference);
                for (int p = 0; p < 5; p++) {
                    final int pp = p;
                    final double l0 = lambda;
                    Path f = h -> dot(w, kernel.leafKernelLanding(tip, axpy(eps0, h, dE[pp]),
                            axpy(sigma0, h, dS[pp]), t, l0 + h * dR[pp], ghatRe, ghatIm));
                    double fd = derivative(f, 1e-3, lambda == 0.0 && p == 3);
                    assertClose("leaf p=" + p + " lambda=" + lambda + " t=" + t, dot(w, vg.gradients[p]), fd, scale);
                }
            }
        }
    }

    public void testPropagateDerivativesAndBitIdenticalValues() {
        final double[] w = weights(12);
        Random r = new Random(5);
        final double[] partial0 = kernel.leafKernelLanding(new double[]{1.0, -0.4}, eps0, sigma0, 0.2, 0.05, ghatRe, ghatIm);
        final double[][] partialGradients = new double[5][n2];
        for (int p = 0; p < 5; p++) for (int k = 0; k < n2; k++) partialGradients[p][k] = 0.3 * r.nextGaussian();
        for (double lambda : LAMBDAS) {
            for (double t : TIMES) {
                final double[][][] dS = dSigma();
                final double[] dR = dRate(lambda);
                TorusFourierKernel.LandingGradientContext ctx = kernel.prepareLandingGradientContext(
                        sigma0, lambda, ghatRe, ghatIm, dS, dR, eps0, null);
                long before = kernel.getLandingNonpositiveCount();
                TorusFourierKernel.ValueGradient vg = kernel.propagateLandingWithGradient(partial0, partialGradients, t, ctx);
                assertEquals(before, kernel.getLandingNonpositiveCount());
                double[] reference = kernel.propagateLanding(partial0, sigma0, t, lambda, ghatRe, ghatIm);
                assertBitIdentical("propagate lambda=" + lambda + " t=" + t, reference, vg.values);
                double scale = absDot(w, reference) + absDot(w, partialGradients[0]);
                for (int p = 0; p < 5; p++) {
                    final int pp = p;
                    final double l0 = lambda;
                    Path f = h -> dot(w, kernel.propagateLanding(axpy(partial0, h, partialGradients[pp]),
                            axpy(sigma0, h, dS[pp]), t, l0 + h * dR[pp], ghatRe, ghatIm));
                    double fd = derivative(f, 1e-3, lambda == 0.0 && p == 3);
                    assertClose("propagate p=" + p + " lambda=" + lambda + " t=" + t,
                            dot(w, vg.gradients[p]), fd, scale);
                }
            }
        }
    }

    public void testZeroLengthBranchReturnsCopies() {
        double[] partial = kernel.leafKernelLanding(tip, eps0, sigma0, 0.3, 0.1, ghatRe, ghatIm);
        double[][] gradients = new double[5][n2];
        gradients[2][7] = 1.5;
        TorusFourierKernel.LandingGradientContext ctx = kernel.prepareLandingGradientContext(
                sigma0, 0.1, ghatRe, ghatIm, dSigma(), dRate(0.1), eps0, null);
        TorusFourierKernel.ValueGradient vg = kernel.propagateLandingWithGradient(partial, gradients, 0.0, ctx);
        assertBitIdentical("t=0", partial, vg.values);
        assertNotSame(partial, vg.values);
        assertEquals(1.5, vg.gradients[2][7], 0.0);
        assertNotSame(gradients[2], vg.gradients[2]);
    }

    public void testRootDerivativesAndBitIdenticalValue() {
        Random r = new Random(9);
        final double[] values0 = kernel.propagateLanding(
                kernel.leafKernelLanding(tip, eps0, sigma0, 0.2, 0.05, ghatRe, ghatIm), sigma0, 0.4, 0.05, ghatRe, ghatIm);
        final double[][] gradients = new double[5][n2];
        for (int p = 0; p < 5; p++) for (int k = 0; k < n2; k++) gradients[p][k] = 0.2 * r.nextGaussian();
        for (double lambda : LAMBDAS) {
            final double[][][] dS = dSigma();
            final double[] dR = dRate(lambda);
            TorusFourierKernel.LandingGradientContext ctx = kernel.prepareLandingGradientContext(
                    sigma0, lambda, ghatRe, ghatIm, dS, dR, eps0, null);
            double[] out = kernel.rootIntegralLandingWithGradient(values0, gradients, ctx);
            double reference = kernel.rootIntegralLanding(values0, sigma0, lambda, ghatRe, ghatIm);
            assertEquals("root value must be the likelihood's own root, bit for bit",
                    Double.doubleToRawLongBits(reference), Double.doubleToRawLongBits(out[0]));
            double scale = 0.0;
            for (double v : values0) scale += Math.abs(v);
            scale /= n2;
            for (int p = 0; p < 5; p++) {
                final int pp = p;
                final double l0 = lambda;
                Path f = h -> kernel.rootIntegralLanding(axpy(values0, h, gradients[pp]),
                        axpy(sigma0, h, dS[pp]), l0 + h * dR[pp], ghatRe, ghatIm);
                double fd = derivative(f, 1e-3, lambda == 0.0 && p == 3);
                assertClose("root p=" + p + " lambda=" + lambda, out[1 + p], fd, scale);
            }
        }
    }

    public void testRateDerivativeAtZeroRateIsTheNonzeroRightDerivative() {
        // At lambda == 0 the value path skips the reset term entirely, but its right
        // derivative is not zero: the rate direction must not be gated on lambda > 0.
        TorusFourierKernel.LandingGradientContext ctx = kernel.prepareLandingGradientContext(
                sigma0, 0.0, ghatRe, ghatIm, dSigma(), dRate(0.0), eps0, null);
        TorusFourierKernel.ValueGradient vg = kernel.leafKernelLandingWithGradient(tip, 0.5, ctx);
        double norm = 0.0;
        for (double v : vg.gradients[3]) norm += Math.abs(v);
        assertTrue("rate derivative at lambda=0 must be nonzero", norm > 1e-6);
    }

    public void testContextRejectsForeignKernelAndBadInputs() {
        TorusFourierKernel other = new TorusFourierKernel(2, N);
        TorusFourierKernel.LandingGradientContext ctx = other.prepareLandingGradientContext(
                sigma0, 0.1, ghatRe, ghatIm, dSigma(), dRate(0.1), eps0, null);
        try {
            kernel.leafKernelLandingWithGradient(tip, 0.3, ctx);
            fail("a context belongs to one kernel");
        } catch (IllegalArgumentException expected) { }
        try {
            kernel.prepareLandingGradientContext(sigma0, -0.1, ghatRe, ghatIm, dSigma(), dRate(0.1), eps0, null);
            fail("negative rate");
        } catch (IllegalArgumentException expected) { }
        try {
            kernel.prepareLandingGradientContext(sigma0, 0.1, ghatRe, ghatIm, dSigma(), new double[]{0, 1}, eps0, null);
            fail("direction counts disagree");
        } catch (IllegalArgumentException expected) { }
    }

    private static void assertBitIdentical(String what, double[] expected, double[] actual) {
        assertEquals(expected.length, actual.length);
        for (int k = 0; k < expected.length; k++) {
            if (Double.doubleToRawLongBits(expected[k]) != Double.doubleToRawLongBits(actual[k])) {
                fail(what + ": value differs at cell " + k + ": " + expected[k] + " vs " + actual[k]);
            }
        }
    }
}
