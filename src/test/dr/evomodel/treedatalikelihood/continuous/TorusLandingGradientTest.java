package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.tree.DefaultTreeModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierGradient;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import junit.framework.TestCase;

import java.util.Collections;

/**
 * Whole-tree landing-law gradient d log L / d(q1, q2, rho, lambda) of TorusFourierTreeLikelihood:
 * finite differences of the engine's own log likelihood on a 9-taxon tree with masked and
 * half-missing tips, K6 landing law, correlated Sigma, small and moderate diffusion and several
 * rates; bit-identity of the sweep's log L; gauge invariance (an independent re-implementation
 * with a different rescaling gauge); read-only behaviour; whole-tip masking of half-missing tips;
 * and the -inf region.
 */
public class TorusLandingGradientTest extends TestCase {
    static final String NEWICK = "((((A:0.31,B:0.12):0.25,(C:0.4,D:0.05):0.11):0.6,(E:0.9,(F:0.2,G:0.33):0.47):0.38)"
            + ":0.2,(H:0.7,I:0.000000001):1.0);";
    static final double EPS = 0.0076154354946700;   // (5 degrees)^2, as in production

    private boolean previousValidation;

    @Override
    protected void setUp() {
        previousValidation = TorusFourierTreeLikelihood.isValidateEveryEvaluation();
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(false);
    }

    @Override
    protected void tearDown() {
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(previousValidation);
    }

    static final class Fixture {
        final DefaultTreeModel tree;
        final Parameter diagonal, correlation, rate, sigmaEps;
        final TorusFourierTreeLikelihood like;
        final TorusFourierKernel kernel;
        final double[][] tips;

        Fixture(int nGrid, double q1, double q2, double rho, double lambda, double[][] tips) throws Exception {
            tree = new DefaultTreeModel("tree", new NewickImporter(NEWICK).importTree(null));
            diagonal = new Parameter.Default(new double[]{q1, q2});
            diagonal.addBounds(new Parameter.DefaultBounds(Double.POSITIVE_INFINITY, 0.0, 2));
            correlation = new Parameter.Default(new double[]{rho});
            correlation.addBounds(new Parameter.DefaultBounds(0.95, -0.95, 1));
            rate = new Parameter.Default(lambda);
            rate.addBounds(new Parameter.DefaultBounds(Double.POSITIVE_INFINITY, 0.0, 1));
            sigmaEps = new Parameter.Default(EPS);
            MatrixParameter precision = new CompoundSymmetricMatrix(diagonal, correlation, true, false);
            this.tips = tips;
            double[][][] data = new double[1][][];
            data[0] = new double[tree.getExternalNodeCount()][];
            for (int i = 0; i < tree.getExternalNodeCount(); i++) {
                NodeRef ext = tree.getExternalNode(i);
                double[] tip = tips[tree.getNodeTaxon(ext).getId().charAt(0) - 'A'];
                data[0][ext.getNumber()] = tip == null ? null : tip.clone();
            }
            kernel = new TorusFourierKernel(2, nGrid);
            like = new TorusFourierTreeLikelihood("landingGradient", tree, null,
                    Collections.singletonList(precision), data, kernel, rate,
                    TorusLandingGradientKernelTest.W, TorusLandingGradientKernelTest.MU,
                    TorusLandingGradientKernelTest.OMEGA, sigmaEps, TorusFourierTreeLikelihood.JumpLaw.LANDING);
        }

        double value(int p) {
            switch (p) {
                case 0: return diagonal.getParameterValue(0);
                case 1: return diagonal.getParameterValue(1);
                case 2: return correlation.getParameterValue(0);
                case 3: return rate.getParameterValue(0);
                default: return sigmaEps.getParameterValue(0);
            }
        }

        void set(int p, double x) {
            switch (p) {
                case 0: diagonal.setParameterValue(0, x); break;
                case 1: diagonal.setParameterValue(1, x); break;
                case 2: correlation.setParameterValue(0, x); break;
                case 3: rate.setParameterValue(0, x); break;
                default: sigmaEps.setParameterValue(0, x); break;
            }
        }

        double logLAt(int p, double x) {
            double old = value(p);
            set(p, x);
            double l = like.getLogLikelihood();
            set(p, old);
            return l;
        }
    }

    /** Tips: two fully missing (D, H), two half-missing (B: psi missing, G: phi missing). */
    static double[][] tips() {
        return new double[][]{
                {-1.15, 2.4}, {-2.0, Double.NaN}, {-1.3, 2.7}, null, {1.1, 0.5},
                {-1.0, -0.6}, {Double.NaN, 2.3}, null, {-2.2, 2.6}};
    }

    private static double richardson(Fixture f, int p, double h) {
        double x = f.value(p);
        double d1 = (f.logLAt(p, x + h) - f.logLAt(p, x - h)) / (2 * h);
        double d2 = (f.logLAt(p, x + h / 2) - f.logLAt(p, x - h / 2)) / h;
        return (4 * d2 - d1) / 3;
    }

    private static double step(int p, double x) {
        return p == 2 ? 1e-3 : 1e-3 * Math.abs(x);
    }

    private static void checkAgainstFiniteDifferences(Fixture f, String label) {
        double logL = f.like.getLogLikelihood();
        assertTrue(label + ": finite logL", Double.isFinite(logL));
        double[] grad = f.like.getResidueGradient(0);
        assertEquals(4, grad.length);
        for (int p = 0; p < 4; p++) {
            double x = f.value(p);
            double h = step(p, x);
            double fd = richardson(f, p, h);
            // FD noise floor: ~1e-13 absolute in logL over a step h (tail-dominated cases are noisier)
            double tol = 1e-6 * Math.max(Math.abs(grad[p]), Math.abs(fd)) + 1e-9 / h;
            assertTrue(label + " p=" + p + ": analytic " + grad[p] + " vs FD " + fd,
                    Math.abs(grad[p] - fd) <= tol);
        }
        assertEquals("FD probing must leave the likelihood state as it was", logL, f.like.getLogLikelihood(), 0.0);
    }

    public void testWholeTreeGradientMatchesFiniteDifferences() throws Exception {
        double[][] settings = {
                // q1, q2, rho, lambda  (Sigma = P^-1; q ~ 60 is ~7 deg, q ~ 1500 is ~1.5 deg)
                {60.0, 90.0, 0.4, 0.08},
                {60.0, 90.0, -0.6, 0.5},
                {40.0, 25.0, 0.2, 0.004},
                {800.0, 1500.0, -0.5, 0.08},
                {1500.0, 700.0, 0.3, 0.02},
        };
        for (double[] s : settings) {
            Fixture f = new Fixture(64, s[0], s[1], s[2], s[3], tips());
            checkAgainstFiniteDifferences(f, java.util.Arrays.toString(s));
        }
    }

    public void testTipErrorDirectionMatchesFiniteDifference() throws Exception {
        Fixture f = new Fixture(64, 60.0, 90.0, 0.4, 0.08, tips());
        double[] grad = f.like.getResidueGradientWithTipError(0);
        assertEquals(5, grad.length);
        double[] plain = f.like.getResidueGradient(0);
        for (int p = 0; p < 4; p++) assertEquals(plain[p], grad[p], 1e-12 * Math.abs(plain[p]) + 1e-15);
        double fd = richardson(f, 4, 1e-3 * EPS);
        assertEquals(fd, grad[4], 1e-6 * Math.abs(fd) + 1e-6);
    }

    public void testSweepLogLikelihoodIsBitIdenticalToTheLikelihood() throws Exception {
        Fixture f = new Fixture(64, 800.0, 1500.0, -0.5, 0.08, tips());
        double[][] moves = {{0, 650.0}, {3, 0.03}, {2, 0.2}, {1, 1900.0}};
        for (double[] move : moves) {
            double logL = f.like.getLogLikelihood();
            double[] g = f.like.getLandingResidueGradientWithLogL(0);
            assertEquals("sweep log L must be the likelihood's, bit for bit",
                    Double.doubleToRawLongBits(logL), Double.doubleToRawLongBits(g[0]));
            double[] plain = f.like.getResidueGradient(0);
            for (int p = 0; p < 4; p++) assertEquals(plain[p], g[1 + p], 0.0);
            f.set((int) move[0], move[1]);
        }
        // Also after a tree change (cached sibling partials reused by the likelihood).
        NodeRef tip = f.tree.getExternalNode(0);
        f.tree.setNodeHeight(tip, f.tree.getNodeHeight(tip) + 0.01);
        double logL = f.like.getLogLikelihood();
        assertEquals(Double.doubleToRawLongBits(logL),
                Double.doubleToRawLongBits(f.like.getLandingResidueGradientWithLogL(0)[0]));
    }

    public void testGradientSweepIsReadOnly() throws Exception {
        Fixture f = new Fixture(64, 60.0, 90.0, 0.4, 0.08, tips());
        double logL = f.like.getLogLikelihood();
        long[][] audit = f.like.getLandingBranchNonpositiveCounts();
        long kernelAudit = f.kernel.getLandingNonpositiveCount();
        long recomputes = f.like.getNodeRecomputeCount();
        long undefined = f.like.getLandingUndefinedEvaluations();
        for (int i = 0; i < 3; i++) f.like.getResidueGradient(0);
        assertEquals(kernelAudit, f.kernel.getLandingNonpositiveCount());
        assertEquals(recomputes, f.like.getNodeRecomputeCount());
        assertEquals(undefined, f.like.getLandingUndefinedEvaluations());
        long[][] after = f.like.getLandingBranchNonpositiveCounts();
        for (int r = 0; r < audit.length; r++) assertTrue(java.util.Arrays.equals(audit[r], after[r]));
        assertEquals(logL, f.like.getLogLikelihood(), 0.0);
        f.like.makeDirty();
        assertEquals(Double.doubleToRawLongBits(logL), Double.doubleToRawLongBits(f.like.getLogLikelihood()));
    }

    public void testHalfMissingTipIsMaskedWholeUnderLanding() throws Exception {
        double[][] half = tips();
        double[][] whole = tips();
        whole[1] = null;     // B was psi-missing
        whole[6] = null;     // G was phi-missing
        Fixture a = new Fixture(64, 60.0, 90.0, 0.4, 0.08, half);
        Fixture b = new Fixture(64, 60.0, 90.0, 0.4, 0.08, whole);
        assertEquals(Double.doubleToRawLongBits(b.like.getLogLikelihood()),
                Double.doubleToRawLongBits(a.like.getLogLikelihood()));
        double[] ga = a.like.getResidueGradient(0), gb = b.like.getResidueGradient(0);
        for (int p = 0; p < 4; p++) assertEquals(gb[p], ga[p], 0.0);
    }

    public void testProviderDispatchesToLanding() throws Exception {
        Fixture f = new Fixture(64, 60.0, 90.0, 0.4, 0.08, tips());
        TorusFourierGradient provider = new TorusFourierGradient(f.like, 0);
        double[] g = provider.getGradientLogDensity();
        double[] direct = f.like.getResidueGradient(0);
        for (int p = 0; p < 4; p++) assertEquals(direct[p], g[p], 0.0);
        assertEquals(4, provider.getParameter().getDimension());
    }

    /**
     * Gauge invariance: re-implement the landing sweep here from public kernel primitives with
     * the ZERO gauge (no log-scale derivative; derivative grids just divided by M) and an
     * independently computed Sigma and dSigma. The result must agree with the engine's
     * signed-peak gauge to rounding.
     */
    public void testGradientIsIndependentOfTheRescalingGauge() throws Exception {
        for (double[] s : new double[][]{{60.0, 90.0, 0.4, 0.08}, {800.0, 1500.0, -0.5, 0.02}}) {
            Fixture f = new Fixture(64, s[0], s[1], s[2], s[3], tips());
            double[] engine = f.like.getResidueGradient(0);
            double[] zeroGauge = zeroGaugeGradient(f, s[0], s[1], s[2], s[3]);
            for (int p = 0; p < 4; p++) {
                assertEquals("p=" + p + " " + java.util.Arrays.toString(s), engine[p], zeroGauge[p],
                        1e-8 * Math.abs(engine[p]) + 1e-10);
            }
        }
    }

    private static double[] zeroGaugeGradient(Fixture f, double q1, double q2, double rho, double lambda) {
        double off = rho * Math.sqrt(q1 * q2);
        double det = q1 * q2 - off * off;
        double[][] sigma = {{q2 / det, -off / det}, {-off / det, q1 / det}};
        double[][][] dP = {
                {{1.0, 0.5 * rho * Math.sqrt(q2 / q1)}, {0.5 * rho * Math.sqrt(q2 / q1), 0.0}},
                {{0.0, 0.5 * rho * Math.sqrt(q1 / q2)}, {0.5 * rho * Math.sqrt(q1 / q2), 1.0}},
                {{0.0, Math.sqrt(q1 * q2)}, {Math.sqrt(q1 * q2), 0.0}},
                {{0.0, 0.0}, {0.0, 0.0}}};
        double[][][] dSigma = new double[4][2][2];
        for (int p = 0; p < 4; p++)
            for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
                for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++)
                    dSigma[p][i][j] -= sigma[i][a] * dP[p][a][b] * sigma[b][j];
        double[][] ghat = f.kernel.computeGhat(TorusLandingGradientKernelTest.W,
                TorusLandingGradientKernelTest.MU, TorusLandingGradientKernelTest.OMEGA);
        double[][] eps = {{EPS, 0.0}, {0.0, EPS}};
        TorusFourierKernel.LandingGradientContext ctx = f.kernel.prepareLandingGradientContext(
                sigma, lambda, ghat[0], ghat[1], dSigma, new double[]{0, 0, 0, 1}, eps, null);
        double[] logScaleGradient = new double[4];
        Object[] root = zeroGaugePartial(f, f.tree.getRoot(), ctx, logScaleGradient);
        double[] values = (double[]) root[0];
        double[][] gradients = (double[][]) root[1];
        double[] r = f.kernel.rootIntegralLandingWithGradient(values, gradients, ctx);
        double[] out = new double[4];
        for (int p = 0; p < 4; p++) out[p] = logScaleGradient[p] + r[1 + p] / r[0];
        return out;
    }

    private static Object[] zeroGaugePartial(Fixture f, NodeRef node, TorusFourierKernel.LandingGradientContext ctx,
                                             double[] logScaleGradient) {
        Tree tree = f.tree;
        int n2 = f.kernel.getGridCount();
        double[] values = new double[n2];
        java.util.Arrays.fill(values, 1.0);
        double[][] gradients = new double[4][n2];
        for (int c = 0; c < tree.getChildCount(node); c++) {
            NodeRef child = tree.getChild(node, c);
            double t = Math.max(tree.getNodeHeight(node) - tree.getNodeHeight(child), 1e-8);
            double[] msg;
            double[][] dmsg;
            if (tree.isExternal(child)) {
                double[] tip = f.tips[tree.getNodeTaxon(child).getId().charAt(0) - 'A'];
                if (tip == null || Double.isNaN(tip[0]) || Double.isNaN(tip[1])) {
                    msg = new double[n2];
                    java.util.Arrays.fill(msg, 1.0);
                    dmsg = new double[4][n2];
                } else {
                    TorusFourierKernel.ValueGradient vg = f.kernel.leafKernelLandingWithGradient(tip, t, ctx);
                    msg = vg.values;
                    dmsg = vg.gradients;
                }
            } else {
                Object[] sub = zeroGaugePartial(f, child, ctx, logScaleGradient);
                TorusFourierKernel.ValueGradient vg = f.kernel.propagateLandingWithGradient(
                        (double[]) sub[0], (double[][]) sub[1], t, ctx);
                msg = vg.values;
                dmsg = vg.gradients;
            }
            double max = 0.0;
            for (int k = 0; k < n2; k++) {
                for (int p = 0; p < 4; p++) gradients[p][k] = gradients[p][k] * msg[k] + values[k] * dmsg[p][k];
                values[k] *= msg[k];
                max = Math.max(max, Math.abs(values[k]));
            }
            for (int k = 0; k < n2; k++) {
                values[k] /= max;
                for (int p = 0; p < 4; p++) gradients[p][k] /= max;     // zero gauge: no log-scale derivative
            }
        }
        return new Object[]{values, gradients};
    }

    /**
     * Where the truncated root sum is non-positive the likelihood is -inf; the gradient must not
     * be finite there: it throws ArithmeticException (HMC rejects the trajectory) and counts it.
     */
    public void testNonPositiveRootThrowsAndIsCounted() throws Exception {
        // Very small diffusion on a coarse grid: the truncated root sum rings negative
        // (found by search; production Sigma_eps, q ~ 1e4, lambda = 1e-4, nGrid 32).
        Fixture f = new Fixture(32, 1.0e4, 7.0e3, 0.2, 1e-4, tips());
        assertEquals(Double.NEGATIVE_INFINITY, f.like.getLogLikelihood());
        assertEquals(1L, f.like.getLandingUndefinedEvaluations());
        long before = f.like.getLandingUndefinedGradients();
        try {
            f.like.getResidueGradient(0);
            fail("a -inf landing state must not yield a finite gradient");
        } catch (ArithmeticException expected) {
            assertTrue(expected.getMessage().contains("non-positive"));
        }
        assertEquals(before + 1, f.like.getLandingUndefinedGradients());
        assertEquals("the read-only sweep must not touch the value-path audit",
                1L, f.like.getLandingUndefinedEvaluations());
    }
}
