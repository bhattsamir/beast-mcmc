package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.tree.DefaultTreeModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood.JumpLaw;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Landing reset-law tests, deliberately separate from the unchanged shift regressions. */
public class TorusLandingTest extends TestCase {
    private static final String NEWICK = "((A:1.0,B:1.0):0.5,C:1.5);";
    private static final double[][] SIGMA = {{0.35, 0.08}, {0.08, 0.22}};
    private static final double[] WEIGHTS = {0.6, 0.4};
    private static final double[][] MEANS = {{0.0, 0.0}, {1.0, -0.5}};
    private static final double[][][] COVARIANCES = {
            {{0.6, 0.0}, {0.0, 0.5}}, {{0.3, 0.0}, {0.0, 0.3}}};

    /** gen_landing_java_refs.py: Fourier n=128 -8.069176600440020;
     * independent real-space winding/quadrature n=48 -8.069176600440668. */
    public void testIndependentThreeTipReference() throws Exception {
        TorusFourierTreeLikelihood like = likelihood(NEWICK, 128, 0.8, JumpLaw.LANDING, null);
        assertEquals(-8.069176600440020, like.getLogLikelihood(), 1.0e-6);
        assertEquals(0L, like.getLandingNonpositiveCount());
        assertEquals(JumpLaw.LANDING, like.getJumpLaw());
    }

    /** A single Fourier mode gives an analytic sign-sensitive root and branch check. */
    public void testAnalyticSingleModeAndNonuniformRoot() {
        TorusFourierKernel kernel = new TorusFourierKernel(1, 64);
        double[][] sigma = {{0.4}};
        double[][] gh = kernel.computeGhat(new double[]{1.0}, new double[][]{{0.7}},
                new double[][][]{{{0.5}}});
        double[] partial = new double[64];
        for (int j = 0; j < partial.length; j++) {
            double x = 2.0 * Math.PI * j / partial.length;
            partial[j] = 1.0 + 0.2 * Math.cos(x) + 0.3 * Math.sin(x);
        }
        double modeMean = Math.exp(-0.25) * (0.2 * Math.cos(0.7) + 0.3 * Math.sin(0.7));
        double lambda = 0.8, q = lambda + 0.2, t = 0.7;
        double expectedRoot = 1.0 + lambda / q * modeMean;
        assertEquals(expectedRoot, kernel.rootIntegralLanding(partial, sigma, lambda, gh[0], gh[1]), 1.0e-13);
        assertTrue(Math.abs(expectedRoot - kernel.rootIntegral(partial)) > 0.1);
        double[] result = kernel.propagateLanding(partial, sigma, t, lambda, gh[0], gh[1]);
        for (int j = 0; j < result.length; j++) {
            double expected = 1.0 + Math.exp(-q * t) * (partial[j] - 1.0)
                    + lambda / q * (-Math.expm1(-q * t)) * modeMean;
            assertEquals(expected, result[j], 1.0e-13);
        }
    }

    public void testSemigroupStationarityAndConstantPreservation() {
        TorusFourierKernel kernel = new TorusFourierKernel(2, 64);
        double[][] gh = kernel.computeGhat(WEIGHTS, MEANS, COVARIANCES);
        double[] partial = new double[kernel.getGridCount()];
        for (int j = 0; j < partial.length; j++) partial[j] = 1.0 + 0.2 * Math.sin(j * 0.01);
        double[] first = kernel.propagateLanding(partial, SIGMA, 0.3, 0.8, gh[0], gh[1]);
        double[] composed = kernel.propagateLanding(first, SIGMA, 0.4, 0.8, gh[0], gh[1]);
        double[] direct = kernel.propagateLanding(partial, SIGMA, 0.7, 0.8, gh[0], gh[1]);
        assertArrayEquals(direct, composed, 1.0e-12);
        assertEquals(kernel.rootIntegralLanding(partial, SIGMA, 0.8, gh[0], gh[1]),
                kernel.rootIntegralLanding(direct, SIGMA, 0.8, gh[0], gh[1]), 1.0e-12);
        Arrays.fill(partial, 1.0);
        assertArrayEquals(partial, kernel.propagateLanding(partial, SIGMA, 20.0, 0.8, gh[0], gh[1]), 1.0e-13);
        assertEquals(1.0, kernel.rootIntegralLanding(partial, SIGMA, 0.8, gh[0], gh[1]), 1.0e-13);
    }

    public void testPureFourierTipMatchesWrappedNormalAndFusedBranch() {
        TorusFourierKernel kernel = new TorusFourierKernel(1, 128);
        double theta = 0.7, variance = 0.4;
        double[] tip = kernel.leafKernelLanding(new double[]{theta}, new double[][]{{variance}});
        for (int j = 0; j < tip.length; j++) {
            double x = 2.0 * Math.PI * j / tip.length;
            double expected = 0.0;
            for (int winding = -4; winding <= 4; winding++) {
                double delta = x - theta + 2.0 * Math.PI * winding;
                expected += Math.exp(-0.5 * delta * delta / variance) / Math.sqrt(2.0 * Math.PI * variance);
            }
            assertEquals(expected, tip[j], 2.0e-14);
        }
        double[][] gh = kernel.computeGhat(new double[]{1.0}, new double[][]{{-0.4}},
                new double[][][]{{{0.5}}});
        double[][] sigma = {{0.3}}, eps = {{variance}};
        assertArrayEquals(kernel.propagateLanding(tip, sigma, 0.7, 0.8, gh[0], gh[1]),
                kernel.leafKernelLanding(new double[]{theta}, eps, sigma, 0.7, 0.8, gh[0], gh[1]), 1.0e-13);
    }

    public void testZeroRateMatchesShiftAndSmallRateLimit() throws Exception {
        double shift = likelihood(NEWICK, 128, 0.0, JumpLaw.SHIFT, null).getLogLikelihood();
        assertEquals(shift, likelihood(NEWICK, 128, 0.0, JumpLaw.LANDING, null).getLogLikelihood(), 1.0e-9);
        assertEquals(shift, likelihood(NEWICK, 128, 1.0e-12, JumpLaw.LANDING, null).getLogLikelihood(), 1.0e-9);
    }

    public void testDefaultShiftIsBitIdenticalToExplicitShift() throws Exception {
        Tree tree = tree(NEWICK);
        double[][][] data = data(tree, 1, null);
        List<MatrixParameter> precisions = precisions(1);
        Parameter rate = new Parameter.Default(0.8), eps = new Parameter.Default(0.04);
        TorusFourierTreeLikelihood implicit = new TorusFourierTreeLikelihood("implicit", tree, null,
                precisions, data, new TorusFourierKernel(2, 128), rate, WEIGHTS, MEANS, COVARIANCES, eps);
        TorusFourierTreeLikelihood explicit = new TorusFourierTreeLikelihood("explicit", tree, null,
                precisions, data, new TorusFourierKernel(2, 128), rate, WEIGHTS, MEANS, COVARIANCES, eps, JumpLaw.SHIFT);
        assertEquals(JumpLaw.SHIFT, implicit.getJumpLaw());
        assertEquals(Double.doubleToLongBits(implicit.getLogLikelihood()), Double.doubleToLongBits(explicit.getLogLikelihood()));
    }

    public void testMissingTipMarginalizesAndAllMissingIsOne() throws Exception {
        double masked = likelihood(NEWICK, 128, 0.8, JumpLaw.LANDING, "C").getLogLikelihood();
        double cherry = likelihood("(A:1.0,B:1.0);", 128, 0.8, JumpLaw.LANDING, null).getLogLikelihood();
        assertEquals(cherry, masked, 1.0e-12);
        assertEquals(0.0, likelihood(NEWICK, 128, 0.8, JumpLaw.LANDING, "*").getLogLikelihood(), 1.0e-12);
    }

    public void testSharedMixtureWithPerResidueRatesAndCache() throws Exception {
        Tree tree = tree(NEWICK);
        List<MatrixParameter> precisions = precisions(2);
        Parameter rates = new Parameter.Default(new double[]{0.2, 0.8});
        TorusFourierTreeLikelihood like = build(tree, 64, precisions, data(tree, 2, null), rates,
                new Parameter.Default(0.04), JumpLaw.LANDING);
        double expected = likelihood(NEWICK, 64, 0.2, JumpLaw.LANDING, null).getLogLikelihood()
                + likelihood(NEWICK, 64, 0.8, JumpLaw.LANDING, null).getLogLikelihood();
        assertEquals(expected, like.getLogLikelihood(), 1.0e-12);
        like.resetRecomputeCount();
        rates.setParameterValue(1, 1.2);
        double changed = like.getLogLikelihood();
        assertEquals(2L, like.getNodeRecomputeCount());
        assertTrue(Math.abs(changed - expected) > 1.0e-6);
        like.makeDirty();
        assertEquals(changed, like.getLogLikelihood(), 1.0e-12);
        like.resetRecomputeCount();
        precisions.get(0).setParameterValue(0, 0, 4.0);
        changed = like.getLogLikelihood();
        assertEquals(2L, like.getNodeRecomputeCount());
        like.makeDirty();
        assertEquals(changed, like.getLogLikelihood(), 1.0e-12);
    }

    public void testCacheObservationErrorTreeAndStoreRestore() throws Exception {
        DefaultTreeModel tree = tree(NEWICK);
        Parameter rate = new Parameter.Default(0.8), eps = new Parameter.Default(0.04);
        TorusFourierTreeLikelihood like = build(tree, 64, precisions(1), data(tree, 1, null), rate, eps, JumpLaw.LANDING);
        double original = like.getLogLikelihood();
        like.storeModelState();
        rate.setParameterValue(0, 0.3);
        assertTrue(Math.abs(like.getLogLikelihood() - original) > 1.0e-6);
        like.restoreModelState();
        assertEquals(original, like.getLogLikelihood(), 1.0e-12);
        like.makeDirty();
        assertEquals(original, like.getLogLikelihood(), 1.0e-12);
        eps.setParameterValue(0, 0.07);
        double changed = like.getLogLikelihood();
        assertTrue(Math.abs(changed - original) > 1.0e-6);
        like.makeDirty();
        assertEquals(changed, like.getLogLikelihood(), 1.0e-12);
        NodeRef tip = tree.getExternalNode(0);
        tree.setNodeHeight(tip, 0.2);
        double heightChanged = like.getLogLikelihood();
        assertTrue(Math.abs(heightChanged - changed) > 1.0e-6);
        like.makeDirty();
        assertEquals(heightChanged, like.getLogLikelihood(), 1.0e-12);
    }

    public void testLandingRequiresPositiveObservationError() throws Exception {
        for (Parameter eps : new Parameter[]{null, new Parameter.Default(0.0),
                new Parameter.Default(new double[]{0.04, 0.0})}) {
            Tree tree = tree(NEWICK);
            try {
                build(tree, 64, precisions(1), data(tree, 1, null), new Parameter.Default(0.8), eps, JumpLaw.LANDING);
                fail("landing must reject missing or degenerate observation error");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("sigmaEpsilon"));
            }
        }
    }

    public void testLandingRejectsInvalidRateAtEvaluation() throws Exception {
        for (double badRate : new double[]{-1.0, Double.NaN, Double.POSITIVE_INFINITY}) {
            try {
                likelihood(NEWICK, 64, badRate, JumpLaw.LANDING, null).getLogLikelihood();
                fail("landing must reject invalid rates");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().toLowerCase().contains("rate"));
            }
        }
    }

    public void testLandingUsesIndependentValidationReference() throws Exception {
        final boolean previous = TorusFourierTreeLikelihood.isValidateEveryEvaluation();
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(true);
        try {
            assertEquals(-8.069176600440668,
                    likelihood(NEWICK, 128, 0.8, JumpLaw.LANDING, null).getLogLikelihood(), 1.0e-6);
        } finally {
            TorusFourierTreeLikelihood.setValidateEveryEvaluation(previous);
        }
    }

    public void testLandingAncestralTraitIncludesInternalNodesOnly() throws Exception {
        Tree tree = tree(NEWICK);
        TorusFourierTreeLikelihood like = build(tree, 128, precisions(1), data(tree, 1, null),
                new Parameter.Default(0.8), new Parameter.Default(0.04), JumpLaw.LANDING);
        dr.evolution.tree.TreeTrait trait = like.getTreeTrait("theta_0");
        for (int i = 0; i < tree.getNodeCount(); i++) {
            NodeRef node = tree.getNode(i);
            double[] draw = (double[]) trait.getTrait(tree, node);
            if (tree.isExternal(node)) assertNull(draw);
            else {
                assertNotNull(draw);
                assertEquals(2, draw.length);
                for (double angle : draw) assertTrue(angle >= -Math.PI && angle < Math.PI);
            }
        }
    }

    public void testEightResidueParallelEvaluationIsDeterministic() throws Exception {
        Tree tree = tree(NEWICK);
        double[] rates = new double[8];
        double expected = 0.0;
        for (int r = 0; r < rates.length; r++) {
            rates[r] = 0.2 + 0.1 * r;
            expected += likelihood(NEWICK, 64, rates[r], JumpLaw.LANDING, null).getLogLikelihood();
        }
        Parameter rate = new Parameter.Default(rates), eps = new Parameter.Default(0.04);
        TorusFourierTreeLikelihood like = build(tree, 64, precisions(8), data(tree, 8, null), rate, eps, JumpLaw.LANDING);
        double parallel = like.getLogLikelihood();
        assertEquals(expected, parallel, 1.0e-12);
        for (int repeat = 0; repeat < 3; repeat++) {
            like.makeDirty();
            assertEquals(Double.doubleToLongBits(parallel), Double.doubleToLongBits(like.getLogLikelihood()));
        }
        like.resetRecomputeCount();
        rate.setParameterValue(5, 1.2);
        double changed = like.getLogLikelihood();
        assertEquals(2L, like.getNodeRecomputeCount());
        like.makeDirty();
        assertEquals(Double.doubleToLongBits(changed), Double.doubleToLongBits(like.getLogLikelihood()));
        like.resetRecomputeCount();
        eps.setParameterValue(0, 0.06);
        changed = like.getLogLikelihood();
        assertEquals(16L, like.getNodeRecomputeCount());
        like.makeDirty();
        assertEquals(Double.doubleToLongBits(changed), Double.doubleToLongBits(like.getLogLikelihood()));
    }

    /**
     * A parameter value at which the landing root integral is non-positive on this grid is
     * UNDEFINED there. By default the likelihood returns -inf so an MCMC proposal into that
     * region is rejected (as the shift path does for an unusable FFT state), and the event is
     * counted; it must not throw and kill the chain. Strict auditing still throws.
     */
    public void testUndefinedRootIntegralIsRejectedNotThrownUnlessStrict() throws Exception {
        final String previous = System.getProperty("torus.landing.strict");
        try {
            System.clearProperty("torus.landing.strict");
            TorusFourierTreeLikelihood lenient = undefinedRootLikelihood();
            assertEquals(Double.NEGATIVE_INFINITY, lenient.getLogLikelihood());
            assertEquals(1L, lenient.getLandingUndefinedEvaluations());

            System.setProperty("torus.landing.strict", "true");
            TorusFourierTreeLikelihood strict = undefinedRootLikelihood();
            try {
                strict.getLogLikelihood();
                fail("strict auditing must reject an undefined landing root integral");
            } catch (ArithmeticException expected) {
                // strict auditing rejects at the first under-resolved message, before the root
                assertTrue(expected.getMessage(), expected.getMessage().contains("Invalid landing root integral")
                        || expected.getMessage().contains("Under-resolved landing branch"));
            }
        } finally {
            if (previous == null) System.clearProperty("torus.landing.strict");
            else System.setProperty("torus.landing.strict", previous);
        }
    }

    /** Found by search: nGrid 8 cannot represent this tree's partials, so the root mass rings negative. */
    private static TorusFourierTreeLikelihood undefinedRootLikelihood() throws Exception {
        Tree tree = tree("((A:0.001,B:0.001):0.001,C:0.002);");
        double[][] tips = {{0.0, 0.0}, {1.5, -1.0}, {-2.0, 2.5}};
        double[][][] data = new double[1][tree.getExternalNodeCount()][];
        for (int e = 0; e < tree.getExternalNodeCount(); e++) {
            data[0][tree.getExternalNode(e).getNumber()] = tips["ABC".indexOf(tree.getNodeTaxon(tree.getExternalNode(e)).getId())];
        }
        List<MatrixParameter> precision = new ArrayList<MatrixParameter>();
        precision.add(new MatrixParameter("undefinedPrecision", new Parameter[]{
                new Parameter.Default(new double[]{1.0e5, 0.0}), new Parameter.Default(new double[]{0.0, 1.0e5})}));
        return new TorusFourierTreeLikelihood("undefinedRoot", tree, null, precision, data,
                new TorusFourierKernel(2, 8), new Parameter.Default(1.0e-3), new double[]{1.0},
                new double[][]{{0.5, 0.5}}, new double[][][]{{{0.2, 0.0}, {0.0, 0.2}}},
                new Parameter.Default(1.0e-4), JumpLaw.LANDING);
    }

    public void testUnderResolvedLandingBranchFailsExplicitly() throws Exception {
        final String previous = System.getProperty("torus.landing.strict");
        System.setProperty("torus.landing.strict", "true");
        try {
            Tree tree = tree("(A:0.00001,B:0.00001);");
            double[][][] observations = data(tree, 1, null);
            for (int tip = 0; tip < observations[0].length; tip++) {
                observations[0][tip] = new double[]{0.4, 1.1};
            }
            TorusFourierTreeLikelihood like = build(tree, 8, precisions(1), observations,
                    new Parameter.Default(0.001), new Parameter.Default(1.0e-6), JumpLaw.LANDING);
            try {
                like.getLogLikelihood();
                fail("strict auditing must reject materially negative branch messages");
            } catch (ArithmeticException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("Under-resolved landing branch"));
                assertTrue(expected.getMessage(), expected.getMessage().contains("nGrid=8"));
                assertTrue(like.getLandingNonpositiveCount() > 0);
            }
        } finally {
            if (previous == null) System.clearProperty("torus.landing.strict");
            else System.setProperty("torus.landing.strict", previous);
        }
    }

    /** Default evaluation records truncation diagnostics while preserving the Fourier result. */
    public void testDefaultLandingRecordsNegativeBranchesWithoutClipping() throws Exception {
        final String previous = System.getProperty("torus.landing.strict");
        System.clearProperty("torus.landing.strict");
        try {
            Tree tree = tree("(A:0.00001,B:0.00001);");
            double[][][] observations = data(tree, 1, null);
            for (int tip = 0; tip < observations[0].length; tip++) {
                observations[0][tip] = new double[]{0.4, 1.1};
            }
            TorusFourierKernel kernel = new TorusFourierKernel(2, 8);
            TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood("landingDiagnosticTest", tree,
                    null, precisions(1), observations, kernel, new Parameter.Default(0.001),
                    WEIGHTS, MEANS, COVARIANCES, new Parameter.Default(1.0e-6), JumpLaw.LANDING);
            assertTrue(Double.isFinite(like.getLogLikelihood()));
            assertTrue(like.getLandingNonpositiveCount() > 0);
            assertTrue(kernel.getLandingMinimumValue() < 0.0);
            assertTrue(kernel.getLandingNegativeMass() > 0.0);
            assertEquals(0L, kernel.getLandingClampCount());
        } finally {
            if (previous == null) System.clearProperty("torus.landing.strict");
            else System.setProperty("torus.landing.strict", previous);
        }
    }

    public void testLandingDiagnosticsCountWithoutClippingAndReset() {
        TorusFourierKernel kernel = new TorusFourierKernel(1, 8);
        double[] values = kernel.leafKernelLanding(new double[]{0.3}, new double[][]{{1.0e-6}});
        long nonpositive = 0, negative = 0;
        double minimum = Double.POSITIVE_INFINITY, negativeMass = 0.0;
        for (double value : values) {
            if (value <= 0.0) nonpositive++;
            if (value < 0.0) { negative++; negativeMass -= value; }
            minimum = Math.min(minimum, value);
        }
        assertTrue("an intentionally under-resolved tip must retain its negative lobes", negative > 0);
        assertEquals(nonpositive, kernel.getLandingNonpositiveCount());
        assertEquals(negative, kernel.getLandingNegativeCount());
        assertEquals(minimum, kernel.getLandingMinimumValue(), 0.0);
        assertEquals(negativeMass, kernel.getLandingNegativeMass(), 1.0e-14);
        assertEquals(0L, kernel.getLandingClampCount());
        kernel.resetLandingNonpositiveCount();
        assertEquals(0L, kernel.getLandingNonpositiveCount());
        assertEquals(0L, kernel.getLandingNegativeCount());
        assertEquals(0.0, kernel.getLandingNegativeMass(), 0.0);
    }

    private static TorusFourierTreeLikelihood likelihood(String newick, int n, double rate, JumpLaw law,
                                                         String missing) throws Exception {
        Tree tree = tree(newick);
        return build(tree, n, precisions(1), data(tree, 1, missing), new Parameter.Default(rate),
                new Parameter.Default(0.04), law);
    }

    private static TorusFourierTreeLikelihood build(Tree tree, int n, List<MatrixParameter> precisions,
                                                    double[][][] data, Parameter rate, Parameter eps, JumpLaw law) {
        return new TorusFourierTreeLikelihood("landingTest", tree, null, precisions, data,
                new TorusFourierKernel(2, n), rate, WEIGHTS, MEANS, COVARIANCES, eps, law);
    }

    private static DefaultTreeModel tree(String newick) throws Exception {
        return new DefaultTreeModel("tree", new NewickImporter(newick).importTree(null));
    }

    private static double[][][] data(Tree tree, int residues, String missing) {
        double[][][] data = new double[residues][tree.getExternalNodeCount()][];
        for (int r = 0; r < residues; r++) {
            for (int e = 0; e < tree.getExternalNodeCount(); e++) {
                NodeRef node = tree.getExternalNode(e);
                String name = tree.getNodeTaxon(node).getId();
                data[r][node.getNumber()] = "*".equals(missing) || name.equals(missing) ? null
                        : "A".equals(name) ? new double[]{0.4, 1.1}
                        : "B".equals(name) ? new double[]{-1.2, 0.6} : new double[]{0.9, -0.7};
            }
        }
        return data;
    }

    private static List<MatrixParameter> precisions(int count) {
        List<MatrixParameter> result = new ArrayList<MatrixParameter>();
        double determinant = SIGMA[0][0] * SIGMA[1][1] - SIGMA[0][1] * SIGMA[1][0];
        for (int i = 0; i < count; i++) {
            result.add(new MatrixParameter("precision" + i, new Parameter[]{
                    new Parameter.Default(new double[]{SIGMA[1][1] / determinant, -SIGMA[0][1] / determinant}),
                    new Parameter.Default(new double[]{-SIGMA[1][0] / determinant, SIGMA[0][0] / determinant})}));
        }
        return result;
    }

    private static void assertArrayEquals(double[] expected, double[] actual, double tolerance) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) assertEquals("grid cell " + i, expected[i], actual[i], tolerance);
    }
}
