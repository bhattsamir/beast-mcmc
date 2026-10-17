package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evolution.tree.TreeTrait;
import dr.evomodel.tree.DefaultTreeModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.evomodel.treedatalikelihood.continuous.TorusLandingPruning;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import dr.math.MathUtils;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.List;

/** Positive landing-reference and ancestral-sampling tests, all using grids at most 128. */
public class TorusLandingAncestralTest extends TestCase {
    private static final String NEWICK = "((A:1.0,B:1.0):1.0,C:2.0);";
    private static final double SIGMA = 0.45;
    private static final double OMEGA = 0.7;
    private static final double MEAN = 0.8;
    private static final double EPSILON = 0.12;
    private static final double RATE = 0.9;
    private static final double TWO_PI = 2.0 * Math.PI;
    private static final int DRAWS = 20000;

    public void testTreeTraitDrawsAreCachedAndSeedReproducible() throws Exception {
        Tree tree = tree();
        TorusFourierTreeLikelihood like = likelihood(tree, 1, new Parameter.Default(RATE),
                new Parameter.Default(EPSILON), false);
        MathUtils.setSeed(71023);
        double[][][] first = traits(like, tree, 1);
        TreeTrait trait = like.getTreeTrait("theta_0");
        Object cached = trait.getTrait(tree, tree.getRoot());
        assertSame(cached, trait.getTrait(tree, tree.getRoot()));
        assertAnglesEqual(first, traits(like, tree, 1));
        for (int tip = 0; tip < tree.getExternalNodeCount(); tip++) {
            assertNull(trait.getTrait(tree, tree.getExternalNode(tip)));
        }
        like.makeDirty();
        MathUtils.setSeed(71023);
        assertAnglesEqual(first, traits(like, tree, 1));
    }

    /** A nonzero basin mean makes sine moments sign-sensitive and distinguishes a reset from a shift. */
    public void testMissingDataRootAndChildHaveDirectedStationaryMoments() throws Exception {
        Tree tree = tree();
        TorusLandingPruning.Result result = reference(tree, RATE, EPSILON, true, 128);
        assertEquals(0.0, result.logLikelihood, 1.0e-7);
        assertTrue(result.gridSize <= 128);
        double magnitude = RATE / (RATE + 0.5 * SIGMA) * Math.exp(-0.5 * OMEGA);
        double expectedCos = magnitude * Math.cos(MEAN);
        double expectedSin = magnitude * Math.sin(MEAN);
        double[][] moments = moments(result, tree, 731);
        for (NodeRef node : new NodeRef[]{tree.getRoot(), internalChild(tree)}) {
            assertEquals(expectedCos, moments[node.getNumber()][0], 0.025);
            assertEquals(expectedSin, moments[node.getNumber()][1], 0.025);
            assertTrue("the absolute landing mean must orient the stationary law", moments[node.getNumber()][1] > 0.25);
        }
    }

    public void testZeroRateRootAndChildAreUniform() throws Exception {
        Tree tree = tree();
        TorusLandingPruning.Result result = reference(tree, 0.0, EPSILON, true, 128);
        assertEquals(0.0, result.logLikelihood, 1.0e-7);
        double[][] moments = moments(result, tree, 9143);
        for (NodeRef node : new NodeRef[]{tree.getRoot(), internalChild(tree)}) {
            assertEquals(0.0, moments[node.getNumber()][0], 0.025);
            assertEquals(0.0, moments[node.getNumber()][1], 0.025);
        }
    }

    /** Independent explicit joint integration: no pruning and no reference-backend quadrature calls. */
    public void testConditionalMomentsAndLikelihoodAgainstDirectJointQuadrature() throws Exception {
        Tree tree = tree();
        TorusLandingPruning.Result result = reference(tree, RATE, EPSILON, false, 128);
        double[] expected = jointReference();
        assertEquals(expected[0], result.logLikelihood, 1.0e-6);
        assertTrue(result.gridSize <= 128);
        double[][] sampled = moments(result, tree, 43389);
        int root = tree.getRoot().getNumber(), child = internalChild(tree).getNumber();
        assertEquals(expected[1], sampled[root][0], 0.025);
        assertEquals(expected[2], sampled[root][1], 0.025);
        assertEquals(expected[3], sampled[child][0], 0.025);
        assertEquals(expected[4], sampled[child][1], 0.025);
    }

    /**
     * The PRODUCTION ancestral path -- the FFT downward sampler reached through getTreeTrait --
     * against direct joint quadrature. The dense-reference test above calls TorusLandingPruning
     * directly and so never exercises the FFT sampler; without this test a sampler that drew
     * from a reflected landing law (conj(ghat) omitted, 2026-10-01) passed every suite while
     * missing the posterior means by 0.52 rad.
     */
    public void testFftAncestralSamplerMatchesDirectJointQuadrature() throws Exception {
        final String previous = System.getProperty("torus.ancestral.reference");
        System.clearProperty("torus.ancestral.reference");
        try {
            Tree tree = tree();
            TorusFourierTreeLikelihood like = likelihood(tree, 1, new Parameter.Default(RATE),
                    new Parameter.Default(EPSILON), false);
            double[] expected = jointReference();
            assertEquals(expected[0], like.getLogLikelihood(), 1.0e-6);
            TreeTrait trait = like.getTreeTrait("theta_0");
            int root = tree.getRoot().getNumber(), child = internalChild(tree).getNumber();
            MathUtils.setSeed(97531L);
            double rc = 0.0, rs = 0.0, cc = 0.0, cs = 0.0;
            for (int draw = 0; draw < DRAWS; draw++) {
                like.forceAncestralResample();
                double[] r = (double[]) trait.getTrait(tree, tree.getNode(root));
                double[] c = (double[]) trait.getTrait(tree, tree.getNode(child));
                assertNotNull(r); assertNotNull(c);
                rc += Math.cos(r[0]) / DRAWS; rs += Math.sin(r[0]) / DRAWS;
                cc += Math.cos(c[0]) / DRAWS; cs += Math.sin(c[0]) / DRAWS;
            }
            assertEquals("root E[cos]", expected[1], rc, 0.025);
            assertEquals("root E[sin]", expected[2], rs, 0.025);
            assertEquals("child E[cos]", expected[3], cc, 0.025);
            assertEquals("child E[sin]", expected[4], cs, 0.025);
        } finally {
            if (previous == null) System.clearProperty("torus.ancestral.reference");
            else System.setProperty("torus.ancestral.reference", previous);
        }
    }

    public void testObservationErrorUpdateAndStoreRestoreRefreshAncestralCache() throws Exception {
        Tree tree = tree();
        Parameter rate = new Parameter.Default(RATE), eps = new Parameter.Default(EPSILON);
        TorusFourierTreeLikelihood like = likelihood(tree, 1, rate, eps, false);
        double originalLogL = like.getLogLikelihood();
        MathUtils.setSeed(8021);
        double[][][] original = traits(like, tree, 1);
        like.storeModelState();
        eps.setParameterValue(0, 0.5);
        assertTrue(Math.abs(like.getLogLikelihood() - originalLogL) > 1.0e-5);
        MathUtils.setSeed(9131);
        double[][][] updated = traits(like, tree, 1);
        Tree freshTree = tree();
        TorusFourierTreeLikelihood fresh = likelihood(freshTree, 1, new Parameter.Default(RATE),
                new Parameter.Default(0.5), false);
        MathUtils.setSeed(9131);
        assertAnglesEqual(updated, traits(fresh, freshTree, 1));
        like.restoreModelState();
        assertEquals(originalLogL, like.getLogLikelihood(), 0.0);
        MathUtils.setSeed(8021);
        assertAnglesEqual(original, traits(like, tree, 1));
    }

    public void testMultipleResiduesUseTheirOwnRatesAndRefreshAfterChange() throws Exception {
        Tree tree = tree();
        Parameter rates = new Parameter.Default(new double[]{0.2, RATE});
        TorusFourierTreeLikelihood like = likelihood(tree, 2, rates, new Parameter.Default(EPSILON), false);
        MathUtils.setSeed(10071);
        traits(like, tree, 2);
        rates.setParameterValue(1, 1.4);
        MathUtils.setSeed(12873);
        double[][][] changed = traits(like, tree, 2);
        Tree freshTree = tree();
        TorusFourierTreeLikelihood fresh = likelihood(freshTree, 2,
                new Parameter.Default(new double[]{0.2, 1.4}), new Parameter.Default(EPSILON), false);
        assertEquals(fresh.getLogLikelihood(), like.getLogLikelihood(), 1.0e-12);
        MathUtils.setSeed(12873);
        assertAnglesEqual(changed, traits(fresh, freshTree, 2));
        double expected = reference(tree, 0.2, EPSILON, false, 128).logLikelihood
                + reference(tree, 1.4, EPSILON, false, 128).logLikelihood;
        assertEquals(expected, like.getLogLikelihood(), 1.0e-6);
    }

    public void testPositiveReferenceFailsExplicitlyWhenGridCapCannotResolveBranches() throws Exception {
        Tree tree = tree();
        try {
            TorusLandingPruning.calculate(tree, child -> branchLength(tree, child), observations(tree, false),
                    new double[][]{{1.0e-6}}, RATE, new double[]{1.0}, new double[][]{{MEAN}},
                    new double[][][]{{{OMEGA}}}, new double[][]{{1.0e-6}}, 8);
            fail("the reference must respect the supplied grid cap and reject unresolved quadrature");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().toLowerCase().contains("accuracy")
                    || expected.getMessage().toLowerCase().contains("grid"));
        }
    }

    private static TorusLandingPruning.Result reference(Tree tree, double rate, double eps,
                                                        boolean missing, int maxGrid) {
        return TorusLandingPruning.calculate(tree, child -> branchLength(tree, child), observations(tree, missing),
                new double[][]{{SIGMA}}, rate, new double[]{1.0}, new double[][]{{MEAN}},
                new double[][][]{{{OMEGA}}}, new double[][]{{eps}}, maxGrid);
    }

    private static double branchLength(Tree tree, NodeRef child) {
        return tree.getNodeHeight(tree.getParent(child)) - tree.getNodeHeight(child);
    }

    private static DefaultTreeModel tree() throws Exception {
        return new DefaultTreeModel("ancestralTree", new NewickImporter(NEWICK).importTree(null));
    }

    private static NodeRef internalChild(Tree tree) {
        for (int c = 0; c < tree.getChildCount(tree.getRoot()); c++) {
            NodeRef child = tree.getChild(tree.getRoot(), c);
            if (!tree.isExternal(child)) return child;
        }
        throw new AssertionError("fixture requires an internal child");
    }

    private static double[][] observations(Tree tree, boolean missing) {
        double[][] observations = new double[tree.getExternalNodeCount()][];
        if (missing) return observations;
        for (int tip = 0; tip < tree.getExternalNodeCount(); tip++) {
            NodeRef node = tree.getExternalNode(tip);
            String name = tree.getNodeTaxon(node).getId();
            observations[node.getNumber()] = new double[]{"A".equals(name) ? 0.1 : "B".equals(name) ? -0.5 : 1.0};
        }
        return observations;
    }

    private static TorusFourierTreeLikelihood likelihood(Tree tree, int residues, Parameter rate,
                                                         Parameter eps, boolean missing) {
        List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        double[][][] data = new double[residues][][];
        for (int r = 0; r < residues; r++) {
            precisions.add(new MatrixParameter("precision" + r,
                    new Parameter[]{new Parameter.Default(new double[]{1.0 / SIGMA})}));
            data[r] = observations(tree, missing);
        }
        return new TorusFourierTreeLikelihood("landingASR", tree, null, precisions, data,
                new TorusFourierKernel(1, 128), rate, new double[]{1.0}, new double[][]{{MEAN}},
                new double[][][]{{{OMEGA}}}, eps, TorusFourierTreeLikelihood.JumpLaw.LANDING);
    }

    private static double[][][] traits(TorusFourierTreeLikelihood like, Tree tree, int residues) {
        double[][][] out = new double[residues][tree.getNodeCount()][];
        for (int r = 0; r < residues; r++) {
            TreeTrait trait = like.getTreeTrait("theta_" + r);
            for (int n = 0; n < tree.getNodeCount(); n++) {
                NodeRef node = tree.getNode(n);
                if (!tree.isExternal(node)) {
                    double[] angles = (double[]) trait.getTrait(tree, node);
                    assertNotNull(angles);
                    assertEquals(1, angles.length);
                    assertTrue(angles[0] >= -Math.PI && angles[0] <= Math.PI);
                    out[r][n] = angles.clone();
                }
            }
        }
        return out;
    }

    private static void assertAnglesEqual(double[][][] expected, double[][][] actual) {
        assertEquals(expected.length, actual.length);
        for (int r = 0; r < expected.length; r++) {
            assertEquals(expected[r].length, actual[r].length);
            for (int n = 0; n < expected[r].length; n++) {
                if (expected[r][n] == null) assertNull(actual[r][n]);
                else assertEquals("residue " + r + " node " + n, expected[r][n][0], actual[r][n][0], 0.0);
            }
        }
    }

    private static double[][] moments(TorusLandingPruning.Result result, Tree tree, long seed) {
        MathUtils.setSeed(seed);
        double[][] sums = new double[tree.getNodeCount()][2];
        double[][] output = new double[tree.getNodeCount()][];
        for (int draw = 0; draw < DRAWS; draw++) {
            result.sample(output);
            for (int n = 0; n < tree.getNodeCount(); n++) {
                if (!tree.isExternal(tree.getNode(n))) {
                    assertNotNull(output[n]);
                    assertTrue(Double.isFinite(output[n][0]));
                    sums[n][0] += Math.cos(output[n][0]) / DRAWS;
                    sums[n][1] += Math.sin(output[n][0]) / DRAWS;
                }
            }
        }
        return sums;
    }

    private static double wrappedNormal(double delta, double variance) {
        double sum = 0.0;
        for (int k = -8; k <= 8; k++) {
            double residual = delta + TWO_PI * k;
            sum += Math.exp(-0.5 * residual * residual / variance) / Math.sqrt(TWO_PI * variance);
        }
        return sum;
    }

    private static double stationary(double x) {
        double sum = 1.0;
        for (int m = 1; m <= 32; m++) {
            sum += 2.0 * Math.exp(-0.5 * OMEGA * m * m) * RATE / (RATE + 0.5 * SIGMA * m * m)
                    * Math.cos(m * (x - MEAN));
        }
        return sum / TWO_PI;
    }

    private static double resetDensity(double y, double t, double observationVariance) {
        double sum = -Math.expm1(-RATE * t);
        for (int m = 1; m <= 32; m++) {
            double q = RATE + 0.5 * SIGMA * m * m;
            sum += 2.0 * Math.exp(-0.5 * (OMEGA + observationVariance) * m * m)
                    * RATE / q * (-Math.expm1(-q * t)) * Math.cos(m * (y - MEAN));
        }
        return sum / TWO_PI;
    }

    private static double tipMessage(double observation, double parent, double t) {
        return Math.exp(-RATE * t) * wrappedNormal(observation - parent, SIGMA * t + EPSILON)
                + resetDensity(observation, t, EPSILON);
    }

    /** Return log normalizer, root cos/sin and internal-child cos/sin. */
    private static double[] jointReference() {
        int grid = 64;
        double[] sums = new double[5];
        for (int r = 0; r < grid; r++) {
            double root = TWO_PI * (r + 0.17) / grid;
            for (int c = 0; c < grid; c++) {
                double child = TWO_PI * (c + 0.43) / grid;
                double transition = Math.exp(-RATE) * wrappedNormal(child - root, SIGMA)
                        + resetDensity(child, 1.0, 0.0);
                double weight = stationary(root) * transition * tipMessage(0.1, child, 1.0)
                        * tipMessage(-0.5, child, 1.0) * tipMessage(1.0, root, 2.0);
                sums[0] += weight;
                sums[1] += weight * Math.cos(root);
                sums[2] += weight * Math.sin(root);
                sums[3] += weight * Math.cos(child);
                sums[4] += weight * Math.sin(child);
            }
        }
        for (int k = 1; k < sums.length; k++) sums[k] /= sums[0];
        sums[0] = Math.log(sums[0] * TWO_PI * TWO_PI / (grid * grid));
        return sums;
    }
}
