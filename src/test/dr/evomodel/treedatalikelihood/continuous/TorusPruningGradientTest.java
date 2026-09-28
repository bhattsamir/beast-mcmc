package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evomodel.tree.DefaultTreeModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.List;

/** Finite differences of the actual production likelihood, including internal pruning. */
public class TorusPruningGradientTest extends TestCase {
    private boolean previousValidation;

    protected void setUp() {
        previousValidation = TorusFourierTreeLikelihood.isValidateEveryEvaluation();
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(false);
    }

    protected void tearDown() {
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(previousValidation);
    }

    public void testBrownianRawPrecisionGradient() throws Exception {
        Fixture fixture = new Fixture(false, false, new double[][]{{1.4, 0.9, 0.3}}, 0);
        assertGradientMatchesLikelihood(fixture, 0);
        assertEquals("absent jump law has no rate derivative", 0.0,
                fixture.like.getResidueGradient(0)[3], 1e-12);
    }

    public void testDirectedMixtureRawGradient() throws Exception {
        Fixture fixture = new Fixture(true, false, new double[][]{{2.1, 0.7, -0.55}}, 0);
        assertGradientMatchesLikelihood(fixture, 0);
    }

    public void testAnisotropicDiffusionWithStrongCorrelation() throws Exception {
        Fixture fixture = new Fixture(true, false, new double[][]{{7.0, 0.7, 0.65}}, 0);
        assertGradientMatchesLikelihood(fixture, 0);
    }

    public void testMissingTipGradient() throws Exception {
        Fixture fixture = new Fixture(true, false, new double[][]{{1.6, 1.1, -0.4}}, 1 << 2);
        assertGradientMatchesLikelihood(fixture, 0);
    }

    public void testEntireMissingSubtreeGradient() throws Exception {
        Fixture fixture = new Fixture(true, false, new double[][]{{1.6, 1.1, -0.4}}, (1 << 2) | (1 << 3));
        assertGradientMatchesLikelihood(fixture, 0);
    }

    public void testEntirelyMissingResidueIsNeutral() throws Exception {
        Fixture fixture = new Fixture(true, false, new double[][]{{1.6, 1.1, -0.4}}, 15);
        assertEquals(0.0, fixture.like.getLogLikelihood(), 1e-12);
        for (double gradient : fixture.like.getResidueGradient(0)) {
            assertEquals("marginalized observations cannot inform a parameter", 0.0, gradient, 1e-12);
        }
    }

    public void testPerResidueRateAndPrecisionGradients() throws Exception {
        Fixture fixture = new Fixture(true, false,
                new double[][]{{1.4, 0.9, 0.3}, {2.1, 0.7, -0.55}}, 0);
        assertGradientMatchesLikelihood(fixture, 0);
        assertGradientMatchesLikelihood(fixture, 1);
    }

    public void testSharedRateGradientSumsAcrossResidues() throws Exception {
        Fixture fixture = new Fixture(true, true,
                new double[][]{{1.4, 0.9, 0.3}, {2.1, 0.7, -0.55}}, 0);
        double sum = fixture.like.getResidueGradient(0)[3] + fixture.like.getResidueGradient(1)[3];
        double numeric = finiteDifference(fixture, 0, 3);
        assertEquals("shared lambda affects all residues", numeric, sum, tolerance(numeric));
    }

    public void testGradientSweepPreservesLikelihoodCache() throws Exception {
        Fixture fixture = new Fixture(true, false, new double[][]{{1.4, 0.9, 0.3}}, 0);
        double original = fixture.like.getLogLikelihood();
        fixture.like.resetRecomputeCount();
        double[] first = fixture.like.getResidueGradient(0);
        double[] second = fixture.like.getResidueGradient(0);
        for (int p = 0; p < 4; p++) assertEquals(first[p], second[p], 0.0);
        assertEquals(original, fixture.like.getLogLikelihood(), 0.0);
        assertEquals("read-only gradient must preserve reusable partials", 0L,
                fixture.like.getNodeRecomputeCount());
    }

    public void testReferenceValidationCannotUseFftGradient() throws Exception {
        Fixture fixture = new Fixture(true, false, new double[][]{{1.4, 0.9, 0.3}}, 0);
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(true);
        try {
            fixture.like.getResidueGradient(0);
            fail("an FFT gradient cannot represent the adaptive reference target");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage().contains("torus.validate=false"));
        }
    }

    public void testRejectedGradientTrajectoryReturnsOriginalGradient() throws Exception {
        Fixture fixture = new Fixture(true, false, new double[][]{{1.4, 0.9, 0.3}}, 0);
        double original = fixture.like.getLogLikelihood();
        double[] gradient = fixture.like.getResidueGradient(0);
        fixture.like.storeModelState();
        for (int step = 0; step < 4; step++) {
            fixture.diagonal[0].setParameterValue(0, 0.7 + 0.13 * step);
            fixture.correlation[0].setParameterValue(0, -0.3 + 0.06 * step);
            fixture.rate.setParameterValue(0, 0.6 + 0.08 * step);
            fixture.like.getResidueGradient(0);
            fixture.like.getLogLikelihood();
        }
        fixture.like.restoreModelState();
        assertEquals(original, fixture.like.getLogLikelihood(), 1e-12);
        double[] restored = fixture.like.getResidueGradient(0);
        for (int p = 0; p < 4; p++) assertEquals(gradient[p], restored[p], 1e-12);
    }

    private static void assertGradientMatchesLikelihood(Fixture fixture, int residue) {
        double original = fixture.like.getLogLikelihood();
        double[] analytic = fixture.like.getResidueGradient(residue);
        assertEquals(4, analytic.length);
        for (int p = 0; p < 4; p++) {
            double numeric = finiteDifference(fixture, residue, p);
            assertEquals("residue=" + residue + " derivative=" + p + " [q1,q2,rho,lambda]",
                    numeric, analytic[p], tolerance(numeric));
        }
        assertEquals("finite differences restore their initial state", original,
                fixture.like.getLogLikelihood(), 1e-12);
    }

    private static double tolerance(double expected) {
        return 2e-6 * Math.max(1.0, Math.abs(expected));
    }

    private static double finiteDifference(Fixture fixture, int residue, int coordinate) {
        Parameter parameter = coordinate < 2 ? fixture.diagonal[residue]
                : coordinate == 2 ? fixture.correlation[residue] : fixture.rate;
        int index = coordinate < 2 ? coordinate
                : coordinate == 2 || fixture.sharedRate ? 0 : residue;
        double value = parameter.getParameterValue(index);
        double h = 1e-5 * Math.max(1.0, Math.abs(value));
        double[] evaluated = new double[4];
        int[] offsets = {-2, -1, 1, 2};
        try {
            for (int i = 0; i < offsets.length; i++) {
                parameter.setParameterValue(index, value + offsets[i] * h);
                evaluated[i] = fixture.like.getLogLikelihood();
            }
        } finally {
            parameter.setParameterValue(index, value);
        }
        return (evaluated[0] - 8 * evaluated[1] + 8 * evaluated[2] - evaluated[3]) / (12 * h);
    }

    private static final class Fixture {
        final Parameter[] diagonal;
        final Parameter[] correlation;
        final Parameter rate;
        final boolean sharedRate;
        final TorusFourierTreeLikelihood like;

        Fixture(boolean jumps, boolean sharedRate, double[][] precisionValues, int missingMask) throws Exception {
            this.sharedRate = sharedRate;
            DefaultTreeModel tree = new DefaultTreeModel("tree", new NewickImporter(
                    "((A:0.8,B:0.8):0.6,(C:1.2,D:1.2):0.2);").importTree(null));
            int residues = precisionValues.length;
            diagonal = new Parameter[residues];
            correlation = new Parameter[residues];
            double[] rates = new double[sharedRate ? 1 : residues];
            for (int r = 0; r < rates.length; r++) rates[r] = 0.35 + 0.12 * r;
            rate = new Parameter.Default(rates);
            List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
            double[][][] data = new double[residues][tree.getExternalNodeCount()][];
            double[][] observed = {{0.4, 1.1}, {-1.2, 0.6}, {0.9, -0.3}, {-0.7, -1.0}};
            for (int r = 0; r < residues; r++) {
                diagonal[r] = new Parameter.Default(new double[]{precisionValues[r][0], precisionValues[r][1]});
                correlation[r] = new Parameter.Default(new double[]{precisionValues[r][2]});
                precisions.add(new CompoundSymmetricMatrix(diagonal[r], correlation[r], true, false));
                for (int tip = 0; tip < tree.getExternalNodeCount(); tip++) {
                    NodeRef node = tree.getExternalNode(tip);
                    int taxon = "ABCD".indexOf(tree.getNodeTaxon(node).getId());
                    if ((missingMask & (1 << taxon)) == 0) {
                        data[r][node.getNumber()] = new double[]{observed[taxon][0] + 0.2 * r,
                                observed[taxon][1] - 0.1 * r};
                    }
                }
            }
            like = new TorusFourierTreeLikelihood("gradient", tree, null, precisions, data,
                    new TorusFourierKernel(2, 32), rate,
                    jumps ? new double[]{0.65, 0.35} : null,
                    jumps ? new double[][]{{0, 0}, {1.2, -0.7}} : null,
                    jumps ? new double[][][]{{{0.35, 0.04}, {0.04, 0.45}}, {{0.45, 0}, {0, 0.35}}} : null);
        }
    }
}
