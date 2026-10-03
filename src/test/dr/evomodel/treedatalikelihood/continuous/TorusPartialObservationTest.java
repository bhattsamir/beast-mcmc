package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evomodel.tree.DefaultTreeModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.evomodel.treedatalikelihood.continuous.TorusLogSpacePruning;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import junit.framework.TestCase;

import java.util.Collections;

/** Partially observed tips must integrate missing coordinates, including their gradients. */
public class TorusPartialObservationTest extends TestCase {
    private static final double NA = Double.NaN;
    private static final double LOG_TWO_PI = Math.log(2.0 * Math.PI);
    private static final String CHERRY = "(A:1.0,B:1.0);";
    private static final String THREE_TIPS = "((A:0.8,B:0.8):0.6,C:1.4);";
    private static final double[][] SIGMA = {{0.8, 0.35}, {0.35, 1.3}};
    private static final double[] WEIGHTS = {0.65, 0.35};
    private static final double[][] DISPLACEMENTS = {{0.0, 0.0}, {1.2, -0.7}};
    private static final double[][][] JUMP_COVARIANCES = {
            {{0.45, 0.08}, {0.08, 0.6}}, {{0.7, -0.1}, {-0.1, 0.5}}};
    private static final double RATE = 0.25;
    private boolean previousValidation;

    protected void setUp() {
        previousValidation = TorusFourierTreeLikelihood.isValidateEveryEvaluation();
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(false);
    }

    protected void tearDown() {
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(previousValidation);
    }

    public void testPhiOnlyTreeEqualsMarginalOneDimensionalJumpModel() throws Exception {
        assertOneCoordinateTreeMatchesProjection(0);
    }

    public void testPsiOnlyTreeEqualsMarginalOneDimensionalJumpModel() throws Exception {
        assertOneCoordinateTreeMatchesProjection(1);
    }

    private static void assertOneCoordinateTreeMatchesProjection(int axis) throws Exception {
        double[][] complete = {{0.4, 1.1}, {-1.2, 0.6}, {0.9, -0.3}};
        double[][] partial = observeOnly(complete, axis);
        double actual = likelihood(THREE_TIPS, partial, SIGMA, true).getLogLikelihood();
        double expected = projectedLikelihood(THREE_TIPS, partial, SIGMA, axis, true).getLogLikelihood();
        assertEquals("unobserved axis integrates out of the whole tree", expected, actual, 2e-10);
        // Sigma_aa is the marginal variance. 1/Q_aa would instead be a conditional variance.
        double[][] wrongCovariance = {{1.0 / inverse(SIGMA)[axis][axis]}};
        TorusFourierTreeLikelihood wrong = likelihoodWithComponents(THREE_TIPS,
                projectObservations(partial, axis), wrongCovariance, RATE, WEIGHTS,
                projectDisplacements(axis), projectJumpCovariances(axis));
        assertTrue("fixture must distinguish marginalization from conditioning",
                Math.abs(expected - wrong.getLogLikelihood()) > 1e-3);
    }

    public void testMixedObservationPatternsFactorForIndependentBrownianAxes() throws Exception {
        String tree = "((A:0.8,B:0.8):0.6,(C:1.0,D:1.0):0.4);";
        double[][] partial = {{0.4, NA}, {NA, -0.6}, {1.1, 0.2}, null};
        double[][] diagonal = {{0.8, 0}, {0, 1.3}};
        double actual = likelihood(tree, partial, diagonal, false).getLogLikelihood();
        double expected = projectedLikelihood(tree, partial, diagonal, 0, false).getLogLikelihood()
                + projectedLikelihood(tree, partial, diagonal, 1, false).getLogLikelihood();
        assertEquals("independent-axis marginal likelihoods multiply", expected, actual, 2e-10);
    }

    public void testAllNaNTipsAreNeutralInProductionAndGradient() throws Exception {
        double[][] partial = {{NA, NA}, null, {NA, NA}};
        Parameter diagonal = new Parameter.Default(new double[]{1.4, 0.9});
        Parameter correlation = new Parameter.Default(new double[]{0.3});
        Parameter rate = new Parameter.Default(RATE);
        TorusFourierTreeLikelihood like = gradientLikelihood(partial, diagonal, correlation, rate);
        assertEquals(0.0, like.getLogLikelihood(), 1e-12);
        for (double gradient : like.getResidueGradient(0)) assertEquals(0.0, gradient, 1e-12);
    }

    public void testPartialTipTreeGradientMatchesFivePointFiniteDifferences() throws Exception {
        double[][] partial = {{0.4, NA}, {NA, -0.6}, {1.1, 0.2}};
        Parameter diagonal = new Parameter.Default(new double[]{1.4, 0.9});
        Parameter correlation = new Parameter.Default(new double[]{-0.4});
        Parameter rate = new Parameter.Default(RATE);
        TorusFourierTreeLikelihood like = gradientLikelihood(partial, diagonal, correlation, rate);
        double initial = like.getLogLikelihood();
        double[] analytic = like.getResidueGradient(0);
        Parameter[] parameters = {diagonal, diagonal, correlation, rate};
        int[] coordinates = {0, 1, 0, 0};
        int[] offsets = {-2, -1, 1, 2};
        for (int p = 0; p < parameters.length; p++) {
            Parameter parameter = parameters[p];
            int coordinate = coordinates[p];
            double value = parameter.getParameterValue(coordinate);
            double h = 1e-5 * Math.max(1, Math.abs(value));
            double[] evaluations = new double[4];
            try {
                for (int step = 0; step < offsets.length; step++) {
                    parameter.setParameterValue(coordinate, value + h * offsets[step]);
                    evaluations[step] = like.getLogLikelihood();
                }
            } finally {
                parameter.setParameterValue(coordinate, value);
            }
            double numeric = (evaluations[0] - 8 * evaluations[1] + 8 * evaluations[2]
                    - evaluations[3]) / (12 * h);
            assertEquals("partial observation derivative [q1,q2,rho,lambda] index=" + p,
                    numeric, analytic[p], 2e-6 * Math.max(1, Math.abs(numeric)));
        }
        assertEquals(initial, like.getLogLikelihood(), 1e-12);
    }

    public void testReferencePhiOnlyJumpModelEqualsProjectedReference() throws Exception {
        assertReferenceMatchesProjection(0, true);
    }

    public void testReferencePsiOnlyBrownianModelEqualsProjectedReference() throws Exception {
        assertReferenceMatchesProjection(1, false);
    }

    private static void assertReferenceMatchesProjection(int axis, boolean jumps) throws Exception {
        double[][] partial = observeOnly(new double[][]{{0.4, 1.1}, {-1.2, 0.6}}, axis);
        double actual = reference(CHERRY, partial, SIGMA, jumps ? RATE : 0,
                jumps ? WEIGHTS : null, jumps ? DISPLACEMENTS : null, jumps ? JUMP_COVARIANCES : null);
        double expected = reference(CHERRY, projectObservations(partial, axis),
                new double[][]{{SIGMA[axis][axis]}}, jumps ? RATE : 0,
                jumps ? WEIGHTS : null, jumps ? projectDisplacements(axis) : null,
                jumps ? projectJumpCovariances(axis) : null);
        assertEquals(expected, actual, 2e-7);
    }

    public void testDifferentObservedAxesHaveAnalyticHaarLikelihood() throws Exception {
        // A observes phi and B observes psi once each. Independent uniform root
        // coordinates make this density (2pi)^-2 even for correlated increments.
        double[][] partial = {{0.4, NA}, {NA, -0.6}};
        assertEquals(-2 * LOG_TWO_PI, likelihood(CHERRY, partial, SIGMA, true).getLogLikelihood(), 2e-10);
        assertEquals(-2 * LOG_TWO_PI,
                reference(CHERRY, partial, SIGMA, RATE, WEIGHTS, DISPLACEMENTS, JUMP_COVARIANCES), 2e-7);
        double[][] oneObserved = {{0.4, NA}, {NA, NA}};
        assertEquals(-LOG_TWO_PI,
                reference(CHERRY, oneObserved, SIGMA, 0, null, null, null), 2e-7);
    }

    public void testAllNaNObservationsAreNeutralInReference() throws Exception {
        assertEquals(0.0, reference(CHERRY, new double[][]{{NA, NA}, {NA, NA}}, SIGMA,
                RATE, WEIGHTS, DISPLACEMENTS, JUMP_COVARIANCES), 1e-12);
    }

    private static TorusFourierTreeLikelihood gradientLikelihood(double[][] observations,
            Parameter diagonal, Parameter correlation, Parameter rate) throws Exception {
        DefaultTreeModel tree = tree(THREE_TIPS);
        MatrixParameter precision = new CompoundSymmetricMatrix(diagonal, correlation, true, false);
        return new TorusFourierTreeLikelihood("partialGradient", tree, null,
                Collections.singletonList(precision), new double[][][]{reorder(tree, observations)},
                new TorusFourierKernel(2, 32), rate, WEIGHTS, DISPLACEMENTS, JUMP_COVARIANCES);
    }

    private static TorusFourierTreeLikelihood likelihood(String newick, double[][] observations,
            double[][] sigma, boolean jumps) throws Exception {
        return likelihoodWithComponents(newick, observations, sigma, jumps ? RATE : 0,
                jumps ? WEIGHTS : null, jumps ? DISPLACEMENTS : null, jumps ? JUMP_COVARIANCES : null);
    }

    private static TorusFourierTreeLikelihood projectedLikelihood(String newick, double[][] observations,
            double[][] sigma, int axis, boolean jumps) throws Exception {
        return likelihoodWithComponents(newick, projectObservations(observations, axis),
                new double[][]{{sigma[axis][axis]}}, jumps ? RATE : 0,
                jumps ? WEIGHTS : null, jumps ? projectDisplacements(axis) : null,
                jumps ? projectJumpCovariances(axis) : null);
    }

    private static TorusFourierTreeLikelihood likelihoodWithComponents(String newick, double[][] observations,
            double[][] sigma, double rate, double[] weights, double[][] displacements,
            double[][][] jumpCovariances) throws Exception {
        DefaultTreeModel tree = tree(newick);
        double[][] precisionValues = inverse(sigma);
        Parameter[] columns = new Parameter[sigma.length];
        for (int j = 0; j < columns.length; j++) {
            double[] column = new double[columns.length];
            for (int i = 0; i < column.length; i++) column[i] = precisionValues[i][j];
            columns[j] = new Parameter.Default(column);
        }
        MatrixParameter precision = new MatrixParameter("precision", columns);
        return new TorusFourierTreeLikelihood("partial", tree, null, Collections.singletonList(precision),
                new double[][][]{reorder(tree, observations)}, new TorusFourierKernel(sigma.length, 32),
                new Parameter.Default(rate), weights, displacements, jumpCovariances);
    }

    private static double reference(String newick, double[][] observations, double[][] sigma, double rate,
            double[] weights, double[][] displacements, double[][][] jumpCovariances) throws Exception {
        DefaultTreeModel tree = tree(newick);
        return TorusLogSpacePruning.calculate(tree,
                child -> tree.getNodeHeight(tree.getParent(child)) - tree.getNodeHeight(child),
                reorder(tree, observations), sigma, rate, weights, displacements, jumpCovariances).logLikelihood;
    }

    private static DefaultTreeModel tree(String newick) throws Exception {
        return new DefaultTreeModel("tree", new NewickImporter(newick).importTree(null));
    }

    private static double[][] reorder(DefaultTreeModel tree, double[][] observations) {
        double[][] reordered = new double[tree.getExternalNodeCount()][];
        for (int tip = 0; tip < reordered.length; tip++) {
            NodeRef node = tree.getExternalNode(tip);
            int index = "ABCD".indexOf(tree.getNodeTaxon(node).getId());
            reordered[node.getNumber()] = observations[index];
        }
        return reordered;
    }

    private static double[][] observeOnly(double[][] observations, int axis) {
        double[][] result = new double[observations.length][2];
        for (int tip = 0; tip < result.length; tip++) {
            result[tip][axis] = observations[tip][axis];
            result[tip][1 - axis] = NA;
        }
        return result;
    }

    private static double[][] projectObservations(double[][] observations, int axis) {
        double[][] result = new double[observations.length][];
        for (int tip = 0; tip < result.length; tip++) {
            if (observations[tip] != null && !Double.isNaN(observations[tip][axis])) {
                result[tip] = new double[]{observations[tip][axis]};
            }
        }
        return result;
    }

    private static double[][] projectDisplacements(int axis) {
        double[][] result = new double[DISPLACEMENTS.length][1];
        for (int component = 0; component < result.length; component++) result[component][0] = DISPLACEMENTS[component][axis];
        return result;
    }

    private static double[][][] projectJumpCovariances(int axis) {
        double[][][] result = new double[JUMP_COVARIANCES.length][1][1];
        for (int component = 0; component < result.length; component++) {
            result[component][0][0] = JUMP_COVARIANCES[component][axis][axis];
        }
        return result;
    }

    private static double[][] inverse(double[][] matrix) {
        if (matrix.length == 1) return new double[][]{{1.0 / matrix[0][0]}};
        double det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        return new double[][]{{matrix[1][1] / det, -matrix[0][1] / det},
                {-matrix[1][0] / det, matrix[0][0] / det}};
    }
}
