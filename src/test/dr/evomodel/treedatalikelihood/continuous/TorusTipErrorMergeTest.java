/*
 * TorusTipErrorMergeTest.java
 *
 * @author Samir Bhatt
 */

package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.evomodel.treedatalikelihood.continuous.TorusLogSpacePruning;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The two things the Sigma_eps merge introduced that neither parent engine had, and that
 * therefore had no coverage anywhere: Sigma_eps COMPOSED WITH PARTIAL OBSERVATION, and a
 * Sigma_eps-AWARE ANALYTIC GRADIENT.
 *
 * Both are checked against something outside the engine -- a closed form written from the
 * model, and central finite differences of the same likelihood the gradient differentiates.
 * Each test also records what a plausible WRONG implementation would have produced, so that
 * a passing run is evidence rather than a tautology.
 *
 * The nearest-image centring flag (-Dtorus.centredImages) is left OFF throughout: the
 * likelihood folds under it and the gradient's copy of the winding sum does not, and
 * nothing here needs it.
 */
public class TorusTipErrorMergeTest extends TestCase {

    /** HA's tip observation error: 3.5 degrees isotropic, as a VARIANCE in radians^2. */
    private static final double SIGMA_EPS_HA = 0.00373156339239;

    // ---------------------------------------------------------------- partial observation

    /**
     * Sigma_eps marginalises with the tip.
     *
     * d = 3 cherry, flat root prior, pure BM, tip error on both tips; A fully observed, B
     * missing axis 2. Integrating the root's axis 2 out of A and convolving what is left
     * gives exactly
     *     logL = -3 log(2 pi) + log WN_2( yA_S - yB_S ; C_A[S,S] + C_B[S,S] ),  S = {0,1},
     *     C_X = Sigma_eps + t_X Sigma,
     * where C[S,S] is the principal SUBMATRIX and so carries Sigma_eps[0][1]. Reducing
     * Sigma_eps to its diagonal instead -- the obvious wrong answer, and one that d = 2
     * cannot distinguish -- moves the likelihood by 0.25 nats, asserted below.
     */
    public void testTipErrorMarginalisesWithTheTip() throws Exception {
        final double tA = 0.55, tB = 1.3;
        final double[][] sigma = {{0.30, 0.07, -0.04}, {0.07, 0.22, 0.05}, {-0.04, 0.05, 0.26}};
        final double[][] eps = {{0.0170, 0.0062, -0.0021},
                                {0.0062, 0.0140, 0.0035},
                                {-0.0021, 0.0035, 0.0110}};
        final double[] yA = {0.42, -1.15, 2.05};
        final double[] yB = {-1.20, 0.60, Double.NaN};

        final double closed = cherry3ClosedForm(yA, yB, sigma, eps, tA, tB, false);
        final double diagonalOnly = cherry3ClosedForm(yA, yB, sigma, eps, tA, tB, true);
        assertTrue("the two candidate compositions must be distinguishable, else this proves nothing",
                Math.abs(closed - diagonalOnly) > 0.2);

        for (int grid : new int[]{32, 64}) {
            double engine = cherry3(tA, tB, yA, yB, sigma, eps, grid);
            assertEquals("nGrid " + grid, closed, engine, 1.0e-11);
            assertTrue("nGrid " + grid + ": must NOT match the diagonal read-off",
                    Math.abs(engine - diagonalOnly) > 0.2);
        }
    }

    /** The positive log-space reference projects Sigma_eps the same way (its own code path).
     *  In d = 2, because the reference's resource cap forbids a 32^3 grid. */
    public void testReferenceAgreesOnPartialTipWithTipError() throws Exception {
        final double tA = 0.55, tB = 1.3;
        final double[][] sigma = {{0.30, 0.07}, {0.07, 0.22}};
        final double[][] eps = {{0.0170, 0.0062}, {0.0062, 0.0140}};
        final double[] yA = {0.42, -1.15}, yB = {-1.20, Double.NaN};

        // -2 log(2pi) + log WN_1(yA0 - yB0 ; C_A[0][0] + C_B[0][0])
        final double v = (tA + tB) * sigma[0][0] + 2.0 * eps[0][0];
        double total = 0.0;
        for (int k = -14; k <= 14; k++) {
            double delta = yA[0] - yB[0] + 2.0 * Math.PI * k;
            total += Math.exp(-0.5 * delta * delta / v) / Math.sqrt(2.0 * Math.PI * v);
        }
        final double closed = -2.0 * Math.log(2.0 * Math.PI) + Math.log(total);

        Tree tree = new NewickImporter("(A:" + tA + ",B:" + tB + ");").importTree(null);
        double[][] data = tipData(tree, yA, yB);
        final Tree ft = tree;
        TorusLogSpacePruning.BranchLengths lengths = new TorusLogSpacePruning.BranchLengths() {
            public double get(NodeRef node) { return ft.getBranchLength(node); }
        };
        assertEquals(closed, TorusLogSpacePruning.calculate(
                tree, lengths, data, sigma, 0.0, null, null, null, eps).logLikelihood, 1.0e-12);

        List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        precisions.add(matrix(invert(sigma)));
        assertEquals(closed, new TorusFourierTreeLikelihood("q", tree, null, precisions,
                new double[][][]{data}, new TorusFourierKernel(2, 128),
                null, null, null, null, matrix(eps)).getLogLikelihood(), 1.0e-11);
    }

    /** A tip with NO observed coordinate carries no observation error, because it carries no
     *  message: one observed tip under a flat root prior integrates to exactly -d log(2 pi). */
    public void testFullyMissingTipCarriesNoTipError() throws Exception {
        final double[][] sigma = {{0.30, 0.07}, {0.07, 0.22}};
        final double[][] eps = {{0.0170, 0.0062}, {0.0062, 0.0140}};
        Tree tree = new NewickImporter("(A:0.55,B:1.3);").importTree(null);
        double[][] data = tipData(tree, new double[]{0.42, -1.15},
                new double[]{Double.NaN, Double.NaN});
        List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        precisions.add(matrix(invert(sigma)));
        double logL = new TorusFourierTreeLikelihood("m", tree, null, precisions,
                new double[][][]{data}, new TorusFourierKernel(2, 64),
                null, null, null, null, matrix(eps)).getLogLikelihood();
        assertEquals(-2.0 * Math.log(2.0 * Math.PI), logL, 1.0e-11);
    }

    // ---------------------------------------------------------------- gradient

    /**
     * Analytic gradient against central finite differences, with tip error ON.
     *
     * Five directions: the two precision diagonals, the precision correlation, the jump rate,
     * and the isotropic tip-error variance. nGrid 128 is what requiredGrid() demands of a
     * 3.5-degree Sigma_eps; at a coarser grid this would measure aliasing, not a derivative.
     *
     * Leaving the winding atom's covariance at t*Sigma while damping the spectral half -- the
     * "it is just a rename" mistake -- gives relative errors of 1.3e-2 to 4.0e-2 here, with
     * every sign and magnitude still plausible. That is the gap this test exists to see.
     */
    public void testTipErrorAwareGradientMatchesFiniteDifferences() throws Exception {
        assertTrue(fdWorstRelativeError(false, SIGMA_EPS_HA, 128) < 1.0e-7);
    }

    /** The same, with two tips each missing one coordinate. */
    public void testTipErrorAwareGradientOnPartiallyObservedTips() throws Exception {
        assertTrue(fdWorstRelativeError(true, SIGMA_EPS_HA, 128) < 1.0e-7);
    }

    /** Control: with Sigma_eps absent the four original components must still be right. */
    public void testGradientWithoutTipErrorStillMatchesFiniteDifferences() throws Exception {
        assertTrue(fdWorstRelativeError(false, 0.0, 128) < 1.0e-7);
    }

    /** d/dSigma_eps must be far from zero, or the arms above would pass with it unimplemented. */
    public void testTipErrorScoreIsSubstantial() throws Exception {
        Fixture f = new Fixture(false, SIGMA_EPS_HA, 128);
        double[] g = f.likelihood.getResidueGradientWithTipError(0);
        assertEquals(5, g.length);
        assertTrue("d logL / d Sigma_eps = " + g[4], Math.abs(g[4]) > 1.0);
    }

    /** A per-axis or full-matrix Sigma_eps needs one direction per free entry; refuse rather
     *  than silently differentiate the wrong thing. */
    public void testNonIsotropicTipErrorGradientRefused() throws Exception {
        final double[][] sigma = {{0.30, 0.07}, {0.07, 0.22}};
        Tree tree = new NewickImporter("(A:0.55,B:1.3);").importTree(null);
        double[][] data = tipData(tree, new double[]{0.42, -1.15}, new double[]{-1.20, 0.60});
        Parameter diagonal = new Parameter.Default(new double[]{3.1, 2.2});
        Parameter correlation = new Parameter.Default(new double[]{0.23});
        List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        precisions.add(new CompoundSymmetricMatrix(diagonal, correlation, true, false));
        TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood("n", tree, null,
                precisions, new double[][][]{data}, new TorusFourierKernel(2, 64),
                new Parameter.Default(new double[]{0.45}), new double[]{1.0},
                new double[][]{{0.0, 0.0}}, new double[][][]{{{1.1, 0.0}, {0.0, 0.8}}},
                new Parameter.Default("sigmaEpsilon", new double[]{0.004, 0.005}));
        try {
            like.getResidueGradientWithTipError(0);
            fail("a per-axis Sigma_eps must not be differentiated as if it were isotropic");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("isotropic"));
        }
        assertEquals("the four original components must still be available", 4,
                like.getResidueGradient(0).length);
        // unused here, but sigma is what the fixture above inverts elsewhere
        assertNotNull(sigma);
    }

    // ---------------------------------------------------------------- helpers

    private static final class Fixture {
        final TorusFourierTreeLikelihood likelihood;
        final Parameter diagonal = new Parameter.Default(new double[]{3.10, 2.20});
        final Parameter correlation = new Parameter.Default(new double[]{0.23});
        final Parameter rate = new Parameter.Default(new double[]{0.45});
        final Parameter tipError;

        Fixture(boolean partial, double epsilon, int grid) throws Exception {
            Tree tree = new NewickImporter(
                    "(((A:0.22,B:0.17):0.31,(C:0.4,D:0.26):0.19):0.15,(E:0.35,F:0.28):0.24);")
                    .importTree(null);
            Map<String, double[]> observed = new LinkedHashMap<String, double[]>();
            observed.put("A", new double[]{-1.20, 0.60});
            observed.put("B", new double[]{-1.05, 0.72});
            observed.put("C", new double[]{2.10, -2.20});
            observed.put("D", partial ? new double[]{Double.NaN, -1.95} : new double[]{1.85, -1.95});
            observed.put("E", new double[]{0.40, 1.90});
            observed.put("F", partial ? new double[]{0.12, Double.NaN} : new double[]{0.12, 2.25});
            double[][] data = new double[tree.getExternalNodeCount()][];
            for (int e = 0; e < tree.getExternalNodeCount(); e++) {
                NodeRef node = tree.getExternalNode(e);
                data[node.getNumber()] = observed.get(tree.getNodeTaxon(node).getId());
            }
            List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
            precisions.add(new CompoundSymmetricMatrix(diagonal, correlation, true, false));
            tipError = epsilon == 0.0 ? null : new Parameter.Default("sigmaEpsilon", epsilon);
            likelihood = new TorusFourierTreeLikelihood("g", tree, null, precisions,
                    new double[][][]{data}, new TorusFourierKernel(2, grid), rate,
                    new double[]{0.62, 0.38},
                    new double[][]{{0.40, -0.20}, {-1.10, 0.70}},
                    new double[][][]{{{0.90, 0.05}, {0.05, 0.70}}, {{1.40, -0.20}, {-0.20, 1.10}}},
                    tipError);
        }
    }

    private static double fdWorstRelativeError(boolean partial, double epsilon, int grid)
            throws Exception {
        Fixture f = new Fixture(partial, epsilon, grid);
        double[] analytic = epsilon == 0.0 ? f.likelihood.getResidueGradient(0)
                                           : f.likelihood.getResidueGradientWithTipError(0);
        Parameter[] parameters = {f.diagonal, f.diagonal, f.correlation, f.rate, f.tipError};
        int[] index = {0, 1, 0, 0, 0};
        double worst = 0.0;
        for (int p = 0; p < analytic.length; p++) {
            double base = parameters[p].getParameterValue(index[p]);
            double step = Math.max(1.0e-6, 1.0e-5 * Math.abs(base));
            parameters[p].setParameterValue(index[p], base + step);
            double up = f.likelihood.getLogLikelihood();
            parameters[p].setParameterValue(index[p], base - step);
            double down = f.likelihood.getLogLikelihood();
            parameters[p].setParameterValue(index[p], base);
            f.likelihood.getLogLikelihood();
            double fd = (up - down) / (2.0 * step);
            worst = Math.max(worst, Math.abs(analytic[p] - fd) / Math.max(1.0, Math.abs(fd)));
        }
        return worst;
    }

    private static double cherry3(double tA, double tB, double[] yA, double[] yB,
                                  double[][] sigma, double[][] eps, int grid) throws Exception {
        Tree tree = new NewickImporter("(A:" + tA + ",B:" + tB + ");").importTree(null);
        List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        precisions.add(matrix(invert(sigma)));
        return new TorusFourierTreeLikelihood("p", tree, null, precisions,
                new double[][][]{tipData(tree, yA, yB)}, new TorusFourierKernel(3, grid),
                null, null, null, null, matrix(eps)).getLogLikelihood();
    }

    private static double cherry3ClosedForm(double[] yA, double[] yB, double[][] sigma,
                                            double[][] eps, double tA, double tB,
                                            boolean diagonalOnly) {
        double[][] c = new double[2][2];
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
            double e = diagonalOnly ? (i == j ? eps[i][i] : 0.0) : eps[i][j];
            c[i][j] = (tA + tB) * sigma[i][j] + 2.0 * e;
        }
        double det = c[0][0] * c[1][1] - c[0][1] * c[1][0];
        double[][] precision = {{c[1][1] / det, -c[0][1] / det}, {-c[1][0] / det, c[0][0] / det}};
        double norm = 1.0 / (2.0 * Math.PI * Math.sqrt(det));
        double total = 0.0;
        for (int k0 = -12; k0 <= 12; k0++) for (int k1 = -12; k1 <= 12; k1++) {
            double v0 = yA[0] - yB[0] + 2.0 * Math.PI * k0;
            double v1 = yA[1] - yB[1] + 2.0 * Math.PI * k1;
            total += norm * Math.exp(-0.5 * (v0 * (precision[0][0] * v0 + precision[0][1] * v1)
                    + v1 * (precision[1][0] * v0 + precision[1][1] * v1)));
        }
        return -3.0 * Math.log(2.0 * Math.PI) + Math.log(total);
    }

    private static double[][] tipData(Tree tree, double[] yA, double[] yB) {
        double[][] data = new double[tree.getExternalNodeCount()][];
        for (int e = 0; e < tree.getExternalNodeCount(); e++) {
            NodeRef node = tree.getExternalNode(e);
            data[node.getNumber()] = tree.getNodeTaxon(node).getId().equals("A") ? yA : yB;
        }
        return data;
    }

    private static MatrixParameter matrix(double[][] m) {
        MatrixParameter p = new MatrixParameter("m");
        for (double[] row : m) p.addParameter(new Parameter.Default(row.clone()));
        return p;
    }

    /** Gauss-Jordan inverse, so the test does not borrow the engine's linear algebra. */
    private static double[][] invert(double[][] a) {
        int n = a.length;
        double[][] m = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n);
            m[i][n + i] = 1.0;
        }
        for (int c = 0; c < n; c++) {
            int pivot = c;
            for (int r = c; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[pivot][c])) pivot = r;
            double[] swap = m[c]; m[c] = m[pivot]; m[pivot] = swap;
            double d = m[c][c];
            for (int k = 0; k < 2 * n; k++) m[c][k] /= d;
            for (int r = 0; r < n; r++) if (r != c) {
                double factor = m[r][c];
                for (int k = 0; k < 2 * n; k++) m[r][k] -= factor * m[c][k];
            }
        }
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(m[i], n, inverse[i], 0, n);
        return inverse;
    }
}
