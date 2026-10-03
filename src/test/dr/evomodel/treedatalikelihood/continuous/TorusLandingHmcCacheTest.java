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

import java.util.Collections;

/**
 * TorusHmcCacheTest for jumpLaw=landing. HMC (trajectory, gradient/step-size checks,
 * fullEvaluation checks) evaluates the likelihood several times between store and
 * accept/restore. Without the stateStored write-buffer rule, the second evaluation
 * overwrites the stored buffer and a rejected trajectory silently restores corrupted
 * partials, which the scalar log L cache hides until a local update.
 */
public class TorusLandingHmcCacheTest extends TestCase {
    private boolean previousValidation;

    protected void setUp() {
        previousValidation = TorusFourierTreeLikelihood.isValidateEveryEvaluation();
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(false);
    }

    protected void tearDown() {
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(previousValidation);
    }

    public void testRejectSeveralEvaluationsThenUpdateOneSubtree() throws Exception {
        Fixture f = new Fixture();
        double original = f.like.getLogLikelihood();
        f.like.storeModelState();
        f.trajectory(2, false);
        f.like.restoreModelState();
        assertEquals(1400.0, f.diagonal.getParameterValue(0), 0.0);
        assertEquals(original, f.like.getLogLikelihood(), 0.0);
        assertLocalUpdateMatchesFresh(f);
    }

    public void testAcceptSeveralEvaluationsThenRejectNextTrajectory() throws Exception {
        Fixture f = new Fixture();
        f.like.getLogLikelihood();
        f.like.storeModelState();
        f.trajectory(3, false);
        double accepted = f.like.getLogLikelihood();
        double acceptedPrecision = f.diagonal.getParameterValue(0);
        f.like.acceptModelState();
        f.like.storeModelState();
        f.trajectory(4, false);
        f.like.restoreModelState();
        assertEquals(acceptedPrecision, f.diagonal.getParameterValue(0), 0.0);
        assertEquals(accepted, f.like.getLogLikelihood(), 0.0);
        assertLocalUpdateMatchesFresh(f);
    }

    public void testGradientCallsInsideATrajectoryDoNotDisturbTheCache() throws Exception {
        Fixture f = new Fixture();
        double original = f.like.getLogLikelihood();
        f.like.storeModelState();
        f.trajectory(3, true);
        f.like.restoreModelState();
        assertEquals(original, f.like.getLogLikelihood(), 0.0);
        assertLocalUpdateMatchesFresh(f);
    }

    private static void assertLocalUpdateMatchesFresh(Fixture f) {
        // Only the A/B side is dirtied. C/D's cached partial must still be the
        // restored state's partial, even though a scalar logL cache hid damage.
        NodeRef tip = f.tree.getExternalNode(0);
        f.tree.setNodeHeight(tip, f.tree.getNodeHeight(tip) + 0.12);
        double cached = f.like.getLogLikelihood();
        f.like.makeDirty();
        assertEquals("rejected trajectory corrupted the untouched sibling partial",
                f.like.getLogLikelihood(), cached, 0.0);
    }

    private static final class Fixture {
        final DefaultTreeModel tree;
        final Parameter diagonal = new Parameter.Default(new double[]{1400.0, 900.0});
        final Parameter correlation = new Parameter.Default(new double[]{0.3});
        final Parameter rate = new Parameter.Default(0.05);
        final TorusFourierTreeLikelihood like;

        Fixture() throws Exception {
            tree = new DefaultTreeModel("tree", new NewickImporter(
                    "((A:1.0,B:1.0):0.6,(C:1.0,D:1.0):0.6);").importTree(null));
            MatrixParameter precision = new CompoundSymmetricMatrix(diagonal, correlation, true, false);
            double[][][] data = {{{-1.2, 2.4}, {-1.9, Double.NaN}, {0.9, 0.3}, {-1.1, -0.5}}};
            like = new TorusFourierTreeLikelihood("landingTrajectory", tree, null,
                    Collections.singletonList(precision), data, new TorusFourierKernel(2, 32),
                    rate, TorusLandingGradientKernelTest.W, TorusLandingGradientKernelTest.MU,
                    TorusLandingGradientKernelTest.OMEGA, new Parameter.Default(0.0076154354946700),
                    TorusFourierTreeLikelihood.JumpLaw.LANDING);
        }

        void trajectory(int evaluations, boolean withGradients) {
            for (int i = 0; i < evaluations; i++) {
                diagonal.setParameterValue(0, 55.0 + 17.0 * i);
                correlation.setParameterValue(0, -0.25 + 0.07 * i);
                rate.setParameterValue(0, 0.4 + 0.08 * i);
                assertTrue(Double.isFinite(like.getLogLikelihood()));
                if (withGradients) {
                    double[] g = like.getResidueGradient(0);
                    for (double v : g) assertTrue(Double.isFinite(v));
                }
            }
        }
    }
}
