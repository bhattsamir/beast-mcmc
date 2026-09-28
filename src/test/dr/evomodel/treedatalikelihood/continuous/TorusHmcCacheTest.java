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

/** Rejection must preserve partials across the many evaluations in an HMC trajectory. */
public class TorusHmcCacheTest extends TestCase {
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
        f.trajectory(2); // two alternating writes used to overwrite the stored buffer
        f.like.restoreModelState();
        assertEquals(1.4, f.diagonal.getParameterValue(0), 0.0);
        assertEquals(original, f.like.getLogLikelihood(), 1e-12);
        assertLocalUpdateMatchesFresh(f);
    }

    public void testAcceptSeveralEvaluationsThenRejectNextTrajectory() throws Exception {
        Fixture f = new Fixture();
        f.like.getLogLikelihood();
        f.like.storeModelState();
        f.trajectory(3);
        double accepted = f.like.getLogLikelihood();
        double acceptedPrecision = f.diagonal.getParameterValue(0);
        f.like.acceptModelState();
        f.like.storeModelState();
        f.trajectory(4);
        f.like.restoreModelState();
        assertEquals(acceptedPrecision, f.diagonal.getParameterValue(0), 0.0);
        assertEquals(accepted, f.like.getLogLikelihood(), 1e-12);
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
                f.like.getLogLikelihood(), cached, 1e-10);
    }

    private static final class Fixture {
        final DefaultTreeModel tree;
        final Parameter diagonal = new Parameter.Default(new double[]{1.4, 0.9});
        final Parameter correlation = new Parameter.Default(new double[]{0.3});
        final Parameter rate = new Parameter.Default(0.25);
        final TorusFourierTreeLikelihood like;

        Fixture() throws Exception {
            tree = new DefaultTreeModel("tree", new NewickImporter(
                    "((A:1.0,B:1.0):0.6,(C:1.0,D:1.0):0.6);").importTree(null));
            MatrixParameter precision = new CompoundSymmetricMatrix(diagonal, correlation, true, false);
            double[][][] data = {{{0.4, 1.1}, {-1.2, 0.6}, {0.9, -0.3}, {-0.7, -1.0}}};
            like = new TorusFourierTreeLikelihood("trajectory", tree, null,
                    Collections.singletonList(precision), data, new TorusFourierKernel(2, 32),
                    rate, new double[]{0.65, 0.35}, new double[][]{{0, 0}, {1.2, -0.7}},
                    new double[][][]{{{0.35, 0.04}, {0.04, 0.45}}, {{0.45, 0}, {0, 0.35}}});
        }

        void trajectory(int evaluations) {
            for (int i = 0; i < evaluations; i++) {
                diagonal.setParameterValue(0, 0.55 + 0.17 * i);
                correlation.setParameterValue(0, -0.25 + 0.07 * i);
                rate.setParameterValue(0, 0.4 + 0.08 * i);
                assertTrue(Double.isFinite(like.getLogLikelihood()));
            }
        }
    }
}
