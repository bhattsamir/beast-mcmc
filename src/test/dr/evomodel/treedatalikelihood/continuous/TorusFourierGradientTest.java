package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.Tree;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierGradient;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.evomodel.treedatalikelihood.continuous.TorusJumpRatePriorGradient;
import dr.inference.distribution.DistributionLikelihood;
import dr.inference.distribution.LogNormalDistributionModel;
import dr.inference.model.CompoundParameter;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.MaskedParameter;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import dr.math.distributions.ExponentialDistribution;
import dr.math.distributions.LogNormalDistribution;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.List;

/** Parameter views and prior scores at the boundary between torus pruning and generic HMC. */
public class TorusFourierGradientTest extends TestCase {
    @Override
    protected void setUp() {
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(false);
    }

    @Override
    protected void tearDown() {
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(false);
    }

    public void testFourParameterBlockUpdatesOnlySelectedResidue() throws Exception {
        Parameter rates = new Parameter.Default(new double[]{0.2, 0.4});
        rates.addBounds(new Parameter.DefaultBounds(Double.POSITIVE_INFINITY, 0.0, 2));
        TorusFourierTreeLikelihood likelihood = likelihood(2, rates, false);
        TorusFourierGradient gradient = new TorusFourierGradient(likelihood, 1);
        CompoundParameter block = gradient.getParameter();
        assertEquals(4, gradient.getDimension());
        assertSame(likelihood, gradient.getLikelihood());
        CompoundSymmetricMatrix precision = (CompoundSymmetricMatrix) likelihood.getResiduePrecision(1);
        assertSame(precision.getDiagonalParameter(), block.getParameter(0));
        assertSame(precision.getOffDiagonalParameter(), block.getParameter(1));
        assertSame(rates, ((MaskedParameter) block.getParameter(2)).getUnmaskedParameter());
        List<Integer> rateEvents = new ArrayList<Integer>();
        rates.addParameterListener((variable, index, type) -> rateEvents.add(index));
        double old = likelihood.getLogLikelihood();
        likelihood.resetRecomputeCount();
        block.setParameterValueQuietly(0, 1.7);
        block.setParameterValueQuietly(1, 2.2);
        block.setParameterValueQuietly(2, 0.12);
        block.setParameterValueQuietly(3, 0.6);
        block.fireParameterChangedEvent();
        assertEquals(0.2, rates.getParameterValue(0), 0.0);
        assertEquals(0.6, rates.getParameterValue(1), 0.0);
        assertEquals(1, rateEvents.size());
        assertEquals(Integer.valueOf(1), rateEvents.get(0));
        double changed = likelihood.getLogLikelihood();
        assertTrue(Math.abs(old - changed) > 1e-7);
        assertEquals("one residue's two internal FFT partials", 2L, likelihood.getNodeRecomputeCount());
        likelihood.makeDirty();
        assertEquals(changed, likelihood.getLogLikelihood(), 1e-11);
    }

    public void testScalarRateViewRestoresUnderlyingVector() throws Exception {
        Parameter rates = new Parameter.Default(new double[]{0.2, 0.4});
        rates.addBounds(new Parameter.DefaultBounds(Double.POSITIVE_INFINITY, 0.0, 2));
        CompoundParameter block = new TorusFourierGradient(likelihood(2, rates, false), 1).getParameter();
        block.storeParameterValues();
        block.setParameterValue(3, 0.8);
        assertEquals(0.8, rates.getParameterValue(1), 0.0);
        block.restoreParameterValues();
        assertEquals(0.2, rates.getParameterValue(0), 0.0);
        assertEquals(0.4, rates.getParameterValue(1), 0.0);
    }

    public void testRejectsSharedRateAcrossResidues() throws Exception {
        Parameter rate = new Parameter.Default(0.3);
        try {
            new TorusFourierGradient(likelihood(2, rate, false), 0);
            fail("shared rate needs a sum over residues");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("one jump-rate parameter per residue"));
        }
    }

    public void testRejectsSharedPrecisionComponentsAcrossResidues() throws Exception {
        for (boolean sharedDiagonal : new boolean[]{true, false}) {
            Parameter rates = new Parameter.Default(new double[]{0.2, 0.4});
            TorusFourierTreeLikelihood shared = likelihood(2, rates, false,
                    sharedDiagonal, !sharedDiagonal);
            try {
                new TorusFourierGradient(shared, 1);
                fail("a shared precision component needs a sum over residue gradients");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains("independent precision parameters across residues"));
            }
        }
    }

    public void testRejectsReferenceModeAndCholesky() throws Exception {
        TorusFourierTreeLikelihood likelihood = likelihood(1, new Parameter.Default(0.3), false);
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(true);
        try {
            new TorusFourierGradient(likelihood, 0);
            fail("FFT gradient must not be paired with adaptive reference likelihood");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("torus.validate=false"));
        }
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(false);
        try {
            new TorusFourierGradient(likelihood(1, new Parameter.Default(0.3), true), 0);
            fail("Cholesky coordinates need a different derivative mapping");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("isCholesky=false"));
        }
    }

    public void testExponentialPriorMatchesFiniteDifferenceAndTargetIdentity() {
        Parameter rates = new Parameter.Default(new double[]{0.2, 0.4});
        Parameter view = scalarView(rates, 1);
        DistributionLikelihood prior = new DistributionLikelihood(new ExponentialDistribution(7.0));
        prior.addData(rates);
        TorusJumpRatePriorGradient gradient = new TorusJumpRatePriorGradient(prior, view);
        assertSame(prior, gradient.getLikelihood());
        assertSame(view, gradient.getParameter());
        assertEquals(-7.0, gradient.getGradientLogDensity()[0], 0.0);
        assertEquals(finiteDifference(prior, view), gradient.getGradientLogDensity()[0], 1e-8);
    }

    public void testLognormalPriorTracksHyperparameterChanges() {
        Parameter rates = new Parameter.Default(new double[]{0.2, 0.4});
        Parameter view = scalarView(rates, 1);
        Parameter mean = new Parameter.Default(-1.0), sd = new Parameter.Default(0.7);
        LogNormalDistributionModel distribution = new LogNormalDistributionModel(mean, sd, 0.0, false);
        DistributionLikelihood prior = new DistributionLikelihood(distribution);
        prior.addData(rates);
        TorusJumpRatePriorGradient gradient = new TorusJumpRatePriorGradient(prior, view);
        double initial = gradient.getGradientLogDensity()[0];
        assertEquals(finiteDifference(prior, view), initial, 1e-8);
        mean.setParameterValue(0, -0.4);
        sd.setParameterValue(0, 1.2);
        assertEquals(finiteDifference(prior, view), gradient.getGradientLogDensity()[0], 1e-8);
        assertTrue(Math.abs(initial - gradient.getGradientLogDensity()[0]) > 1e-5);
        DistributionLikelihood fixed = new DistributionLikelihood(new LogNormalDistribution(-0.4, 1.2));
        fixed.addData(rates);
        assertEquals(gradient.getGradientLogDensity()[0],
                new TorusJumpRatePriorGradient(fixed, view).getGradientLogDensity()[0], 1e-12);
    }

    public void testPriorRejectsUnrelatedParameter() {
        Parameter rate = new Parameter.Default(0.2);
        DistributionLikelihood prior = new DistributionLikelihood(new ExponentialDistribution(7.0));
        prior.addData(rate);
        try {
            new TorusJumpRatePriorGradient(prior, new Parameter.Default(0.2));
            fail("equal values do not imply parameter identity");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("refer to the parameter"));
        }
    }

    private static Parameter scalarView(Parameter rates, int index) {
        Parameter mask = new Parameter.Default(rates.getDimension(), 0.0);
        mask.setParameterValueQuietly(index, 1.0);
        return new MaskedParameter(rates, mask, true);
    }

    private static double finiteDifference(DistributionLikelihood prior, Parameter parameter) {
        double x = parameter.getParameterValue(0), h = 1e-6;
        parameter.setParameterValue(0, x + h);
        double plus = prior.getLogLikelihood();
        parameter.setParameterValue(0, x - h);
        double minus = prior.getLogLikelihood();
        parameter.setParameterValue(0, x);
        return (plus - minus) / (2.0 * h);
    }

    private static TorusFourierTreeLikelihood likelihood(int residues, Parameter rates,
                                                        boolean cholesky) throws Exception {
        return likelihood(residues, rates, cholesky, false, false);
    }

    private static TorusFourierTreeLikelihood likelihood(int residues, Parameter rates,
                                                        boolean cholesky, boolean sharedDiagonal,
                                                        boolean sharedCorrelation) throws Exception {
        Tree tree = new NewickImporter("((A:1,B:1):0.5,C:1.5);").importTree(null);
        List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        double[][][] data = new double[residues][tree.getExternalNodeCount()][2];
        Parameter firstDiagonal = null, firstOff = null;
        for (int r = 0; r < residues; r++) {
            Parameter diagonal = new Parameter.Default(new double[]{1.5, 2.0});
            diagonal.addBounds(new Parameter.DefaultBounds(Double.POSITIVE_INFINITY, 0.0, 2));
            Parameter off = new Parameter.Default(0.1);
            off.addBounds(new Parameter.DefaultBounds(0.95, -0.95, 1));
            if (r == 0) {
                firstDiagonal = diagonal;
                firstOff = off;
            } else {
                if (sharedDiagonal) diagonal = firstDiagonal;
                if (sharedCorrelation) off = firstOff;
            }
            precisions.add(new CompoundSymmetricMatrix(diagonal, off, true, cholesky));
            for (int i = 0; i < data[r].length; i++) data[r][i] = new double[]{0.4 - 0.3 * i, 0.1 + 0.5 * i};
        }
        return new TorusFourierTreeLikelihood("providerTest", tree, null, precisions, data,
                new TorusFourierKernel(2, 16), rates, new double[]{1.0},
                new double[][]{{0.6, -0.2}}, new double[][][]{{{0.3, 0.0}, {0.0, 0.4}}});
    }
}
