package dr.evomodel.treedatalikelihood.continuous;

import dr.inference.distribution.DistributionLikelihood;
import dr.inference.distribution.LogNormalDistributionModel;
import dr.inference.hmc.GradientWrtParameterProvider;
import dr.inference.model.GradientProvider;
import dr.inference.model.Likelihood;
import dr.inference.model.MaskedParameter;
import dr.inference.model.Parameter;
import dr.math.distributions.Distribution;
import dr.math.distributions.ExponentialDistribution;
import dr.math.distributions.LogNormalDistribution;

/**
 * Prior score for one scalar rate, keeping the actual target prior as its
 * likelihood. This fills the missing GradientProvider on BEAST's exponential
 * distribution and also supports an unshifted hierarchical lognormal prior.
 * The XML parser rejects shifted, scaled, or range-restricted prior definitions.
 */
public final class TorusJumpRatePriorGradient implements GradientWrtParameterProvider {
    private final DistributionLikelihood prior;
    private final Parameter parameter;

    public TorusJumpRatePriorGradient(DistributionLikelihood prior, Parameter parameter) {
        if (prior == null || parameter == null || parameter.getDimension() != 1) {
            throw new IllegalArgumentException("A jump-rate prior gradient requires a prior and a scalar parameter view");
        }
        if (prior.getDataList().size() != 1) {
            throw new IllegalArgumentException("Jump-rate prior must have exactly one parameter as its data");
        }
        Object data = prior.getDataList().get(0);
        boolean sameData = data == parameter;
        if (parameter instanceof MaskedParameter) {
            sameData |= data == ((MaskedParameter) parameter).getUnmaskedParameter();
        }
        if (!sameData) {
            throw new IllegalArgumentException("Scalar rate view must refer to the parameter evaluated by its prior");
        }
        Distribution distribution = prior.getDistribution();
        if (!(distribution instanceof ExponentialDistribution)
                && !(distribution instanceof LogNormalDistribution)
                && !(distribution instanceof LogNormalDistributionModel)) {
            throw new IllegalArgumentException("Supported jump-rate priors are unshifted exponential or lognormal distributions");
        }
        this.prior = prior;
        this.parameter = parameter;
    }

    @Override
    public Likelihood getLikelihood() { return prior; }

    @Override
    public Parameter getParameter() { return parameter; }

    @Override
    public int getDimension() { return 1; }

    @Override
    public double[] getGradientLogDensity() {
        double value = parameter.getParameterValue(0);
        if (!(value > 0.0) || !Double.isFinite(value)) {
            throw new ArithmeticException("Torus HMC jump rate must be finite and positive");
        }
        Distribution distribution = prior.getDistribution();
        double derivative;
        if (distribution instanceof ExponentialDistribution) {
            derivative = ((ExponentialDistribution) distribution).gradLogPdf(value);
        } else {
            derivative = ((GradientProvider) distribution).getGradientLogDensity(new double[]{value})[0];
        }
        if (!Double.isFinite(derivative)) throw new ArithmeticException("Nonfinite jump-rate prior gradient");
        return new double[]{derivative};
    }
}
