package dr.evomodelxml.treedatalikelihood.continuous;

import dr.inference.hmc.GradientWrtParameterProvider;
import dr.inference.model.Bounds;
import dr.inference.model.Parameter;
import dr.util.Transform;
import dr.xml.*;
import java.util.Arrays;

/** Log transforms for q1, q2 and lambda; bounded logit for precision correlation. */
public class TorusHmcTransformParser extends AbstractXMLObjectParser {
    public String getParserName() { return "torusHmcTransform"; }

    public Object parseXMLObject(XMLObject xo) throws XMLParseException {
        GradientWrtParameterProvider gradient =
                (GradientWrtParameterProvider) xo.getChild(GradientWrtParameterProvider.class);
        Parameter p = gradient.getParameter();
        if (p.getDimension() != 4) throw new XMLParseException("Torus HMC requires four parameters");
        Bounds<Double> b = p.getBounds();
        for (int i : new int[]{0, 1, 3}) {
            if (b.getLowerLimit(i) != 0.0 || b.getUpperLimit(i) != Double.POSITIVE_INFINITY)
                throw new XMLParseException("Torus precision diagonals and jump rate require bounds (0, infinity)");
        }
        double lo = b.getLowerLimit(2), hi = b.getUpperLimit(2);
        if (!Double.isFinite(lo) || !Double.isFinite(hi) || lo < -1.0 || hi > 1.0 || lo >= hi)
            throw new XMLParseException("Torus correlation requires finite bounds within [-1, 1]");
        Transform.Array transform = new Transform.Array(Arrays.asList(
                new FiniteLog(), new FiniteLog(), new BoundedLogit(hi, lo), new FiniteLog()), p);
        if (!transform.isInInteriorDomain(p.getParameterValues(), 0, 4))
            throw new XMLParseException("Initial torus HMC parameters must be strictly inside their bounds");
        return transform;
    }

    public static class FiniteLog extends Transform.LogTransform {
        @Override public boolean isInInteriorDomain(double x) {
            return x > 0.0 && Double.isFinite(x);
        }
    }

    /** BEAST's ScaledLogitTransform lacks HMC derivatives in the supported checkout.
     * Here x is the original coordinate, z = log((x-lo)/(hi-x)). */
    public static class BoundedLogit extends Transform.ScaledLogitTransform {
        private final double lo, hi, width;
        public BoundedLogit(double hi, double lo) {
            super(hi, lo);
            this.lo = lo; this.hi = hi; this.width = hi - lo;
        }
        @Override public double inverse(double z) {
            double e = Math.exp(-Math.abs(z));
            return z >= 0.0 ? hi - width * e / (1.0 + e) : lo + width * e / (1.0 + e);
        }
        @Override public double gradientInverse(double z) {
            double e = Math.exp(-Math.abs(z));
            return width * e / ((1.0 + e) * (1.0 + e));
        }
        @Override public double updateGradientLogDensity(double gradient, double x) {
            return gradient * (x - lo) * (hi - x) / width + (hi + lo - 2.0 * x) / width;
        }
        @Override public double gradientLogJacobianInverse(double z) {
            return -Math.tanh(0.5 * z);
        }
        @Override public double updateDiagonalHessianLogDensity(double hessian, double gradient, double x) {
            double d = (x - lo) * (hi - x) / width;
            return hessian * d * d + gradient * d * (hi + lo - 2.0 * x) / width - 2.0 * d / width;
        }
        @Override public String getTransformName() { return "torusBoundedLogit"; }
    }

    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{new ElementRule(GradientWrtParameterProvider.class)};
    }
    public String getParserDescription() { return "Transforms a torus residue HMC block to unconstrained coordinates."; }
    public Class getReturnType() { return Transform.MultivariableTransformWithParameter.class; }
}
