/*
 * TorusResidueBlockOperatorParser.java
 *
 * @author Samir Bhatt
 */

package dr.evomodelxml.treedatalikelihood.continuous.gpu;

import dr.evomodel.treedatalikelihood.continuous.gpu.TorusLandingGpuLikelihood;
import dr.evomodel.treedatalikelihood.continuous.gpu.TorusResidueBlockOperator;
import dr.inference.model.Likelihood;
import dr.inference.operators.MCMCOperator;
import dr.xml.*;

/**
 * &lt;torusResidueBlockOperator weight="1" adaptStart="200" adaptUntil="5000" initialScale="0.1"&gt;
 *     &lt;torusLandingGpuLikelihood idref="tfl"/&gt;
 *     &lt;prior&gt;&lt;prior idref="prior"/&gt;&lt;/prior&gt;
 *     &lt;posterior&gt;&lt;joint idref="joint"/&gt;&lt;/posterior&gt;     (optional: per-step consistency guard)
 * &lt;/torusResidueBlockOperator&gt;
 * ignoreLikelihood="true" samples the prior only (for testing; the posterior must then be the prior).
 *
 * @author Samir Bhatt
 */
public class TorusResidueBlockOperatorParser extends AbstractXMLObjectParser {

    public static final String PRIOR = "prior";
    public static final String POSTERIOR = "posterior";

    public String getParserName() { return TorusResidueBlockOperator.OPERATOR_NAME; }

    public Object parseXMLObject(XMLObject xo) throws XMLParseException {
        final TorusLandingGpuLikelihood like = (TorusLandingGpuLikelihood) xo.getChild(TorusLandingGpuLikelihood.class);
        final Likelihood prior = (Likelihood) xo.getElementFirstChild(PRIOR);
        final Likelihood posterior = xo.hasChildNamed(POSTERIOR) ? (Likelihood) xo.getElementFirstChild(POSTERIOR) : null;
        final long adaptStart = xo.getAttribute("adaptStart", 200);
        final long adaptUntil = xo.getAttribute("adaptUntil", 5000);
        if (adaptStart < 0 || adaptUntil < adaptStart) throw new XMLParseException("need 0 <= adaptStart <= adaptUntil");
        try {
            return new TorusResidueBlockOperator(like, prior, posterior, xo.getAttribute("ignoreLikelihood", false),
                    adaptStart, adaptUntil, xo.getAttribute("initialScale", 0.1), xo.getAttribute("guardTolerance", 1e-8),
                    xo.getDoubleAttribute(MCMCOperator.WEIGHT));
        } catch (IllegalArgumentException e) {
            throw new XMLParseException(e.getMessage());
        }
    }

    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{
                AttributeRule.newDoubleRule(MCMCOperator.WEIGHT),
                AttributeRule.newIntegerRule("adaptStart", true),
                AttributeRule.newIntegerRule("adaptUntil", true),
                AttributeRule.newDoubleRule("initialScale", true),
                AttributeRule.newDoubleRule("guardTolerance", true),
                AttributeRule.newBooleanRule("ignoreLikelihood", true),
                new ElementRule(TorusLandingGpuLikelihood.class),
                new ElementRule(PRIOR, new XMLSyntaxRule[]{new ElementRule(Likelihood.class)}),
                new ElementRule(POSTERIOR, new XMLSyntaxRule[]{new ElementRule(Likelihood.class)}, true),
        };
    }

    public String getParserDescription() {
        return "Metropolis-within-Gibbs update of every residue of a torusLandingGpuLikelihood, with all proposals "
                + "evaluated in one GPU pass (adaptive proposals during burn-in only).";
    }

    public Class getReturnType() { return MCMCOperator.class; }
}
