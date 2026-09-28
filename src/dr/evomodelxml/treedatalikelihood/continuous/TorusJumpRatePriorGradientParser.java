package dr.evomodelxml.treedatalikelihood.continuous;

import dr.evomodel.treedatalikelihood.continuous.TorusJumpRatePriorGradient;
import dr.inference.distribution.DistributionLikelihood;
import dr.inference.model.Parameter;
import dr.xml.AbstractXMLObjectParser;
import dr.xml.ElementRule;
import dr.xml.Reference;
import dr.xml.XMLObject;
import dr.xml.XMLParseException;
import dr.xml.XMLSyntaxRule;

/** Parse a scalar rate-prior score without duplicating the prior's parameters. */
public final class TorusJumpRatePriorGradientParser extends AbstractXMLObjectParser {
    public static final String NAME = "torusJumpRatePriorGradient";

    @Override
    public String getParserName() { return NAME; }

    @Override
    public Object parseXMLObject(XMLObject xo) throws XMLParseException {
        DistributionLikelihood prior = (DistributionLikelihood) xo.getChild(DistributionLikelihood.class);
        Parameter parameter = (Parameter) xo.getChild(Parameter.class);
        // DistributionLikelihood does not expose offset/scale/range getters.
        // Inspect the original XML even for idrefs, rather than silently assuming
        // that its density is the unmodified distribution's density.
        for (int i = 0; i < xo.getChildCount(); i++) {
            if (xo.getChild(i) == prior) checkDefinition(original(xo.getRawChild(i)));
        }
        try {
            return new TorusJumpRatePriorGradient(prior, parameter);
        } catch (IllegalArgumentException e) {
            throw new XMLParseException(e.getMessage());
        }
    }

    private static XMLObject original(Object child) throws XMLParseException {
        if (child instanceof Reference) return ((Reference) child).getReferenceObject();
        if (child instanceof XMLObject) return (XMLObject) child;
        throw new XMLParseException("Cannot inspect the jump-rate prior definition");
    }

    private static void checkDefinition(XMLObject definition) throws XMLParseException {
        if ((definition.hasAttribute("offset") && definition.getDoubleAttribute("offset") != 0.0)
                || (definition.hasAttribute("scale") && definition.getDoubleAttribute("scale") != 1.0)
                || definition.hasAttribute("from") || definition.hasAttribute("to")) {
            throw new XMLParseException("torusJumpRatePriorGradient requires an unshifted, unscaled prior over its full parameter");
        }
        for (int i = 0; i < definition.getChildCount(); i++) {
            Object child = definition.getRawChild(i);
            if (child instanceof XMLObject || child instanceof Reference) {
                XMLObject nested = original(child);
                // A parameter reference is data, not a distribution wrapper. Its
                // model bounds are respected by HMC and need no density transform.
                if (!(nested.getNativeObject() instanceof Parameter)) checkDefinition(nested);
            }
        }
    }

    @Override
    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{
                new ElementRule(DistributionLikelihood.class),
                new ElementRule(Parameter.class)
        };
    }

    @Override
    public String getParserDescription() {
        return "Gradient of an exponential or lognormal jump-rate prior for one scalar rate view";
    }

    @Override
    public Class getReturnType() { return TorusJumpRatePriorGradient.class; }
}
