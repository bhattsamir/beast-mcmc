package dr.evomodelxml.treedatalikelihood.continuous;

import dr.evomodel.treedatalikelihood.continuous.TorusFourierGradient;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.xml.AbstractXMLObjectParser;
import dr.xml.AttributeRule;
import dr.xml.ElementRule;
import dr.xml.XMLObject;
import dr.xml.XMLParseException;
import dr.xml.XMLSyntaxRule;

/** Connect the custom FFT likelihood gradient to BEAST's generic HMC operator. */
public final class TorusFourierGradientParser extends AbstractXMLObjectParser {
    public static final String NAME = "torusFourierGradient";
    public static final String RESIDUE = "residue";

    @Override
    public String getParserName() { return NAME; }

    @Override
    public Object parseXMLObject(XMLObject xo) throws XMLParseException {
        TorusFourierTreeLikelihood likelihood =
                (TorusFourierTreeLikelihood) xo.getChild(TorusFourierTreeLikelihood.class);
        int residue = xo.getIntegerAttribute(RESIDUE);
        if (residue < 1) throw new XMLParseException("torusFourierGradient residue is one based and must be >= 1");
        try {
            return new TorusFourierGradient(likelihood, residue - 1);
        } catch (IllegalArgumentException e) {
            throw new XMLParseException(e.getMessage());
        }
    }

    @Override
    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{
                AttributeRule.newIntegerRule(RESIDUE),
                new ElementRule(TorusFourierTreeLikelihood.class)
        };
    }

    @Override
    public String getParserDescription() {
        return "Raw FFT likelihood gradient for one torus residue's two precision diagonals, correlation, and jump rate";
    }

    @Override
    public Class getReturnType() { return TorusFourierGradient.class; }
}
