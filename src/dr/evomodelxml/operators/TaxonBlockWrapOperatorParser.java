/*
 * TaxonBlockWrapOperatorParser.java
 *
 * @author Samir Bhatt
 */

package dr.evomodelxml.operators;

import dr.evomodel.operators.TaxonBlockWrapOperator;
import dr.evomodel.treedatalikelihood.continuous.WrappedContinuousTraitDataModel;
import dr.inference.operators.MCMCOperator;
import dr.xml.*;

public class TaxonBlockWrapOperatorParser extends AbstractXMLObjectParser {

    public static final String TAXON_BLOCK_WRAP_OPERATOR = "taxonBlockWrapOperator";

    @Override
    public String getParserName() {
        return TAXON_BLOCK_WRAP_OPERATOR;
    }

    @Override
    public Object parseXMLObject(XMLObject xo) throws XMLParseException {
        double weight = xo.getDoubleAttribute(MCMCOperator.WEIGHT);
        WrappedContinuousTraitDataModel dataModel =
                (WrappedContinuousTraitDataModel) xo.getChild(WrappedContinuousTraitDataModel.class);
        return new TaxonBlockWrapOperator(dataModel, weight);
    }

    @Override
    public String getParserDescription() {
        return "MCMC operator: pick a tip and flip all its wraps (across trait dims) by +/-1.";
    }

    @Override
    public Class getReturnType() {
        return TaxonBlockWrapOperator.class;
    }

    @Override
    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{
                AttributeRule.newDoubleRule(MCMCOperator.WEIGHT),
                new ElementRule(WrappedContinuousTraitDataModel.class),
        };
    }
}
