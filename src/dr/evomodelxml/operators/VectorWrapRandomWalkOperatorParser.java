/*
 * VectorWrapRandomWalkOperatorParser.java
 *
 * @author Samir Bhatt
 */

package dr.evomodelxml.operators;

import dr.evomodel.operators.VectorWrapRandomWalkOperator;
import dr.evomodel.treedatalikelihood.continuous.WrappedContinuousTraitDataModel;
import dr.inference.operators.MCMCOperator;
import dr.xml.*;

public class VectorWrapRandomWalkOperatorParser extends AbstractXMLObjectParser {

    public static final String VECTOR_WRAP_RANDOM_WALK_OPERATOR = "vectorWrapRandomWalkOperator";
    public static final String BLOCK_SIZE = "blockSize";

    @Override
    public String getParserName() {
        return VECTOR_WRAP_RANDOM_WALK_OPERATOR;
    }

    @Override
    public Object parseXMLObject(XMLObject xo) throws XMLParseException {
        double weight = xo.getDoubleAttribute(MCMCOperator.WEIGHT);
        int blockSize = xo.getAttribute(BLOCK_SIZE, 10);
        if (blockSize < 1) {
            throw new XMLParseException(BLOCK_SIZE + " must be >= 1");
        }
        WrappedContinuousTraitDataModel dataModel =
                (WrappedContinuousTraitDataModel) xo.getChild(WrappedContinuousTraitDataModel.class);
        return new VectorWrapRandomWalkOperator(dataModel, blockSize, weight);
    }

    @Override
    public String getParserDescription() {
        return "MCMC operator: pick K distinct non-pinned wraps and flip each by independent +/-1.";
    }

    @Override
    public Class getReturnType() {
        return VectorWrapRandomWalkOperator.class;
    }

    @Override
    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{
                AttributeRule.newDoubleRule(MCMCOperator.WEIGHT),
                AttributeRule.newIntegerRule(BLOCK_SIZE, true),
                new ElementRule(WrappedContinuousTraitDataModel.class),
        };
    }
}
