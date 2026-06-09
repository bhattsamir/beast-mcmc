/*
 * SubtreeWrapOperatorParser.java
 *
 * @author Samir Bhatt
 */

package dr.evomodelxml.operators;

import dr.evomodel.operators.SubtreeWrapOperator;
import dr.evomodel.tree.TreeModel;
import dr.evomodel.treedatalikelihood.continuous.WrappedContinuousTraitDataModel;
import dr.inference.operators.MCMCOperator;
import dr.xml.*;

public class SubtreeWrapOperatorParser extends AbstractXMLObjectParser {

    public static final String SUBTREE_WRAP_OPERATOR = "subtreeWrapOperator";

    @Override
    public String getParserName() {
        return SUBTREE_WRAP_OPERATOR;
    }

    @Override
    public Object parseXMLObject(XMLObject xo) throws XMLParseException {
        double weight = xo.getDoubleAttribute(MCMCOperator.WEIGHT);
        TreeModel treeModel = (TreeModel) xo.getChild(TreeModel.class);
        WrappedContinuousTraitDataModel dataModel =
                (WrappedContinuousTraitDataModel) xo.getChild(WrappedContinuousTraitDataModel.class);
        return new SubtreeWrapOperator(treeModel, dataModel, weight);
    }

    @Override
    public String getParserDescription() {
        return "Block-flip MCMC operator: pick a clade, pick a (residue, dim), "
             + "flip all wraps in the clade by +/-1.";
    }

    @Override
    public Class getReturnType() {
        return SubtreeWrapOperator.class;
    }

    @Override
    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{
                AttributeRule.newDoubleRule(MCMCOperator.WEIGHT),
                new ElementRule(TreeModel.class),
                new ElementRule(WrappedContinuousTraitDataModel.class),
        };
    }
}
