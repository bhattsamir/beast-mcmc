/*
 * WrappedContinuousTraitDataModelParser.java
 *
 * @author Samir Bhatt
 */

package dr.evomodelxml.treedatalikelihood.continuous;

import dr.evolution.tree.Tree;
import dr.evomodel.treedatalikelihood.continuous.WrappedContinuousTraitDataModel;
import dr.evomodel.treedatalikelihood.continuous.cdi.PrecisionType;
import dr.evomodelxml.continuous.ContinuousTraitDataModelParser;
import dr.evomodelxml.treelikelihood.TreeTraitParserUtilities;
import dr.inference.model.CompoundParameter;
import dr.inference.model.Parameter;
import dr.xml.*;

public class WrappedContinuousTraitDataModelParser extends AbstractXMLObjectParser {

    public static final String WRAPPED_TRAITS = "wrappedContinuousTraitDataModel";
    public static final String WRAPS = "wraps";
    public static final String NUM_TRAITS = ContinuousTraitDataModelParser.NUM_TRAITS;

    @Override
    public Object parseXMLObject(XMLObject xo) throws XMLParseException {

        Tree treeModel = (Tree) xo.getChild(Tree.class);
        TreeTraitParserUtilities utilities = new TreeTraitParserUtilities();
        TreeTraitParserUtilities.TraitsAndMissingIndices returnValue =
                utilities.parseTraitsFromTaxonAttributes(xo, treeModel, true);

        CompoundParameter traitParameter = returnValue.traitParameter;
        boolean[] missingIndicators = returnValue.getMissingIndicators();
        String traitName = returnValue.traitName;
        boolean useMissingIndices = returnValue.useMissingIndices;

        int dimAll = traitParameter.getParameter(0).getDimension();
        int numTraits = xo.getAttribute(NUM_TRAITS, 1);
        int dimTrait = dimAll / numTraits;

        // FULL precision required because we shift the mean per-coordinate at the tips.
        PrecisionType precisionType = PrecisionType.FULL;

        if (xo.hasChildNamed(TreeTraitParserUtilities.JITTER)) {
            utilities.jitter(xo, dimTrait, missingIndicators);
        }

        XMLObject wrapsCxo = xo.getChild(WRAPS);
        Parameter wraps = (Parameter) wrapsCxo.getChild(Parameter.class);

        int expected = traitParameter.getParameterCount() * numTraits * dimTrait;
        if (wraps.getDimension() != expected) {
            // Auto-size if dimension is 1: expand and zero-fill.
            if (wraps.getDimension() == 1) {
                wraps.setDimension(expected);
                for (int i = 0; i < expected; ++i) wraps.setParameterValue(i, 0.0);
            } else {
                throw new XMLParseException(
                        "wraps parameter dimension " + wraps.getDimension()
                        + " does not match nTaxa(" + traitParameter.getParameterCount()
                        + ") * numTraits(" + numTraits + ") * dimTrait(" + dimTrait + ") = " + expected);
            }
        }

        // Bounds: window for the integer wraps. +/- 3 covers sigma^2 * depth
        // up to ~30 (the K_max=3 truncated wrapped-normal density has rel error
        // 4e-16 at our regime of sigma^2*depth <= 5; <=3e-5 even at sat=63%).
        wraps.addBounds(new Parameter.DefaultBounds(3.0, -3.0, wraps.getDimension()));

        return new WrappedContinuousTraitDataModel(traitName, traitParameter,
                missingIndicators, useMissingIndices, dimTrait, numTraits,
                precisionType, wraps);
    }

    @Override
    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{
                new ElementRule(Tree.class),
                new ElementRule(TreeTraitParserUtilities.TRAIT_PARAMETER, new XMLSyntaxRule[]{
                        new ElementRule(Parameter.class)
                }),
                new ElementRule(WRAPS, new XMLSyntaxRule[]{
                        new ElementRule(Parameter.class)
                }),
                AttributeRule.newIntegerRule(NUM_TRAITS, true),
                AttributeRule.newStringRule(TreeTraitParserUtilities.TRAIT_NAME, true)
        };
    }

    @Override
    public String getParserDescription() {
        return "Continuous trait data model with integer-valued wrap multipliers (data augmentation for wrapped BM on the torus).";
    }

    @Override
    public Class getReturnType() {
        return WrappedContinuousTraitDataModel.class;
    }

    @Override
    public String getParserName() {
        return WRAPPED_TRAITS;
    }
}
