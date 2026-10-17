/*
 * TorusLogNormalRegressionPriorParser.java
 *
 * @author Samir Bhatt
 */

package dr.evomodelxml.treedatalikelihood.continuous.gpu;

import dr.evomodel.treedatalikelihood.continuous.gpu.TorusLogNormalRegressionPrior;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import dr.xml.*;

/**
 * &lt;torusLogNormalRegressionPrior id="lambdaPrior"&gt;
 *     &lt;data&gt;&lt;parameter idref="lambda"/&gt;&lt;/data&gt;
 *     &lt;coefficients&gt;&lt;parameter idref="beta"/&gt;&lt;/coefficients&gt;      (intercept first)
 *     &lt;sd&gt;&lt;parameter idref="tau"/&gt;&lt;/sd&gt;
 *     &lt;designMatrix&gt;&lt;matrixParameter ...&gt;&lt;/designMatrix&gt;         (optional; R x p, rows = data elements)
 * &lt;/torusLogNormalRegressionPrior&gt;
 *
 * @author Samir Bhatt
 */
public class TorusLogNormalRegressionPriorParser extends AbstractXMLObjectParser {

    public String getParserName() { return TorusLogNormalRegressionPrior.NAME; }

    public Object parseXMLObject(XMLObject xo) throws XMLParseException {
        final Parameter x = (Parameter) xo.getElementFirstChild("data");
        final Parameter beta = (Parameter) xo.getElementFirstChild("coefficients");
        final Parameter tau = (Parameter) xo.getElementFirstChild("sd");
        final MatrixParameter design = xo.hasChildNamed("designMatrix")
                ? (MatrixParameter) xo.getElementFirstChild("designMatrix") : null;
        try {
            return new TorusLogNormalRegressionPrior(xo.getId(), x, beta, tau, design);
        } catch (IllegalArgumentException e) {
            throw new XMLParseException(e.getMessage());
        }
    }

    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{
                new ElementRule("data", new XMLSyntaxRule[]{new ElementRule(Parameter.class)}),
                new ElementRule("coefficients", new XMLSyntaxRule[]{new ElementRule(Parameter.class)}),
                new ElementRule("sd", new XMLSyntaxRule[]{new ElementRule(Parameter.class)}),
                new ElementRule("designMatrix", new XMLSyntaxRule[]{new ElementRule(MatrixParameter.class)}, true),
        };
    }

    public String getParserDescription() {
        return "log x_r ~ Normal(z_r' beta, tau^2) for the elements of a positive parameter (Jacobian included).";
    }

    public Class getReturnType() { return TorusLogNormalRegressionPrior.class; }
}
