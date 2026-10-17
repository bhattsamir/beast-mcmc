/*
 * TorusLogNormalRegressionPrior.java
 *
 * @author Samir Bhatt
 */

package dr.evomodel.treedatalikelihood.continuous.gpu;

import dr.inference.model.AbstractModelLikelihood;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Model;
import dr.inference.model.Parameter;
import dr.inference.model.Variable;
import dr.xml.Reportable;

/**
 * Hierarchical prior for per-residue positive rates: log x_r ~ Normal(z_r' beta, tau^2) independently given
 * beta and tau, with z_r row r of a fixed design matrix (an intercept column and any covariates). As a density
 * of x_r it includes the Jacobian -log x_r:
 *     log p(x | beta, tau) = sum_r [ -log x_r - log tau - log(2 pi)/2 - (log x_r - z_r' beta)^2 / (2 tau^2) ].
 * This is the second-stage model of results_pipeline/partial_pooling.py, so a pooled MCMC run and the post hoc
 * pooling of an unpooled run target the same posterior. Priors on beta and tau are separate BEAST elements.
 *
 * @author Samir Bhatt
 */
public class TorusLogNormalRegressionPrior extends AbstractModelLikelihood implements Reportable {

    public static final String NAME = "torusLogNormalRegressionPrior";
    private static final double HALF_LOG_2PI = 0.5 * Math.log(2.0 * Math.PI);

    private final Parameter x;
    private final Parameter beta;
    private final Parameter tau;
    private final double[][] z;            // [R][p], fixed
    private double logL, storedLogL;
    private boolean known = false, storedKnown = false;

    public TorusLogNormalRegressionPrior(String name, Parameter x, Parameter beta, Parameter tau, MatrixParameter design) {
        super(name);
        this.x = x;
        this.beta = beta;
        this.tau = tau;
        final int R = x.getDimension(), p = beta.getDimension();
        if (tau.getDimension() != 1) throw new IllegalArgumentException(NAME + ": tau must be a scalar");
        if (design == null) {
            if (p != 1) throw new IllegalArgumentException(NAME + ": without a design matrix beta must be the intercept only");
            z = new double[R][1];
            for (int r = 0; r < R; r++) z[r][0] = 1.0;
        } else {
            if (design.getRowDimension() != R || design.getColumnDimension() != p) {
                throw new IllegalArgumentException(NAME + ": design matrix must be " + R + " x " + p + ", got "
                        + design.getRowDimension() + " x " + design.getColumnDimension());
            }
            z = new double[R][p];
            for (int r = 0; r < R; r++) for (int j = 0; j < p; j++) z[r][j] = design.getParameterValue(r, j);
        }
        addVariable(x);
        addVariable(beta);
        addVariable(tau);
    }

    public Model getModel() { return this; }

    public double getLogLikelihood() {
        if (!known) {
            logL = calculate();
            known = true;
        }
        return logL;
    }

    private double calculate() {
        final double t = tau.getParameterValue(0);
        if (!(t > 0.0)) return Double.NEGATIVE_INFINITY;
        final double logT = Math.log(t);
        double s = 0.0;
        for (int r = 0; r < z.length; r++) {
            final double v = x.getParameterValue(r);
            if (!(v > 0.0)) return Double.NEGATIVE_INFINITY;
            double mean = 0.0;
            for (int j = 0; j < z[r].length; j++) mean += z[r][j] * beta.getParameterValue(j);
            final double lv = Math.log(v);
            final double e = (lv - mean) / t;
            s += -lv - logT - HALF_LOG_2PI - 0.5 * e * e;
        }
        return s;
    }

    public void makeDirty() { known = false; }
    protected void handleModelChangedEvent(Model model, Object object, int index) { known = false; }
    protected void handleVariableChangedEvent(Variable variable, int index, Parameter.ChangeType type) { known = false; }
    protected void storeState() { storedLogL = logL; storedKnown = known; }
    protected void restoreState() { logL = storedLogL; known = storedKnown; }
    protected void acceptState() { }

    public String getReport() {
        return NAME + "(" + getId() + "): logL = " + getLogLikelihood() + ", " + z.length + " elements, "
                + beta.getDimension() + " coefficients";
    }
}
