package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evomodelxml.treedatalikelihood.continuous.TorusHmcTransformParser.BoundedLogit;
import junit.framework.TestCase;

/** Check the bounded correlation transform, including its density Jacobian. */
public class TorusHmcTransformTest extends TestCase {
    public void testTransformedDensityGradient() {
        BoundedLogit transform = new BoundedLogit(0.95, -0.95);
        for (double z : new double[]{-9.0, -3.0, -0.7, 0.0, 2.1, 8.0}) {
            double x = transform.inverse(z), eps = 1e-5;
            double plus = transform.inverse(z + eps), minus = transform.inverse(z - eps);
            // Nonuniform original density proportional to exp(-2*x^2 + x).
            double numeric = ((-2 * plus * plus + plus - transform.logJacobian(plus))
                    - (-2 * minus * minus + minus - transform.logJacobian(minus))) / (2 * eps);
            assertEquals(numeric, transform.updateGradientLogDensity(-4 * x + 1, x), 2e-7);
            assertEquals(z, transform.transform(x), 1e-11);
            assertEquals((plus - minus) / (2 * eps), transform.gradientInverse(z), 1e-10);
        }
    }

    public void testStableInverseAndBounds() {
        BoundedLogit transform = new BoundedLogit(0.8, -0.6);
        assertEquals(-0.6, transform.inverse(-1000.0), 0.0);
        assertEquals(0.8, transform.inverse(1000.0), 0.0);
        assertEquals(0.0, transform.gradientInverse(-1000.0), 0.0);
        assertEquals(0.0, transform.gradientInverse(1000.0), 0.0);
        assertFalse(transform.isInInteriorDomain(-0.6));
        assertFalse(transform.isInInteriorDomain(0.8));
        assertTrue(transform.isInInteriorDomain(0.0));
    }
}
