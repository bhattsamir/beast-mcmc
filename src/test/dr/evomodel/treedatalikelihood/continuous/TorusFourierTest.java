/*
 * TorusFourierTest.java
 *
 * @author Samir Bhatt
 */

package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Validates the FFT torus pruning against reference values from the standalone
 * Python prototype (torus_fourier_pruning.py): the kernel reproduces the analytic
 * two-tip likelihood, and the full tree likelihood matches the prototype's
 * pruning at p = 1, 2, 3.
 *
 * @author Samir Bhatt
 */
public class TorusFourierTest extends TestCase {

    private static final String NEWICK = "((A:1.0,B:1.0):0.5,C:1.5);";

    /** Kernel only: analytic two-tip likelihood, grid-independent. */
    public void testKernelTwoTip() {
        TorusFourierKernel g = new TorusFourierKernel(2, 128);
        double[][] sigma = {{0.35, 0.08}, {0.08, 0.22}};   // covariance
        double[] mA = g.leafKernel(new double[]{0.4, 1.1}, sigma, 0.7);
        double[] mB = g.leafKernel(new double[]{-1.2, 0.6}, sigma, 1.4);
        int n = g.getGridCount();
        double[] v = new double[n];
        Arrays.fill(v, 1.0);
        double logScale = 0.0;
        logScale += multiplyInto(v, mA);
        logScale += multiplyInto(v, mB);
        double ll = logScale + Math.log(g.rootIntegral(v));
        assertEquals(-6.69298747149, ll, 1.0e-6);
    }

    public void testLikelihoodP1() throws Exception {
        Map<String, double[]> obs = new HashMap<String, double[]>();
        obs.put("A", new double[]{0.4});
        obs.put("B", new double[]{-1.2});
        obs.put("C", new double[]{0.9});
        double ll = logLikelihood(1, 64, new double[][]{{1.0}}, obs);
        assertEquals(-5.4429860970, ll, 1.0e-6);
    }

    public void testLikelihoodP2() throws Exception {
        Map<String, double[]> obs = new HashMap<String, double[]>();
        obs.put("A", new double[]{0.4, 1.1});
        obs.put("B", new double[]{-1.2, 0.6});
        obs.put("C", new double[]{0.9, -0.3});
        double ll = logLikelihood(2, 64, new double[][]{{1.0, 0.1}, {0.1, 2.0}}, obs);
        assertEquals(-9.8717597169, ll, 1.0e-6);
    }

    public void testLikelihoodP3() throws Exception {
        Map<String, double[]> obs = new HashMap<String, double[]>();
        obs.put("A", new double[]{0.4, 1.1, -0.5});
        obs.put("B", new double[]{-1.2, 0.6, 0.8});
        obs.put("C", new double[]{0.9, -0.3, 0.2});
        double[][] prec = {{1.0, 0.05, 0.05}, {0.05, 2.0, 0.05}, {0.05, 0.05, 1.5}};
        double ll = logLikelihood(3, 16, prec, obs);
        assertEquals(-14.5216588673, ll, 1.0e-6);
    }

    // ---- helpers ----

    private static double multiplyInto(double[] v, double[] msg) {
        double max = 0.0;
        for (int k = 0; k < v.length; k++) {
            v[k] *= msg[k];
            if (v[k] > max) max = v[k];
        }
        double inv = 1.0 / max;
        for (int k = 0; k < v.length; k++) v[k] *= inv;
        return Math.log(max);
    }

    /** precision is given row-major; build the column-major MatrixParameter. */
    private static MatrixParameter precisionParameter(double[][] p) {
        int d = p.length;
        Parameter[] cols = new Parameter[d];
        for (int j = 0; j < d; j++) {
            double[] c = new double[d];
            for (int i = 0; i < d; i++) c[i] = p[i][j];
            cols[j] = new Parameter.Default(c);
        }
        return new MatrixParameter("p", cols);
    }

    private static double logLikelihood(int dim, int nGrid, double[][] precision,
                                        Map<String, double[]> obs) throws Exception {
        Tree tree = new NewickImporter(NEWICK).importTree(null);
        int nExt = tree.getExternalNodeCount();
        double[][] data = new double[nExt][];
        for (int e = 0; e < nExt; e++) {
            NodeRef ext = tree.getExternalNode(e);
            data[ext.getNumber()] = obs.get(tree.getNodeTaxon(ext).getId());
        }
        List<MatrixParameter> precs = new ArrayList<MatrixParameter>();
        precs.add(precisionParameter(precision));
        TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood(
                "test", tree, null, precs, new double[][][]{data}, new TorusFourierKernel(dim, nGrid));
        return like.getLogLikelihood();
    }
}
