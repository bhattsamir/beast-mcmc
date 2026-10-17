package test.dr.evomodel.treedatalikelihood.continuous.gpu;

import dr.app.beast.BeastParser;
import dr.app.beast.BeastVersion;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.evomodel.treedatalikelihood.continuous.gpu.TorusLandingGpuEngine;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import dr.xml.XMLObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Evaluates one residue's production landing log-likelihood at many parameter points: on the GPU (all points, in
 * batches) and, for every k-th point, with the production CPU class as well. The model is built by BEAST's own
 * parser from the production XML; the precision matrix of each point is the production compoundSymmetricMatrix's.
 * For importance-sampling checks of a sampler against an MCMC-free answer.
 *
 * usage: TorusLandingPointEvaluator residue.xml points.csv out.csv [cpuEvery]
 *   points.csv: header q1,q2,rho,lambda then one point per line
 *   out.csv:    q1,q2,rho,lambda,gpu_logL,status,cpu_logL (cpu NaN where not computed)
 */
public class TorusLandingPointEvaluator {

    public static void main(String[] args) throws Exception {
        try {
            run(args);
        } catch (Throwable t) {
            t.printStackTrace();
            System.exit(3);
        }
        System.exit(0);
    }

    static void run(String[] args) throws Exception {
        final File xml = new File(args[0]);
        final int cpuEvery = args.length > 3 ? Integer.parseInt(args[3]) : 0;
        final BeastParser parser = new BeastParser(new String[]{xml.getPath()}, null, false, false, false, new BeastVersion());
        final Map<String, XMLObject> store = parser.parse(new FileReader(xml), false);
        final TorusFourierTreeLikelihood tfl = (TorusFourierTreeLikelihood) store.get("tfl").getNativeObject();
        final Parameter diag = (Parameter) store.get("diag_0").getNativeObject();
        final Parameter off = (Parameter) store.get("off_0").getNativeObject();
        final Parameter lambda = (Parameter) store.get("lambda").getNativeObject();
        final MatrixParameter prec = (MatrixParameter) store.get("prec_0").getNativeObject();

        final double[][][] tipData = (double[][][]) field(tfl, "tipData");
        final Tree tree = (Tree) field(tfl, "tree");
        final TorusFourierKernel kernel = (TorusFourierKernel) field(tfl, "kernel");
        final Method bl = TorusFourierTreeLikelihood.class.getDeclaredMethod("branchLength", NodeRef.class);
        bl.setAccessible(true);
        final Method se = TorusFourierTreeLikelihood.class.getDeclaredMethod("sigmaEpsilonMatrix");
        se.setAccessible(true);
        final int nc = tree.getNodeCount();
        final int[][] ch = new int[nc][];
        final double[] br = new double[nc];
        for (int i = 0; i < nc; i++) {
            final NodeRef node = tree.getNode(i);
            ch[i] = new int[tree.getChildCount(node)];
            for (int c = 0; c < ch[i].length; c++) ch[i][c] = tree.getChild(node, c).getNumber();
            if (!tree.isRoot(node)) br[i] = (Double) bl.invoke(tfl, node);
        }

        final List<double[]> pts = new ArrayList<double[]>();
        try (BufferedReader in = new BufferedReader(new FileReader(args[1]))) {
            String line = in.readLine();                       // header
            while ((line = in.readLine()) != null) {
                if (line.trim().isEmpty()) continue;
                final String[] f = line.split(",");
                pts.add(new double[]{Double.parseDouble(f[0]), Double.parseDouble(f[1]), Double.parseDouble(f[2]),
                        Double.parseDouble(f[3])});
            }
        }
        final int K = pts.size();
        final double[][][] P = new double[K][2][2];
        for (int k = 0; k < K; k++) {
            final double[] p = pts.get(k);
            diag.setParameterValue(0, p[0]);
            diag.setParameterValue(1, p[1]);
            off.setParameterValue(0, p[2]);
            for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) P[k][a][b] = prec.getParameterValue(a, b);
        }
        final int batch = Math.min(K, Integer.getInteger("batch", 256));
        final TorusLandingGpuEngine engine = new TorusLandingGpuEngine(kernel.getGridSize(), ch,
                tree.getRoot().getNumber(), br, tipData, (double[]) field(tfl, "jumpW"), (double[][]) field(tfl, "jumpD"),
                (double[][][]) field(tfl, "jumpS"), (double[][]) se.invoke(tfl), Integer.getInteger("torus.gpu.device", 0), batch);
        final double[] gpu = new double[K];
        final int[] status = new int[K];
        final long t0 = System.nanoTime();
        for (int s = 0; s < K; s += batch) {
            final int b = Math.min(batch, K - s);
            final int[] idx = new int[b];
            final double[][][] sig = new double[b][][];
            final double[] lam = new double[b];
            for (int i = 0; i < b; i++) {
                sig[i] = TorusLandingGpuEngine.invertPrecision(P[s + i]);
                lam[i] = pts.get(s + i)[3];
            }
            final double[] ll = new double[b];
            final int[] st = new int[b];
            engine.logLikelihood(idx, sig, lam, ll, st);
            System.arraycopy(ll, 0, gpu, s, b);
            System.arraycopy(st, 0, status, s, b);
        }
        engine.close();
        System.out.printf("GPU: %d points in %.1f s%n", K, (System.nanoTime() - t0) / 1e9);

        final double[] cpu = new double[K];
        java.util.Arrays.fill(cpu, Double.NaN);
        if (cpuEvery > 0) {
            final long t1 = System.nanoTime();
            for (int k = 0; k < K; k += cpuEvery) {
                final double[] p = pts.get(k);
                diag.setParameterValue(0, p[0]);
                diag.setParameterValue(1, p[1]);
                off.setParameterValue(0, p[2]);
                lambda.setParameterValue(0, p[3]);
                try {
                    cpu[k] = tfl.getLogLikelihood();
                } catch (RuntimeException e) {
                    cpu[k] = Double.NaN;
                }
            }
            System.out.printf("CPU: %d points in %.1f s%n", (K + cpuEvery - 1) / cpuEvery, (System.nanoTime() - t1) / 1e9);
        }
        try (PrintWriter pw = new PrintWriter(args[2])) {
            pw.println("q1,q2,rho,lambda,gpu_logL,status,cpu_logL");
            for (int k = 0; k < K; k++) {
                final double[] p = pts.get(k);
                pw.printf("%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g%n", p[0], p[1], p[2], p[3], gpu[k], status[k], cpu[k]);
            }
        }
    }

    static Object field(Object o, String name) throws Exception {
        final Field f = o.getClass().getDeclaredField(name);
        f.setAccessible(true);
        return f.get(o);
    }
}
