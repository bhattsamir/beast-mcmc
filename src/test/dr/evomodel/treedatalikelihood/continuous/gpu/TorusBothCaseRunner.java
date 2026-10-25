/*
 * TorusBothCaseRunner.java
 *
 * @author Samir Bhatt
 */

package test.dr.evomodel.treedatalikelihood.continuous.gpu;

import dr.evomodel.treedatalikelihood.continuous.gpu.TorusLandingGpuEngine;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Evaluates {@link TorusLandingGpuEngine} (the GPU likelihood) on the case files of make_h3_cases.py / write_cases.py,
 * for the combined model (landing AND additive jumps): every line
 *     evalb S00 S01 S10 S11 lambda_L lambda_A s_J
 * (covariance, landing rate, additive rate, additive jump SD in radians) is evaluated, and lambda_A = 0 lines
 * through the landing-only entry point as well, so the two paths can be compared. Other case lines ("eval", "frac")
 * are ignored. Prints one line per evaluation:  case index logL status [logL_landing_entry].
 *
 * usage: java -cp ... test.dr.evomodel.treedatalikelihood.continuous.gpu.TorusBothCaseRunner DEVICE CASE...
 *
 * @author Samir Bhatt
 */
public final class TorusBothCaseRunner {

    private TorusBothCaseRunner() { }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: TorusBothCaseRunner DEVICE CASE...");
            System.exit(2);
        }
        final int device = Integer.parseInt(args[0]);
        for (int i = 1; i < args.length; i++) runCase(new File(args[i]), device);
    }

    private static void runCase(File caseFile, int device) throws IOException {
        String name = null;
        int n = 0, nn = 0, root = -1;
        final List<double[]> nodeLines = new ArrayList<double[]>();
        final List<double[]> comps = new ArrayList<double[]>();
        final List<double[]> evals = new ArrayList<double[]>();
        double[][] se = null;
        try (BufferedReader in = new BufferedReader(new FileReader(caseFile))) {
            String ln;
            while ((ln = in.readLine()) != null) {
                final String[] f = ln.trim().split("\\s+");
                if (f.length == 0 || f[0].isEmpty()) continue;
                switch (f[0]) {
                    case "case": name = f[1]; break;
                    case "n": n = Integer.parseInt(f[1]); break;
                    case "nodes": nn = Integer.parseInt(f[1]); root = Integer.parseInt(f[3]); break;
                    case "node": nodeLines.add(doubles(f, 1, 7)); break;
                    case "sigma_eps": { final double[] v = doubles(f, 1, 4); se = new double[][]{{v[0], v[1]}, {v[2], v[3]}}; break; }
                    case "comp": comps.add(doubles(f, 1, 7)); break;
                    case "evalb": evals.add(doubles(f, 1, 7)); break;
                    case "frac": case "eval": break;
                    default: throw new IOException("unknown line in " + caseFile + ": " + ln);
                }
            }
        }
        final double[] branch = new double[nn];
        final List<List<Integer>> ch = new ArrayList<List<Integer>>();
        for (int v = 0; v < nn; v++) ch.add(new ArrayList<Integer>());
        final double[][][] tips = new double[1][nn][];
        for (double[] d : nodeLines) {                   // children of a node: its node lines in file order
            final int v = (int) d[0], p = (int) d[1];
            branch[v] = d[2];
            if (p >= 0) ch.get(p).add(v);
            if (d[3] != 0 && d[4] != 0) tips[0][v] = new double[]{d[5], d[6]};
        }
        final int[][] children = new int[nn][];
        for (int v = 0; v < nn; v++) {
            children[v] = new int[ch.get(v).size()];
            for (int i = 0; i < children[v].length; i++) children[v][i] = ch.get(v).get(i);
        }
        final double[] w = new double[comps.size()];
        final double[][] mu = new double[comps.size()][];
        final double[][][] s = new double[comps.size()][][];
        for (int k = 0; k < w.length; k++) {
            final double[] c = comps.get(k);
            w[k] = c[0];
            mu[k] = new double[]{c[1], c[2]};
            s[k] = new double[][]{{c[3], c[4]}, {c[5], c[6]}};
        }
        try (TorusLandingGpuEngine eng = new TorusLandingGpuEngine(n, children, root, branch, tips, w, mu, s, se, device, 1)) {
            for (int e = 0; e < evals.size(); e++) {
                final double[] v = evals.get(e);
                final double[][][] sigma = {{{v[0], v[1]}, {v[2], v[3]}}};
                final double[] ll = new double[1];
                final int[] st = new int[1];
                eng.logLikelihood(new int[]{0}, sigma, new double[]{v[4]}, new double[]{v[5]}, new double[]{v[6]}, ll, st);
                String extra = "";
                if (v[5] == 0.0) {                       // the landing-only entry point at the same point
                    final double[] l2 = new double[1];
                    final int[] s2 = new int[1];
                    eng.logLikelihood(new int[]{0}, sigma, new double[]{v[4]}, l2, s2);
                    extra = " " + Double.toString(l2[0]);
                }
                System.out.println(String.format(Locale.ROOT, "%s %d %s %d%s", name, e, Double.toString(ll[0]), st[0], extra));
            }
        }
    }

    private static double[] doubles(String[] f, int from, int count) {
        final double[] v = new double[count];
        for (int i = 0; i < count; i++) v[i] = Double.parseDouble(f[from + i]);
        return v;
    }
}
