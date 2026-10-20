/*
 * TorusLandingCountCaseRunner.java
 *
 * @author Samir Bhatt
 */

package test.dr.evomodel.treedatalikelihood.continuous.gpu;

import dr.evomodel.treedatalikelihood.continuous.gpu.TorusLandingCountEngine;
import dr.evomodel.treedatalikelihood.continuous.gpu.TorusLandingGpuEngine;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Runs {@link TorusLandingCountEngine} on the validation harness's synthetic case files (write_cases.py format) and
 * writes one count_io run per case (format torus-counts-v1, validation mode: records.tsv + N.bin), for
 * check_counts_java.py check --set case --cases DIR.
 *
 * usage: java -cp ... test.dr.evomodel.treedatalikelihood.continuous.gpu.TorusLandingCountCaseRunner OUT_DIR DEVICE CASE...
 *
 * @author Samir Bhatt
 */
public final class TorusLandingCountCaseRunner {

    private TorusLandingCountCaseRunner() { }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("usage: TorusLandingCountCaseRunner OUT_DIR DEVICE CASE...");
            System.exit(2);
        }
        final File outDir = new File(args[0]);
        final int device = Integer.parseInt(args[1]);
        for (int i = 2; i < args.length; i++) runCase(new File(args[i]), outDir, device);
    }

    private static void runCase(File caseFile, File outDir, int device) throws IOException {
        String name = null;
        int n = 0, nn = 0, root = -1;
        final List<double[]> nodeLines = new ArrayList<double[]>();
        final List<double[]> comps = new ArrayList<double[]>();
        final List<double[]> evals = new ArrayList<double[]>();
        double[][] se = null;
        String fracFile = null;
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
                    case "frac": fracFile = f[1]; break;
                    case "eval": evals.add(doubles(f, 1, 8)); break;
                    default: throw new IOException("unknown line in " + caseFile + ": " + ln);
                }
            }
        }
        final int[] parent = new int[nn];
        final double[] branch = new double[nn];
        final List<List<Integer>> ch = new ArrayList<List<Integer>>();
        for (int v = 0; v < nn; v++) ch.add(new ArrayList<Integer>());
        final double[][][] tips = new double[1][nn][];
        final boolean[] isTip = new boolean[nn];
        for (double[] d : nodeLines) {                   // children of a node: its node lines in file order
            final int v = (int) d[0], p = (int) d[1];
            parent[v] = p;
            branch[v] = d[2];
            if (p >= 0) ch.get(p).add(v);
            isTip[v] = d[3] != 0;
            if (isTip[v] && d[4] != 0) tips[0][v] = new double[]{d[5], d[6]};
        }
        final int[][] children = new int[nn][];
        for (int v = 0; v < nn; v++) {
            children[v] = new int[ch.get(v).size()];
            for (int i = 0; i < children[v].length; i++) children[v][i] = ch.get(v).get(i);
        }
        final int K = TorusLandingCountEngine.K;
        final double[] w = new double[comps.size()];
        final double[][] mu = new double[comps.size()][];
        final double[][][] s = new double[comps.size()][][];
        for (int k = 0; k < w.length; k++) {
            final double[] c = comps.get(k);
            w[k] = c[0];
            mu[k] = new double[]{c[1], c[2]};
            s[k] = new double[][]{{c[3], c[4]}, {c[5], c[6]}};
        }
        final double[][] frac = readFrac(new File(caseFile.getParentFile(), fracFile), K, n);

        final File run = new File(outDir, name);
        if (!run.mkdirs() && !run.isDirectory()) throw new IOException("cannot create " + run);
        final String[] clade = cladeKeys(children, root, nn);
        try (PrintWriter meta = new PrintWriter(new File(run, "meta.json"), "UTF-8")) {
            final StringBuilder b = new StringBuilder();
            b.append("{\"format\": \"torus-counts-v1\", \"K\": 3, \"regions\": [\"helical\", \"extended\", \"alpha_L\"], ");
            b.append("\"nodes\": ").append(nn).append(", \"root\": ").append(root).append(", \"parent\": [");
            for (int v = 0; v < nn; v++) b.append(v == 0 ? "" : ", ").append(parent[v]);
            b.append("], \"branch\": [");
            for (int v = 0; v < nn; v++) b.append(v == 0 ? "" : ", ").append('"').append(Double.toString(branch[v])).append('"');
            b.append("], \"clade\": [");
            for (int v = 0; v < nn; v++) b.append(v == 0 ? "" : ", ").append('"').append(clade[v]).append('"');
            b.append("], \"residues\": [\"").append(name).append("\"], \"grid_run\": ").append(n)
                    .append(", \"count_grid\": ").append(n).append(", \"refine_grid\": ").append(2 * n)
                    .append(", \"neg_tol\": 1e-9, \"region_md5\": {}, \"mode\": \"validation\", ")
                    .append("\"device_names\": [\"cuda:").append(device).append("\"], \"build\": \"case runner\"}");
            meta.println(b);
        }
        final TreeMap<Double, TorusLandingCountEngine> engines = new TreeMap<Double, TorusLandingCountEngine>();
        try (PrintWriter rec = new PrintWriter(new File(run, "records.tsv"), "UTF-8");
             DataOutputStream nbin = new DataOutputStream(new FileOutputStream(new File(run, "N.bin")))) {
            rec.println("state\tresidue_index\tresidue_id\tP00\tP01\tP10\tP11\tlambda\tS00\tS01\tS10\tS11\tgrid_used\t"
                    + "logL_count\tlogL_engine\tk0\tneg\tn_panels\tn_terms\tstatus\tsec_inside\tsec_total");
            for (int e = 0; e < evals.size(); e++) {
                final double[] v = evals.get(e);
                final double[][] sigma = {{v[0], v[1]}, {v[2], v[3]}};
                final double lambda = v[4], c0 = v[5], c1 = v[6];
                final int refine = (int) v[7];
                TorusLandingCountEngine eng = engines.get(c0);
                if (eng == null) {
                    eng = new TorusLandingCountEngine(n, children, root, branch, tips, w, mu, s, se, frac, c0, device);
                    engines.put(c0, eng);
                }
                final TorusLandingCountEngine.Result r = eng.count(0, sigma, lambda, c1, refine);
                double min = 0.0, sum = 0.0;
                final ByteBuffer bb = ByteBuffer.allocate(8 * nn * K * K).order(ByteOrder.LITTLE_ENDIAN);
                for (int u = 0; u < nn; u++) {
                    for (int a = 0; a < K; a++) {
                        for (int c = 0; c < K; c++) {
                            final double x = r.N[u][a][c];
                            bb.putDouble(x);
                            min = Math.min(min, x);
                            sum += x;
                        }
                    }
                }
                nbin.write(bb.array());
                final double neg = Math.min(0.0, min) / Math.max(sum, 1e-300);
                final double[][] P = TorusLandingGpuEngine.invertPrecision(sigma);
                rec.println(String.join("\t", Integer.toString(e), "0", name, d(P[0][0]), d(P[0][1]), d(P[1][0]), d(P[1][1]),
                        d(lambda), d(sigma[0][0]), d(sigma[0][1]), d(sigma[1][0]), d(sigma[1][1]), Integer.toString(n),
                        d(r.logL), "NaN", d(r.k0), d(neg), Long.toString(r.nPanels), Long.toString(r.nTerms),
                        Integer.toString(r.status), d(r.secInside), d(r.secTotal)));
                System.out.println(String.format(Locale.ROOT, "%s eval %d: logL %.12g, k0 %.1e, changes %.6g, panels %d, terms %d, %.2f s [model %.2f inside %.2f launch %.2f sync %.2f cross %.2f = plan %.2f upload %.2f launch %.2f wait %.2f]",
                        name, e, r.logL, r.k0, changes(r.N), r.nPanels, r.nTerms, r.secTotal, r.secModel, r.secInside, r.secLaunch, r.secSync, r.secCross, r.secPlan, r.secUpload, r.secKernel, r.secWait));
            }
        } finally {
            for (TorusLandingCountEngine eng : engines.values()) eng.close();
        }
    }

    static double changes(double[][][] N) {
        double c = 0.0;
        for (double[][] Nv : N) for (int a = 0; a < Nv.length; a++) for (int b = 0; b < Nv.length; b++) if (a != b) c += Nv[a][b];
        return c;
    }

    private static String d(double x) { return Double.toString(x); }

    private static double[] doubles(String[] f, int from, int count) {
        final double[] v = new double[count];
        for (int i = 0; i < count; i++) v[i] = Double.parseDouble(f[from + i]);
        return v;
    }

    /** float64 little-endian (K, n, n) region shares, C order */
    static double[][] readFrac(File f, int K, int n) throws IOException {
        final byte[] raw = new byte[(int) f.length()];
        try (FileInputStream in = new FileInputStream(f)) {
            int off = 0, r;
            while (off < raw.length && (r = in.read(raw, off, raw.length - off)) > 0) off += r;
        }
        if (raw.length != 8L * K * n * n) throw new IOException(f + ": " + raw.length + " bytes, expected " + 8L * K * n * n);
        final ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        final double[][] frac = new double[K][n * n];
        for (int a = 0; a < K; a++) for (int i = 0; i < n * n; i++) frac[a][i] = bb.getDouble();
        return frac;
    }

    /** count_io.clade_keys with tip names t<v> */
    static String[] cladeKeys(int[][] children, int root, int nn) {
        final String[] key = new String[nn];
        final List<Integer> order = new ArrayList<Integer>();
        final java.util.ArrayDeque<Integer> st = new java.util.ArrayDeque<Integer>();
        st.push(root);
        while (!st.isEmpty()) { final int v = st.pop(); order.add(v); for (int c : children[v]) st.push(c); }
        final List<TreeSet<String>> sets = new ArrayList<TreeSet<String>>();
        for (int v = 0; v < nn; v++) sets.add(new TreeSet<String>());
        for (int i = order.size() - 1; i >= 0; i--) {
            final int v = order.get(i);
            if (children[v].length == 0) sets.get(v).add("t" + v);
            else for (int c : children[v]) sets.get(v).addAll(sets.get(c));
            key[v] = String.join("|", sets.get(v));
        }
        return key;
    }
}
