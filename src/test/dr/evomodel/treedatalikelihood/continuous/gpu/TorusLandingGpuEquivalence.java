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
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;

/**
 * Gate G1a/G1b of the GPU engine: the production landing likelihood, evaluated in this JVM by the production
 * classes on objects built by BEAST's own XML parser from the production XMLs, against the GPU engine fed with
 * the arrays those very objects hold (read by reflection: tip data, landing mixture, Sigma_eps, and branch
 * lengths from the production branchLength method). Also compares with the likelihood the production jar
 * logged on HX2 at the same states.
 *
 * Parameter sets per residue: the XML's start values; four states from the HX2 trace log (30% burn-in,
 * evenly spaced); four random sets over a wide box; lambda = 0; a huge-diffusion and a tiny-diffusion extreme.
 *
 * usage: java ... TorusLandingGpuEquivalence xmlDir logDir threads out.csv [nGridOverride] [residue,...]
 * The production values are cached in out.csv.cache; with -Dtorus.gpu.cachedCpu=true the CPU phase is skipped
 * and the GPU engine is checked against the cached production values (for re-validating kernel changes).
 */
public class TorusLandingGpuEquivalence {

    static final class Residue implements java.io.Serializable {
        String name;
        double[][] tips;                   // [node][2] or null
        List<double[]> sets = new ArrayList<double[]>();     // q1, q2, rho, lambda
        List<String> labels = new ArrayList<String>();
        List<Double> logged = new ArrayList<Double>();       // jar's logged likelihood, NaN if none
        double[] cpu;                      // production logL per set
        String[] cpuError;                 // exception text where production threw
        double[][][] precision;            // [set][2][2] production precision values
    }

    public static void main(String[] args) throws Exception {
        try {
            run(args);
        } catch (Throwable t) {
            t.printStackTrace();
            System.exit(3);
        }
    }

    static void run(String[] args) throws Exception {
        final File xmlDir = new File(args[0]);
        final File logDir = new File(args[1]);
        final int threads = Integer.parseInt(args[2]);
        final File out = new File(args[3]);
        final int ngridOverride = args.length > 4 ? Integer.parseInt(args[4]) : 0;
        final List<String> only = args.length > 5 ? Arrays.asList(args[5].split(",")) : null;

        final File cache = new File(out.getPath() + ".cache");
        final boolean cached = Boolean.getBoolean("torus.gpu.cachedCpu");
        final File[] xmls = xmlDir.listFiles((d, n) -> n.matches("res_.+_c1\\.xml"));
        Arrays.sort(xmls);
        final List<Residue> residues = new ArrayList<Residue>();
        final ExecutorService pool = Executors.newFixedThreadPool(threads);
        final Semaphore inflight = new Semaphore(2 * threads);
        final List<Future<?>> futures = new ArrayList<Future<?>>();

        int[][] children = null;
        int root = -1;
        double[] branch = null;
        double[] w = null;
        double[][] mu = null;
        double[][][] s = null;
        double[][] sigmaEps = null;
        int ngrid = -1;
        final long t0 = System.nanoTime();

        if (cached) {
            try (java.io.ObjectInputStream in = new java.io.ObjectInputStream(new java.io.FileInputStream(cache))) {
                final Object[] o = (Object[]) in.readObject();
                children = (int[][]) o[0]; root = (Integer) o[1]; branch = (double[]) o[2]; w = (double[]) o[3];
                mu = (double[][]) o[4]; s = (double[][][]) o[5]; sigmaEps = (double[][]) o[6]; ngrid = (Integer) o[7];
                @SuppressWarnings("unchecked") final List<Residue> rs = (List<Residue>) o[8];
                for (Residue r : rs) if (only == null || only.contains(r.name)) residues.add(r);
            }
            System.out.printf("production values from cache %s (%d residues)%n", cache, residues.size());
        }
        for (File xml : cached ? new File[0] : xmls) {
            final String name = xml.getName().replaceAll("^res_(.+)_c1\\.xml$", "$1");
            if (only != null && !only.contains(name)) continue;
            String text = new String(java.nio.file.Files.readAllBytes(xml.toPath()), "UTF-8");
            if (ngridOverride > 0) text = text.replaceFirst("nGrid=\"\\d+\"", "nGrid=\"" + ngridOverride + "\"");
            final BeastParser parser = new BeastParser(new String[]{xml.getPath()}, null, false, false, false,
                    new BeastVersion());
            final Map<String, XMLObject> store = parser.parse(new java.io.StringReader(text), false);
            final TorusFourierTreeLikelihood tfl = (TorusFourierTreeLikelihood) store.get("tfl").getNativeObject();
            final Parameter diag = (Parameter) store.get("diag_0").getNativeObject();
            final Parameter off = (Parameter) store.get("off_0").getNativeObject();
            final Parameter lambda = (Parameter) store.get("lambda").getNativeObject();
            final MatrixParameter prec = (MatrixParameter) store.get("prec_0").getNativeObject();

            // production arrays, by reflection
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
            final double[] wi = (double[]) field(tfl, "jumpW");
            final double[][] mui = (double[][]) field(tfl, "jumpD");
            final double[][][] si = (double[][][]) field(tfl, "jumpS");
            final double[][] sei = (double[][]) se.invoke(tfl);
            if (children == null) {
                children = ch; root = tree.getRoot().getNumber(); branch = br; w = wi; mu = mui; s = si;
                sigmaEps = sei; ngrid = kernel.getGridSize();
            } else {
                if (!Arrays.deepEquals(children, ch) || root != tree.getRoot().getNumber() || !Arrays.equals(branch, br)) {
                    throw new IllegalStateException(name + ": tree differs from the first residue's");
                }
                if (!Arrays.equals(w, wi) || !Arrays.deepEquals(mu, mui) || !Arrays.deepEquals(s, si)
                        || !Arrays.deepEquals(sigmaEps, sei) || ngrid != kernel.getGridSize()) {
                    throw new IllegalStateException(name + ": landing mixture, Sigma_eps or grid differs");
                }
            }

            final Residue res = new Residue();
            res.name = name;
            res.tips = tipData[0];
            buildSets(res, diag, off, lambda, new File(logDir, "res_" + name + "_c1.log"), residues.size());
            residues.add(res);

            inflight.acquire();
            futures.add(pool.submit(() -> {
                try {
                    final int k = res.sets.size();
                    res.cpu = new double[k];
                    res.cpuError = new String[k];
                    res.precision = new double[k][2][2];
                    for (int j = 0; j < k; j++) {
                        final double[] p = res.sets.get(j);
                        diag.setParameterValue(0, p[0]);
                        diag.setParameterValue(1, p[1]);
                        off.setParameterValue(0, p[2]);
                        lambda.setParameterValue(0, p[3]);
                        for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) res.precision[j][a][b] = prec.getParameterValue(a, b);
                        try {
                            res.cpu[j] = tfl.getLogLikelihood();
                        } catch (RuntimeException e) {
                            res.cpu[j] = Double.NaN;
                            res.cpuError[j] = e.getClass().getSimpleName() + ": " + e.getMessage();
                        }
                    }
                } finally {
                    // BEAST keeps static references to parsed models (citations), so the production
                    // partials (2 x nodes x nGrid^2 doubles per residue) must be released by hand
                    try {
                        final Field f = TorusFourierTreeLikelihood.class.getDeclaredField("partials");
                        f.setAccessible(true);
                        f.set(tfl, null);
                    } catch (ReflectiveOperationException e) {
                        throw new IllegalStateException(e);
                    }
                    inflight.release();
                }
                return null;
            }));
        }
        for (Future<?> f : futures) f.get();
        pool.shutdown();
        if (!cached) {
            try (java.io.ObjectOutputStream oo = new java.io.ObjectOutputStream(new java.io.FileOutputStream(cache))) {
                oo.writeObject(new Object[]{children, root, branch, w, mu, s, sigmaEps, ngrid, new ArrayList<Residue>(residues)});
            }
        }
        final double tCpu = (System.nanoTime() - t0) / 1e9;
        System.out.printf("production (CPU): %d residues x %d sets in %.1f s on %d threads, nGrid %d%n",
                residues.size(), residues.get(0).sets.size(), tCpu, threads, ngrid);

        // GPU: one engine for every residue, one batched call per parameter set
        final int R = residues.size();
        final double[][][] tips = new double[R][][];
        for (int r = 0; r < R; r++) tips[r] = residues.get(r).tips;
        final long t1 = System.nanoTime();
        final TorusLandingGpuEngine engine = new TorusLandingGpuEngine(ngrid, children, root, branch, tips, w, mu, s,
                sigmaEps, Integer.getInteger("torus.gpu.device", 0), R);
        System.out.printf("GPU engine: %d residues, %d partial buffers alive at most, built in %.1f s%n", R,
                engine.getPartialBufferCount(), (System.nanoTime() - t1) / 1e9);
        final int K = residues.get(0).sets.size();
        final double[][] gpu = new double[R][K];
        final int[][] status = new int[R][K];
        final int[] all = new int[R];
        for (int r = 0; r < R; r++) all[r] = r;
        double gpuSeconds = 0.0;
        for (int j = 0; j < K; j++) {
            final double[][][] sig = new double[R][][];
            final double[] lam = new double[R];
            for (int r = 0; r < R; r++) {
                sig[r] = TorusLandingGpuEngine.invertPrecision(residues.get(r).precision[j]);
                lam[r] = residues.get(r).sets.get(j)[3];
            }
            final double[] ll = new double[R];
            final int[] st = new int[R];
            final long ts = System.nanoTime();
            engine.logLikelihood(all, sig, lam, ll, st);
            gpuSeconds += (System.nanoTime() - ts) / 1e9;
            for (int r = 0; r < R; r++) { gpu[r][j] = ll[r]; status[r][j] = st[r]; }
        }
        engine.close();
        System.out.printf("GPU: %d sets x %d residues, %.2f s per batched evaluation of all residues%n", K, R, gpuSeconds / K);

        // compare
        int compared = 0, infAgree = 0, mismatch = 0, throwAgree = 0, loggedCompared = 0;
        double maxAbs = 0.0, maxRel = 0.0, maxLogged = 0.0, maxLoggedCpu = 0.0;
        String worst = "";
        try (PrintWriter pw = new PrintWriter(out)) {
            pw.println("residue,set,label,q1,q2,rho,lambda,cpu,gpu,status,diff,logged,cpu_error");
            for (int r = 0; r < R; r++) {
                final Residue res = residues.get(r);
                for (int j = 0; j < K; j++) {
                    final double c = res.cpu[j], g = gpu[r][j];
                    final double[] p = res.sets.get(j);
                    final String err = res.cpuError[j];
                    double diff = Double.NaN;
                    if (err != null) {
                        // production threw: the GPU must report the matching failure, not a number
                        final boolean match = err.startsWith("ArithmeticException") && Double.isNaN(g)
                                && (status[r][j] & (TorusLandingGpuEngine.STATUS_INVALID_PARTIAL
                                | TorusLandingGpuEngine.STATUS_NONFINITE_MESSAGE)) != 0;
                        if (match) throwAgree++; else { mismatch++; worst += "\n  THROW " + res.name + " set " + j + ": " + err + " gpu " + g; }
                    } else if (Double.isInfinite(c) || Double.isInfinite(g)) {
                        if (c == g) infAgree++; else { mismatch++; worst += "\n  INF " + res.name + " set " + j + " cpu " + c + " gpu " + g; }
                    } else if (Double.isNaN(g)) {
                        mismatch++; worst += "\n  NAN " + res.name + " set " + j + " cpu " + c + " status " + status[r][j];
                    } else {
                        diff = g - c;
                        compared++;
                        maxAbs = Math.max(maxAbs, Math.abs(diff));
                        maxRel = Math.max(maxRel, Math.abs(diff) / Math.max(1.0, Math.abs(c)));
                    }
                    final double lg = res.logged.get(j);
                    if (!Double.isNaN(lg) && err == null && Double.isFinite(c)) {
                        loggedCompared++;
                        maxLogged = Math.max(maxLogged, Math.abs(g - lg));
                        maxLoggedCpu = Math.max(maxLoggedCpu, Math.abs(c - lg));
                    }
                    pw.printf("%s,%d,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.3e,%.17g,%s%n", res.name, j, res.labels.get(j),
                            p[0], p[1], p[2], p[3], c, g, status[r][j], diff, lg, err == null ? "" : err.replace(',', ';'));
                }
            }
        }
        System.out.printf("finite pairs: %d, max |gpu - cpu| %.3e nats (max relative %.3e)%n", compared, maxAbs, maxRel);
        System.out.printf("-inf agreed: %d, production throws matched by GPU failure status: %d, mismatches: %d%n",
                infAgree, throwAgree, mismatch);
        System.out.printf("against the jar's logged likelihood (HX2 job 981654): %d states, max |gpu - logged| %.3e, "
                + "max |cpu - logged| %.3e%n", loggedCompared, maxLogged, maxLoggedCpu);
        if (!worst.isEmpty()) System.out.println("mismatches:" + worst);
        final boolean pass = mismatch == 0 && maxAbs < 1e-8 && compared > 0;
        System.out.println(pass ? "G1 PASS" : "G1 FAIL");
        System.exit(pass ? 0 : 1);
    }

    static void buildSets(Residue res, Parameter diag, Parameter off, Parameter lambda, File log, int seed)
            throws Exception {
        add(res, "start", diag.getParameterValue(0), diag.getParameterValue(1), off.getParameterValue(0),
                lambda.getParameterValue(0), Double.NaN);
        // four logged states after 30% burn-in, evenly spaced
        final List<double[]> rows = new ArrayList<double[]>();
        if (log.exists()) {
            try (BufferedReader in = new BufferedReader(new FileReader(log))) {
                String line;
                String[] header = null;
                while ((line = in.readLine()) != null) {
                    if (line.startsWith("#") || line.isEmpty()) continue;
                    final String[] f = line.split("\t");
                    if (header == null) { header = f; continue; }
                    if (f.length != header.length) continue;              // half-written last line
                    final double[] v = new double[f.length];
                    for (int i = 0; i < f.length; i++) v[i] = Double.parseDouble(f[i]);
                    rows.add(v);
                }
            }
        }
        if (rows.size() >= 10) {
            // columns: state joint likelihood prior lambda prec_000 prec_001 prec_010 prec_011
            final int from = (int) (0.3 * rows.size());
            for (int k = 0; k < 4; k++) {
                final double[] v = rows.get(from + (int) Math.round(k * (rows.size() - 1 - from) / 3.0));
                final double q1 = v[5], q2 = v[8], rho = v[6] / Math.sqrt(q1 * q2);
                add(res, "logged_" + (long) v[0], q1, q2, rho, v[4], v[2]);
            }
        } else {
            for (int k = 0; k < 4; k++) {                  // keep K equal across residues
                final double[] st = res.sets.get(0);
                add(res, "start_dup" + k, st[0], st[1], st[2], st[3], Double.NaN);
            }
        }
        final Random rng = new Random(20261009L + 7919L * seed);
        for (int k = 0; k < 4; k++) {
            add(res, "random" + k, Math.exp(Math.log(1e-3) + rng.nextDouble() * Math.log(1e10)),
                    Math.exp(Math.log(1e-3) + rng.nextDouble() * Math.log(1e10)), -0.95 + 1.9 * rng.nextDouble(),
                    Math.exp(Math.log(1e-7) + rng.nextDouble() * Math.log(3e7)), Double.NaN);
        }
        final double[] base = res.sets.get(Math.min(1, res.sets.size() - 1));
        add(res, "lambda0", base[0], base[1], base[2], 0.0, Double.NaN);
        add(res, "huge_diffusion", 1e-5, 1e-5, 0.9, 5.0, Double.NaN);
        add(res, "tiny_diffusion", 1e9, 1e9, -0.9, 1e-6, Double.NaN);
    }

    static void add(Residue res, String label, double q1, double q2, double rho, double lam, double logged) {
        res.sets.add(new double[]{q1, q2, rho, lam});
        res.labels.add(label);
        res.logged.add(logged);
    }

    static Object field(Object o, String name) throws Exception {
        final Field f = o.getClass().getDeclaredField(name);
        f.setAccessible(true);
        return f.get(o);
    }
}
