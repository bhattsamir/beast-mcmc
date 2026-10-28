/*
 * TorusBoxGpuCaseRunner.java
 *
 * @author Samir Bhatt
 */

package test.dr.evomodel.treedatalikelihood.continuous.gpu;

import dr.evomodel.treedatalikelihood.continuous.gpu.TorusBoxGpuEngine;
import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Reader;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Pattern;

import static jcuda.driver.JCudaDriver.*;

/**
 * Runs {@link TorusBoxGpuEngine} on every case of gpu_build/box_gpu/cases (format box-gpu-case/1, cases/FORMAT.md,
 * written by gpu_build/box_engine/export_gpu_cases.py from the frozen numpy reference box_landing.py) and compares
 * the GPU with the reference:
 * <ol>
 *   <li><b>logL</b> against the case's prec_check=False value. A finite value passes within the per-case tolerance
 *       max(1e-9, 0.3 x the case's PRECISION bound) (PLAN_NOTES GB3; the two implementations differ by their own
 *       round-off where that bound is large). A non-finite expected value (NONPOSITIVE) must be reproduced exactly,
 *       with the NONPOSITIVE bit, when the case is well conditioned (an exact zero at a node or a robust root sign);
 *       when it is not, a different outcome is reported as NOT-REQUIRED.</li>
 *   <li><b>Guard values</b> (LEAK, RESOLUTION, rho_min, the status bits) against prec_check=False, by agreement class
 *       (exact, close = 1e-6 relative or 1e-13 absolute, magnitude = within 10x, differ). They need not be bitwise.</li>
 *   <li><b>Init tables</b> (GB1) of the cases that carry them: Gp (gam_b), E1, E2 and the box modes as the engine holds
 *       them on the device ({@link TorusBoxGpuEngine#debugSlotTable}); LQ rebuilt from the device modes; h1 h2; the tip
 *       phase tables of the engine's own error-free reduction against exact rational fractions; the tip coefficients
 *       and tip amplitudes as the engine's own kernels (tip_axis, tip_build, from its PTX) build them; gsub on the
 *       device; musub from the engine's kernel. The engine itself is not modified.</li>
 *   <li><b>Invariance</b>, all bitwise: two repeats; the call order reversed; the smallest maxBatch (the most boxes of
 *       a site, or padding when the group is one site) with all sites in one call and with one site per call; a second
 *       device; the sites in reversed construction order.</li>
 *   <li><b>Timing</b> per site-grid (one (site, box) slot through the whole tree), with an H100 extrapolation.</li>
 * </ol>
 * Cases with the same batch_key share the tree, n, the tip error and the mixture (checked bitwise): they run as ONE
 * engine, each case its own site.
 *
 * <p>usage: TorusBoxGpuCaseRunner CASES_DIR OUT_PREFIX [--devices 0,1] [--only REGEX] [--tiers t1,t2]
 * [--no-invariance]. Writes OUT_PREFIX.log (also echoed to stdout) and OUT_PREFIX.tsv (one row per case).
 * Exit status 0 when every case passes and every invariance and table check holds, 1 otherwise.
 *
 * @author Samir Bhatt
 */
public final class TorusBoxGpuCaseRunner {

    private TorusBoxGpuCaseRunner() { }

    private static final double LEAK_TOL = 1e-12, RES_TOL = 1e-12;      // the cases' leak_tol and res_tol
    private static final double GB3_FLOOR = 1e-9, GB3_FACTOR = 0.3;
    private static final double TWO_PI = 2.0 * Math.PI;
    private static final double H100_FP64_RATIO = 55.0;                  // H100 FP64 / RTX A6000 FP64
    private static final int BLOCK = 256;
    private static final long CACHE_BYTES = 1L << 20;

    private static PrintStream log;

    // =========================================================================================== case files

    /** One case directory: header (case.properties), arrays read on demand (raw little-endian float64). */
    static final class Case {
        final String name, tier, batchKey;
        final File dir;
        final Properties h = new Properties();
        private final Map<String, double[]> cache = new LinkedHashMap<String, double[]>();
        final int n, nodes, root, boxes;
        final double lam, expLogL, expLeak, expRes, expRho, prec, tolField, numpySeconds;
        final int expStatus;
        final boolean wellConditioned, exactZero, rootRobust, initTables, dropPath;
        final String nonpositive;

        Case(File casesDir, String name, String tier, String batchKey) throws IOException {
            this.name = name;
            this.tier = tier;
            this.batchKey = batchKey;
            this.dir = new File(casesDir, name);
            try (Reader r = new InputStreamReader(new FileInputStream(new File(dir, "case.properties")),
                    StandardCharsets.ISO_8859_1)) {
                h.load(r);
            }
            if (!"box-gpu-case/1".equals(h.getProperty("format"))) throw new IOException(name + ": format " + h.getProperty("format"));
            if (!name.equals(h.getProperty("case"))) throw new IOException(name + ": header names case " + h.getProperty("case"));
            n = integer("n");
            nodes = integer("nodes");
            root = integer("root");
            boxes = integer("boxes");
            lam = dbl("lam");
            expLogL = dbl("expected.prec_off.logL");
            expLeak = dbl("expected.prec_off.leak_max");
            expRes = dbl("expected.prec_off.res_max");
            expRho = dbl("expected.prec_off.rho_min");
            expStatus = integer("expected.prec_off.status");
            prec = dbl("expected.prec_on.prec");
            tolField = "null".equals(prop("expected.compare.logL_tol")) ? Double.NaN : dbl("expected.compare.logL_tol");
            wellConditioned = bool("expected.compare.well_conditioned");
            exactZero = bool("expected.compare.nonpositive_exact_zero");
            rootRobust = bool("expected.compare.nonpositive_root_robust");
            dropPath = bool("expected.compare.drop_path");
            nonpositive = prop("expected.compare.nonpositive");
            initTables = bool("init_tables");
            numpySeconds = dbl("expected.prec_off.eval_seconds");
            // expected.bin holds the same numbers with no parsing involved: they must agree bitwise
            final double[] e = arr("expected");
            final double[] fromHeader = {expLogL, prec, expLeak, expRes, expRho, expStatus};
            final int[] at = {0, 2, 3, 4, 5, 6};
            for (int i = 0; i < at.length; i++) {
                if (Double.doubleToLongBits(e[at[i]]) != Double.doubleToLongBits(fromHeader[i])) {
                    throw new IOException(name + ": expected.bin[" + at[i] + "] = " + e[at[i]] + " but the header has " + fromHeader[i]);
                }
            }
        }

        String prop(String key) {
            final String v = h.getProperty(key);
            if (v == null) throw new IllegalStateException(name + ": header has no " + key);
            return v;
        }

        double dbl(String key) { return Double.parseDouble(prop(key)); }

        int integer(String key) { return Integer.parseInt(prop(key)); }

        boolean bool(String key) { return Boolean.parseBoolean(prop(key)); }

        boolean has(String array) { return h.getProperty("arrays." + array + ".file") != null; }

        int[] shape(String array) {
            final String s = prop("arrays." + array + ".shape");
            if (s.isEmpty()) return new int[0];
            final String[] f = s.split(",");
            final int[] out = new int[f.length];
            for (int i = 0; i < f.length; i++) out[i] = Integer.parseInt(f[i].trim());
            return out;
        }

        double[] arr(String array) throws IOException {
            final double[] hit = cache.get(array);
            if (hit != null) return hit;
            final String file = prop("arrays." + array + ".file");
            final long bytes = Long.parseLong(prop("arrays." + array + ".bytes"));
            final byte[] b = Files.readAllBytes(new File(dir, file).toPath());
            if (b.length != bytes) throw new IOException(name + "/" + file + ": " + b.length + " bytes, header says " + bytes);
            long count = 1;
            for (int s : shape(array)) count *= s;
            if (count * 8 != bytes) throw new IOException(name + "/" + file + ": shape does not match " + bytes + " bytes");
            final double[] a = new double[b.length / 8];
            ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().get(a);
            if (bytes <= CACHE_BYTES) cache.put(array, a);
            return a;
        }

        int[] ints(String array) throws IOException {
            final double[] a = arr(array);
            final int[] out = new int[a.length];
            for (int i = 0; i < a.length; i++) {
                out[i] = (int) a[i];
                if (out[i] != a[i]) throw new IOException(name + ": int array " + array + " holds " + a[i]);
            }
            return out;
        }
    }

    // =========================================================================================== engine inputs

    /** The engine inputs of a group of cases (one case = one site), from the case files alone. */
    static final class Inputs {
        final int k;
        int[][] children;
        int root, n, nSlots, mostBoxes;
        double marginSd;
        double[] blen, w, sigma4, lambda;
        double[][] sigmaEps, mu;
        double[][][] S;
        TorusBoxGpuEngine.Box[][] boxes;            // [case][box]
        int[][] tipBox;                             // [case][node]
        double[][][] tipOff;                        // [case][node][2]
        String hostNote = "";
        int hostProblems = 0;

        Inputs(int k) { this.k = k; }
    }

    private static final String[] SHARED = {"children_ptr", "children_idx", "blen", "sigma_eps", "comps_w", "comps_mu", "comps_S"};

    static Inputs inputs(List<Case> cases) throws IOException {
        final Case c0 = cases.get(0);
        final Inputs in = new Inputs(cases.size());
        for (Case c : cases) {
            if (c.n != c0.n || c.root != c0.root || c.nodes != c0.nodes) throw new IOException(c.name + ": n, root or nodes differ within its batch_key");
            for (String a : SHARED) {
                final double[] x = c.arr(a), y = c0.arr(a);
                if (x.length != y.length) throw new IOException(c.name + ": " + a + " differs within its batch_key");
                for (int i = 0; i < x.length; i++) {
                    if (Double.doubleToLongBits(x[i]) != Double.doubleToLongBits(y[i])) {
                        throw new IOException(c.name + ": " + a + " differs within its batch_key (element " + i + ")");
                    }
                }
            }
        }
        final int nn = c0.nodes;
        final int[] ptr = c0.ints("children_ptr"), idx = c0.ints("children_idx");
        if (ptr.length != nn + 1) throw new IOException(c0.name + ": children_ptr has " + ptr.length + " entries for " + nn + " nodes");
        in.children = new int[nn][];
        for (int v = 0; v < nn; v++) in.children[v] = Arrays.copyOfRange(idx, ptr[v], ptr[v + 1]);
        in.root = c0.root;
        in.n = c0.n;
        in.blen = c0.arr("blen").clone();
        final double[] se = c0.arr("sigma_eps");
        in.sigmaEps = new double[][]{{se[0], se[1]}, {se[2], se[3]}};
        in.w = c0.arr("comps_w").clone();
        final int K = in.w.length;
        final double[] mu = c0.arr("comps_mu"), S = c0.arr("comps_S");
        in.mu = new double[K][];
        in.S = new double[K][][];
        for (int k = 0; k < K; k++) {
            in.mu[k] = new double[]{mu[2 * k], mu[2 * k + 1]};
            in.S[k] = new double[][]{{S[4 * k], S[4 * k + 1]}, {S[4 * k + 2], S[4 * k + 3]}};
        }
        in.marginSd = c0.dbl("engine.margin_sd");
        in.boxes = new TorusBoxGpuEngine.Box[in.k][];
        in.tipBox = new int[in.k][];
        in.tipOff = new double[in.k][][];
        in.sigma4 = new double[4 * in.k];
        in.lambda = new double[in.k];
        int recBad = 0, recAll = 0, asgBad = 0, hhBad = 0;
        final List<String> notes = new ArrayList<String>();
        for (int ci = 0; ci < in.k; ci++) {
            final Case c = cases.get(ci);
            final int B = c.boxes;
            final double[] o = c.arr("box_o"), L = c.arr("box_L"), deg = c.arr("box_deg"), hh = c.arr("box_hh");
            final int[] npts = c.ints("box_n"), full = c.ints("box_full");
            in.boxes[ci] = new TorusBoxGpuEngine.Box[B];
            for (int b = 0; b < B; b++) {
                final TorusBoxGpuEngine.Box bx = new TorusBoxGpuEngine.Box(o[2 * b], o[2 * b + 1], L[2 * b], L[2 * b + 1],
                        npts[2 * b], npts[2 * b + 1], full[2 * b] != 0, full[2 * b + 1] != 0);
                in.boxes[ci][b] = bx;
                in.nSlots++;
                // host port of _parse_record must give Python's radians bitwise
                final TorusBoxGpuEngine.Box r = TorusBoxGpuEngine.Box.fromDegrees(deg[4 * b], deg[4 * b + 1], deg[4 * b + 2],
                        deg[4 * b + 3], npts[2 * b], npts[2 * b + 1]);
                recAll++;
                if (!sameBits(r.phi0, bx.phi0) || !sameBits(r.psi0, bx.psi0) || !sameBits(r.widthPhi, bx.widthPhi)
                        || !sameBits(r.widthPsi, bx.widthPsi) || r.fullPhi != bx.fullPhi || r.fullPsi != bx.fullPsi) {
                    recBad++;
                    notes.add(c.name + " box " + b + ": Box.fromDegrees differs from box_o/box_L");
                }
                final double hhJ = (bx.widthPhi / (double) bx.nPhi) * (bx.widthPsi / (double) bx.nPsi);
                if (!sameBits(hhJ, hh[b])) {
                    hhBad++;
                    notes.add(c.name + " box " + b + ": h1 h2 " + hhJ + " vs box_hh " + hh[b]);
                }
            }
            in.mostBoxes = Math.max(in.mostBoxes, B);
            final int[] tb = c.ints("tip_box");
            final double[] td = c.arr("tip_d"), y = c.arr("tips");
            in.tipBox[ci] = tb;
            in.tipOff[ci] = new double[nn][];
            final double[][] raw = new double[nn][];
            for (int v = 0; v < nn; v++) {
                if (tb[v] >= 0) in.tipOff[ci][v] = new double[]{td[2 * v], td[2 * v + 1]};
                if (in.children[v].length == 0 && !Double.isNaN(y[2 * v]) && !Double.isNaN(y[2 * v + 1])) {
                    raw[v] = new double[]{y[2 * v], y[2 * v + 1]};
                }
            }
            // host port of _layout_from_records (validation and tip assignment) must give Python's tip_box and tip_d
            try {
                final TorusBoxGpuEngine.TipAssignment ta = TorusBoxGpuEngine.assignTips(in.boxes[ci], raw, in.sigmaEps, in.marginSd);
                int bad = 0;
                for (int v = 0; v < nn; v++) {
                    if (ta.box[v] != tb[v] || (tb[v] >= 0 && (!sameBits(ta.offset[v][0], td[2 * v])
                            || !sameBits(ta.offset[v][1], td[2 * v + 1])))) {
                        bad++;
                    }
                }
                if (bad > 0) {
                    asgBad++;
                    notes.add(c.name + ": assignTips differs from tip_box/tip_d at " + bad + " nodes");
                }
            } catch (IllegalArgumentException e) {
                asgBad++;
                notes.add(c.name + ": assignTips refused the layout: " + e.getMessage());
            }
            final double[] p = c.arr("params");
            System.arraycopy(p, 0, in.sigma4, 4 * ci, 4);
            in.lambda[ci] = p[4];
        }
        in.hostProblems = recBad + asgBad + hhBad;
        in.hostNote = String.format(Locale.ROOT, "Box.fromDegrees bitwise %d/%d, h1 h2 bitwise %d/%d, assignTips bitwise %d/%d sites",
                recAll - recBad, recAll, recAll - hhBad, recAll, in.k - asgBad, in.k);
        for (String s : notes) in.hostNote += "\n      ! " + s;
        return in;
    }

    // =========================================================================================== evaluation

    /** Results per case of the group (indexed by case, whatever the call or site order). */
    static final class Res {
        final double[] logL, leak, res, rho, seconds;
        final int[] status;
        double total;

        Res(int k) {
            logL = new double[k];
            leak = new double[k];
            res = new double[k];
            rho = new double[k];
            seconds = new double[k];
            status = new int[k];
            Arrays.fill(logL, Double.NaN);
        }
    }

    static TorusBoxGpuEngine construct(Inputs in, int[] siteOfCase, int device, int maxBatch) {
        final TorusBoxGpuEngine.Box[][] bx = new TorusBoxGpuEngine.Box[in.k][];
        final int[][] tb = new int[in.k][];
        final double[][][] to = new double[in.k][][];
        for (int c = 0; c < in.k; c++) {
            final int s = siteOfCase[c];
            bx[s] = in.boxes[c];
            tb[s] = in.tipBox[c];
            to[s] = in.tipOff[c];
        }
        final TorusBoxGpuEngine eng = new TorusBoxGpuEngine(in.children, in.root, in.blen, bx, tb, to, in.sigmaEps, in.w,
                in.mu, in.S, device, maxBatch);
        eng.setTolerances(LEAK_TOL, RES_TOL);
        return eng;
    }

    /** One engine call with the given cases in the given order. */
    static Res call(TorusBoxGpuEngine eng, Inputs in, int[] order, int[] siteOfCase, Res into) {
        final int m = order.length;
        final int[] sites = new int[m];
        final double[] s4 = new double[4 * m], lam = new double[m];
        for (int i = 0; i < m; i++) {
            final int c = order[i];
            sites[i] = siteOfCase[c];
            System.arraycopy(in.sigma4, 4 * c, s4, 4 * i, 4);
            lam[i] = in.lambda[c];
        }
        final double[] ll = new double[m], lk = new double[m], rs = new double[m], rh = new double[m];
        final int[] st = new int[m];
        eng.logLikelihood(sites, s4, lam, ll, st, lk, rs, rh);
        final Res r = into != null ? into : new Res(in.k);
        final double sec = eng.getLastSeconds();
        for (int i = 0; i < m; i++) {
            final int c = order[i];
            r.logL[c] = ll[i];
            r.status[c] = st[i];
            r.leak[c] = lk[i];
            r.res[c] = rs[i];
            r.rho[c] = rh[i];
            r.seconds[c] = sec;
        }
        r.total += sec;
        return r;
    }

    static int[] identity(int k) {
        final int[] a = new int[k];
        for (int i = 0; i < k; i++) a[i] = i;
        return a;
    }

    static int[] reversed(int k) {
        final int[] a = new int[k];
        for (int i = 0; i < k; i++) a[i] = k - 1 - i;
        return a;
    }

    /** Bitwise differences between two result sets (NaN payloads canonical). Empty = identical. */
    static List<String> bitwise(Res a, Res b, List<Case> cases) {
        final List<String> out = new ArrayList<String>();
        for (int c = 0; c < cases.size(); c++) {
            final List<String> what = new ArrayList<String>();
            if (!sameBits(a.logL[c], b.logL[c])) what.add("logL " + a.logL[c] + " vs " + b.logL[c]);
            if (a.status[c] != b.status[c]) what.add("status " + a.status[c] + " vs " + b.status[c]);
            if (!sameBits(a.leak[c], b.leak[c])) what.add("leak " + a.leak[c] + " vs " + b.leak[c]);
            if (!sameBits(a.res[c], b.res[c])) what.add("res " + a.res[c] + " vs " + b.res[c]);
            if (!sameBits(a.rho[c], b.rho[c])) what.add("rho " + a.rho[c] + " vs " + b.rho[c]);
            if (!what.isEmpty()) out.add(cases.get(c).name + ": " + String.join(", ", what));
        }
        return out;
    }

    static boolean sameBits(double a, double b) { return Double.doubleToLongBits(a) == Double.doubleToLongBits(b); }

    /** |a / b - 1|, 0 when both are 0, infinite when only b is 0. */
    static double relDiff(double a, double b) {
        if (sameBits(a, b)) return 0.0;
        if (b == 0.0) return Double.POSITIVE_INFINITY;
        return Math.abs(a / b - 1.0);
    }

    // =========================================================================================== verdicts

    static final class Row {
        final Case c;
        final int group;
        double gpu = Double.NaN, leak = Double.NaN, res = Double.NaN, rho = Double.NaN, tol = Double.NaN, diff = Double.NaN;
        int status = -1;
        String verdict = "ERROR", note = "", leakClass = "-", resClass = "-", rhoClass = "-";
        boolean statusMatch;
        double gpuSeconds = Double.NaN;

        Row(Case c, int group) {
            this.c = c;
            this.group = group;
        }

        boolean passed() { return verdict.equals("PASS") || verdict.equals("NOT-REQUIRED"); }
    }

    /** PASS, FAIL or NOT-REQUIRED (a non-finite outcome that round-off may flip, per the case's well_conditioned). */
    static void judge(Row r) {
        final Case c = r.c;
        r.tol = Math.max(GB3_FLOOR, GB3_FACTOR * c.prec);
        if (Double.isFinite(c.expLogL)) {
            if (!(Double.isNaN(c.tolField) || sameBits(c.tolField, r.tol))) {
                r.note += "logL_tol field " + c.tolField + " != recomputed " + r.tol + "; ";
            }
            r.diff = Math.abs(r.gpu - c.expLogL);
            if (Double.isFinite(r.gpu) && r.diff <= r.tol) {
                r.verdict = "PASS";
            } else {
                r.verdict = "FAIL";
                if (!Double.isFinite(r.gpu)) r.note += "GPU gave " + r.gpu + " (status " + r.status + ") where Python is finite; ";
                if (!c.wellConditioned) r.note += "case is not well conditioned; ";
            }
        } else {
            final boolean same = (Double.isNaN(c.expLogL) && Double.isNaN(r.gpu)) || c.expLogL == r.gpu;
            final boolean nonpos = (r.status & TorusBoxGpuEngine.STATUS_NONPOSITIVE) != 0;
            r.diff = same ? 0.0 : Double.POSITIVE_INFINITY;
            if (same && nonpos) {
                r.verdict = "PASS";
                r.note += "NONPOSITIVE reproduced (" + c.nonpositive + (c.exactZero ? ", exact zero" : "")
                        + (c.rootRobust ? ", robust root" : "") + "); ";
            } else if (!c.wellConditioned) {
                r.verdict = "NOT-REQUIRED";
                r.note += "Python NONPOSITIVE (" + c.nonpositive + ") with a sign round-off can flip; GPU gave " + r.gpu
                        + " status " + r.status + "; ";
            } else {
                r.verdict = "FAIL";
                r.note += "Python NONPOSITIVE (" + c.nonpositive + ", well conditioned) not reproduced: GPU " + r.gpu
                        + " status " + r.status + "; ";
            }
        }
        r.statusMatch = r.status == c.expStatus;
        if (!r.statusMatch) {
            final int d = r.status ^ c.expStatus;
            if ((d & TorusBoxGpuEngine.STATUS_LEAK) != 0) r.note += "LEAK bit differs (GPU " + r.leak + ", Python " + c.expLeak + ", tol 1e-12); ";
            if ((d & TorusBoxGpuEngine.STATUS_RESOLUTION) != 0) r.note += "RESOLUTION bit differs (GPU " + r.res + ", Python " + c.expRes + ", tol 1e-12); ";
            if ((d & TorusBoxGpuEngine.STATUS_NONPOSITIVE) != 0) r.note += "NONPOSITIVE bit differs; ";
        }
        r.leakClass = guardClass(r.leak, c.expLeak);
        r.resClass = guardClass(r.res, c.expRes);
        r.rhoClass = guardClass(r.rho, c.expRho);
    }

    /** exact | close (1e-6 relative or 1e-13 absolute) | magnitude (within 10x) | differ. */
    static String guardClass(double g, double p) {
        if (sameBits(g, p) || (Double.isNaN(g) && Double.isNaN(p))) return "exact";
        if (Double.isNaN(g) || Double.isNaN(p)) return "differ";
        final double d = Math.abs(g - p), m = Math.max(Math.abs(g), Math.abs(p));
        if (d <= 1e-6 * m || d <= 1e-13) return "close";
        if (g > 0 && p > 0 && Math.max(g / p, p / g) <= 10.0) return "magnitude";
        return "differ";
    }

    // =========================================================================================== init tables (GB1)

    /** Worst value of each table check over every box checked. */
    static final class TableStats {
        final Map<String, double[]> worst = new LinkedHashMap<String, double[]>();     // name -> {value, count}
        final Map<String, String> where = new LinkedHashMap<String, String>();
        final Map<String, Double> limit = new LinkedHashMap<String, Double>();
        final List<String> failures = new ArrayList<String>();

        void see(String table, double value, double lim, String at) {
            double[] w = worst.get(table);
            if (w == null) {
                w = new double[]{Double.NEGATIVE_INFINITY, 0};
                worst.put(table, w);
                limit.put(table, lim);
            }
            w[1]++;
            if (Double.isNaN(value) || value > w[0]) {
                w[0] = value;
                where.put(table, at);
            }
            if (!(value <= lim)) failures.add(table + " at " + at + ": " + sci(value) + " > " + sci(lim));
        }
    }

    /** max |a - b| over interleaved complex (or real) arrays, and max |b|. */
    static double[] absDiff(double[] gpu, double[] py, boolean complex) {
        if (gpu.length != py.length) throw new IllegalStateException("table lengths differ: " + gpu.length + " vs " + py.length);
        double dmax = 0.0, bmax = 0.0;
        int equal = 0;
        final int step = complex ? 2 : 1;
        for (int i = 0; i < gpu.length; i += step) {
            final double d, m;
            if (complex) {
                d = Math.hypot(gpu[i] - py[i], gpu[i + 1] - py[i + 1]);
                m = Math.hypot(py[i], py[i + 1]);
                if (sameBits(gpu[i], py[i]) && sameBits(gpu[i + 1], py[i + 1])) equal++;
            } else {
                d = Math.abs(gpu[i] - py[i]);
                m = Math.abs(py[i]);
                if (sameBits(gpu[i], py[i])) equal++;
            }
            if (Double.isNaN(d) || d > dmax) dmax = d;
            if (m > bmax) bmax = m;
        }
        return new double[]{dmax, bmax, equal, gpu.length / step};
    }

    /** The device kernels of the engine's own PTX, launched on the engine's own tables (tip coefficients, musub). */
    static final class Kernels implements AutoCloseable {
        private final CUdevice dev = new CUdevice();
        private final CUcontext ctx = new CUcontext();
        private final CUmodule mod = new CUmodule();
        private final CUfunction tipAxis = new CUfunction(), tipBuild = new CUfunction(), musubBuild = new CUfunction();
        private final List<CUdeviceptr> live = new ArrayList<CUdeviceptr>();

        Kernels(int device) {
            JCudaDriver.setExceptionsEnabled(true);
            cuInit(0);
            cuDeviceGet(dev, device);
            cuDevicePrimaryCtxRetain(ctx, dev);
            cuCtxSetCurrent(ctx);
            cuModuleLoadData(mod, ptx());
            cuModuleGetFunction(tipAxis, mod, "tip_axis");
            cuModuleGetFunction(tipBuild, mod, "tip_build");
            cuModuleGetFunction(musubBuild, mod, "musub_build");
        }

        /** tip_axis + tip_build for one tip alone in slot 0: the raw coefficients (n1 x n2h complex, interleaved). */
        double[] tip(double[] phase, double[] om1, double[] om2, double hh, double pi00h, double pi01, double pi11h) {
            cuCtxSetCurrent(ctx);
            final int n1 = om1.length, n2h = om2.length;
            final long N2 = (long) n1 * n2h, NG = (long) n1 * 2L * (n2h - 1);
            final CUdeviceptr dT = alloc(16L * (n1 + n2h)), dX = alloc(16L * N2), dPres = alloc(4L);
            final CUdeviceptr dSlot = up(new int[]{0, 0, 0, 0}), dSite = up(new int[]{0, 0, 1, 0}), dDead = up(new int[]{0});
            final CUdeviceptr dTipBox = up(new int[]{0}), dOff = up(new long[]{0L}), dPh = up(phase), dOm1 = up(om1);
            final CUdeviceptr dOm2 = up(om2), dHH = up(new double[]{hh});
            final CUdeviceptr dCode = new CUdeviceptr();
            cuMemAlloc(dCode, 8L);
            live.add(dCode);
            cuMemcpyHtoD(dCode, Pointer.to(new byte[]{1}), 1L);
            launch(tipAxis, (n1 + n2h + BLOCK - 1) / BLOCK, Pointer.to(dT), Pointer.to(dSlot), Pointer.to(dSite),
                    Pointer.to(dCode), Pointer.to(dDead), Pointer.to(dTipBox), Pointer.to(dOff), Pointer.to(dPh),
                    Pointer.to(new int[]{n1}), Pointer.to(new int[]{n2h}));
            final int nblk = (int) Math.max(1L, Math.min(128L, NG / (BLOCK * 4L)));
            launch(tipBuild, nblk, Pointer.to(dX), Pointer.to(dPres), Pointer.to(dT), Pointer.to(dSlot), Pointer.to(dSite),
                    Pointer.to(dCode), Pointer.to(dDead), Pointer.to(dTipBox), Pointer.to(dOm1), Pointer.to(dOm2),
                    Pointer.to(dHH), Pointer.to(new double[]{pi00h}), Pointer.to(new double[]{pi01}),
                    Pointer.to(new double[]{pi11h}), Pointer.to(new int[]{n1}), Pointer.to(new int[]{n2h}));
            cuCtxSynchronize();
            final double[] x = new double[(int) (2 * N2)];
            cuMemcpyDtoH(Pointer.to(x), dX, 16L * N2);
            freeAll();
            return x;
        }

        /** musub_build for one site: lam gsub / (1/2 m'Sigma m + lam), M x M complex interleaved. */
        double[] musub(double[] gsub, double S00, double s01, double S11, double lam, int M, int Mcut) {
            cuCtxSetCurrent(ctx);
            final long MM = (long) M * M;
            final CUdeviceptr dMU = alloc(16L * MM), dG = up(gsub);
            final CUdeviceptr dP = up(new double[]{S00, s01, S11, 0.5 * S00, 0.5 * S11, lam, 0.0, 0.0});
            launch(musubBuild, (int) Math.min(1024L, (MM + BLOCK - 1) / BLOCK), Pointer.to(dMU), Pointer.to(dG),
                    Pointer.to(dP), Pointer.to(new int[]{M}), Pointer.to(new int[]{Mcut}));
            cuCtxSynchronize();
            final double[] out = new double[(int) (2 * MM)];
            cuMemcpyDtoH(Pointer.to(out), dMU, 16L * MM);
            freeAll();
            return out;
        }

        /** Copy a device array of the (shared primary) context to the host. */
        double[] download(CUdeviceptr p, int len) {
            cuCtxSetCurrent(ctx);
            final double[] out = new double[len];
            cuMemcpyDtoH(Pointer.to(out), p, 8L * len);
            return out;
        }

        private void launch(CUfunction f, int gridX, Pointer... args) {
            cuLaunchKernel(f, gridX, 1, 1, BLOCK, 1, 1, 0, null, Pointer.to(args), null);
        }

        private CUdeviceptr alloc(long bytes) {
            final CUdeviceptr p = new CUdeviceptr();
            cuMemAlloc(p, Math.max(8L, bytes));
            cuMemsetD8(p, (byte) 0, Math.max(8L, bytes));
            live.add(p);
            return p;
        }

        private CUdeviceptr up(double[] a) {
            final CUdeviceptr p = alloc(8L * a.length);
            cuMemcpyHtoD(p, Pointer.to(a), 8L * a.length);
            return p;
        }

        private CUdeviceptr up(int[] a) {
            final CUdeviceptr p = alloc(4L * a.length);
            cuMemcpyHtoD(p, Pointer.to(a), 4L * a.length);
            return p;
        }

        private CUdeviceptr up(long[] a) {
            final CUdeviceptr p = alloc(8L * a.length);
            cuMemcpyHtoD(p, Pointer.to(a), 8L * a.length);
            return p;
        }

        private void freeAll() {
            for (CUdeviceptr p : live) cuMemFree(p);
            live.clear();
        }

        @Override
        public void close() {
            cuCtxSetCurrent(ctx);
            freeAll();
            cuModuleUnload(mod);
            cuDevicePrimaryCtxRelease(dev);
        }

        /** The engine's PTX, found exactly as TorusBoxGpuEngine finds it. */
        private static byte[] ptx() {
            final String override = System.getProperty("torus.box.ptx");
            try (InputStream in = override != null ? new FileInputStream(override)
                    : TorusBoxGpuEngine.class.getResourceAsStream("torus_box_kernels.ptx")) {
                if (in == null) throw new IllegalStateException("torus_box_kernels.ptx not found next to TorusBoxGpuEngine");
                final ByteArrayOutputStream out = new ByteArrayOutputStream();
                final byte[] buf = new byte[8192];
                int r;
                while ((r = in.read(buf)) > 0) out.write(buf, 0, r);
                out.write(0);
                return out.toByteArray();
            } catch (IOException e) {
                throw new IllegalStateException(e);
            }
        }
    }

    /** numpy fftfreq(n, 1 / n) and rfftfreq(n, 1 / n), computed as numpy computes them. */
    static double[] npFftfreq(int n) {
        final double val = 1.0 / (n * (1.0 / n));
        final double[] out = new double[n];
        final int N = (n - 1) / 2 + 1;
        for (int i = 0; i < N; i++) out[i] = (double) i * val;
        for (int i = N; i < n; i++) out[i] = (double) (i - n) * val;
        return out;
    }

    static double[] npRfftfreq(int n) {
        final double val = 1.0 / (n * (1.0 / n));
        final double[] out = new double[n / 2 + 1];
        for (int i = 0; i < out.length; i++) out[i] = (double) i * val;
        return out;
    }

    private static Method tipPhaseMethod;

    static double tipPhase(double p, double d, double L) throws Exception {
        if (tipPhaseMethod == null) {
            tipPhaseMethod = TorusBoxGpuEngine.class.getDeclaredMethod("tipPhase", double.class, double.class, double.class);
            tipPhaseMethod.setAccessible(true);
        }
        return (Double) tipPhaseMethod.invoke(null, p, d, L);
    }

    /** |th - 2 pi f| reduced to the nearest multiple of 2 pi (a fraction just below 1 may round to 1.0). */
    static double phaseErr(double th, double f) {
        double e = th - TWO_PI * f;
        e -= TWO_PI * Math.rint(e / TWO_PI);
        return Math.abs(e);
    }

    static void initTables(TorusBoxGpuEngine eng, Case c, int site, int caseIdx, Inputs in, Kernels kern, TableStats ts)
            throws Exception {
        final String at0 = c.name;
        final int Mcut = c.integer("init.Mcut"), M = 2 * Mcut + 1;
        ts.see("Mcut |engine - case|", Math.abs(eng.getMcut() - Mcut), 0.0, at0);
        // gsub as the engine holds it on the device (private field read by reflection; parameter-free)
        final Field fg = TorusBoxGpuEngine.class.getDeclaredField("dGsub");
        fg.setAccessible(true);
        eng.debugSlotTable("om1", eng.getSiteFirstSlot(site));          // makes the engine's context current
        final double[] gsub = kern.download((CUdeviceptr) fg.get(eng), 2 * M * M);
        final double[] gd = absDiff(gsub, c.arr("gsub"), true);
        ts.see("gsub max|d| / max|gsub|", gd[0] / gd[1], 1e-14, at0);
        final double[] s4 = Arrays.copyOfRange(in.sigma4, 4 * caseIdx, 4 * caseIdx + 4);
        final double lam = in.lambda[caseIdx];
        final double s01 = s4[1] != s4[2] ? 0.5 * (s4[1] + s4[2]) : s4[1];
        if (lam > 0) {
            final double[] mu = kern.musub(gsub, s4[0], s01, s4[3], lam, M, Mcut);
            final double[] md = absDiff(mu, c.arr("musub"), true);
            ts.see("musub max|d| / max|musub|", md[0] / md[1], 1e-14, at0);
        }
        final double pi00h = 0.5 * in.sigmaEps[0][0], pi01 = in.sigmaEps[0][1], pi11h = 0.5 * in.sigmaEps[1][1];
        final int[] nodeBox = in.tipBox[caseIdx];
        for (int b = 0; b < c.boxes; b++) {
            final String at = c.name + " box " + b;
            final int slot = eng.getSiteFirstSlot(site) + b;
            final TorusBoxGpuEngine.Box bx = in.boxes[caseIdx][b];
            final int n1 = bx.nPhi, n2 = bx.nPsi, n2h = n2 / 2 + 1;
            // gam_b, E1, E2 as the engine holds them
            final double[] gp = absDiff(eng.debugSlotTable("gp", slot), c.arr("box" + b + "_Gp"), true);
            ts.see("Gp (gam_b) max|d| / max|Gp|", gp[0] / gp[1], 1e-14, at);
            final double[] e1 = absDiff(eng.debugSlotTable("e1", slot), c.arr("box" + b + "_E1"), true);
            ts.see("E1 max|d|", e1[0], 1e-14, at);
            final double[] e2 = absDiff(eng.debugSlotTable("e2", slot), c.arr("box" + b + "_E2"), true);
            ts.see("E2 max|d|", e2[0], 1e-14, at);
            // box modes: numpy's om = 2 pi p / L bitwise, and LQ (this case's Sigma, lambda) rebuilt from them bitwise
            final double[] om1 = eng.debugSlotTable("om1", slot), om2 = eng.debugSlotTable("om2", slot);
            final double[] p = npFftfreq(n1), q = npRfftfreq(n2);
            int omBad = 0;
            for (int i = 0; i < n1; i++) if (!sameBits(om1[i], (TWO_PI * p[i]) / bx.widthPhi)) omBad++;
            for (int j = 0; j < n2h; j++) if (!sameBits(om2[j], (TWO_PI * q[j]) / bx.widthPsi)) omBad++;
            ts.see("om1, om2 entries not bitwise numpy", omBad, 0.0, at);
            final double[] LQ = c.arr("box" + b + "_LQ");
            int lqBad = 0;
            for (int i = 0; i < n1; i++) {
                final double w1 = om1[i];
                for (int j = 0; j < n2h; j++) {
                    final double w2 = om2[j];
                    final double Q = (((0.5 * s4[0]) * (w1 * w1) + (s4[1] * w1) * w2) + (0.5 * s4[3]) * (w2 * w2)) + lam;
                    if (!sameBits(Q, LQ[i * n2h + j])) lqBad++;
                }
            }
            ts.see("LQ from device modes, entries not bitwise", lqBad, 0.0, at);
            // the tip of this box (export_gpu_cases.init_tables: the box's first tip)
            final String tn = c.prop("init.boxes." + b + ".tip_node");
            if ("null".equals(tn)) continue;
            final int v = Integer.parseInt(tn);
            final String[] tdS = c.prop("init.boxes." + b + ".tip_d").split(",");
            final double[] d = {Double.parseDouble(tdS[0]), Double.parseDouble(tdS[1])};
            if (nodeBox[v] != b || !sameBits(in.tipOff[caseIdx][v][0], d[0]) || !sameBits(in.tipOff[caseIdx][v][1], d[1])) {
                ts.failures.add(at + ": init tip " + v + " is not in this box at tip_d");
            }
            // phase tables of the engine's error-free reduction, against the exact rational fractions
            final double[] th = new double[n1 + n2h];
            for (int i = 0; i < n1; i++) th[i] = tipPhase(p[i], d[0], bx.widthPhi);
            for (int j = 0; j < n2h; j++) th[n1 + j] = tipPhase(q[j], d[1], bx.widthPsi);
            final double[] f1 = c.arr("box" + b + "_tipfrac1"), f2 = c.arr("box" + b + "_tipfrac2");
            final double[] l1 = c.arr("box" + b + "_tipfrac1_ld"), l2 = c.arr("box" + b + "_tipfrac2_ld");
            double pe = 0.0, pl = 0.0;
            for (int i = 0; i < n1; i++) {
                pe = Math.max(pe, phaseErr(th[i], f1[i]));
                pl = Math.max(pl, phaseErr(th[i], l1[i]));
            }
            for (int j = 0; j < n2h; j++) {
                pe = Math.max(pe, phaseErr(th[n1 + j], f2[j]));
                pl = Math.max(pl, phaseErr(th[n1 + j], l2[j]));
            }
            ts.see("tip phase vs 2 pi exact fraction (rad)", pe, 2e-15, at);
            ts.see("tip phase vs 2 pi long-double fraction (rad)", pl, 2e-15, at);
            // the tip's coefficients from the engine's own kernels, and its amplitudes (zero phase: X = tipamp)
            final double hh = (bx.widthPhi / (double) n1) * (bx.widthPsi / (double) n2);
            final double[] X = kern.tip(th, om1, om2, hh, pi00h, pi01, pi11h);
            final double[] amp = c.arr("box" + b + "_tipamp");
            double ampMax = 0.0;
            for (double a : amp) ampMax = Math.max(ampMax, a);
            final double[] xd = absDiff(X, c.arr("box" + b + "_tipcoef"), true);
            ts.see("tip coefficients max|d| / max tipamp", xd[0] / ampMax, 1e-14, at);
            final double[] A = kern.tip(new double[n1 + n2h], om1, om2, hh, pi00h, pi01, pi11h);
            // relative error where the amplitude is a normal number (a subnormal carries few significant bits), and the
            // absolute error relative to the largest amplitude everywhere (underflow to 0 included)
            double ar = 0.0, aa = 0.0, aimag = 0.0;
            int ampBits = 0;
            for (int i = 0; i < amp.length; i++) {
                final double g = A[2 * i], py = amp[i];
                aimag = Math.max(aimag, Math.abs(A[2 * i + 1]));
                if (sameBits(g, py)) ampBits++;
                if (py >= Double.MIN_NORMAL) ar = Math.max(ar, Math.abs(g / py - 1.0));
                final double dd = Math.abs(g - py);
                if (Double.isNaN(dd) || dd > aa) aa = dd;
            }
            ts.see("tip amplitude max|gpu/python - 1| (normal values)", ar, 1e-14, at);
            ts.see("tip amplitude max|d| / max tipamp", aa / ampMax, 1e-15, at);
            ts.see("tip amplitude imaginary part", aimag, 0.0, at);
            log.printf(Locale.ROOT, "      %s: Gp %s, E1 %s, E2 %s, om/LQ bitwise %s, tip phase err %s rad (vs long double %s), "
                            + "tip coef %s, tipamp rel %s (%d/%d bitwise)%n", at, sci(gp[0] / gp[1]), sci(e1[0]), sci(e2[0]),
                    (omBad == 0 && lqBad == 0) ? "yes" : ("NO (" + omBad + ", " + lqBad + ")"), sci(pe), sci(pl),
                    sci(xd[0] / ampMax), sci(ar), ampBits, amp.length);
        }
    }

    // =========================================================================================== one group

    static final class Invariance {
        final List<String> passed = new ArrayList<String>(), failed = new ArrayList<String>(), skipped = new ArrayList<String>();
    }

    static final class Timing {
        String group;
        int n, nodes, slots, sites;
        double construct, perCall;
        final List<double[]> perSite = new ArrayList<double[]>();      // {boxes, seconds}
        double numpySeconds;
    }

    static void runGroup(int gi, List<Case> cases, int[] devices, boolean invariance, List<Row> rows, Invariance inv,
                         TableStats ts, List<Timing> timings) throws Exception {
        final Inputs in = inputs(cases);
        final int k = in.k;
        final Case c0 = cases.get(0);
        log.printf(Locale.ROOT, "%n== group %d  batch_key %s  tier %s: %d case(s) %s .. %s, %d nodes, n %d, %d slots (most boxes %d)%n",
                gi, c0.batchKey, c0.tier, k, c0.name, cases.get(k - 1).name, c0.nodes, in.n, in.nSlots, in.mostBoxes);
        log.println("   host: " + in.hostNote);
        if (in.hostProblems > 0) inv.failed.add("group " + gi + " host checks: " + in.hostProblems + " problem(s)");
        final int[] ident = identity(k), rev = reversed(k);
        final String tag = "group " + gi + " (" + c0.tier + ", " + k + " sites)";

        // ---- engine A: primary device, all slots in one chunk
        final Res a1, a2, aRev;
        final Timing tm = new Timing();
        long t0 = System.nanoTime();
        try (TorusBoxGpuEngine eng = construct(in, ident, devices[0], in.nSlots)) {
            tm.construct = (System.nanoTime() - t0) * 1e-9;
            // parameter-free values against the case headers
            // (the engine evaluates them in Java: Math.exp may differ from numpy's exp by an ulp, so relative, not bitwise)
            int mcutBad = 0, tlBits = 0, trBits = 0;
            for (int ci = 0; ci < k; ci++) {
                final Case c = cases.get(ci);
                if (eng.getMcut() != c.integer("Mcut")) mcutBad++;
                ts.see("g_sup |engine/case - 1|", Math.abs(eng.getGSup() / c.dbl("g_sup") - 1.0), 1e-12, c.name);
                final double[] tl = c.arr("tip_leak"), tr = c.arr("box_tipres");
                double tlMax = 0.0, trMax = 0.0;
                for (int v = 0; v < c.nodes; v++) if (in.tipBox[ci][v] >= 0 && tl[v] > tlMax) tlMax = tl[v];
                for (double x : tr) if (x > trMax) trMax = x;
                if (sameBits(eng.getTipLeak(ci), tlMax)) tlBits++;
                if (sameBits(eng.getTipResolution(ci), trMax)) trBits++;
                ts.see("tip LEAK (parameter-free) |engine/case - 1|", relDiff(eng.getTipLeak(ci), tlMax), 1e-14, c.name);
                ts.see("tip RESOLUTION (parameter-free) |engine/case - 1|", relDiff(eng.getTipResolution(ci), trMax), 1e-14, c.name);
            }
            ts.see("Mcut cases differing", mcutBad, 0.0, c0.name);
            log.printf(Locale.ROOT, "   engine A (device %d, maxBatch %d): construct %.2f s; Mcut %d (%s), g_sup %.17g (case %.17g), "
                            + "tip LEAK bitwise %d/%d, tip RESOLUTION bitwise %d/%d, buffers %d%n", devices[0], in.nSlots,
                    tm.construct, eng.getMcut(), mcutBad == 0 ? "= cases" : mcutBad + " differ", eng.getGSup(), c0.dbl("g_sup"),
                    tlBits, k, trBits, k, eng.getBufferCount());

            a1 = call(eng, in, ident, ident, null);
            a2 = call(eng, in, ident, ident, null);
            aRev = call(eng, in, rev, ident, null);
            tm.perCall = Math.min(a1.total, a2.total);
            log.printf(Locale.ROOT, "   evaluations A: %.3f s, %.3f s (repeat), %.3f s (reversed call order) for %d sites / %d slots%n",
                    a1.total, a2.total, aRev.total, k, in.nSlots);
            check(inv, tag + ": repeat bitwise", bitwise(a1, a2, cases));
            check(inv, tag + ": reversed call order bitwise", k > 1 ? bitwise(a1, aRev, cases) : null);

            // ---- GB1 init tables
            boolean any = false;
            for (int ci = 0; ci < k; ci++) any |= cases.get(ci).initTables;
            if (any) {
                log.println("   init tables (GB1), against the case's dumped tables:");
                try (Kernels kern = new Kernels(devices[0])) {
                    for (int ci = 0; ci < k; ci++) {
                        if (cases.get(ci).initTables) initTables(eng, cases.get(ci), ci, ci, in, kern, ts);
                    }
                }
            }
        }

        // ---- verdicts (engine A, the reference configuration), recorded before the invariance engines run
        for (int ci = 0; ci < k; ci++) {
            final Row r = new Row(cases.get(ci), gi);
            r.gpu = a1.logL[ci];
            r.status = a1.status[ci];
            r.leak = a1.leak[ci];
            r.res = a1.res[ci];
            r.rho = a1.rho[ci];
            r.gpuSeconds = a1.total;
            judge(r);
            rows.add(r);
            log.printf(Locale.ROOT, "   %-30s %-12s gpu %-22s py %-22s |d| %-9s tol %-9s st %d/%d  leak %-9s res %-9s rho %-9s %s%n",
                    r.c.name, r.verdict, Double.toString(r.gpu), Double.toString(r.c.expLogL), sci(r.diff), sci(r.tol),
                    r.status, r.c.expStatus, r.leakClass, r.resClass, r.rhoClass, r.note);
        }
        tm.group = tag;
        tm.n = in.n;
        tm.nodes = c0.nodes;
        tm.slots = in.nSlots;
        tm.sites = k;
        for (Case c : cases) tm.numpySeconds += c.numpySeconds;
        timings.add(tm);
        if (!invariance) return;

        // ---- engine B: smallest maxBatch (one site per chunk when every site has the most boxes) or padding
        final int mb = k > 1 ? in.mostBoxes : in.nSlots + 3;
        try (TorusBoxGpuEngine eng = construct(in, ident, devices[0], mb)) {
            final Res b1 = call(eng, in, ident, ident, null);
            check(inv, tag + ": maxBatch " + mb + " (all sites in one call) vs " + in.nSlots + " bitwise", bitwise(a1, b1, cases));
            final Res b2 = new Res(k);
            for (int ci = 0; ci < k; ci++) call(eng, in, new int[]{ci}, ident, b2);
            check(inv, tag + ": maxBatch " + mb + ", one site per call, bitwise", bitwise(a1, b2, cases));
            for (int ci = 0; ci < k; ci++) tm.perSite.add(new double[]{cases.get(ci).boxes, b2.seconds[ci]});
            log.printf(Locale.ROOT, "   engine B (maxBatch %d): all sites %.3f s, one site per call %.3f s in total%n", mb, b1.total, b2.total);
        } catch (RuntimeException e) {
            inv.failed.add(tag + ": engine B (maxBatch " + mb + ") raised " + e);
            log.println("   ENGINE B RAISED: " + e);
        }
        // ---- engine C: the second device
        if (devices.length > 1) {
            try (TorusBoxGpuEngine eng = construct(in, ident, devices[1], in.nSlots)) {
                final Res c1 = call(eng, in, ident, ident, null);
                check(inv, tag + ": device " + devices[1] + " vs device " + devices[0] + " bitwise", bitwise(a1, c1, cases));
                log.printf(Locale.ROOT, "   engine C (device %d): %.3f s%n", devices[1], c1.total);
            } catch (RuntimeException e) {
                inv.failed.add(tag + ": engine C (device " + devices[1] + ") raised " + e);
                log.println("   ENGINE C RAISED: " + e);
            }
        } else {
            inv.skipped.add(tag + ": second device (only one free device given)");
        }
        // ---- engine D: sites in reversed construction order
        if (k > 1) {
            try (TorusBoxGpuEngine eng = construct(in, rev, devices[0], in.nSlots)) {
                final Res d1 = call(eng, in, ident, rev, null);
                check(inv, tag + ": sites in reversed construction order bitwise", bitwise(a1, d1, cases));
                log.printf(Locale.ROOT, "   engine D (sites reversed at construction): %.3f s%n", d1.total);
            } catch (RuntimeException e) {
                inv.failed.add(tag + ": engine D (sites reversed) raised " + e);
                log.println("   ENGINE D RAISED: " + e);
            }
        }
    }

    static void check(Invariance inv, String what, List<String> diffs) {
        if (diffs == null) {
            inv.skipped.add(what + " (one site)");
            return;
        }
        if (diffs.isEmpty()) {
            inv.passed.add(what);
            log.println("   invariance ok: " + what);
        } else {
            inv.failed.add(what + ": " + diffs.size() + " case(s) differ, e.g. " + diffs.get(0));
            log.println("   INVARIANCE FAILED: " + what);
            for (String d : diffs) log.println("      " + d);
        }
    }

    // =========================================================================================== main

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: TorusBoxGpuCaseRunner CASES_DIR OUT_PREFIX [--devices 0,1] [--only REGEX] [--tiers t1,t2] [--no-invariance]");
            System.exit(2);
        }
        final File casesDir = new File(args[0]);
        final String out = args[1];
        int[] devices = {0};
        Pattern only = null;
        Set<String> tiers = null;
        boolean invariance = true;
        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--devices": {
                    final String[] f = args[++i].split(",");
                    devices = new int[f.length];
                    for (int j = 0; j < f.length; j++) devices[j] = Integer.parseInt(f[j].trim());
                    break;
                }
                case "--only": only = Pattern.compile(args[++i]); break;
                case "--tiers": tiers = new HashSet<String>(Arrays.asList(args[++i].split(","))); break;
                case "--no-invariance": invariance = false; break;
                default: throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        log = new PrintStream(new Tee(System.out, new FileOutputStream(out + ".log")), true, "UTF-8");
        log.println("TorusBoxGpuCaseRunner  cases " + casesDir.getAbsolutePath() + "  devices " + Arrays.toString(devices)
                + "  invariance " + invariance + "  CUDA_VISIBLE_DEVICES=" + System.getenv("CUDA_VISIBLE_DEVICES"));
        log.println("engine class " + TorusBoxGpuEngine.class.getProtectionDomain().getCodeSource().getLocation()
                + "  ptx " + (System.getProperty("torus.box.ptx") != null ? System.getProperty("torus.box.ptx") : "(resource)"));
        log.println("rule: finite logL passes iff |GPU - Python(prec_check False)| <= max(1e-9, 0.3 x PRECISION bound); a "
                + "non-finite expected value must be reproduced with NONPOSITIVE when well conditioned (else NOT-REQUIRED)");

        // ---- cases from index.tsv, grouped by batch_key in index order
        final List<String> lines = Files.readAllLines(new File(casesDir, "index.tsv").toPath(), StandardCharsets.UTF_8);
        final List<String> head = Arrays.asList(lines.get(0).split("\t", -1));
        final int iCase = head.indexOf("case"), iTier = head.indexOf("tier"), iKey = head.indexOf("batch_key");
        if (iCase < 0 || iTier < 0 || iKey < 0) throw new IOException("index.tsv lacks case / tier / batch_key");
        final Map<String, List<Case>> groups = new LinkedHashMap<String, List<Case>>();
        int nCases = 0;
        for (int r = 1; r < lines.size(); r++) {
            if (lines.get(r).trim().isEmpty()) continue;
            final String[] f = lines.get(r).split("\t", -1);
            if (only != null && !only.matcher(f[iCase]).find()) continue;
            if (tiers != null && !tiers.contains(f[iTier])) continue;
            final Case c = new Case(casesDir, f[iCase], f[iTier], f[iKey]);
            if (!groups.containsKey(c.batchKey)) groups.put(c.batchKey, new ArrayList<Case>());
            groups.get(c.batchKey).add(c);
            nCases++;
        }
        log.printf(Locale.ROOT, "%d cases in %d batch groups%n", nCases, groups.size());

        final List<Row> rows = new ArrayList<Row>();
        final Invariance inv = new Invariance();
        final TableStats ts = new TableStats();
        final List<Timing> timings = new ArrayList<Timing>();
        int gi = 0;
        final long tAll = System.nanoTime();
        for (List<Case> g : groups.values()) {
            gi++;
            try {
                runGroup(gi, g, devices, invariance, rows, inv, ts, timings);
            } catch (Throwable t) {
                log.println("   GROUP ERROR: " + t);
                t.printStackTrace(log);
                inv.failed.add("group " + gi + " raised " + t);
                for (Case c : g) {
                    boolean seen = false;
                    for (Row r : rows) seen |= r.c == c;
                    if (!seen) {
                        final Row r = new Row(c, gi);
                        r.note = "group raised " + t;
                        rows.add(r);
                    }
                }
            }
        }
        final double wall = (System.nanoTime() - tAll) * 1e-9;
        writeTsv(new File(out + ".tsv"), rows);
        final boolean ok = summary(rows, inv, ts, timings, wall);
        log.close();
        System.exit(ok ? 0 : 1);
    }

    // =========================================================================================== output

    static void writeTsv(File f, List<Row> rows) throws IOException {
        try (PrintWriter w = new PrintWriter(f, "UTF-8")) {
            w.println("case\ttier\tgroup\tn\tboxes\tverdict\tlogL_gpu\tlogL_gpu_hex\tlogL_py\tabs_diff\ttol\tdiff_over_tol\tprec"
                    + "\twell_conditioned\tstatus_gpu\tstatus_py\tleak_gpu\tleak_py\tleak_class\tres_gpu\tres_py\tres_class"
                    + "\trho_gpu\trho_py\trho_class\tgroup_call_seconds\tnumpy_eval_seconds\tnote");
            for (Row r : rows) {
                w.println(String.join("\t", r.c.name, r.c.tier, Integer.toString(r.group), Integer.toString(r.c.n),
                        Integer.toString(r.c.boxes), r.verdict, Double.toString(r.gpu), Double.toHexString(r.gpu),
                        Double.toString(r.c.expLogL), Double.toString(r.diff), Double.toString(r.tol),
                        Double.toString(r.diff / r.tol), Double.toString(r.c.prec), Boolean.toString(r.c.wellConditioned),
                        Integer.toString(r.status), Integer.toString(r.c.expStatus), Double.toString(r.leak),
                        Double.toString(r.c.expLeak), r.leakClass, Double.toString(r.res), Double.toString(r.c.expRes),
                        r.resClass, Double.toString(r.rho), Double.toString(r.c.expRho), r.rhoClass,
                        Double.toString(r.gpuSeconds), Double.toString(r.c.numpySeconds), r.note.trim()));
            }
        }
    }

    static boolean summary(List<Row> rows, Invariance inv, TableStats ts, List<Timing> timings, double wall) {
        log.printf(Locale.ROOT, "%n================================ SUMMARY (%.1f s wall) ================================%n", wall);
        final Map<String, int[]> byTier = new LinkedHashMap<String, int[]>();
        int pass = 0, notReq = 0, fail = 0, err = 0, stOk = 0;
        for (Row r : rows) {
            int[] t = byTier.get(r.c.tier);
            if (t == null) byTier.put(r.c.tier, t = new int[5]);
            t[0]++;
            switch (r.verdict) {
                case "PASS": pass++; t[1]++; break;
                case "NOT-REQUIRED": notReq++; t[2]++; break;
                case "FAIL": fail++; t[3]++; break;
                default: err++; t[4]++;
            }
            if (r.statusMatch) stOk++;
        }
        log.printf(Locale.ROOT, "logL: %d cases: %d PASS, %d NOT-REQUIRED, %d FAIL, %d ERROR%n", rows.size(), pass, notReq, fail, err);
        for (Map.Entry<String, int[]> e : byTier.entrySet()) {
            final int[] t = e.getValue();
            log.printf(Locale.ROOT, "   %-10s %3d cases: %3d pass, %d not-required, %d fail, %d error%n", e.getKey(), t[0], t[1], t[2], t[3], t[4]);
        }
        // failures with numbers
        for (Row r : rows) {
            if (!r.passed()) {
                log.printf(Locale.ROOT, "   %s %s: GPU %s, Python %s, |d| %s, tol %s (prec %s, well conditioned %s), status %d vs %d. %s%n",
                        r.verdict, r.c.name, r.gpu, r.c.expLogL, sci(r.diff), sci(r.tol), sci(r.c.prec), r.c.wellConditioned,
                        r.status, r.c.expStatus, r.note);
            }
        }
        for (Row r : rows) {
            if (r.verdict.equals("NOT-REQUIRED")) log.printf(Locale.ROOT, "   NOT-REQUIRED %s: %s%n", r.c.name, r.note);
        }
        // worst finite agreements relative to their tolerance
        final List<Row> fin = new ArrayList<Row>();
        for (Row r : rows) if (Double.isFinite(r.c.expLogL) && Double.isFinite(r.diff)) fin.add(r);
        fin.sort((x, y) -> Double.compare(y.diff / y.tol, x.diff / x.tol));
        log.println("worst |d| / tol (finite expected logL):");
        for (int i = 0; i < Math.min(12, fin.size()); i++) {
            final Row r = fin.get(i);
            log.printf(Locale.ROOT, "   %-30s |d| %-9s tol %-9s ratio %-9s (prec %s, rho_min %s)%n", r.c.name, sci(r.diff), sci(r.tol),
                    sci(r.diff / r.tol), sci(r.c.prec), sci(r.c.expRho));
        }
        fin.sort((x, y) -> Double.compare(y.diff, x.diff));
        log.println("largest |d| (finite expected logL):");
        for (int i = 0; i < Math.min(8, fin.size()); i++) {
            final Row r = fin.get(i);
            log.printf(Locale.ROOT, "   %-30s |d| %-9s tol %-9s (prec %s)%n", r.c.name, sci(r.diff), sci(r.tol), sci(r.c.prec));
        }
        int exactL = 0, under1e12 = 0, under1e10 = 0;
        for (Row r : fin) {
            if (r.diff == 0) exactL++;
            if (r.diff <= 1e-12) under1e12++;
            if (r.diff <= 1e-10) under1e10++;
        }
        log.printf(Locale.ROOT, "finite cases: %d bitwise equal, %d within 1e-12, %d within 1e-10, of %d%n", exactL, under1e12, under1e10, fin.size());

        // guards
        log.printf(Locale.ROOT, "status bits equal in %d of %d cases%n", stOk, rows.size());
        for (Row r : rows) if (!r.statusMatch && !r.verdict.equals("ERROR")) log.printf(Locale.ROOT, "   status %s: GPU %d, Python %d: %s%n", r.c.name, r.status, r.c.expStatus, r.note);
        for (String g : new String[]{"leak", "res", "rho"}) {
            final Map<String, Integer> cnt = new LinkedHashMap<String, Integer>();
            for (String k : new String[]{"exact", "close", "magnitude", "differ", "-"}) cnt.put(k, 0);
            for (Row r : rows) {
                final String k = g.equals("leak") ? r.leakClass : g.equals("res") ? r.resClass : r.rhoClass;
                cnt.put(k, cnt.get(k) + 1);
            }
            log.printf(Locale.ROOT, "guard %-4s: %s%n", g, cnt);
            for (Row r : rows) {
                final String k = g.equals("leak") ? r.leakClass : g.equals("res") ? r.resClass : r.rhoClass;
                if (k.equals("magnitude") || k.equals("differ")) {
                    final double gv = g.equals("leak") ? r.leak : g.equals("res") ? r.res : r.rho;
                    final double pv = g.equals("leak") ? r.c.expLeak : g.equals("res") ? r.c.expRes : r.c.expRho;
                    log.printf(Locale.ROOT, "   %-9s %-30s GPU %-24s Python %-24s%n", k, r.c.name, Double.toString(gv), Double.toString(pv));
                }
            }
        }

        // init tables
        log.println("init tables (GB1): worst over every box checked");
        for (Map.Entry<String, double[]> e : ts.worst.entrySet()) {
            log.printf(Locale.ROOT, "   %-52s %-10s (limit %s, %d checks, worst at %s)%n", e.getKey(), sci(e.getValue()[0]),
                    sci(ts.limit.get(e.getKey())), (int) e.getValue()[1], ts.where.get(e.getKey()));
        }
        for (String f : ts.failures) log.println("   TABLE CHECK FAILED: " + f);

        // invariance
        log.printf(Locale.ROOT, "invariance: %d checks passed, %d failed, %d skipped%n", inv.passed.size(), inv.failed.size(), inv.skipped.size());
        for (String f : inv.failed) log.println("   FAILED: " + f);
        final Map<String, Integer> why = new LinkedHashMap<String, Integer>();
        for (String s : inv.skipped) {
            final String reason = s.replaceFirst("^group \\d+ \\([^)]*\\): ", "");
            why.put(reason, why.containsKey(reason) ? why.get(reason) + 1 : 1);
        }
        for (Map.Entry<String, Integer> e : why.entrySet()) log.printf(Locale.ROOT, "   skipped in %d group(s): %s%n", e.getValue(), e.getKey());
        final Map<String, Integer> kinds = new LinkedHashMap<String, Integer>();
        for (String s : inv.passed) {
            final String kind = s.replaceFirst("^group \\d+ \\([^)]*\\): ", "").replaceAll("\\d+", "#");
            kinds.put(kind, kinds.containsKey(kind) ? kinds.get(kind) + 1 : 1);
        }
        for (Map.Entry<String, Integer> e : kinds.entrySet()) log.printf(Locale.ROOT, "   passed in %d group(s): %s%n", e.getValue(), e.getKey());

        // timing
        log.println("timing (RTX A6000 FP64; one site-grid = one (site, box) slot of n x n through the whole tree):");
        for (Timing t : timings) {
            final double perSlot = t.perCall / t.slots;
            String perSite = "";
            if (!t.perSite.isEmpty()) {
                double sx = 0, sy = 0;
                for (double[] p : t.perSite) {
                    sx += p[0];
                    sy += p[1];
                }
                perSite = String.format(Locale.ROOT, ", one site per call %.4f s per site-grid", sy / sx);
            }
            log.printf(Locale.ROOT, "   %-34s n %4d, %3d nodes, %2d sites, %2d slots: construct %.2f s, one call %.3f s = %.4f s per site-grid%s; numpy (1 thread) %.1f s%n",
                    t.group, t.n, t.nodes, t.sites, t.slots, t.construct, t.perCall, perSlot, perSite, t.numpySeconds);
            if (t.n >= 512) {
                // device memory traffic per node and slot, in passes over one n x n double grid (the half spectrum is
                // about the same size): Z2D (column pass read + write, c2r read + write: ~4), D2Z of an internal child
                // (~4, on about half the nodes: ~2), propagate (read X and Gp, write X: 3), combine (read R and P,
                // write P: 3), scale_parts (2), tip_build (1 on the other half: 0.5) = ~14.5 passes
                final double gridBytes = 8.0 * t.n * t.n;
                final double bytes = 14.5 * gridBytes * (t.nodes - 1);
                log.printf(Locale.ROOT, "      H100 at n %d: %.4f s per site-grid if FP64-bound (A6000 / %.0f); but the memory traffic is "
                                + "~%.0f GB per site-grid, a floor of %.4f s at 3.35 TB/s (H100 SXM; the A6000's 768 GB/s gives %.4f s, "
                                + "%.1fx below what it measures, so the A6000 is FP64-bound and the H100 will be bandwidth-bound): expect "
                                + "~%.3f s per site-grid or a little more%n", t.n, perSlot / H100_FP64_RATIO, H100_FP64_RATIO,
                        bytes / 1e9, bytes / 3.35e12, bytes / 768e9, perSlot / (bytes / 768e9),
                        Math.max(perSlot / H100_FP64_RATIO, bytes / 3.35e12));
            }
        }
        final boolean ok = fail == 0 && err == 0 && inv.failed.isEmpty() && ts.failures.isEmpty();
        log.println(ok ? "RESULT: ALL CHECKS PASS" : "RESULT: FAILURES (see above)");
        return ok;
    }

    static String sci(double x) {
        if (Double.isNaN(x)) return "NaN";
        if (Double.isInfinite(x)) return x > 0 ? "inf" : "-inf";
        if (x == 0) return "0";
        return String.format(Locale.ROOT, "%.3g", x);
    }

    /** Writes to two streams (the log file and stdout). */
    static final class Tee extends OutputStream {
        private final OutputStream a, b;

        Tee(OutputStream a, OutputStream b) {
            this.a = a;
            this.b = b;
        }

        @Override
        public void write(int x) throws IOException {
            a.write(x);
            b.write(x);
        }

        @Override
        public void write(byte[] buf, int off, int len) throws IOException {
            a.write(buf, off, len);
            b.write(buf, off, len);
        }

        @Override
        public void flush() throws IOException {
            a.flush();
            b.flush();
        }

        @Override
        public void close() throws IOException {
            flush();
            b.close();
        }
    }
}
