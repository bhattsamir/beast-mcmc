/*
 * TorusLandingCountLogger.java
 *
 * @author Samir Bhatt
 */

package dr.evomodel.treedatalikelihood.continuous.gpu;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.inference.loggers.Logger;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.TreeSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Exact expected labelled landing counts of every residue of a {@link TorusLandingGpuLikelihood}, computed inside the
 * MCMC at its own posterior states: the estimand of results_pipeline/labelled_counts.py, as the offline replay
 * (h3_alphafold/labelled_counts_h3.py) computes it from the logged states, with the same per-entry clipping.
 *
 * <p>At every state that is a multiple of logEvery and at least startState, the logger copies each residue's
 * precision entries and jump rate (pure reads, exactly as the likelihood reads them) and counts every residue that is
 * not conserved with {@link TorusLandingCountEngine}, one engine per device, residues split in contiguous blocks.
 * Conserved residues (decided once, from the tip data, before the run: all observed models within a circular SD of
 * conservedSd degrees on both angles and none more than conservedMax degrees from the circular mean, the rule that
 * defined the H3 zero-by-rule positions) are recorded with status CONSERVED and not counted.
 *
 * <p>With async="true" the counting runs on the logger's own threads (and normally its own devices), on the copied
 * parameters, and the chain continues at once. At most two states wait in the queue: if counting falls further behind,
 * the chain counts the next due state itself (it waits, nothing is dropped), so records can then appear out of state
 * order (every record carries its state; in_mcmc_counts_export.py sorts). Otherwise log() returns after the counts
 * are written.
 *
 * <p>The logger never changes the chain: it draws no random numbers, writes no parameter, calls no likelihood method
 * that computes anything (the chain's cached residue log-likelihood is read by reflection for the record), uses its
 * own CUDA contexts, and catches every Throwable. A failure is recorded in that state's records and the chain goes on.
 *
 * <p>Outputs in outDir: meta.json (tree, node map by clade, residues, conserved set, grid, settings), records.tsv (one
 * line per counted state and residue: parameters, log-likelihoods, raw 3x3 tree totals, clipped amounts, flags,
 * timing) and branch.bin (float32, per record and node: clipped landings and clipped state changes on the branch above
 * the node, the replay's branch_all and branch_change). in_mcmc_counts_export.py turns a run into the replay's files.
 *
 * @author Samir Bhatt
 */
public final class TorusLandingCountLogger implements Logger {

    public static final int STATUS_CONSERVED = 1 << 15;     // not counted: zero by rule
    public static final int STATUS_FAILED = 1 << 4;         // the count threw; see the log
    public static final int STATUS_LOGL = 1 << 5;           // |count log L - chain's cached residue log L| > logLTol
    public static final int STATUS_CACHE = 1 << 13;         // the chain's cached value was not current (not compared)
    private static final double NEG_TOL = 1e-9;

    private final int K;                                // regions (3 coarse states or 6 basins), from the region file
    private final String[] regionNames;
    private final TorusLandingGpuLikelihood likelihood;
    private final boolean combined;                     // landing AND additive jumps: records gain lambdaA, sJ
    private final Tree tree;
    private final String[] residueIds;
    private final boolean[] conserved;
    private final long logEvery, startState;
    private final File outDir;
    private final boolean async;
    private final double logLTol;
    private final int nodeCount, nResidues;
    private final int[] engineOf, localIndex;
    private final TorusLandingCountEngine[] engines;
    private final ExecutorService[] devicePools;       // one thread per engine (CUDA contexts are per thread use)
    private final ExecutorService writer;              // async: counts states in order, one at a time
    private final View view;
    private final String settings;
    private PrintWriter records;
    private OutputStream branchOut;
    private long countedStates = 0, failures = 0;
    private boolean disabled = false;
    private final java.util.logging.Logger log = java.util.logging.Logger.getLogger("dr.evomodel");

    public TorusLandingCountLogger(TorusLandingGpuLikelihood likelihood, Tree tree, String[] residueIds, int nGrid,
                                   double[][][] tipData, double[] w, double[][] mu, double[][][] s, double[][] sigmaEps,
                                   double[][] frac, String[] regionNames, int[] devices, long logEvery, long startState, File outDir,
                                   boolean async, double logLTol, double conservedSd, double conservedMax,
                                   String settings) {
        this.likelihood = likelihood;
        this.combined = likelihood.isCombined();
        this.tree = tree;
        this.residueIds = residueIds.clone();
        this.nResidues = residueIds.length;
        this.logEvery = logEvery;
        this.startState = startState;
        this.outDir = outDir;
        this.async = async;
        this.logLTol = logLTol;
        this.settings = settings;
        this.K = frac.length;
        if (regionNames.length != K) throw new IllegalArgumentException(K + " regions but " + regionNames.length + " region names");
        this.regionNames = regionNames.clone();
        this.nodeCount = tree.getNodeCount();
        if (likelihood.getResidueCount() != nResidues) throw new IllegalArgumentException("residue count differs from the likelihood");
        if (logEvery < 1) throw new IllegalArgumentException("logEvery must be >= 1");

        // conserved residues, from the data alone, before the run
        this.conserved = new boolean[nResidues];
        for (int r = 0; r < nResidues; r++) conserved[r] = isConserved(tipData[r], conservedSd, conservedMax);

        // engines: contiguous blocks of residues per device (the likelihood's rule)
        final int[][] children = new int[nodeCount][];
        for (int i = 0; i < nodeCount; i++) {
            final NodeRef node = tree.getNode(i);
            children[i] = new int[tree.getChildCount(node)];
            for (int c = 0; c < children[i].length; c++) children[i][c] = tree.getChild(node, c).getNumber();
        }
        final double[] branch = likelihood.getBranchLengths();
        final int E = Math.min(devices.length, nResidues);
        this.engines = new TorusLandingCountEngine[E];
        this.engineOf = new int[nResidues];
        this.localIndex = new int[nResidues];
        this.devicePools = new ExecutorService[E];
        int start = 0;
        for (int e = 0; e < E; e++) {
            final int size = nResidues / E + (e < nResidues % E ? 1 : 0);
            final double[][][] block = new double[size][][];
            for (int k = 0; k < size; k++) {
                engineOf[start + k] = e;
                localIndex[start + k] = k;
                block[k] = tipData[start + k];
            }
            engines[e] = new TorusLandingCountEngine(nGrid, children, tree.getRoot().getNumber(), branch, block, w, mu, s,
                    sigmaEps, frac, TorusLandingCountEngine.C0, devices[e]);
            final int dev = devices[e];
            devicePools[e] = Executors.newSingleThreadExecutor(r -> {
                final Thread t = new Thread(r, "torus-count-" + dev);
                t.setDaemon(true);
                return t;
            });
            start += size;
        }
        this.writer = async ? new ThreadPoolExecutor(1, 1, 0L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<Runnable>(2),
                r -> { final Thread t = new Thread(r, "torus-count-writer"); t.setDaemon(true); return t; },
                new ThreadPoolExecutor.CallerRunsPolicy()) : null;     // a full queue makes the chain wait, never drops
        this.view = new View(likelihood);
        int nc = 0;
        for (boolean c : conserved) if (c) nc++;
        log.info("TorusLandingCountLogger: " + nResidues + " residues (" + nc + " conserved, not counted), nGrid " + nGrid
                + ", devices " + Arrays.toString(Arrays.copyOf(devices, E)) + ", every " + logEvery + " states from "
                + startState + (async ? ", asynchronous" : "") + ", output " + outDir);
    }

    // ---- the conserved rule (h3_alphafold/cap_sensitivity_20261017: sd_phi, sd_psi <= 3 deg and n_out8 == 0)

    /** circular SD (deg) of both angles <= sd and no model with either angle > max deg from the circular mean */
    static boolean isConserved(double[][] tips, double sd, double max) {
        if (!(sd > 0.0)) return false;
        final List<double[]> y = new ArrayList<double[]>();
        for (double[] t : tips) if (t != null) y.add(t);
        if (y.size() < 2) return false;
        final double[] mean = new double[2];
        for (int ax = 0; ax < 2; ax++) {
            double c = 0.0, s = 0.0;
            for (double[] t : y) { c += Math.cos(t[ax]); s += Math.sin(t[ax]); }
            c /= y.size(); s /= y.size();
            final double R = Math.hypot(c, s);
            if (!(Math.toDegrees(Math.sqrt(-2.0 * Math.log(R))) <= sd)) return false;
            mean[ax] = Math.atan2(s, c);
        }
        for (double[] t : y) {
            for (int ax = 0; ax < 2; ax++) {
                final double d = Math.abs(Math.atan2(Math.sin(t[ax] - mean[ax]), Math.cos(t[ax] - mean[ax])));
                if (Math.toDegrees(d) > max) return false;
            }
        }
        return true;
    }

    public boolean[] getConserved() { return conserved.clone(); }

    // ---- Logger

    @Override
    public void startLogging() {
        try {
            if (outDir.exists() && !Boolean.getBoolean("log.allow.overwrite")) {
                throw new IllegalStateException(outDir + " exists (set -Dlog.allow.overwrite=true to reuse it)");
            }
            if (!outDir.mkdirs() && !outDir.isDirectory()) throw new IOException("cannot create " + outDir);
            writeMeta(false);
            records = new PrintWriter(new java.io.OutputStreamWriter(new FileOutputStream(new File(outDir, "records.tsv")),
                    StandardCharsets.UTF_8));
            final StringBuilder h = new StringBuilder("state\tresidue_index\tresidue_id\tP00\tP01\tP10\tP11\tlambda\tS00\tS01\tS10\tS11"
                    + "\tgrid\tlogL_count\tlogL_chain\tk0\tneg");
            for (int a = 0; a < K; a++) for (int b = 0; b < K; b++) h.append("\tN_").append(a).append(b);
            for (int a = 0; a < K; a++) for (int b = 0; b < K; b++) h.append("\tNc_").append(a).append(b);
            h.append("\tclip_all\tclip_change\tC\tQ\tn_panels\tn_terms\tstatus\tsec");
            if (combined) h.append("\tlambdaA\tsJ");           // combined model only: landing-only format unchanged
            records.println(h);
            records.flush();
            branchOut = new BufferedOutputStream(new FileOutputStream(new File(outDir, "branch.bin")), 1 << 20);
        } catch (IOException e) {
            throw new RuntimeException("TorusLandingCountLogger: " + e.getMessage(), e);
        }
    }

    @Override
    public void log(long state) {
        if (disabled || state < startState || state % logEvery != 0) return;
        try {
            final double[][] P = new double[nResidues][];
            final double[] lam = new double[nResidues];
            final double[] lamA = new double[nResidues], sJ = new double[nResidues];
            final double[] chainLogL = new double[nResidues];
            final Parameter jr = likelihood.getJumpRate();
            final Parameter ar = likelihood.getAdditiveRate(), as = likelihood.getAdditiveSize();
            for (int r = 0; r < nResidues; r++) {
                final MatrixParameter p = likelihood.getPrecision(r);
                P[r] = new double[]{p.getParameterValue(0, 0), p.getParameterValue(0, 1), p.getParameterValue(1, 0),
                        p.getParameterValue(1, 1)};
                lam[r] = jr.getParameterValue(jr.getDimension() == 1 ? 0 : r);
                if (combined) {
                    lamA[r] = ar.getParameterValue(ar.getDimension() == 1 ? 0 : r);
                    sJ[r] = as.getParameterValue(as.getDimension() == 1 ? 0 : r);
                }
                chainLogL[r] = view.cachedResidueLogL(r, P[r], lam[r], lamA[r], sJ[r]);
            }
            if (async) writer.submit(() -> countState(state, P, lam, lamA, sJ, chainLogL));
            else countState(state, P, lam, lamA, sJ, chainLogL);
        } catch (Throwable t) {
            failures++;
            log.severe("TorusLandingCountLogger: state " + state + " not counted: " + t);
        }
    }

    @Override
    public void stopLogging() {
        try {
            if (writer != null) {
                writer.shutdown();
                writer.awaitTermination(7, TimeUnit.DAYS);
            }
            for (ExecutorService p : devicePools) p.shutdown();
            if (records != null) records.close();
            if (branchOut != null) branchOut.close();
            writeMeta(true);
        } catch (Throwable t) {
            log.severe("TorusLandingCountLogger: stopping: " + t);
        } finally {
            for (TorusLandingCountEngine e : engines) {
                try { e.close(); } catch (Throwable ignored) { }
            }
        }
    }

    // ---- counting one state

    private void countState(long state, double[][] P, double[] lam, double[] lamA, double[] sJ, double[] chainLogL) {
        final long t0 = System.nanoTime();
        final TorusLandingCountEngine.Result[] res = new TorusLandingCountEngine.Result[nResidues];
        final double[][][] sig = new double[nResidues][][];
        final String[] err = new String[nResidues];
        final List<Future<?>> jobs = new ArrayList<Future<?>>();
        for (int e = 0; e < engines.length; e++) {
            final int ee = e;
            jobs.add(devicePools[e].submit(() -> {
                for (int r = 0; r < nResidues; r++) {
                    if (engineOf[r] != ee) continue;
                    try {
                        sig[r] = TorusLandingGpuEngine.invertPrecision(new double[][]{{P[r][0], P[r][1]}, {P[r][2], P[r][3]}});
                        if (!conserved[r]) {
                            res[r] = combined
                                    ? engines[ee].count(localIndex[r], sig[r], lam[r], lamA[r], sJ[r], TorusLandingCountEngine.C1, 1)
                                    : engines[ee].count(localIndex[r], sig[r], lam[r], TorusLandingCountEngine.C1, 1);
                        }
                    } catch (Throwable t) {
                        err[r] = t.toString();
                    }
                }
            }));
        }
        for (Future<?> f : jobs) {
            try { f.get(); } catch (Throwable t) { log.severe("TorusLandingCountLogger: device task: " + t); }
        }
        write(state, P, lam, lamA, sJ, chainLogL, sig, res, err, (System.nanoTime() - t0) * 1e-9);
    }

    private synchronized void write(long state, double[][] P, double[] lam, double[] lamA, double[] sJ, double[] chainLogL,
                                    double[][][] sig, TorusLandingCountEngine.Result[] res, String[] err, double sec) {
        final ByteBuffer bb = ByteBuffer.allocate(4 * 2 * nodeCount).order(ByteOrder.LITTLE_ENDIAN);
        try {
            for (int r = 0; r < nResidues; r++) {
                final StringBuilder b = new StringBuilder();
                b.append(state).append('\t').append(r).append('\t').append(residueIds[r]);
                for (double x : P[r]) b.append('\t').append(x);
                b.append('\t').append(lam[r]);
                for (int i = 0; i < 4; i++) b.append('\t').append(sig[r] == null ? Double.NaN : sig[r][i / 2][i % 2]);
                final TorusLandingCountEngine.Result x = res[r];
                int status = 0;
                final double[][] tot = new double[K][K], totc = new double[K][K];
                double clipAll = 0.0, clipChange = 0.0, C = 0.0, Q = 0.0, min = 0.0, sum = 0.0;
                bb.clear();
                if (conserved[r]) {
                    status |= STATUS_CONSERVED;
                    for (int i = 0; i < 2 * nodeCount; i++) bb.putFloat(0f);
                } else if (x == null) {
                    status |= STATUS_FAILED;
                    log.severe("TorusLandingCountLogger: state " + state + " residue " + residueIds[r] + ": " + err[r]);
                    for (int i = 0; i < 2 * nodeCount; i++) bb.putFloat(Float.NaN);
                } else {
                    status |= x.status;
                    for (int v = 0; v < nodeCount; v++) {
                        double all = 0.0, change = 0.0;
                        for (int a = 0; a < K; a++) {
                            for (int c = 0; c < K; c++) {
                                final double n = x.N[v][a][c];
                                tot[a][c] += n;
                                min = Math.min(min, n);
                                sum += n;
                                if (n < 0) { clipAll -= n; if (a != c) clipChange -= n; }
                                final double nc = Math.max(n, 0.0);       // labelled_counts_h3: np.maximum(N, 0) per entry
                                totc[a][c] += nc;
                                all += nc;
                                if (a != c) change += nc;
                            }
                        }
                        C += change;
                        Q += all;
                        bb.putFloat((float) all);
                        bb.putFloat((float) change);
                    }
                    if (!Double.isFinite(chainLogL[r])) status |= STATUS_CACHE;
                    else if (!(Math.abs(x.logL - chainLogL[r]) <= logLTol)) status |= STATUS_LOGL;
                }
                final double neg = Math.min(0.0, min) / Math.max(sum, 1e-300);
                b.append('\t').append(engines[0].getGridSize());
                b.append('\t').append(x == null ? Double.NaN : x.logL).append('\t').append(chainLogL[r]);
                b.append('\t').append(x == null ? Double.NaN : x.k0).append('\t').append(neg);
                for (int a = 0; a < K; a++) for (int c = 0; c < K; c++) b.append('\t').append(tot[a][c]);
                for (int a = 0; a < K; a++) for (int c = 0; c < K; c++) b.append('\t').append(totc[a][c]);
                b.append('\t').append(clipAll).append('\t').append(clipChange).append('\t').append(C).append('\t').append(Q);
                b.append('\t').append(x == null ? 0 : x.nPanels).append('\t').append(x == null ? 0 : x.nTerms);
                b.append('\t').append(status).append('\t').append(x == null ? 0.0 : x.secTotal);
                if (combined) b.append('\t').append(lamA[r]).append('\t').append(sJ[r]);
                records.println(b);
                branchOut.write(bb.array());
            }
            records.flush();
            branchOut.flush();
            countedStates++;
            log.info(String.format(Locale.ROOT, "TorusLandingCountLogger: state %d counted in %.1f s", state, sec));
        } catch (IOException e) {
            failures++;
            log.severe("TorusLandingCountLogger: writing state " + state + ": " + e);
        }
    }

    // ---- meta.json

    private void writeMeta(boolean stopped) throws IOException {
        final StringBuilder b = new StringBuilder("{\"format\": \"torus-counts-inmcmc-v1\", \"K\": ").append(K).append(", \"regions\": [");
        for (int a = 0; a < K; a++) b.append(a == 0 ? "" : ", ").append(json(regionNames[a]));
        b.append("], \"nodes\": ").append(nodeCount);
        b.append(", \"root\": ").append(tree.getRoot().getNumber()).append(", \"parent\": [");
        for (int v = 0; v < nodeCount; v++) {
            final NodeRef n = tree.getNode(v);
            b.append(v == 0 ? "" : ", ").append(tree.isRoot(n) ? -1 : tree.getParent(n).getNumber());
        }
        final double[] br = likelihood.getBranchLengths();
        b.append("], \"branch\": [");
        for (int v = 0; v < nodeCount; v++) b.append(v == 0 ? "" : ", ").append('"').append(br[v]).append('"');
        b.append("], \"clade\": [");
        final String[] clade = cladeKeys();
        for (int v = 0; v < nodeCount; v++) b.append(v == 0 ? "" : ", ").append(json(clade[v]));
        b.append("], \"residues\": [");
        for (int r = 0; r < nResidues; r++) b.append(r == 0 ? "" : ", ").append(json(residueIds[r]));
        b.append("], \"conserved\": [");
        boolean first = true;
        for (int r = 0; r < nResidues; r++) if (conserved[r]) { b.append(first ? "" : ", ").append(json(residueIds[r])); first = false; }
        b.append("], \"grid\": ").append(engines[0].getGridSize()).append(", \"log_every\": ").append(logEvery);
        b.append(", \"start_state\": ").append(startState).append(", \"async\": ").append(async);
        b.append(", \"logL_tol\": ").append(logLTol).append(", \"neg_tol\": ").append(NEG_TOL);
        b.append(", \"settings\": ").append(json(settings));
        if (combined) {
            b.append(", \"model\": \"combined: landing AND additive jumps; counts are landings only\"");
            b.append(", \"additive_rate\": ").append(json(likelihood.getAdditiveRate().getId()));
            b.append(", \"additive_size\": ").append(json(likelihood.getAdditiveSize().getId()));
        }
        b.append(", \"branch_bin\": \"float32 little-endian, per record (records.tsv order) and node: clipped landings, clipped state changes\"");
        if (stopped) b.append(", \"completed\": true, \"counted_states\": ").append(countedStates).append(", \"failures\": ").append(failures);
        b.append("}\n");
        try (PrintWriter w = new PrintWriter(new java.io.OutputStreamWriter(new FileOutputStream(new File(outDir, "meta.json")),
                StandardCharsets.UTF_8))) {
            w.print(b);
        }
    }

    /** per node: '|'-joined sorted descendant taxon ids (count_io.clade_keys) */
    private String[] cladeKeys() {
        final String[] key = new String[nodeCount];
        final List<TreeSet<String>> sets = new ArrayList<TreeSet<String>>();
        for (int v = 0; v < nodeCount; v++) sets.add(null);
        fill(tree.getRoot(), sets);
        for (int v = 0; v < nodeCount; v++) key[v] = String.join("|", sets.get(v));
        return key;
    }

    private TreeSet<String> fill(NodeRef node, List<TreeSet<String>> sets) {
        final TreeSet<String> s = new TreeSet<String>();
        if (tree.isExternal(node)) s.add(tree.getNodeTaxon(node).getId().replace("'", ""));
        for (int c = 0; c < tree.getChildCount(node); c++) s.addAll(fill(tree.getChild(node, c), sets));
        sets.set(node.getNumber(), s);
        return s;
    }

    private static String json(String s) {
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    /** Read-only view of the likelihood's cached residue log-likelihoods (reflection; nothing is ever set). */
    static final class View {
        private final TorusLandingGpuLikelihood lik;
        private final Field residueLogL, key, keyValid, known;

        View(TorusLandingGpuLikelihood lik) {
            this.lik = lik;
            try {
                residueLogL = field("residueLogL");
                key = field("key");
                keyValid = field("keyValid");
                known = field("likelihoodKnown");
            } catch (NoSuchFieldException e) {
                throw new IllegalStateException("TorusLandingGpuLikelihood has no field " + e.getMessage());
            }
        }

        private static Field field(String name) throws NoSuchFieldException {
            final Field f = TorusLandingGpuLikelihood.class.getDeclaredField(name);
            f.setAccessible(true);
            return f;
        }

        /** the chain's cached log L of residue r if it is current for exactly (P, lambda[, lambdaA, sJ]), else NaN */
        double cachedResidueLogL(int r, double[] P, double lambda, double lambdaA, double sJ) {
            try {
                if (!known.getBoolean(lik)) return Double.NaN;
                if (!((boolean[]) keyValid.get(lik))[r]) return Double.NaN;
                final double[] k = ((double[][]) key.get(lik))[r];
                if (Double.doubleToRawLongBits(k[0]) != Double.doubleToRawLongBits(P[0])
                        || Double.doubleToRawLongBits(k[1]) != Double.doubleToRawLongBits(P[1])
                        || Double.doubleToRawLongBits(k[2]) != Double.doubleToRawLongBits(P[2])
                        || Double.doubleToRawLongBits(k[3]) != Double.doubleToRawLongBits(P[3])
                        || Double.doubleToRawLongBits(k[4]) != Double.doubleToRawLongBits(lambda)) return Double.NaN;
                if (k.length == 7 && (Double.doubleToRawLongBits(k[5]) != Double.doubleToRawLongBits(lambdaA)
                        || Double.doubleToRawLongBits(k[6]) != Double.doubleToRawLongBits(sJ))) return Double.NaN;
                return ((double[]) residueLogL.get(lik))[r];
            } catch (IllegalAccessException e) {
                return Double.NaN;
            }
        }
    }
}
