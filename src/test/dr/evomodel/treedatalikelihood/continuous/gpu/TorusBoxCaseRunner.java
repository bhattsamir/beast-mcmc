/*
 * TorusBoxCaseRunner.java
 *
 * @author Samir Bhatt
 */

package test.dr.evomodel.treedatalikelihood.continuous.gpu;

import dr.evomodel.treedatalikelihood.continuous.gpu.TorusBoxGpuEngine;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Runs {@link TorusBoxGpuEngine} on a case file written by gpu_build/box_gpu/smoke_box_gpu.py and writes one TSV row
 * per (maxBatch, repetition, evaluation point), with every double in hexadecimal (exact).
 *
 * <p>Case file lines (doubles in any form Double.parseDouble reads, radians unless stated):
 * <pre>
 *   case NAME
 *   nodes NN root R
 *   children V C1 C2 ...          one line per node, in node order (tips: no children)
 *   blen V T
 *   sigma_eps A B C D
 *   comp W MU0 MU1 S00 S01 S10 S11
 *   site J NBOXES
 *   box J B PHI0 PSI0 WPHI WPSI NPHI NPSI FULLPHI FULLPSI      (as box_landing.py's _Box holds it)
 *   boxdeg J B PHI0 PSI0 WPHI WPSI NPHI NPSI                   (the record in degrees, checks Box.fromDegrees)
 *   tip J V B D1 D2 Y1 Y2                                     (observed tip: box, Python's offset, raw angles)
 *   eval SITE S00 S01 S10 S11 LAMBDA
 * </pre>
 * usage: TorusBoxCaseRunner CASE OUT_TSV DEVICE REPS MAXBATCH [MAXBATCH ...] [--dump DIR]
 *
 * @author Samir Bhatt
 */
public final class TorusBoxCaseRunner {

    private TorusBoxCaseRunner() { }

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            System.err.println("usage: TorusBoxCaseRunner CASE OUT_TSV DEVICE REPS MAXBATCH [MAXBATCH ...] [--dump DIR]");
            System.exit(2);
        }
        final File caseFile = new File(args[0]);
        final File outFile = new File(args[1]);
        final int device = Integer.parseInt(args[2]);
        final int reps = Integer.parseInt(args[3]);
        final List<Integer> batches = new ArrayList<Integer>();
        File dump = null;
        for (int i = 4; i < args.length; i++) {
            if (args[i].equals("--dump")) dump = new File(args[++i]);
            else batches.add(Integer.parseInt(args[i]));
        }

        int nn = 0, root = -1;
        final List<int[]> childLines = new ArrayList<int[]>();
        final List<double[]> blenLines = new ArrayList<double[]>(), comps = new ArrayList<double[]>();
        final List<double[]> boxLines = new ArrayList<double[]>(), boxDegLines = new ArrayList<double[]>();
        final List<double[]> tipLines = new ArrayList<double[]>(), evals = new ArrayList<double[]>();
        final List<int[]> siteLines = new ArrayList<int[]>();
        double[][] se = null;
        String name = "?";
        try (BufferedReader in = new BufferedReader(new FileReader(caseFile))) {
            String ln;
            while ((ln = in.readLine()) != null) {
                final String[] f = ln.trim().split("\\s+");
                if (f.length == 0 || f[0].isEmpty()) continue;
                switch (f[0]) {
                    case "case": name = f[1]; break;
                    case "nodes": nn = Integer.parseInt(f[1]); root = Integer.parseInt(f[3]); break;
                    case "children": {
                        final int[] c = new int[f.length - 1];
                        for (int i = 1; i < f.length; i++) c[i - 1] = Integer.parseInt(f[i]);
                        childLines.add(c);
                        break;
                    }
                    case "blen": blenLines.add(doubles(f, 1, 2)); break;
                    case "sigma_eps": { final double[] v = doubles(f, 1, 4); se = new double[][]{{v[0], v[1]}, {v[2], v[3]}}; break; }
                    case "comp": comps.add(doubles(f, 1, 7)); break;
                    case "site": siteLines.add(new int[]{Integer.parseInt(f[1]), Integer.parseInt(f[2])}); break;
                    case "box": boxLines.add(doubles(f, 1, 10)); break;
                    case "boxdeg": boxDegLines.add(doubles(f, 1, 8)); break;
                    case "tip": tipLines.add(doubles(f, 1, 7)); break;
                    case "eval": evals.add(doubles(f, 1, 6)); break;
                    default: throw new IOException("unknown line in " + caseFile + ": " + ln);
                }
            }
        }
        final int[][] children = new int[nn][];
        for (int[] c : childLines) {
            children[c[0]] = new int[c.length - 1];
            System.arraycopy(c, 1, children[c[0]], 0, c.length - 1);
        }
        final double[] blen = new double[nn];
        for (double[] b : blenLines) blen[(int) b[0]] = b[1];
        final int K = comps.size();
        final double[] w = new double[K];
        final double[][] mu = new double[K][];
        final double[][][] s = new double[K][][];
        for (int k = 0; k < K; k++) {
            final double[] c = comps.get(k);
            w[k] = c[0];
            mu[k] = new double[]{c[1], c[2]};
            s[k] = new double[][]{{c[3], c[4]}, {c[5], c[6]}};
        }
        final int nSites = siteLines.size();
        final TorusBoxGpuEngine.Box[][] boxes = new TorusBoxGpuEngine.Box[nSites][];
        for (int[] sl : siteLines) boxes[sl[0]] = new TorusBoxGpuEngine.Box[sl[1]];
        for (double[] b : boxLines) {
            boxes[(int) b[0]][(int) b[1]] = new TorusBoxGpuEngine.Box(b[2], b[3], b[4], b[5], (int) b[6], (int) b[7],
                    b[8] != 0, b[9] != 0);
        }
        final int[][] tipBox = new int[nSites][nn];
        final double[][][] tipOff = new double[nSites][nn][];
        final double[][][] raw = new double[nSites][nn][];
        for (int[] row : tipBox) java.util.Arrays.fill(row, -1);
        for (double[] t : tipLines) {
            final int j = (int) t[0], v = (int) t[1];
            tipBox[j][v] = (int) t[2];
            tipOff[j][v] = new double[]{t[3], t[4]};
            raw[j][v] = new double[]{t[5], t[6]};
        }

        // host helpers against Python: degree records -> radians, tip assignment -> (box, offset), bitwise
        int recMismatch = 0, asgMismatch = 0;
        for (double[] r : boxDegLines) {
            final TorusBoxGpuEngine.Box a = TorusBoxGpuEngine.Box.fromDegrees(r[2], r[3], r[4], r[5], (int) r[6], (int) r[7]);
            final TorusBoxGpuEngine.Box b = boxes[(int) r[0]][(int) r[1]];
            if (a.phi0 != b.phi0 || a.psi0 != b.psi0 || a.widthPhi != b.widthPhi || a.widthPsi != b.widthPsi
                    || a.fullPhi != b.fullPhi || a.fullPsi != b.fullPsi) {
                recMismatch++;
                System.err.println("record mismatch site " + (int) r[0] + " box " + (int) r[1]);
            }
        }
        for (int j = 0; j < nSites; j++) {
            final TorusBoxGpuEngine.TipAssignment ta = TorusBoxGpuEngine.assignTips(boxes[j], raw[j], se, 8.0);
            for (int v = 0; v < nn; v++) {
                if (ta.box[v] != tipBox[j][v] || (ta.box[v] >= 0 && (ta.offset[v][0] != tipOff[j][v][0]
                        || ta.offset[v][1] != tipOff[j][v][1]))) {
                    asgMismatch++;
                    System.err.println("assignment mismatch site " + j + " node " + v);
                }
            }
        }

        final int ne = evals.size();
        final int[] sites = new int[ne];
        final double[] sigma4 = new double[4 * ne], lambda = new double[ne];
        for (int e = 0; e < ne; e++) {
            final double[] v = evals.get(e);
            sites[e] = (int) v[0];
            for (int i = 0; i < 4; i++) sigma4[4 * e + i] = v[1 + i];
            lambda[e] = v[5];
        }
        try (PrintWriter out = new PrintWriter(outFile, "UTF-8")) {
            out.println("# case " + name + " records_mismatch " + recMismatch + " assign_mismatch " + asgMismatch);
            out.println("max_batch\trep\teval\tsite\tlogL\tstatus\tleak\tres\trho_min\tseconds\tmcut\tgsup\tbuffers\tslots");
            for (int mb : batches) {
                final long tc = System.nanoTime();
                try (TorusBoxGpuEngine eng = new TorusBoxGpuEngine(children, root, blen, boxes, tipBox, tipOff, se, w, mu,
                        s, device, mb)) {
                    final double construct = (System.nanoTime() - tc) * 1e-9;
                    System.err.printf("case %s maxBatch %d: construct %.2f s, Mcut %d, gsup %s, buffers %d, slots %d%n",
                            name, mb, construct, eng.getMcut(), Double.toString(eng.getGSup()), eng.getBufferCount(),
                            eng.getSlotCount());
                    if (dump != null && mb == batches.get(0)) dumpTables(eng, dump);
                    for (int rep = 0; rep < reps; rep++) {
                        final double[] ll = new double[ne], leak = new double[ne], res = new double[ne], rho = new double[ne];
                        final int[] st = new int[ne];
                        eng.logLikelihood(sites, sigma4, lambda, ll, st, leak, res, rho);
                        final double sec = eng.getLastSeconds();
                        for (int e = 0; e < ne; e++) {
                            out.println(mb + "\t" + rep + "\t" + e + "\t" + sites[e] + "\t" + Double.toHexString(ll[e]) + "\t"
                                    + st[e] + "\t" + Double.toHexString(leak[e]) + "\t" + Double.toHexString(res[e]) + "\t"
                                    + Double.toHexString(rho[e]) + "\t" + sec + "\t" + eng.getMcut() + "\t"
                                    + Double.toHexString(eng.getGSup()) + "\t" + eng.getBufferCount() + "\t" + eng.getSlotCount());
                        }
                        System.err.printf("  rep %d: %d points in %.3f s%n", rep, ne, sec);
                    }
                }
            }
        }
    }

    private static void dumpTables(TorusBoxGpuEngine eng, File dir) throws IOException {
        if (!dir.mkdirs() && !dir.isDirectory()) throw new IOException("cannot create " + dir);
        for (int g = 0; g < eng.getSlotCount(); g++) {
            for (String t : new String[]{"gp", "e1", "e2", "om1", "om2"}) {
                final double[] a = eng.debugSlotTable(t, g);
                final ByteBuffer bb = ByteBuffer.allocate(8 * a.length).order(ByteOrder.LITTLE_ENDIAN);
                for (double x : a) bb.putDouble(x);
                try (DataOutputStream o = new DataOutputStream(new FileOutputStream(new File(dir, t + "_" + g + ".bin")))) {
                    o.write(bb.array());
                }
            }
        }
    }

    private static double[] doubles(String[] f, int from, int count) throws IOException {
        if (f.length < from + count) throw new IOException("line too short: " + String.join(" ", f));
        final double[] out = new double[count];
        for (int i = 0; i < count; i++) out[i] = Double.parseDouble(f[from + i]);
        return out;
    }
}
