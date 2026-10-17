package test.dr.evomodel.treedatalikelihood.continuous.gpu;

import dr.app.beast.BeastParser;
import dr.app.beast.BeastVersion;
import dr.evomodel.treedatalikelihood.continuous.gpu.TorusLandingGpuLikelihood;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.Parameter;
import dr.xml.XMLObject;

import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * For every residue of a torusLandingGpuLikelihood XML, find a starting point where the likelihood can be
 * computed on the XML's grid: the XML's own start first, then a small grid of diffusion precisions and jump rates
 * (correlation 0). Keeps the XML start if it is computable, otherwise the computable candidate with the highest
 * log-likelihood. All residues are evaluated together, one GPU pass per candidate.
 * usage: TorusLandingStartFinder in.xml starts.csv      (columns: index,q1,q2,rho,lambda,logL,source)
 */
public class TorusLandingStartFinder {
    public static void main(String[] args) throws Exception {
        try {
            final BeastParser parser = new BeastParser(new String[]{args[0]}, null, false, false, false, new BeastVersion());
            final Map<String, XMLObject> store = parser.parse(new FileReader(args[0]), false);
            final TorusLandingGpuLikelihood like = (TorusLandingGpuLikelihood) store.get("tfl").getNativeObject();
            final int R = like.getResidueCount();
            final Parameter lam = like.getJumpRate();
            final Parameter[] diag = new Parameter[R], off = new Parameter[R];
            for (int r = 0; r < R; r++) {
                final CompoundSymmetricMatrix c = (CompoundSymmetricMatrix) like.getPrecision(r);
                diag[r] = c.getDiagonalParameter();
                off[r] = c.getOffDiagonalParameter();
            }
            final List<double[]> cand = new ArrayList<double[]>();
            final double[] own = new double[4 * R];
            for (int r = 0; r < R; r++) {
                own[4 * r] = diag[r].getParameterValue(0); own[4 * r + 1] = diag[r].getParameterValue(1);
                own[4 * r + 2] = off[r].getParameterValue(0); own[4 * r + 3] = lam.getParameterValue(r);
            }
            for (double q : new double[]{1e4, 1e3, 1e2, 1e1}) {
                for (double l : new double[]{1e-3, 1e-2, 1e-1}) cand.add(new double[]{q, q, 0.0, l});
            }
            final double[] bestLL = new double[R];
            final double[][] best = new double[R][];
            final String[] source = new String[R];
            final int[] all = new int[R];
            for (int r = 0; r < R; r++) { all[r] = r; bestLL[r] = Double.NEGATIVE_INFINITY; }
            for (int k = -1; k < cand.size(); k++) {
                for (int r = 0; r < R; r++) {
                    final double[] p = k < 0 ? new double[]{own[4 * r], own[4 * r + 1], own[4 * r + 2], own[4 * r + 3]} : cand.get(k);
                    diag[r].setParameterValue(0, p[0]); diag[r].setParameterValue(1, p[1]);
                    off[r].setParameterValue(0, p[2]); lam.setParameterValue(r, p[3]);
                }
                final double[] ll = new double[R];
                final int[] st = new int[R];
                like.evaluateProposals(all, ll, st);
                int ok = 0;
                for (int r = 0; r < R; r++) {
                    if (!Double.isFinite(ll[r])) continue;
                    ok++;
                    final boolean keepOwn = source[r] != null && source[r].equals("xml");
                    if (!keepOwn && ll[r] > bestLL[r]) {
                        bestLL[r] = ll[r];
                        best[r] = k < 0 ? new double[]{own[4 * r], own[4 * r + 1], own[4 * r + 2], own[4 * r + 3]} : cand.get(k).clone();
                        source[r] = k < 0 ? "xml" : "grid";
                    }
                }
                System.out.printf("candidate %s: computable at %d of %d residues%n",
                        k < 0 ? "XML start" : java.util.Arrays.toString(cand.get(k)), ok, R);
            }
            int none = 0;
            try (PrintWriter pw = new PrintWriter(args[1])) {
                pw.println("index,q1,q2,rho,lambda,logL,source");
                for (int r = 0; r < R; r++) {
                    if (best[r] == null) { none++; pw.printf("%d,,,,,,none%n", r); continue; }
                    pw.printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%s%n", r, best[r][0], best[r][1], best[r][2], best[r][3],
                            bestLL[r], source[r]);
                }
            }
            System.out.printf("residues with no computable start among the candidates: %d%n", none);
        } catch (Throwable t) {
            t.printStackTrace();
            System.exit(3);
        }
        System.exit(0);
    }
}
