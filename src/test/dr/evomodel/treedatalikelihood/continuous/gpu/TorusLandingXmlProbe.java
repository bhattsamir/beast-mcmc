package test.dr.evomodel.treedatalikelihood.continuous.gpu;

import dr.app.beast.BeastParser;
import dr.app.beast.BeastVersion;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.treedatalikelihood.continuous.gpu.TorusLandingGpuLikelihood;
import dr.xml.XMLObject;

import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Map;

/**
 * Parses a BEAST XML with a torusLandingGpuLikelihood (id "tfl") and writes what the likelihood holds, for
 * checking against an independent implementation: every tip's root-to-tip distance through the likelihood's own
 * branch lengths, and every residue's log-likelihood at the XML's parameter values.
 * usage: TorusLandingXmlProbe in.xml tips.csv residues.csv
 */
public class TorusLandingXmlProbe {
    public static void main(String[] args) throws Exception {
        try {
            final BeastParser parser = new BeastParser(new String[]{args[0]}, null, false, false, false, new BeastVersion());
            final Map<String, XMLObject> store = parser.parse(new FileReader(args[0]), false);
            final TorusLandingGpuLikelihood like = (TorusLandingGpuLikelihood) store.get("tfl").getNativeObject();
            final Tree tree = (Tree) store.get("treeModel").getNativeObject();
            final double[] b = like.getBranchLengths();
            try (PrintWriter pw = new PrintWriter(args[1])) {
                pw.println("taxon,root_to_tip");
                for (int i = 0; i < tree.getExternalNodeCount(); i++) {
                    NodeRef v = tree.getExternalNode(i);
                    double d = 0.0;
                    for (NodeRef u = v; !tree.isRoot(u); u = tree.getParent(u)) d += b[u.getNumber()];
                    pw.printf("%s,%.17g%n", tree.getNodeTaxon(v).getId(), d);
                }
            }
            final long t0 = System.nanoTime();
            final double total = like.getLogLikelihood();
            System.out.printf("total logL %.10f for %d residues in %.2f s%n", total, like.getResidueCount(),
                    (System.nanoTime() - t0) / 1e9);
            try (PrintWriter pw = new PrintWriter(args[2])) {
                pw.println("index,logL,q1,q2,rho,lambda");
                for (int r = 0; r < like.getResidueCount(); r++) {
                    final double q1 = like.getPrecision(r).getParameterValue(0, 0);
                    final double q2 = like.getPrecision(r).getParameterValue(1, 1);
                    final double rho = like.getPrecision(r).getParameterValue(0, 1) / Math.sqrt(q1 * q2);
                    pw.printf("%d,%.17g,%.17g,%.17g,%.17g,%.17g%n", r, like.getResidueLogLikelihood(r), q1, q2, rho,
                            like.getJumpRate().getParameterValue(r));
                }
            }
        } catch (Throwable t) {
            t.printStackTrace();
            System.exit(3);
        }
        System.exit(0);
    }
}
