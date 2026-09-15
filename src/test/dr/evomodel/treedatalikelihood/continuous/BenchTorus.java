package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.tree.DefaultTreeModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;

import java.util.ArrayList;
import java.util.List;

/**
 * Microbenchmark isolating the per-evaluation likelihood cost that caching changes:
 *   full     = recompute every residue's whole tree (makeDirty; what the no-cache build does every move)
 *   incr     = after a single residue's precision move (what the cached build does on a precision move)
 * Reports per-eval time and the ratio, for a given (nGrid, nResidues, nTips).
 *
 * Usage: java ... BenchTorus [nGrid] [nResidues] [nTips]
 */
public class BenchTorus {

    public static void main(String[] args) throws Exception {
        int nGrid = args.length > 0 ? Integer.parseInt(args[0]) : 128;
        int nResidues = args.length > 1 ? Integer.parseInt(args[1]) : 100;
        int nTips = args.length > 2 ? Integer.parseInt(args[2]) : 32;

        String newick = balancedNewick(nTips);
        DefaultTreeModel tree = new DefaultTreeModel("t", new NewickImporter(newick).importTree(null));

        // one CompoundSymmetricMatrix precision per residue + random tip angles
        java.util.Random rng = new java.util.Random(42);
        List<MatrixParameter> precs = new ArrayList<MatrixParameter>();
        List<Parameter> diags = new ArrayList<Parameter>();
        for (int r = 0; r < nResidues; r++) {
            Parameter diag = new Parameter.Default(new double[]{0.5 + rng.nextDouble(), 0.5 + rng.nextDouble()});
            Parameter off = new Parameter.Default(new double[]{0.4 * (rng.nextDouble() - 0.5)});
            precs.add(new CompoundSymmetricMatrix(diag, off, true, false));
            diags.add(diag);
        }
        int nExt = tree.getExternalNodeCount();
        double[][][] data = new double[nResidues][nExt][];
        for (int r = 0; r < nResidues; r++)
            for (int e = 0; e < nExt; e++)
                data[r][e] = new double[]{Math.PI * (2 * rng.nextDouble() - 1), Math.PI * (2 * rng.nextDouble() - 1)};

        TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood(
                "bench", tree, null, precs, data, new TorusFourierKernel(2, nGrid));

        System.out.printf("nGrid=%d nResidues=%d nTips=%d nodes=%d%n",
                nGrid, nResidues, nTips, tree.getNodeCount());

        // warm up JIT
        for (int i = 0; i < 3; i++) { like.makeDirty(); like.getLogLikelihood(); }

        // (a) full recompute per eval
        int N = 12;
        like.resetRecomputeCount();
        long t0 = System.nanoTime();
        for (int i = 0; i < N; i++) { like.makeDirty(); like.getLogLikelihood(); }
        double fullMs = (System.nanoTime() - t0) / 1e6 / N;
        double fullNodes = like.getNodeRecomputeCount() / (double) N;

        // (b) incremental: a single residue's precision move, then eval
        like.getLogLikelihood();
        int N2 = 200;
        like.resetRecomputeCount();
        long t1 = System.nanoTime();
        for (int i = 0; i < N2; i++) {
            Parameter diag = diags.get(i % nResidues);
            diag.setParameterValue(0, 0.5 + (i % 7) * 0.1);   // perturb one residue
            like.getLogLikelihood();
        }
        double incrMs = (System.nanoTime() - t1) / 1e6 / N2;
        double incrNodes = like.getNodeRecomputeCount() / (double) N2;

        // (c) tree move: nudge one internal node's height -> dirty path to root, in every residue
        NodeRef mover = null;
        for (int i = 0; i < tree.getNodeCount(); i++) {
            NodeRef nd = tree.getNode(i);
            if (!tree.isExternal(nd) && !tree.isRoot(nd)) { mover = nd; break; }
        }
        like.getLogLikelihood();
        like.resetRecomputeCount();
        long t2 = System.nanoTime();
        for (int i = 0; i < N2; i++) {
            tree.setNodeHeight(mover, tree.getNodeHeight(mover));   // fires a node event (dirty path)
            like.getLogLikelihood();
        }
        double treeMs = (System.nanoTime() - t2) / 1e6 / N2;
        double treeNodes = like.getNodeRecomputeCount() / (double) N2;

        System.out.printf("full       : %7.2f ms/eval, %6.0f node-recomputes%n", fullMs, fullNodes);
        System.out.printf("precision  : %7.2f ms/eval, %6.0f node-recomputes  (one residue, all its nodes)%n", incrMs, incrNodes);
        System.out.printf("tree move  : %7.2f ms/eval, %6.0f node-recomputes  (dirty path x all residues)%n", treeMs, treeNodes);
        System.out.printf("=> CPU-work speedup full/precision: %.0fx ; full/tree: %.0fx%n", fullNodes/incrNodes, fullNodes/Math.max(1,treeNodes));
        System.out.printf("=> wall speedup full/precision: %.1fx ; full/tree: %.1fx%n", fullMs/incrMs, fullMs/treeMs);
    }

    /** A balanced bifurcating newick with n tips (n rounded down to a power of two), unit-ish branches. */
    private static String balancedNewick(int n) {
        int p = 1; while (p * 2 <= n) p *= 2;   // largest power of two <= n
        return build(0, p, 1.0)[0] + ";";
    }
    // returns {newickSubtree} for `count` leaves starting at id `start`, each branch length `bl`
    private static String[] build(int start, int count, double bl) {
        if (count == 1) return new String[]{"T" + start + ":" + bl};
        String left = build(start, count / 2, bl / 1.3)[0];
        String right = build(start + count / 2, count / 2, bl / 1.3)[0];
        return new String[]{"(" + left + "," + right + "):" + bl};
    }
}
