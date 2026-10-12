package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evomodel.operators.ExchangeOperator;
import dr.evomodel.operators.SubtreeSlideOperator;
import dr.evomodel.operators.WilsonBalding;
import dr.evomodel.tree.DefaultTreeModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood.JumpLaw;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import dr.inference.operators.AdaptationMode;
import dr.math.MathUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Wall time per evaluation on a random tree, by proposal type, through a fixed seeded sequence of
 * BEAST tree operators and parameter moves with store, evaluate, accept or restore (by default the
 * problem size of Marc Suchard's plan: 100 residues, 32 tips, nGrid 64). It uses only the engine's
 * public API, so the same harness runs on the paper engine (838e4ebfb, with TorusCacheFuzzTest.java
 * copied in) and on this branch. Also prints a digest of every log-likelihood, so two engines can be
 * checked for identical values on the same sequence. Set the thread count with
 * -Djava.util.concurrent.ForkJoinPool.common.parallelism.
 *
 * Usage: RandomTreeBench shift|landing [nGrid=64] [residues=100] [tips=32] [proposals=420]
 */
public class RandomTreeBench {

    static final String[] MOVES = {"node height", "subtree slide", "narrow exchange", "wide exchange",
            "Wilson-Balding", "one residue's precision", "full recompute"};

    public static void main(String[] args) throws Exception {
        final JumpLaw law = args[0].equals("landing") ? JumpLaw.LANDING : JumpLaw.SHIFT;
        final int nGrid = args.length > 1 ? Integer.parseInt(args[1]) : 64;
        final int residues = args.length > 2 ? Integer.parseInt(args[2]) : 100;
        final int tips = args.length > 3 ? Integer.parseInt(args[3]) : 32;
        final int proposals = args.length > 4 ? Integer.parseInt(args[4]) : 420;
        java.util.logging.Logger.getLogger("dr.evomodel").setLevel(java.util.logging.Level.OFF);
        final Random rng = new Random(17);
        final DefaultTreeModel tree = new DefaultTreeModel("benchTree",
                new NewickImporter(TorusCacheFuzzTest.randomNewick(tips, rng, true)).importTree(null));
        final List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        final List<Parameter> diagonals = new ArrayList<Parameter>();
        for (int r = 0; r < residues; r++) {
            final Parameter diagonal = new Parameter.Default(new double[]{0.5 + rng.nextDouble(), 0.5 + rng.nextDouble()});
            final Parameter correlation = new Parameter.Default(new double[]{0.4 * (rng.nextDouble() - 0.5)});
            precisions.add(new CompoundSymmetricMatrix(diagonal, correlation, true, false));
            diagonals.add(diagonal);
        }
        final int external = tree.getExternalNodeCount();
        final double[][][] data = new double[residues][external][];
        for (int r = 0; r < residues; r++) {
            for (int e = 0; e < external; e++) {
                data[r][e] = rng.nextInt(25) == 0 ? null
                        : new double[]{Math.PI * (2 * rng.nextDouble() - 1), Math.PI * (2 * rng.nextDouble() - 1)};
            }
        }
        final int k = 13;
        final double[] w = new double[k];
        final double[][] d = new double[k][2];
        final double[][][] s = new double[k][2][2];
        for (int i = 0; i < k; i++) {
            w[i] = 1.0 / k;
            d[i][0] = Math.PI * (2 * rng.nextDouble() - 1);
            d[i][1] = Math.PI * (2 * rng.nextDouble() - 1);
            final double a = 0.05 + 0.3 * rng.nextDouble(), b = 0.05 + 0.3 * rng.nextDouble();
            final double c = 0.5 * Math.sqrt(a * b) * (2 * rng.nextDouble() - 1);
            s[i][0][0] = a;
            s[i][1][1] = b;
            s[i][0][1] = c;
            s[i][1][0] = c;
        }
        w[k - 1] = 1.0;
        for (int i = 0; i < k - 1; i++) w[k - 1] -= w[i];
        final Parameter jumpRate = new Parameter.Default(0.5);
        final Parameter tipError = new Parameter.Default(0.02);
        final TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood("bench", tree, null, precisions, data,
                new TorusFourierKernel(2, nGrid), jumpRate, w, d, s, law == JumpLaw.LANDING ? tipError : null, law);
        final SubtreeSlideOperator slide = new SubtreeSlideOperator(tree, 1.0, 0.05, true, false, false, false,
                AdaptationMode.ADAPTATION_OFF, 0.234);
        final ExchangeOperator narrow = new ExchangeOperator(ExchangeOperator.NARROW, tree, 1.0);
        final ExchangeOperator wide = new ExchangeOperator(ExchangeOperator.WIDE, tree, 1.0);
        final WilsonBalding wb = new WilsonBalding(tree, 1.0);
        MathUtils.setSeed(4242);
        final Random moves = new Random(99);
        for (int i = 0; i < 3; i++) {                       // warm-up: full recomputes
            like.makeDirty();
            like.getLogLikelihood();
        }
        final List<List<Double>> times = new ArrayList<List<Double>>();
        for (int i = 0; i < MOVES.length; i++) times.add(new ArrayList<Double>());
        long digest = 1469598103934665603L;
        for (int step = 0; step < proposals; step++) {
            final int move = step % MOVES.length;
            like.storeModelState();
            boolean ok = true;
            switch (move) {
                case 0: {
                    NodeRef v;
                    do {
                        v = tree.getNode(moves.nextInt(tree.getNodeCount()));
                    } while (tree.isExternal(v));
                    double low = 0.0;
                    for (int c = 0; c < tree.getChildCount(v); c++) low = Math.max(low, tree.getNodeHeight(tree.getChild(v, c)));
                    final double high = tree.isRoot(v) ? low + 1.2 * (tree.getNodeHeight(v) - low) : tree.getNodeHeight(tree.getParent(v));
                    tree.setNodeHeight(v, low + (high - low) * (0.05 + 0.9 * moves.nextDouble()));
                    break;
                }
                case 1: ok = slide.doOperation() != Double.NEGATIVE_INFINITY; break;
                case 2: ok = narrow.doOperation() != Double.NEGATIVE_INFINITY; break;
                case 3: ok = wide.doOperation() != Double.NEGATIVE_INFINITY; break;
                case 4: ok = wb.doOperation() != Double.NEGATIVE_INFINITY; break;
                case 5: {
                    final Parameter p = diagonals.get(moves.nextInt(residues));
                    p.setParameterValue(0, p.getParameterValue(0) * Math.exp(0.3 * (moves.nextDouble() - 0.5)));
                    break;
                }
                default:
                    like.makeDirty();
            }
            if (ok) {
                final long t0 = System.nanoTime();
                final double v = like.getLogLikelihood();
                times.get(move).add((System.nanoTime() - t0) * 1e-6);
                digest = (digest ^ Double.doubleToRawLongBits(v)) * 1099511628211L;
            }
            if (ok && moves.nextInt(3) == 0) like.acceptModelState();
            else like.restoreModelState();
            digest = (digest ^ Double.doubleToRawLongBits(like.getLogLikelihood())) * 1099511628211L;
        }
        System.out.printf("%s, nGrid %d, %d residues, %d tips, %d proposals, common pool parallelism %d; "
                        + "log-likelihood digest %016x%n", law, nGrid, residues, tips, proposals,
                java.util.concurrent.ForkJoinPool.commonPool().getParallelism(), digest);
        for (int i = 0; i < MOVES.length; i++) {
            final double[] t = new double[times.get(i).size()];
            for (int j = 0; j < t.length; j++) t[j] = times.get(i).get(j);
            Arrays.sort(t);
            double mean = 0.0;
            for (double x : t) mean += x;
            mean /= Math.max(1, t.length);
            System.out.printf("  %-26s n=%3d  median %8.1f ms  mean %8.1f ms%n", MOVES[i], t.length,
                    t.length == 0 ? Double.NaN : t[t.length / 2], mean);
        }
    }
}
