package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.TreeTrait;
import dr.evomodel.branchratemodel.ArbitraryBranchRates;
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
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Incremental evaluation against full recomputation, through random sequences of proposals
 * made by BEAST's own tree operators and by direct parameter changes, with store, accept and
 * restore in between, as an MCMC chain would call them.
 *
 * Each step stores the model state, makes one proposal (two in the double-evaluation tests),
 * evaluates the likelihood under test, and compares it with an independent instance on the same
 * tree and parameters that is made dirty before every evaluation, so it always recomputes
 * everything. The step then accepts or restores, and the restored value is compared again.
 * Every few steps the ancestral draws of both instances are compared under the same seed, which
 * checks the cached partials themselves and not only the total they sum to.
 *
 * Agreement must be exact (identical bits). Both instances run the same code on the same
 * inputs, so any difference means the cache served a stale or overwritten partial.
 */
public class TorusCacheFuzzTest extends TestCase {

    private static final int NODE_HEIGHT = 0, TIP_HEIGHT = 1, SLIDE = 2, NARROW = 3, WIDE = 4,
            WILSON_BALDING = 5, PRECISION_DIAGONAL = 6, PRECISION_CORRELATION = 7, JUMP_RATE = 8,
            TIP_ERROR = 9, ONE_BRANCH_RATE = 10, ALL_BRANCH_RATES = 11;
    private static final String[] MOVE_NAMES = {"node height", "tip height", "subtree slide",
            "narrow exchange", "wide exchange", "Wilson-Balding", "precision diagonal",
            "precision correlation", "jump rate", "tip error", "one branch rate", "all branch rates"};

    /** One configuration of the fuzz. */
    private static final class Config {
        JumpLaw law = JumpLaw.LANDING;
        boolean jumps = true;
        boolean tipError = true;
        boolean perResidueRate = true;
        boolean branchRates = false;
        boolean heterochronous = true;
        int tips = 10, residues = 3, nGrid = 32, steps = 120, evaluationsPerProposal = 1;
        long seed = 1L;
    }

    /** The tree, parameters and data shared by the instance under test and the reference. */
    private static final class World {
        final Config config;
        final DefaultTreeModel tree;
        final List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        final List<Parameter> diagonals = new ArrayList<Parameter>();
        final List<Parameter> correlations = new ArrayList<Parameter>();
        final double[][][] data;
        final Parameter jumpRate;
        final Parameter tipError;
        final Parameter branchRateValues;
        final ArbitraryBranchRates branchRates;
        final double[] w;
        final double[][] d;
        final double[][][] s;
        final SubtreeSlideOperator slide;
        final ExchangeOperator narrow, wide;
        final WilsonBalding wilsonBalding;

        World(Config config) throws Exception {
            this.config = config;
            final Random rng = new Random(config.seed);
            tree = new DefaultTreeModel("fuzzTree",
                    new NewickImporter(randomNewick(config.tips, rng, config.heterochronous)).importTree(null));
            for (int r = 0; r < config.residues; r++) {
                final Parameter diagonal = new Parameter.Default(new double[]{0.5 + 1.5 * rng.nextDouble(),
                        0.5 + 1.5 * rng.nextDouble()});
                final Parameter correlation = new Parameter.Default(new double[]{1.2 * (rng.nextDouble() - 0.5)});
                precisions.add(new CompoundSymmetricMatrix(diagonal, correlation, true, false));
                diagonals.add(diagonal);
                correlations.add(correlation);
            }
            final int external = tree.getExternalNodeCount();
            data = new double[config.residues][external][];
            for (int r = 0; r < config.residues; r++) {
                for (int e = 0; e < external; e++) {
                    // a few masked observations, never at the first two taxa
                    data[r][e] = e >= 2 && rng.nextInt(12) == 0 ? null
                            : new double[]{Math.PI * (2.0 * rng.nextDouble() - 1.0),
                                           Math.PI * (2.0 * rng.nextDouble() - 1.0)};
                }
            }
            if (config.law == JumpLaw.LANDING) {
                w = new double[]{0.4, 0.3, 0.2, 0.1};
            } else {
                w = new double[]{0.5, 0.3, 0.2};
            }
            d = new double[w.length][2];
            s = new double[w.length][2][2];
            for (int k = 0; k < w.length; k++) {
                d[k][0] = Math.PI * (2.0 * rng.nextDouble() - 1.0);
                d[k][1] = Math.PI * (2.0 * rng.nextDouble() - 1.0);
                final double a = 0.05 + 0.3 * rng.nextDouble(), b = 0.05 + 0.3 * rng.nextDouble();
                final double c = 0.8 * Math.sqrt(a * b) * (2.0 * rng.nextDouble() - 1.0);
                s[k][0][0] = a;
                s[k][1][1] = b;
                s[k][0][1] = c;
                s[k][1][0] = c;
            }
            if (config.jumps) {
                final double[] rates = new double[config.perResidueRate ? config.residues : 1];
                for (int i = 0; i < rates.length; i++) rates[i] = 0.1 + 1.5 * rng.nextDouble();
                jumpRate = new Parameter.Default(rates);
            } else {
                jumpRate = null;
            }
            tipError = config.tipError ? new Parameter.Default(0.03 + 0.03 * rng.nextDouble()) : null;
            if (config.branchRates) {
                final double[] values = new double[tree.getNodeCount() - 1];
                for (int i = 0; i < values.length; i++) values[i] = 0.5 + rng.nextDouble();
                branchRateValues = new Parameter.Default(values);
                branchRates = new ArbitraryBranchRates(tree, branchRateValues,
                        new ArbitraryBranchRates.BranchRateTransform.None(), false);
            } else {
                branchRateValues = null;
                branchRates = null;
            }
            slide = new SubtreeSlideOperator(tree, 1.0, 0.1, true, false, false, false,
                    AdaptationMode.ADAPTATION_OFF, 0.234);
            narrow = new ExchangeOperator(ExchangeOperator.NARROW, tree, 1.0);
            wide = new ExchangeOperator(ExchangeOperator.WIDE, tree, 1.0);
            wilsonBalding = new WilsonBalding(tree, 1.0);
        }

        TorusFourierTreeLikelihood likelihood(String name) {
            return new TorusFourierTreeLikelihood(name, tree, branchRates, precisions, data,
                    new TorusFourierKernel(2, config.nGrid),
                    config.jumps ? jumpRate : null,
                    config.jumps ? w : null, config.jumps ? d : null, config.jumps ? s : null,
                    tipError, config.law);
        }

        int[] moves() {
            final List<Integer> moves = new ArrayList<Integer>();
            moves.add(NODE_HEIGHT);
            if (config.heterochronous) moves.add(TIP_HEIGHT);
            moves.add(SLIDE);
            moves.add(NARROW);
            moves.add(WIDE);
            moves.add(WILSON_BALDING);
            moves.add(PRECISION_DIAGONAL);
            moves.add(PRECISION_CORRELATION);
            if (config.jumps) moves.add(JUMP_RATE);
            if (config.tipError) moves.add(TIP_ERROR);
            if (config.branchRates) {
                moves.add(ONE_BRANCH_RATE);
                moves.add(ONE_BRANCH_RATE);
                moves.add(ALL_BRANCH_RATES);
            }
            final int[] out = new int[moves.size()];
            for (int i = 0; i < out.length; i++) out[i] = moves.get(i);
            return out;
        }

        /** Make one proposal. Returns false if an operator declined (the chain would reject). */
        boolean propose(int move, Random rng) {
            switch (move) {
                case NODE_HEIGHT: {
                    NodeRef v;
                    do {
                        v = tree.getNode(rng.nextInt(tree.getNodeCount()));
                    } while (tree.isExternal(v));
                    double low = 0.0;
                    for (int c = 0; c < tree.getChildCount(v); c++) {
                        low = Math.max(low, tree.getNodeHeight(tree.getChild(v, c)));
                    }
                    final double high = tree.isRoot(v) ? low + 1.5 * (tree.getNodeHeight(v) - low)
                            : tree.getNodeHeight(tree.getParent(v));
                    tree.setNodeHeight(v, low + (high - low) * (0.05 + 0.9 * rng.nextDouble()));
                    return true;
                }
                case TIP_HEIGHT: {
                    final NodeRef tip = tree.getExternalNode(rng.nextInt(tree.getExternalNodeCount()));
                    final double high = tree.getNodeHeight(tree.getParent(tip));
                    tree.setNodeHeight(tip, high * 0.9 * rng.nextDouble());
                    return true;
                }
                case SLIDE: return slide.doOperation() != Double.NEGATIVE_INFINITY;
                case NARROW: return narrow.doOperation() != Double.NEGATIVE_INFINITY;
                case WIDE: return wide.doOperation() != Double.NEGATIVE_INFINITY;
                case WILSON_BALDING: return wilsonBalding.doOperation() != Double.NEGATIVE_INFINITY;
                case PRECISION_DIAGONAL: {
                    final Parameter p = diagonals.get(rng.nextInt(diagonals.size()));
                    final int i = rng.nextInt(2);
                    p.setParameterValue(i, p.getParameterValue(i) * Math.exp(0.6 * (rng.nextDouble() - 0.5)));
                    return true;
                }
                case PRECISION_CORRELATION: {
                    final Parameter p = correlations.get(rng.nextInt(correlations.size()));
                    p.setParameterValue(0, 1.4 * (rng.nextDouble() - 0.5));
                    return true;
                }
                case JUMP_RATE: {
                    final int i = rng.nextInt(jumpRate.getDimension());
                    jumpRate.setParameterValue(i, jumpRate.getParameterValue(i) * Math.exp(rng.nextDouble() - 0.5));
                    return true;
                }
                case TIP_ERROR:
                    tipError.setParameterValue(0, 0.03 + 0.03 * rng.nextDouble());
                    return true;
                case ONE_BRANCH_RATE: {
                    final int i = rng.nextInt(branchRateValues.getDimension());
                    branchRateValues.setParameterValue(i, 0.5 + rng.nextDouble());
                    return true;
                }
                case ALL_BRANCH_RATES: {
                    final int i = rng.nextInt(branchRateValues.getDimension());
                    branchRateValues.setParameterValueNotifyChangedAll(i, 0.5 + rng.nextDouble());
                    return true;
                }
                default:
                    throw new IllegalArgumentException("move " + move);
            }
        }
    }

    /** Random rooted binary tree, merging random pairs; tips at random heights when heterochronous. */
    static String randomNewick(int tips, Random rng, boolean heterochronous) {
        final List<String> labels = new ArrayList<String>();
        final List<Double> heights = new ArrayList<Double>();
        for (int i = 0; i < tips; i++) {
            labels.add("t" + i);
            heights.add(heterochronous && i > 0 ? 0.4 * rng.nextDouble() : 0.0);
        }
        while (labels.size() > 1) {
            final int a = rng.nextInt(labels.size());
            int b;
            do {
                b = rng.nextInt(labels.size());
            } while (b == a);
            final double h = Math.max(heights.get(a), heights.get(b)) + 0.05 + 0.4 * rng.nextDouble();
            final String merged = "(" + labels.get(a) + ":" + (h - heights.get(a)) + ","
                    + labels.get(b) + ":" + (h - heights.get(b)) + ")";
            final int high = Math.max(a, b), low = Math.min(a, b);
            labels.remove(high);
            heights.remove(high);
            labels.remove(low);
            heights.remove(low);
            labels.add(merged);
            heights.add(h);
        }
        return labels.get(0) + ";";
    }

    /** Outcome of a fuzz run. */
    private static final class Tally {
        int checks, mismatches, ancestralChecks, ancestralMismatches, declined, accepted, restored;
        String first;

        void record(boolean equal, String what) {
            checks++;
            if (!equal) {
                mismatches++;
                if (first == null) first = what;
            }
        }

        public String toString() {
            return checks + " likelihood checks (" + mismatches + " mismatched), " + ancestralChecks
                    + " ancestral checks (" + ancestralMismatches + " mismatched), " + accepted + " accepted, "
                    + restored + " restored, " + declined + " declined" + (first == null ? "" : "; first: " + first);
        }
    }

    private static double full(TorusFourierTreeLikelihood reference) {
        reference.makeDirty();
        return reference.getLogLikelihood();
    }

    private static boolean same(double a, double b) {
        return Double.doubleToLongBits(a) == Double.doubleToLongBits(b);
    }

    /** Draw ancestral states from both instances under the same seed and compare every angle. */
    private static boolean sameAncestors(World world, TorusFourierTreeLikelihood under,
                                         TorusFourierTreeLikelihood reference, long seed) {
        final List<double[]> a = draw(world, under, seed, false);
        final List<double[]> b = draw(world, reference, seed, true);
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) {
            if (a.get(i).length != b.get(i).length) return false;
            for (int j = 0; j < a.get(i).length; j++) {
                if (!same(a.get(i)[j], b.get(i)[j])) return false;
            }
        }
        return true;
    }

    private static List<double[]> draw(World world, TorusFourierTreeLikelihood like, long seed, boolean dirty) {
        if (dirty) like.makeDirty();
        like.getLogLikelihood();
        MathUtils.setSeed(seed);
        final List<double[]> out = new ArrayList<double[]>();
        try {
            like.forceAncestralResample();
        } catch (RuntimeException e) {
            out.add(new double[]{e.getClass().getName().hashCode(), String.valueOf(e.getMessage()).hashCode()});
            return out;
        }
        for (int r = 0; r < world.config.residues; r++) {
            final TreeTrait trait = like.getTreeTrait("theta_" + r);
            for (int i = 0; i < world.tree.getNodeCount(); i++) {
                final double[] v = (double[]) trait.getTrait(world.tree, world.tree.getNode(i));
                if (v != null) out.add(v.clone());
            }
        }
        return out;
    }

    private static Tally fuzz(Config config) throws Exception {
        final World world = new World(config);
        MathUtils.setSeed(1000L + config.seed);
        final Random rng = new Random(2000L + config.seed);
        final TorusFourierTreeLikelihood under = world.likelihood("underTest");
        final TorusFourierTreeLikelihood reference = world.likelihood("reference");
        final int[] moves = world.moves();
        final Tally tally = new Tally();
        tally.record(same(under.getLogLikelihood(), full(reference)), "initial evaluation");
        for (int step = 0; step < config.steps; step++) {
            under.storeModelState();                     // cascades to the tree and the parameters
            boolean ok = true;
            final boolean evaluate = rng.nextInt(8) != 0;   // sometimes restore without evaluating
            String last = "";
            for (int e = 0; e < config.evaluationsPerProposal && ok; e++) {
                final int move = moves[rng.nextInt(moves.length)];
                last = MOVE_NAMES[move];
                ok = world.propose(move, rng);
                if (ok && evaluate) {
                    final double got = under.getLogLikelihood();
                    final double want = full(reference);
                    tally.record(same(got, want), "step " + step + " after " + last + " (evaluation " + (e + 1)
                            + "): incremental " + got + ", full " + want);
                }
            }
            if (!ok) tally.declined++;
            if (ok && evaluate && rng.nextInt(3) == 0) {
                under.acceptModelState();
                tally.accepted++;
            } else {
                under.restoreModelState();
                tally.restored++;
            }
            final double got = under.getLogLikelihood();
            final double want = full(reference);
            tally.record(same(got, want), "step " + step + " after restoring " + last
                    + ": incremental " + got + ", full " + want);
            if (step % 10 == 9) {
                tally.ancestralChecks++;
                if (!sameAncestors(world, under, reference, 5000L + step)) {
                    tally.ancestralMismatches++;
                    if (tally.first == null) tally.first = "ancestral draws differ at step " + step;
                }
            }
        }
        return tally;
    }

    private static void check(String label, Config config) throws Exception {
        final Tally tally = fuzz(config);
        System.out.println("  " + label + ": " + tally);
        assertEquals(label + ": " + tally, 0, tally.mismatches + tally.ancestralMismatches);
        assertTrue(label + ": too few checks", tally.checks > config.steps);
    }

    public void testLandingPerResidueRates() throws Exception {
        final Config c = new Config();
        c.seed = 11L;
        check("landing, per-residue rates", c);
    }

    public void testLandingParallelResiduesWithBranchRates() throws Exception {
        final Config c = new Config();
        c.residues = 9;               // at and above 8 the residue loop runs in parallel
        c.perResidueRate = false;
        c.branchRates = true;
        c.steps = 80;
        c.seed = 12L;
        check("landing, 9 residues in parallel, shared rate, branch rates", c);
    }

    public void testShiftWithJumpsAndTipError() throws Exception {
        final Config c = new Config();
        c.law = JumpLaw.SHIFT;
        c.branchRates = true;
        c.seed = 13L;
        check("shift, jumps, tip error, branch rates", c);
    }

    public void testShiftBrownianDiracTips() throws Exception {
        final Config c = new Config();
        c.law = JumpLaw.SHIFT;
        c.jumps = false;
        c.tipError = false;
        c.seed = 14L;
        check("shift, Brownian, Dirac tips", c);
    }

    /** A second evaluation between store and restore must not overwrite the stored partials. */
    public void testLandingTwoEvaluationsPerProposal() throws Exception {
        final Config c = new Config();
        c.evaluationsPerProposal = 2;
        c.seed = 15L;
        check("landing, two evaluations per proposal", c);
    }

    public void testShiftTwoEvaluationsPerProposal() throws Exception {
        final Config c = new Config();
        c.law = JumpLaw.SHIFT;
        c.evaluationsPerProposal = 2;
        c.branchRates = true;
        c.seed = 16L;
        check("shift, two evaluations per proposal, branch rates", c);
    }
}
