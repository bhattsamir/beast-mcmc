package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evolution.tree.TreeTrait;
import dr.evomodel.tree.DefaultTreeModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood.JumpLaw;
import dr.evomodel.treedatalikelihood.continuous.TorusLogSpacePruning;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import dr.math.MathUtils;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.List;

/**
 * The production ancestral sampler (split downward pass over FFT partials), for both jump laws,
 * against the EXACT CONTINUOUS posterior of (root, internal child), quantised to grid cells --
 * what a perfect grid-reporting sampler would produce.
 *
 * Expected values come from ancestral_continuous_truth.py, which shares no code or representation
 * with the engine: every kernel is a fully resolved spectral sum on a 4096-point grid, the upward
 * pass is a continuous convolution, and the joint posterior is binned into the 128 cells centred
 * on the engine's grid points. Each case first asserts that the engine's log-likelihood equals the
 * truth's (they agree to 1e-15), so the two are provably the same model before their posteriors
 * are compared.
 *
 * Statistics are COARSER than a cell, because placement within a cell is below any grid
 * sampler's resolution: P(child within two cells of the root) and circular moments. In the
 * ringing cases (internal branch t*Sigma = 3e-4) the alternatives fail this test by many standard
 * errors at 80000 draws (SE ~0.0008 on the two-cell probability): a truncated FFT branch kernel
 * misses by ~0.019, unnormalised exact point values by ~0.006 (sub-cell point values carry 1.17x
 * the mass, over-weighting "no jump"); the normalised exact atom misses by 1e-4. The directed case
 * catches an orientation error in the shift jump kernel: a reflected kernel moves the child's
 * E[sin] by 0.26.
 */
public class TorusAncestralSamplerTest extends TestCase {

    private static final String NEWICK = "((A:1.0,B:1.0):1.0,C:2.0);";
    private static final int GRID = 128;
    private static final double CELL = 2.0 * Math.PI / GRID;

    // ---- exact continuous truth (ancestral_continuous_truth.py, 2026-10-01) ----
    // {logL, rootCos, rootSin, childCos, childSin, P(|delta| <= 2 cells), E[cos delta]}
    private static final double[] LANDING_RINGING = {-0.7913885310870031, 0.9085436639105311,
            0.36293737261596937, 0.9074154982728286, 0.37068476301322795, 0.9415761667496815,
            0.9935825286313522};
    private static final double[] SHIFT_RINGING = {-2.485554011707154, 0.9199814413716987,
            0.32170275735202897, 0.9177329427201057, 0.33720680275284365, 0.9525164841295041,
            0.9927972164676104};
    private static final double[] SHIFT_DIRECTED = {-4.758986827069575, 0.795223318851514,
            -0.13874062348430832, 0.818880978816988, 0.32166376395003343, 0.19317311387346206,
            0.6959370635587732};

    @Override
    protected void tearDown() {
        System.clearProperty("torus.ancestral.reference");
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(false);
    }

    public void testLandingSamplerMatchesContinuousPosteriorWhereFftKernelsRing() throws Exception {
        assertMatchesTruth(fixture(JumpLaw.LANDING, 3e-4, 0.1,
                new double[]{1.0}, new double[][]{{0.5}}, new double[][][]{{{0.2}}},
                new double[]{0.40, 0.45, 0.20}), LANDING_RINGING, 80000, 0.003);
    }

    public void testShiftSamplerMatchesContinuousPosteriorWhereFftKernelsRing() throws Exception {
        assertMatchesTruth(fixture(JumpLaw.SHIFT, 3e-4, 0.1,
                new double[]{0.6, 0.4}, new double[][]{{0.0}, {1.0}}, new double[][][]{{{0.2}}, {{0.2}}},
                new double[]{0.40, 0.45, 0.20}), SHIFT_RINGING, 80000, 0.003);
    }

    public void testShiftSamplerOrientationForDirectedJumps() throws Exception {
        assertMatchesTruth(fixture(JumpLaw.SHIFT, 0.05, 0.8,
                new double[]{0.6, 0.4}, new double[][]{{0.0}, {1.0}}, new double[][][]{{{0.2}}, {{0.2}}},
                new double[]{0.40, 1.20, -0.30}), SHIFT_DIRECTED, 20000, 0.012);
    }

    /** Kernel level: the zero-jump branch kernel is the exact wrapped normal up to ONE constant
     *  (no truncated-series ringing), carries unit mass on the grid, and equals the exact values
     *  once the kernel is resolved. */
    public void testBranchDiffusionKernelIsNormalisedExactWrappedNormal() {
        TorusFourierKernel k = new TorusFourierKernel(1, GRID);
        for (double v : new double[]{1e-5, 3e-4, 1e-3, 1e-2, 0.3}) {
            double[] g = k.branchDiffusionKernel(new double[][]{{v}}, 1.0);
            double mass = 0.0, ratio0 = Double.NaN;
            for (int i = 0; i < GRID; i++) {
                double z = 2.0 * Math.PI * i / GRID, exact = 0.0;
                for (int w = -4; w <= 4; w++) {
                    double d = z + 2.0 * Math.PI * w;
                    exact += Math.exp(-0.5 * d * d / v) / Math.sqrt(2.0 * Math.PI * v);
                }
                mass += g[i] * CELL;
                if (exact > 1e-250) {
                    double ratio = g[i] / exact;
                    if (Double.isNaN(ratio0)) ratio0 = ratio;
                    assertEquals("v=" + v + " shape, cell " + i, ratio0, ratio, 1e-10 * ratio0);
                } else {
                    assertTrue("v=" + v + " no spurious far-field mass at cell " + i, g[i] < 1e-250);
                }
                if (v >= 1e-2) assertEquals("v=" + v + " resolved value, cell " + i, exact, g[i], 1e-12 * Math.max(1.0, exact));
            }
            assertEquals("v=" + v + " unit mass", 1.0, mass, 1e-12);
        }
    }

    /** Kernel level, 2-D correlated: unit mass, symmetric, and the exact correlated wrapped normal
     *  once resolved. */
    public void testBranchDiffusionKernelTwoDimensionalCorrelated() {
        TorusFourierKernel k = new TorusFourierKernel(2, 64);
        double[][] sig = {{0.05, 0.03}, {0.03, 0.04}};
        double[] g = k.branchDiffusionKernel(sig, 1.0);
        double c = 2.0 * Math.PI / 64, det = sig[0][0] * sig[1][1] - sig[0][1] * sig[1][0], mass = 0.0;
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 64; j++) {
                double exact = 0.0;
                for (int a = -3; a <= 3; a++) {
                    for (int b = -3; b <= 3; b++) {
                        double x = i * c + 2 * Math.PI * a, y = j * c + 2 * Math.PI * b;
                        double q = (sig[1][1] * x * x - 2 * sig[0][1] * x * y + sig[0][0] * y * y) / det;
                        exact += Math.exp(-0.5 * q) / (2 * Math.PI * Math.sqrt(det));
                    }
                }
                double got = g[i * 64 + j];
                assertEquals("cell " + i + "," + j, exact, got, 1e-10 * Math.max(1.0, exact));
                mass += got * c * c;
            }
        }
        assertEquals("unit mass", 1.0, mass, 1e-12);
    }

    /** Kernel level: a directed jump of +1.0 moves the child to +1.0, not -1.0. */
    public void testShiftJumpKernelOrientation() {
        TorusFourierKernel k = new TorusFourierKernel(1, GRID);
        double[][] sigma = {{1e-3}};
        double[] psiRe = new double[GRID], psiIm = new double[GRID];
        k.computePsiComplex(sigma, 1.0, new double[]{1.0}, new double[][]{{1.0}},
                new double[][][]{{{0.01}}}, psiRe, psiIm);
        double[] d = k.shiftJumpKernel(sigma, 0.5, 1.0, psiRe, psiIm);
        int plus = (int) Math.round(1.0 / CELL) % GRID, minus = ((int) Math.round(-1.0 / CELL) + GRID) % GRID;
        assertTrue("D(+1)=" + d[plus] + " D(-1)=" + d[minus], d[plus] > 1.0e3 * d[minus]);
    }

    /** Under per-evaluation validation a positive reference exists, and draws must come from it. */
    public void testValidationModeDrawsFromThePositiveReference() throws Exception {
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(true);
        assertSameDraws(fixture(JumpLaw.SHIFT, 0.05, 0.8,
                        new double[]{0.6, 0.4}, new double[][]{{0.0}, {1.0}},
                        new double[][][]{{{0.2}}, {{0.2}}}, new double[]{0.40, 1.20, -0.30}),
                true);
    }

    /** -Dtorus.ancestral.reference=true forces the positive reference without validation. */
    public void testForcedReferenceDrawsFromThePositiveReference() throws Exception {
        System.setProperty("torus.ancestral.reference", "true");
        assertSameDraws(fixture(JumpLaw.SHIFT, 0.05, 0.8,
                        new double[]{0.6, 0.4}, new double[][]{{0.0}, {1.0}},
                        new double[][][]{{{0.2}}, {{0.2}}}, new double[]{0.40, 1.20, -0.30}),
                true);
    }

    /** Without validation or the property, draws do NOT come from the reference. */
    public void testProductionDrawsUseTheFftSampler() throws Exception {
        assertSameDraws(fixture(JumpLaw.SHIFT, 0.05, 0.8,
                        new double[]{0.6, 0.4}, new double[][]{{0.0}, {1.0}},
                        new double[][][]{{{0.2}}, {{0.2}}}, new double[]{0.40, 1.20, -0.30}),
                false);
    }

    // ------------------------------------------------------------------ helpers

    private static void assertMatchesTruth(Fixture f, double[] truth, int draws, double tolNear) {
        assertEquals("same model as the truth", truth[0], f.like.getLogLikelihood(), 1.0e-8);
        double[] s = statistics(f, 20261001L, draws);
        assertEquals("root E[cos]", truth[1], s[0], 0.01);
        assertEquals("root E[sin]", truth[2], s[1], 0.01);
        assertEquals("child E[cos]", truth[3], s[2], 0.01);
        assertEquals("child E[sin]", truth[4], s[3], 0.01);
        assertEquals("P(child within two cells of root)", truth[5], s[4], tolNear);
        assertEquals("E[cos delta]", truth[6], s[5], 0.01);
    }

    /** {rootCos, rootSin, childCos, childSin, P(|delta| <= 2 cells), E[cos delta]}, delta = child - root. */
    private static double[] statistics(Fixture f, long seed, int draws) {
        NodeRef root = f.tree.getRoot(), child = internalChild(f.tree);
        TreeTrait trait = f.like.getTreeTrait("theta_0");
        MathUtils.setSeed(seed);
        double[] s = new double[6];
        for (int i = 0; i < draws; i++) {
            f.like.forceAncestralResample();
            double r = ((double[]) trait.getTrait(f.tree, root))[0];
            double c = ((double[]) trait.getTrait(f.tree, child))[0];
            double d = Math.IEEEremainder(c - r, 2.0 * Math.PI);
            s[0] += Math.cos(r); s[1] += Math.sin(r); s[2] += Math.cos(c); s[3] += Math.sin(c);
            if (Math.abs(Math.round(d / CELL)) <= 2) s[4] += 1.0;
            s[5] += Math.cos(d);
        }
        for (int i = 0; i < s.length; i++) s[i] /= draws;
        return s;
    }

    /** Draws via the likelihood equal the positive reference's own draws iff `expectReference`.
     *  The reference is rebuilt here from the same inputs, independently of the likelihood. */
    private static void assertSameDraws(Fixture f, boolean expectReference) {
        f.like.getLogLikelihood();
        NodeRef root = f.tree.getRoot(), child = internalChild(f.tree);
        TreeTrait trait = f.like.getTreeTrait("theta_0");
        MathUtils.setSeed(777L);
        double[] viaLikelihood = new double[8];
        for (int i = 0; i < 4; i++) {
            f.like.forceAncestralResample();
            viaLikelihood[2 * i] = ((double[]) trait.getTrait(f.tree, root))[0];
            viaLikelihood[2 * i + 1] = ((double[]) trait.getTrait(f.tree, child))[0];
        }
        final Tree tree = f.tree;
        TorusLogSpacePruning.Result ref = TorusLogSpacePruning.calculate(tree,
                node -> tree.getNodeHeight(tree.getParent(node)) - tree.getNodeHeight(node),
                f.data[0], new double[][]{{f.sigma2}}, f.rate, f.w, f.d, f.s, new double[][]{{0.12}});
        MathUtils.setSeed(777L);
        double[] viaReference = new double[8];
        double[][] out = new double[tree.getNodeCount()][];
        for (int i = 0; i < 4; i++) {
            ref.sample(out);
            viaReference[2 * i] = out[root.getNumber()][0];
            viaReference[2 * i + 1] = out[child.getNumber()][0];
        }
        assertEquals("draws come from the positive reference", expectReference,
                java.util.Arrays.equals(viaLikelihood, viaReference));
    }

    private static NodeRef internalChild(Tree tree) {
        NodeRef root = tree.getRoot();
        for (int c = 0; c < tree.getChildCount(root); c++) {
            if (!tree.isExternal(tree.getChild(root, c))) return tree.getChild(root, c);
        }
        throw new IllegalStateException("no internal child");
    }

    /** Inputs and the likelihood built from them, so tests can rebuild the reference themselves. */
    private static final class Fixture {
        Tree tree; TorusFourierTreeLikelihood like; double[][][] data;
        double sigma2, rate; double[] w; double[][] d; double[][][] s;
    }

    private static Fixture fixture(JumpLaw law, double sigma2, double rate, double[] w, double[][] d,
                                   double[][][] s, double[] tips) throws Exception {
        Fixture f = new Fixture();
        f.tree = new DefaultTreeModel("tree", new NewickImporter(NEWICK).importTree(null));
        f.data = new double[1][f.tree.getExternalNodeCount()][];
        for (int e = 0; e < f.tree.getExternalNodeCount(); e++) {
            NodeRef node = f.tree.getExternalNode(e);
            f.data[0][node.getNumber()] = new double[]{tips["ABC".indexOf(f.tree.getNodeTaxon(node).getId())]};
        }
        List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        precisions.add(new MatrixParameter("precision0",
                new Parameter[]{new Parameter.Default(new double[]{1.0 / sigma2})}));
        f.like = new TorusFourierTreeLikelihood("sampler", f.tree, null, precisions, f.data,
                new TorusFourierKernel(1, GRID), new Parameter.Default(rate), w, d, s,
                new Parameter.Default(0.12), law);
        f.like.setTraitNames(new String[]{"theta_0"});
        f.sigma2 = sigma2; f.rate = rate; f.w = w; f.d = d; f.s = s;
        return f;
    }
}
