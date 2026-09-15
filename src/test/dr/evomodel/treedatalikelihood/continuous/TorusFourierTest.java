/*
 * TorusFourierTest.java
 *
 * @author Samir Bhatt
 */

package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.tree.DefaultTreeModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusLogSpacePruning;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Validates the FFT torus pruning against reference values from the standalone
 * Python prototype (torus_fourier_pruning.py): the kernel reproduces the analytic
 * two-tip likelihood, and the full tree likelihood matches the prototype's
 * pruning at p = 1, 2, 3.
 *
 * @author Samir Bhatt
 */
public class TorusFourierTest extends TestCase {

    private static final String NEWICK = "((A:1.0,B:1.0):0.5,C:1.5);";

    /** Kernel only: analytic two-tip likelihood, grid-independent. */
    public void testKernelTwoTip() {
        TorusFourierKernel g = new TorusFourierKernel(2, 128);
        double[][] sigma = {{0.35, 0.08}, {0.08, 0.22}};   // covariance
        double[] mA = g.leafKernel(new double[]{0.4, 1.1}, sigma, 0.7);
        double[] mB = g.leafKernel(new double[]{-1.2, 0.6}, sigma, 1.4);
        int n = g.getGridCount();
        double[] v = new double[n];
        Arrays.fill(v, 1.0);
        double logScale = 0.0;
        logScale += multiplyInto(v, mA);
        logScale += multiplyInto(v, mB);
        double ll = logScale + Math.log(g.rootIntegral(v));
        assertEquals(-6.69298747149, ll, 1.0e-6);
    }

    /** Kernel with compound-Poisson jumps: two-tip likelihood vs the prototype. */
    public void testKernelTwoTipJumps() {
        TorusFourierKernel g = new TorusFourierKernel(2, 128);
        double[][] sigma = {{0.35, 0.08}, {0.08, 0.22}};
        double[][] sigmaJump = {{0.6, 0.0}, {0.0, 0.5}};
        double rate = 0.8;
        double[] mA = g.leafKernel(new double[]{0.4, 1.1}, sigma, 0.7, rate, sigmaJump);
        double[] mB = g.leafKernel(new double[]{-1.2, 0.6}, sigma, 1.4, rate, sigmaJump);
        int n = g.getGridCount();
        double[] v = new double[n];
        Arrays.fill(v, 1.0);
        double logScale = 0.0;
        logScale += multiplyInto(v, mA);
        logScale += multiplyInto(v, mB);
        double ll = logScale + Math.log(g.rootIntegral(v));
        assertEquals(-6.72524483142, ll, 1.0e-6);
    }

    /** Kernel with a DIRECTED jump mixture (nonzero displacements d_k -> complex multiplier):
     *  two-tip likelihood vs the Python prototype (torus_fourier_pruning, jump_mix). Covers the
     *  complex transferMultiplier / leafKernel path. */
    public void testKernelTwoTipDirectedJump() {
        TorusFourierKernel g = new TorusFourierKernel(2, 128);
        double[][] sigma = {{0.35, 0.08}, {0.08, 0.22}};
        double rate = 0.8;
        double[] w = {0.6, 0.4};
        double[][] d = {{0.0, 0.0}, {1.0, -0.5}};                 // 2nd component is directed
        double[][][] s = {{{0.6, 0.0}, {0.0, 0.5}}, {{0.3, 0.0}, {0.0, 0.3}}};
        double[] mA = g.leafKernel(new double[]{0.4, 1.1}, sigma, 0.7, rate, w, d, s);
        double[] mB = g.leafKernel(new double[]{-1.2, 0.6}, sigma, 1.4, rate, w, d, s);
        int n = g.getGridCount();
        double[] v = new double[n];
        Arrays.fill(v, 1.0);
        double logScale = 0.0;
        logScale += multiplyInto(v, mA);
        logScale += multiplyInto(v, mB);
        double ll = logScale + Math.log(g.rootIntegral(v));
        assertEquals(-6.86537128899406, ll, 1.0e-6);
    }

    /** Full tree likelihood with a DIRECTED jump mixture, via the constructor: 3-tip p=2 vs the
     *  Python prototype. Covers the complex propagate and the likelihood's jump wiring. */
    public void testLikelihoodDirectedJump() throws Exception {
        Tree tree = new NewickImporter(NEWICK).importTree(null);
        int nExt = tree.getExternalNodeCount();
        Map<String, double[]> obs = new HashMap<String, double[]>();
        obs.put("A", new double[]{0.4, 1.1});
        obs.put("B", new double[]{-1.2, 0.6});
        obs.put("C", new double[]{0.9, -0.3});
        double[][] data = new double[nExt][];
        for (int e = 0; e < nExt; e++) {
            NodeRef ext = tree.getExternalNode(e);
            data[ext.getNumber()] = obs.get(tree.getNodeTaxon(ext).getId());
        }
        List<MatrixParameter> precs = new ArrayList<MatrixParameter>();
        precs.add(precisionParameter(new double[][]{{1.0, 0.1}, {0.1, 2.0}}));
        Parameter rate = new Parameter.Default(0.8);
        double[] w = {0.6, 0.4};
        double[][] d = {{0.0, 0.0}, {1.0, -0.5}};
        double[][][] s = {{{0.6, 0.0}, {0.0, 0.5}}, {{0.3, 0.0}, {0.0, 0.3}}};
        TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood(
                "test", tree, null, precs, new double[][][]{data}, new TorusFourierKernel(2, 64),
                rate, w, d, s);
        assertEquals(-10.165673018093234, like.getLogLikelihood(), 1.0e-6);
    }

    public void testLikelihoodP1() throws Exception {
        Map<String, double[]> obs = new HashMap<String, double[]>();
        obs.put("A", new double[]{0.4});
        obs.put("B", new double[]{-1.2});
        obs.put("C", new double[]{0.9});
        double ll = logLikelihood(1, 64, new double[][]{{1.0}}, obs);
        assertEquals(-5.4429860970, ll, 1.0e-6);
    }

    public void testLikelihoodP2() throws Exception {
        Map<String, double[]> obs = new HashMap<String, double[]>();
        obs.put("A", new double[]{0.4, 1.1});
        obs.put("B", new double[]{-1.2, 0.6});
        obs.put("C", new double[]{0.9, -0.3});
        double ll = logLikelihood(2, 64, new double[][]{{1.0, 0.1}, {0.1, 2.0}}, obs);
        assertEquals(-9.8717597169, ll, 1.0e-6);
    }

    public void testLikelihoodP3() throws Exception {
        Map<String, double[]> obs = new HashMap<String, double[]>();
        obs.put("A", new double[]{0.4, 1.1, -0.5});
        obs.put("B", new double[]{-1.2, 0.6, 0.8});
        obs.put("C", new double[]{0.9, -0.3, 0.2});
        double[][] prec = {{1.0, 0.05, 0.05}, {0.05, 2.0, 0.05}, {0.05, 0.05, 1.5}};
        double ll = logLikelihood(3, 16, prec, obs);
        assertEquals(-14.5216588673, ll, 1.0e-6);
    }

    /** A masked (missing) tip -- tipData[r][tip] == null -- must emit a flat leaf and be
     *  marginalised out, like an internal node. Masking C in ((A,B):0.5, C:1.5) must therefore
     *  equal the 2-tip (A,B) likelihood (convolution preserves the integral), which the Python
     *  prototype confirms to 1e-15. Reference is that masked-C value. */
    public void testLikelihoodMaskedTip() throws Exception {
        Tree tree = new NewickImporter(NEWICK).importTree(null);
        int nExt = tree.getExternalNodeCount();
        Map<String, double[]> obs = new HashMap<String, double[]>();
        obs.put("A", new double[]{0.4, 1.1});
        obs.put("B", new double[]{-1.2, 0.6});
        obs.put("C", null);                                   // masked: residue absent at C
        double[][] data = new double[nExt][];
        for (int e = 0; e < nExt; e++) {
            NodeRef ext = tree.getExternalNode(e);
            data[ext.getNumber()] = obs.get(tree.getNodeTaxon(ext).getId());
        }
        List<MatrixParameter> precs = new ArrayList<MatrixParameter>();
        precs.add(precisionParameter(new double[][]{{1.0, 0.1}, {0.1, 2.0}}));
        TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood(
                "test", tree, null, precs, new double[][][]{data}, new TorusFourierKernel(2, 64));
        assertEquals(-6.658529227132618, like.getLogLikelihood(), 1.0e-6);
    }

    /** Per-residue jump rate (jumpRate dimension == nResidues): two residues sharing the same data
     *  and precision with lambda = [0, 0.8]. Residue 0 (lambda=0) must reproduce the pure-BM value
     *  (testLikelihoodP2) and residue 1 (lambda=0.6) the directed-jump value
     *  (testLikelihoodDirectedJump), so the total is their sum -- confirming lambda is applied
     *  per residue. */
    public void testPerResidueJumpRate() throws Exception {
        Tree tree = new NewickImporter(NEWICK).importTree(null);
        int nExt = tree.getExternalNodeCount();
        Map<String, double[]> obs = new HashMap<String, double[]>();
        obs.put("A", new double[]{0.4, 1.1});
        obs.put("B", new double[]{-1.2, 0.6});
        obs.put("C", new double[]{0.9, -0.3});
        double[][] data = new double[nExt][];
        for (int e = 0; e < nExt; e++) {
            NodeRef ext = tree.getExternalNode(e);
            data[ext.getNumber()] = obs.get(tree.getNodeTaxon(ext).getId());
        }
        List<MatrixParameter> precs = new ArrayList<MatrixParameter>();
        precs.add(precisionParameter(new double[][]{{1.0, 0.1}, {0.1, 2.0}}));
        precs.add(precisionParameter(new double[][]{{1.0, 0.1}, {0.1, 2.0}}));
        Parameter rate = new Parameter.Default(new double[]{0.0, 0.8});   // per-residue lambda (res1 matches the directed test, lambda=0.8)
        double[] w = {0.6, 0.4};
        double[][] d = {{0.0, 0.0}, {1.0, -0.5}};
        double[][][] s = {{{0.6, 0.0}, {0.0, 0.5}}, {{0.3, 0.0}, {0.0, 0.3}}};
        TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood(
                "test", tree, null, precs, new double[][][]{data, data}, new TorusFourierKernel(2, 64),
                rate, w, d, s);
        assertEquals(-9.8717597169 + -10.165673018093234, like.getLogLikelihood(), 1.0e-6);
    }

    // ---- caching: cached logL must equal a from-scratch recompute, and store/restore must revert ----

    /** A single residue's precision move must recompute only that residue; cached == fresh. */
    public void testCachePrecisionChange() throws Exception {
        List<MatrixParameter> precs = twoResiduePrecisions();
        TorusFourierTreeLikelihood like = buildLikelihood(newTreeModel(), 2, 64, precs);
        double ll0 = like.getLogLikelihood();
        precs.get(1).setParameterValue(0, 0, 0.9);   // perturb residue 1 only
        double llCached = like.getLogLikelihood();
        like.makeDirty();
        double llFresh = like.getLogLikelihood();
        assertEquals(llFresh, llCached, 1.0e-9);
        assertTrue("precision change should move the likelihood", Math.abs(llCached - ll0) > 1.0e-6);
    }

    /** A node-height move must recompute only the dirty path to the root; cached == fresh. */
    public void testCacheTreeChange() throws Exception {
        DefaultTreeModel tree = newTreeModel();
        TorusFourierTreeLikelihood like = buildLikelihood(tree, 2, 64, twoResiduePrecisions());
        double ll0 = like.getLogLikelihood();
        NodeRef root = tree.getRoot();
        NodeRef internalChild = null;                // the (A,B) ancestor, height 1.0, under root at 1.5
        for (int i = 0; i < tree.getChildCount(root); i++) {
            NodeRef c = tree.getChild(root, i);
            if (!tree.isExternal(c)) internalChild = c;
        }
        assertNotNull(internalChild);
        tree.setNodeHeight(internalChild, 1.2);
        double llCached = like.getLogLikelihood();
        like.makeDirty();
        double llFresh = like.getLogLikelihood();
        assertEquals(llFresh, llCached, 1.0e-9);
        assertTrue("height change should move the likelihood", Math.abs(llCached - ll0) > 1.0e-6);
    }

    /** Restore must revert the cached buffers: an unchanged residue stays reusable, a
     *  reverted residue recomputes to its original value. */
    public void testStoreRestoreReverts() throws Exception {
        List<MatrixParameter> precs = twoResiduePrecisions();
        TorusFourierTreeLikelihood like = buildLikelihood(newTreeModel(), 2, 64, precs);
        double ll0 = like.getLogLikelihood();
        double orig = precs.get(1).getParameterValue(0, 0);
        like.storeModelState();
        precs.get(1).setParameterValue(0, 0, 0.5);
        double llChanged = like.getLogLikelihood();
        assertTrue("change should move the likelihood", Math.abs(llChanged - ll0) > 1.0e-6);
        like.restoreModelState();
        precs.get(1).setParameterValue(0, 0, orig);  // operator-reject restores the parameter too
        double llBack = like.getLogLikelihood();
        assertEquals(ll0, llBack, 1.0e-9);
    }

    /** With real compoundSymmetricMatrix wiring, a move on one residue's diagonal must
     *  recompute only that residue's nodes (the per-residue caching win). The 3-tip tree
     *  has 2 internal nodes, so a single-residue recompute == 2 node recomputes. */
    public void testPerResidueDirtyingCompoundSymmetric() throws Exception {
        dr.inference.model.Parameter diag0 = new Parameter.Default(new double[]{2.0, 3.0});
        dr.inference.model.Parameter off0 = new Parameter.Default(new double[]{0.3});
        dr.inference.model.Parameter diag1 = new Parameter.Default(new double[]{1.5, 2.5});
        dr.inference.model.Parameter off1 = new Parameter.Default(new double[]{-0.2});
        List<MatrixParameter> precs = new ArrayList<MatrixParameter>();
        precs.add(new dr.inference.model.CompoundSymmetricMatrix(diag0, off0, true, false));
        precs.add(new dr.inference.model.CompoundSymmetricMatrix(diag1, off1, true, false));
        TorusFourierTreeLikelihood like = buildLikelihood(newTreeModel(), 2, 64, precs);
        double ll0 = like.getLogLikelihood();

        like.resetRecomputeCount();
        diag1.setParameterValue(0, 1.8);          // perturb residue 1 only, via its sub-parameter
        double llCached = like.getLogLikelihood();
        long recomputes = like.getNodeRecomputeCount();

        like.makeDirty();
        double llFresh = like.getLogLikelihood();
        assertEquals(llFresh, llCached, 1.0e-9);
        assertTrue("diagonal change should move the likelihood", Math.abs(llCached - ll0) > 1.0e-6);
        assertEquals("a one-residue precision move should recompute only that residue's 2 internal nodes",
                2L, recomputes);
    }

    /** Regression for the parameter-passthrough bug: a node-height move must recompute only
     *  the dirty path to the root, not every node. On an 8-tip tree (7 internal nodes), moving
     *  a cherry-parent must recompute strictly fewer than all 7 internal nodes per residue. */
    public void testTreeMoveRecomputesDirtyPathOnly() throws Exception {
        String newick = "(((A:1,B:1):1,(C:1,D:1):1):1,((E:1,F:1):1,(G:1,H:1):1):1);";
        DefaultTreeModel tree = new DefaultTreeModel("t", new NewickImporter(newick).importTree(null));
        int nExt = tree.getExternalNodeCount();
        List<MatrixParameter> precs = new ArrayList<MatrixParameter>();
        precs.add(precisionParameter(new double[][]{{1.0, 0.1}, {0.1, 2.0}}));
        double[][][] data = new double[1][nExt][];
        for (int e = 0; e < nExt; e++) {
            NodeRef ext = tree.getExternalNode(e);
            data[0][ext.getNumber()] = new double[]{0.2 * e - 0.5, 0.7 - 0.1 * e};
        }
        TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood(
                "t", tree, null, precs, data, new TorusFourierKernel(2, 32));
        like.getLogLikelihood();
        int internal = tree.getInternalNodeCount();   // 7

        NodeRef cherryParent = null;                   // an internal node whose children are tips
        for (int i = 0; i < tree.getNodeCount(); i++) {
            NodeRef nd = tree.getNode(i);
            if (!tree.isExternal(nd) && tree.isExternal(tree.getChild(nd, 0))) { cherryParent = nd; break; }
        }
        assertNotNull(cherryParent);

        like.resetRecomputeCount();
        tree.setNodeHeight(cherryParent, tree.getNodeHeight(cherryParent));
        like.getLogLikelihood();
        long recomputes = like.getNodeRecomputeCount();
        assertTrue("a single cherry-parent move must recompute fewer than all " + internal
                + " internal nodes (got " + recomputes + ")", recomputes < internal);
        assertTrue("must recompute at least the moved node", recomputes >= 1);
    }

    /** Bug 1 regression: a tip (external-branch / sampled-date) height change must force the
     *  parent to recompute. Tips hold no cached partial, so the parent must consult the tip's
     *  own dirty flag; otherwise the parent stays stale. cached must equal fresh. */
    public void testCacheTipHeightChange() throws Exception {
        DefaultTreeModel tree = newTreeModel();
        TorusFourierTreeLikelihood like = buildLikelihood(tree, 2, 64, twoResiduePrecisions());
        double ll0 = like.getLogLikelihood();
        NodeRef tip = tree.getExternalNode(0);
        double parentH = tree.getNodeHeight(tree.getParent(tip));
        tree.setNodeHeight(tip, 0.4 * parentH);   // serially-sampled tip: nonzero, below parent
        double llCached = like.getLogLikelihood();
        like.makeDirty();
        double llFresh = like.getLogLikelihood();
        assertEquals(llFresh, llCached, 1.0e-9);
        assertTrue("tip height change should move the likelihood", Math.abs(llCached - ll0) > 1.0e-6);
    }

    /** Bug 2 regression: storing before the first evaluation (state fully dirty, buffers
     *  uninitialised) then restoring must not leave the indices pointing at never-computed
     *  buffers. Restoring the dirty flags forces a correct recompute instead of reading zeros. */
    public void testStoreBeforeFirstEvalThenRestore() throws Exception {
        TorusFourierTreeLikelihood like = buildLikelihood(newTreeModel(), 2, 64, twoResiduePrecisions());
        like.storeModelState();                       // store while still dirty / uninitialised
        double ll = like.getLogLikelihood();
        assertFalse("first eval must be finite", Double.isInfinite(ll) || Double.isNaN(ll));
        like.restoreModelState();
        double ll2 = like.getLogLikelihood();
        assertEquals("restore-to-dirty must recompute, not read uninitialised buffers", ll, ll2, 1.0e-9);
    }


    // Hard regressions: these are analytic small-tree Gaussian likelihoods. FFT
    // clipping used to report roughly -36 and -706 for the first two cases.
    @Override
    protected void tearDown() throws Exception {
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(false);
        super.tearDown();
    }

    /** Production must not ship the validating path: it costs ~1e3x and can fail outright
     *  on a 13-component jump mixture. Guards against silently re-enabling it. */
    public void testProductionDefaultIsUnvalidated() {
        assertFalse("per-evaluation validation must default to off",
                TorusFourierTreeLikelihood.isValidateEveryEvaluation());
    }

    public void testSmallSigmaInternalPropagation() throws Exception {
        // Asserts REFERENCE accuracy: the production FFT path cannot reach these tails
        // (see verify_checks.py F2/J2), so validation must be enabled explicitly.
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(true);
        TorusFourierTreeLikelihood like = smallSigmaLikelihood(
                "((A:1,B:1):1,C:1);", 0.001, new double[]{0.0, 0.02, 1.0}, 64);
        assertEquals(-193.692717810054, like.getLogLikelihood(), 1.0e-7);
        // Draw ancestors too: tiny transition probabilities cannot be exponentiated
        // before conditioning or replaced with the unconditioned child partial.
        dr.evolution.tree.TreeTrait trait = like.getTreeTrait("theta_0");
        assertNotNull(trait);
        Tree tree = new NewickImporter("((A:1,B:1):1,C:1);").importTree(null);
        double[] rootDraw = (double[]) trait.getTrait(tree, tree.getRoot());
        assertEquals(1, rootDraw.length);
        assertTrue(Double.isFinite(rootDraw[0]));
    }

    public void testSmallSigmaLikelihoodBelowDoubleFloor() throws Exception {
        // Asserts REFERENCE accuracy: the production FFT path cannot reach these tails
        // (see verify_checks.py F2/J2), so validation must be enabled explicitly.
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(true);
        TorusFourierTreeLikelihood like = smallSigmaLikelihood(
                "(A:1,B:1);", 0.0001, new double[]{0.0, 0.6}, 64);
        assertEquals(-898.498219003906, like.getLogLikelihood(), 1.0e-7);
    }

    public void testSmallSigmaPeaksBetweenGridPoints() throws Exception {
        // Asserts REFERENCE accuracy: the production FFT path cannot reach these tails
        // (see verify_checks.py F2/J2), so validation must be enabled explicitly.
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(true);
        TorusFourierTreeLikelihood like = smallSigmaLikelihood(
                "(A:1,B:1);", 0.0001, new double[]{0.03, 0.05}, 64);
        double exact = -Math.log(2.0 * Math.PI)
                - 0.5 * Math.log(2.0 * Math.PI * 0.0002) - 1.0;
        assertEquals(exact, like.getLogLikelihood(), 1.0e-7);
    }

    public void testStableTransitionAngleInvarianceAndMissingJumpLaw() {
        double[][] sigma = {{0.001}};
        double[] density = TorusLogSpacePruning.logTransition(
                new double[][]{{0.17}, {0.17 + 6.0 * Math.PI}}, sigma, 1.0, 3.0,
                null, null, null);
        double exact = -0.5 * Math.log(2.0 * Math.PI * 0.001) - 0.17 * 0.17 / 0.002;
        assertEquals(exact, density[0], 1.0e-10);
        assertEquals(density[0], density[1], 1.0e-10);
    }

    public void testStableDirectedJumpSign() {
        // A very narrow jump displaced by +0.7 puts its one-jump mass at +0.7.
        double[] density = TorusLogSpacePruning.logTransition(new double[][]{{0.7}, {-0.7}},
                new double[][]{{0.0001}}, 1.0, 0.2, new double[]{1.0},
                new double[][]{{0.7}}, new double[][][]{{{0.0001}}});
        double exactOneJump = -0.2 + Math.log(0.2)
                - 0.5 * Math.log(2.0 * Math.PI * 0.0002);
        assertEquals(exactOneJump, density[0], 1.0e-7);
        assertTrue("directed displacement must have the physical child-parent sign",
                density[0] - density[1] > 10.0);
    }

    public void testStableDeterministicJumpAndUnderflowingPoissonMean() {
        double[] deterministic = TorusLogSpacePruning.logTransition(new double[][]{{0.7}},
                new double[][]{{0.0001}}, 1.0, 0.2, new double[]{1.0},
                new double[][]{{0.7}}, new double[][][]{{{0.0}}});
        assertEquals(-0.2 + Math.log(0.2) - 0.5 * Math.log(2.0 * Math.PI * 0.0001),
                deterministic[0], 1.0e-7);
        double[] rare = TorusLogSpacePruning.logTransition(new double[][]{{0.6}},
                new double[][]{{10000.0}}, 1e-8, 1e-320, new double[]{1.0},
                new double[][]{{0.6}}, new double[][][]{{{0.01}}});
        double exact = Math.log(1e-320) + Math.log(1e-8)
                - 0.5 * Math.log(2.0 * Math.PI * 0.0101);
        assertEquals(exact, rare[0], 1.0e-9);
    }

    public void testStableLargeJumpRate() {
        double[] density = TorusLogSpacePruning.logTransition(new double[][]{{0.0}, {2.0}},
                new double[][]{{0.0001}}, 1.0, 5000.0, new double[]{1.0},
                new double[][]{{0.7}}, new double[][][]{{{0.5}}});
        assertEquals(-Math.log(2.0 * Math.PI), density[0], 1.0e-10);
        assertEquals(density[0], density[1], 1.0e-10);
    }

    public void testStableRareJumpCountsControlDistantTails() {
        // Independent high-precision Poisson/wrapped-Gaussian sum references.
        double[] density = TorusLogSpacePruning.logTransition(new double[][]{{0.6}, {3.14}},
                new double[][]{{0.0001}}, 1.0, 0.5, new double[]{1.0},
                new double[][]{{0.0}}, new double[][][]{{{0.0001}}});
        assertEquals(-138.719459179258024, density[0], 1.0e-8);
        assertEquals(-916.958864509519799, density[1], 1.0e-8);
    }

    public void testStableAllMissingSubtreeIsExactlyFlat() throws Exception {
        TorusFourierTreeLikelihood like = smallSigmaLikelihood(
                "((A:100):1,B:2);", 0.4, new double[]{Double.NaN, Double.NaN}, 8);
        assertEquals(0.0, like.getLogLikelihood(), 1.0e-12);
    }

    public void testStableMissingSubtreeMarginalizesExactly() throws Exception {
        TorusFourierTreeLikelihood full = smallSigmaLikelihood(
                "((A:100):1,B:2,C:2);", 0.4, new double[]{Double.NaN, 0.4, -0.8}, 8);
        TorusFourierTreeLikelihood pruned = smallSigmaLikelihood(
                "(A:2,B:2);", 0.4, new double[]{0.4, -0.8}, 8);
        assertEquals(pruned.getLogLikelihood(), full.getLogLikelihood(), 1.0e-8);
    }

    public void testStableInternalGridMassAliasing() throws Exception {
        TorusFourierTreeLikelihood like = smallSigmaLikelihood(
                "((A:1):0.0064,B:1);", 1.0, new double[]{0.17, Double.NaN}, 64);
        assertEquals(-Math.log(2.0 * Math.PI), like.getLogLikelihood(), 1.0e-8);
    }

    public void testStableResourceCapIsExplicit() throws Exception {
        // Asserts REFERENCE accuracy: the production FFT path cannot reach these tails
        // (see verify_checks.py F2/J2), so validation must be enabled explicitly.
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(true);
        TorusFourierTreeLikelihood like = smallSigmaLikelihood(
                "(A:1,B:1);", 1.0e-12, new double[]{0.03, 0.05}, 64);
        try {
            like.getLogLikelihood();
            fail("unresolved quadrature must not return a likelihood");
        } catch (TorusLogSpacePruning.AccuracyException expected) {
            assertTrue(expected.getMessage().contains("resource cap"));
        }
    }

    private static TorusFourierTreeLikelihood smallSigmaLikelihood(
            String newick, double variance, double[] observations, int grid) throws Exception {
        Tree tree = new NewickImporter(newick).importTree(null);
        double[][] data = new double[tree.getExternalNodeCount()][];
        for (int i = 0; i < tree.getExternalNodeCount(); i++) {
            NodeRef node = tree.getExternalNode(i);
            int observation = tree.getNodeTaxon(node).getId().charAt(0) - 'A';
            data[node.getNumber()] = Double.isNaN(observations[observation])
                    ? null : new double[]{observations[observation]};
        }
        List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        precisions.add(precisionParameter(new double[][]{{1.0 / variance}}));
        return new TorusFourierTreeLikelihood("smallSigma", tree, null, precisions,
                new double[][][]{data}, new TorusFourierKernel(1, grid));
    }

    public void testAncestralTraitRefreshesDirtyLikelihood() throws Exception {
        DefaultTreeModel tree = newTreeModel();
        List<MatrixParameter> precisions = twoResiduePrecisions();
        TorusFourierTreeLikelihood like = buildLikelihood(tree, 2, 64, precisions);
        dr.evolution.tree.TreeTrait trait = like.getTreeTrait("theta_0");
        trait.getTrait(tree, tree.getRoot());
        like.resetRecomputeCount();
        precisions.get(0).setParameterValue(0, 0, 1.2);
        // A logger may ask for an ancestral trait before any likelihood logger.
        trait.getTrait(tree, tree.getRoot());
        assertTrue("ancestral access must refresh changed parameter state",
                like.getNodeRecomputeCount() > 0);
    }

    // ---- helpers ----

    private static final java.util.Map<String, double[]> P2_OBS = new HashMap<String, double[]>() {{
        put("A", new double[]{0.4, 1.1});
        put("B", new double[]{-1.2, 0.6});
        put("C", new double[]{0.9, -0.3});
    }};

    private static DefaultTreeModel newTreeModel() throws Exception {
        return new DefaultTreeModel("treeModel", new NewickImporter(NEWICK).importTree(null));
    }

    private static List<MatrixParameter> twoResiduePrecisions() {
        List<MatrixParameter> precs = new ArrayList<MatrixParameter>();
        precs.add(precisionParameter(new double[][]{{1.0, 0.1}, {0.1, 2.0}}));
        precs.add(precisionParameter(new double[][]{{1.5, -0.2}, {-0.2, 0.8}}));
        return precs;
    }

    /** p=2 likelihood over `precs.size()` residues that share the P2_OBS tip data. */
    private static TorusFourierTreeLikelihood buildLikelihood(DefaultTreeModel tree, int dim, int nGrid,
                                                              List<MatrixParameter> precs) {
        int nExt = tree.getExternalNodeCount();
        double[][][] data = new double[precs.size()][nExt][];
        for (int r = 0; r < precs.size(); r++) {
            for (int e = 0; e < nExt; e++) {
                NodeRef ext = tree.getExternalNode(e);
                data[r][ext.getNumber()] = P2_OBS.get(tree.getNodeTaxon(ext).getId());
            }
        }
        return new TorusFourierTreeLikelihood(
                "test", tree, null, precs, data, new TorusFourierKernel(dim, nGrid));
    }

    private static double multiplyInto(double[] v, double[] msg) {
        double max = 0.0;
        for (int k = 0; k < v.length; k++) {
            v[k] *= msg[k];
            if (v[k] > max) max = v[k];
        }
        double inv = 1.0 / max;
        for (int k = 0; k < v.length; k++) v[k] *= inv;
        return Math.log(max);
    }

    /** precision is given row-major; build the column-major MatrixParameter. */
    private static MatrixParameter precisionParameter(double[][] p) {
        int d = p.length;
        Parameter[] cols = new Parameter[d];
        for (int j = 0; j < d; j++) {
            double[] c = new double[d];
            for (int i = 0; i < d; i++) c[i] = p[i][j];
            cols[j] = new Parameter.Default(c);
        }
        return new MatrixParameter("p", cols);
    }

    private static double logLikelihood(int dim, int nGrid, double[][] precision,
                                        Map<String, double[]> obs) throws Exception {
        Tree tree = new NewickImporter(NEWICK).importTree(null);
        int nExt = tree.getExternalNodeCount();
        double[][] data = new double[nExt][];
        for (int e = 0; e < nExt; e++) {
            NodeRef ext = tree.getExternalNode(e);
            data[ext.getNumber()] = obs.get(tree.getNodeTaxon(ext).getId());
        }
        List<MatrixParameter> precs = new ArrayList<MatrixParameter>();
        precs.add(precisionParameter(precision));
        TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood(
                "test", tree, null, precs, new double[][][]{data}, new TorusFourierKernel(dim, nGrid));
        return like.getLogLikelihood();
    }

    // ---------------- ancestral reconstruction (downward joint sampling) ----------------

    /** The kernel used to sample a child given its parent MUST be the same kernel the inward pass
     *  propagates with: transitionKernel(t) == propagateFromPsi(delta_at_0, t). If these ever
     *  diverge, ancestral samples would be drawn from a different process than the one fitted. */
    public void testTransitionKernelMatchesPropagate() {
        TorusFourierKernel k = new TorusFourierKernel(2, 32);
        double[][] sigma = {{0.2, 0.05}, {0.05, 0.3}};
        double[] w = {0.6, 0.4};
        double[][] d = {{0.0, 0.0}, {0.8, -0.5}};
        double[][][] s = {{{0.3, 0.0}, {0.0, 0.3}}, {{0.2, 0.0}, {0.0, 0.2}}};
        int n2 = k.getGridCount();
        double[] psiRe = new double[n2], psiIm = new double[n2];
        k.computePsiComplex(sigma, 0.7, w, d, s, psiRe, psiIm);
        double t = 1.3;
        double[] kt = k.transitionKernel(t, psiRe, psiIm);
        double[] delta = new double[n2]; delta[0] = 1.0;
        double[] prop = k.propagateFromPsi(delta, t, psiRe, psiIm);
        double mx = 0.0;
        for (int i = 0; i < n2; i++) mx = Math.max(mx, Math.abs(kt[i] - prop[i]));
        assertEquals(0.0, mx, 1.0e-9);
    }

    /** sampleConditionalCell must weight child cell j by K_t(j - parent): a delta kernel at a known
     *  displacement (with a flat child partial) must place every draw at parent + displacement,
     *  exercising the per-axis modular displacement indexing. */
    public void testSampleConditionalDisplacement() {
        TorusFourierKernel k = new TorusFourierKernel(2, 16);
        int n = k.getGridSize(), n2 = k.getGridCount();
        double[] kernel = new double[n2]; kernel[3 * n + 2] = 1.0;     // displacement (3, 2)
        double[] childPartial = new double[n2]; Arrays.fill(childPartial, 1.0);
        int parent = 5 * n + 7;                                         // parent cell (5, 7)
        int expected = ((5 + 3) % n) * n + ((7 + 2) % n);
        for (double u : new double[]{0.0, 0.3, 0.7, 0.999}) {
            assertEquals(expected, k.sampleConditionalCell(kernel, parent, childPartial, u));
        }
    }

    /** End-to-end: sampled ancestral angles are length-2 and in (-pi, pi] at every internal node,
     *  and null at tips (tips are observed, not reconstructed). */
    public void testAncestralSampleInRange() throws Exception {
        Tree tree = new NewickImporter(NEWICK).importTree(null);
        int nExt = tree.getExternalNodeCount();
        Map<String, double[]> obs = new HashMap<String, double[]>();
        obs.put("A", new double[]{0.4, 1.1});
        obs.put("B", new double[]{-1.2, 0.6});
        obs.put("C", new double[]{0.9, -0.3});
        double[][] data = new double[nExt][];
        for (int e = 0; e < nExt; e++) {
            NodeRef ext = tree.getExternalNode(e);
            data[ext.getNumber()] = obs.get(tree.getNodeTaxon(ext).getId());
        }
        List<MatrixParameter> precs = new ArrayList<MatrixParameter>();
        precs.add(precisionParameter(new double[][]{{1.0, 0.1}, {0.1, 2.0}}));
        Parameter rate = new Parameter.Default(0.8);
        double[] w = {0.6, 0.4};
        double[][] d = {{0.0, 0.0}, {1.0, -0.5}};
        double[][][] s = {{{0.6, 0.0}, {0.0, 0.5}}, {{0.3, 0.0}, {0.0, 0.3}}};
        TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood(
                "test", tree, null, precs, new double[][][]{data}, new TorusFourierKernel(2, 64),
                rate, w, d, s);
        like.getLogLikelihood();
        dr.evolution.tree.TreeTrait tt = like.getTreeTrait("theta_0");
        assertNotNull(tt);
        int internal = 0;
        for (int i = 0; i < tree.getNodeCount(); i++) {
            NodeRef node = tree.getNode(i);
            Object v = tt.getTrait(tree, node);
            if (tree.isExternal(node)) {
                assertNull(v);
            } else {
                double[] a = (double[]) v;
                assertEquals(2, a.length);
                assertTrue(a[0] > -Math.PI - 1e-9 && a[0] <= Math.PI + 1e-9);
                assertTrue(a[1] > -Math.PI - 1e-9 && a[1] <= Math.PI + 1e-9);
                internal++;
            }
        }
        assertTrue(internal > 0);
    }

    /** Deep-tree regression (8 tips, depth 3) — the earlier tests were all <=4 tips, which hid
     *  accumulation issues. Reference logL from the validated numpy prototype (torus_fourier_pruning)
     *  on the same tree/obs/Sigma; precision passed = inv(Sigma). */
    public void testDeepTree8Tip() throws Exception {
        String nwk = "(((A:10.0,B:10.0):5.0,(C:10.0,D:10.0):5.0):8.0,"
                   + "((E:10.0,F:10.0):5.0,(G:10.0,H:10.0):5.0):8.0);";
        Tree tree = new NewickImporter(nwk).importTree(null);
        int nExt = tree.getExternalNodeCount();
        Map<String, double[]> obs = new HashMap<String, double[]>();
        obs.put("A", new double[]{0.4, 1.1});   obs.put("B", new double[]{-1.2, 0.6});
        obs.put("C", new double[]{0.9, -0.3});  obs.put("D", new double[]{2.1, -1.5});
        obs.put("E", new double[]{-2.0, 2.5});  obs.put("F", new double[]{0.1, 0.2});
        obs.put("G", new double[]{1.5, -2.0});  obs.put("H", new double[]{-0.7, 0.8});
        double[][] data = new double[nExt][];
        for (int e = 0; e < nExt; e++) {
            NodeRef ext = tree.getExternalNode(e);
            data[ext.getNumber()] = obs.get(tree.getNodeTaxon(ext).getId());
        }
        List<MatrixParameter> precs = new ArrayList<MatrixParameter>();
        precs.add(precisionParameter(new double[][]{{20.51282051282051, -2.5641025641025643},
                                                    {-2.5641025641025643, 12.820512820512821}}));
        TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood(
                "test", tree, null, precs, new double[][][]{data}, new TorusFourierKernel(2, 32));
        assertEquals(-36.15001596266041, like.getLogLikelihood(), 1.0e-6);
    }
}
