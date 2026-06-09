/*
 * TorusFourierTreeLikelihood.java
 *
 * @author Samir Bhatt
 */

package dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evomodel.branchratemodel.BranchRateModel;
import dr.inference.model.*;

import java.util.Arrays;
import java.util.List;

/**
 * Felsenstein pruning likelihood for independent 2D wrapped-Brownian diffusions
 * (one per residue) on a flat torus, computed by FFT heat-kernel propagation on a
 * power-of-two grid (see TorusFourierKernel). The windings are marginalised exactly
 * by the FFT, so no wrap augmentation is required. v1 recomputes all partials on
 * each evaluation; caching/threading are a later optimisation.
 *
 * @author Samir Bhatt
 */
public class TorusFourierTreeLikelihood extends AbstractModelLikelihood {

    public static final String TORUS_FOURIER_LIKELIHOOD = "torusFourierTreeLikelihood";
    private static final double MIN_BRANCH = 1.0e-8;   // floor for zero-length branches (replicate tips)

    private final Tree tree;
    private final BranchRateModel branchRateModel;     // nullable -> rate 1
    private final List<MatrixParameter> precisions;    // one 2x2 precision per residue
    private final double[][][] tipData;                // [residue][externalNodeNumber][2], radians
    private final TorusFourierKernel kernel;
    private final int nResidues;
    private final int n2;

    private double logLikelihood;
    private boolean likelihoodKnown = false;
    private double storedLogLikelihood;
    private boolean storedLikelihoodKnown = false;

    public TorusFourierTreeLikelihood(String name, Tree tree, BranchRateModel branchRateModel,
                                      List<MatrixParameter> precisions, double[][][] tipData,
                                      TorusFourierKernel kernel) {
        super(name);
        this.tree = tree;
        this.branchRateModel = branchRateModel;
        this.precisions = precisions;
        this.tipData = tipData;
        this.kernel = kernel;
        this.nResidues = precisions.size();
        this.n2 = kernel.getGridCount();

        for (int r = 0; r < nResidues; r++) {
            if (tipData[r].length != tree.getExternalNodeCount()) {
                throw new IllegalArgumentException("tipData residue " + r + " has "
                        + tipData[r].length + " taxa, tree has " + tree.getExternalNodeCount());
            }
            if (precisions.get(r).getRowDimension() != 2 || precisions.get(r).getColumnDimension() != 2) {
                throw new IllegalArgumentException("precision " + r + " must be 2x2");
            }
        }
        if (tree instanceof Model) addModel((Model) tree);
        if (branchRateModel != null) addModel(branchRateModel);
        for (MatrixParameter p : precisions) addVariable(p);
    }

    // ---- Likelihood ----

    public Model getModel() { return this; }

    public double getLogLikelihood() {
        if (!likelihoodKnown) {
            logLikelihood = calculateLogLikelihood();
            likelihoodKnown = true;
        }
        return logLikelihood;
    }

    public void makeDirty() { likelihoodKnown = false; }

    private double calculateLogLikelihood() {
        final NodeRef root = tree.getRoot();
        double total = 0.0;
        for (int r = 0; r < nResidues; r++) {
            final double[] sigma = invertPrecision(precisions.get(r));   // {s00, s01, s11}
            final double[] logScale = new double[1];
            final double[] rootValues = nodeValues(r, root, sigma, logScale);
            total += logScale[0] + Math.log(kernel.rootIntegral(rootValues));
        }
        return total;
    }

    /** Post-order product of child messages at an internal node (mirrors prune_loglik). */
    private double[] nodeValues(int r, NodeRef node, double[] sigma, double[] logScale) {
        final double[] values = new double[n2];
        Arrays.fill(values, 1.0);
        for (int c = 0; c < tree.getChildCount(node); c++) {
            final NodeRef child = tree.getChild(node, c);
            final double bl = branchLength(child);
            final double[] msg;
            if (tree.isExternal(child)) {
                final double[] y = tipData[r][child.getNumber()];
                msg = kernel.leafKernel(y[0], y[1], sigma[0], sigma[1], sigma[2], bl);
            } else {
                final double[] childValues = nodeValues(r, child, sigma, logScale);
                msg = kernel.propagate(childValues, sigma[0], sigma[1], sigma[2], bl);
            }
            double max = 0.0;
            for (int k = 0; k < n2; k++) {
                values[k] *= msg[k];
                if (values[k] > max) max = values[k];
            }
            if (!(max > 0.0)) {
                throw new RuntimeException("Torus pruning underflowed at node "
                        + node.getNumber() + ", residue " + r);
            }
            final double inv = 1.0 / max;
            for (int k = 0; k < n2; k++) values[k] *= inv;
            logScale[0] += Math.log(max);
        }
        return values;
    }

    private double branchLength(NodeRef node) {
        double bl = tree.getNodeHeight(tree.getParent(node)) - tree.getNodeHeight(node);
        if (branchRateModel != null) bl *= branchRateModel.getBranchRate(tree, node);
        return bl > MIN_BRANCH ? bl : MIN_BRANCH;
    }

    /** Invert a 2x2 precision matrix to a covariance {s00, s01, s11}. */
    private double[] invertPrecision(MatrixParameter p) {
        final double p00 = p.getParameterValue(0, 0);
        final double p01 = p.getParameterValue(0, 1);
        final double p11 = p.getParameterValue(1, 1);
        final double det = p00 * p11 - p01 * p01;
        if (!(det > 0.0)) throw new RuntimeException("Non-PD precision matrix, det = " + det);
        return new double[]{p11 / det, -p01 / det, p00 / det};
    }

    // ---- Model bookkeeping (v1: any change invalidates the whole likelihood) ----

    protected void handleModelChangedEvent(Model model, Object object, int index) { likelihoodKnown = false; }

    protected void handleVariableChangedEvent(Variable variable, int index, Parameter.ChangeType type) { likelihoodKnown = false; }

    protected void storeState() { storedLogLikelihood = logLikelihood; storedLikelihoodKnown = likelihoodKnown; }

    protected void restoreState() { logLikelihood = storedLogLikelihood; likelihoodKnown = storedLikelihoodKnown; }

    protected void acceptState() { }
}
