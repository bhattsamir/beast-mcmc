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
import dr.xml.Reportable;

import java.util.Arrays;
import java.util.List;

/**
 * Felsenstein pruning likelihood for independent d-dimensional wrapped-Brownian
 * diffusions (one per residue) on a flat torus, computed by FFT heat-kernel
 * propagation on a power-of-two grid (see TorusFourierKernel). The windings are
 * marginalised exactly by the FFT, so no wrap augmentation is required. v1
 * recomputes all partials on each evaluation; caching/threading are a later
 * optimisation.
 *
 * @author Samir Bhatt
 */
public class TorusFourierTreeLikelihood extends AbstractModelLikelihood implements Reportable {

    public static final String TORUS_FOURIER_LIKELIHOOD = "torusFourierTreeLikelihood";
    private static final double MIN_BRANCH = 1.0e-8;   // floor for zero-length branches (replicate tips)
    private static final int PARALLEL_THRESHOLD = 8;   // parallelise the residue loop at/above this many residues

    private final Tree tree;
    private final BranchRateModel branchRateModel;     // nullable -> rate 1
    private final List<MatrixParameter> precisions;    // one dim x dim precision per residue
    private final double[][][] tipData;                // [residue][externalNodeNumber][dim], radians
    private final TorusFourierKernel kernel;
    private final int nResidues;
    private final int n2;
    private final int dim;

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
        this.dim = kernel.getDimension();

        for (int r = 0; r < nResidues; r++) {
            if (tipData[r].length != tree.getExternalNodeCount()) {
                throw new IllegalArgumentException("tipData residue " + r + " has "
                        + tipData[r].length + " taxa, tree has " + tree.getExternalNodeCount());
            }
            final MatrixParameter p = precisions.get(r);
            if (p.getRowDimension() != dim || p.getColumnDimension() != dim) {
                throw new IllegalArgumentException("precision " + r + " must be " + dim + "x" + dim);
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

    public String getReport() {
        return getClass().getSimpleName() + ": logL = " + getLogLikelihood();
    }

    private double calculateLogLikelihood() {
        final NodeRef root = tree.getRoot();
        // Residues are independent; the kernel and all per-residue state are read-only
        // during evaluation, so the residue loop parallelises safely. Above a threshold
        // we use a parallel stream (helps many-residue datasets such as the p=2 sweeps).
        java.util.stream.IntStream stream = java.util.stream.IntStream.range(0, nResidues);
        if (nResidues >= PARALLEL_THRESHOLD) stream = stream.parallel();
        return stream.mapToDouble(r -> residueLogLikelihood(r, root)).sum();
    }

    private double residueLogLikelihood(int r, NodeRef root) {
        final double[][] sigma = invertPrecision(precisions.get(r));
        final double[] logScale = new double[1];
        final double[] rootValues = nodeValues(r, root, sigma, logScale);
        return logScale[0] + Math.log(kernel.rootIntegral(rootValues));
    }

    /** Post-order product of child messages at an internal node (mirrors prune_loglik). */
    private double[] nodeValues(int r, NodeRef node, double[][] sigma, double[] logScale) {
        final double[] values = new double[n2];
        Arrays.fill(values, 1.0);
        for (int c = 0; c < tree.getChildCount(node); c++) {
            final NodeRef child = tree.getChild(node, c);
            final double bl = branchLength(child);
            final double[] msg;
            if (tree.isExternal(child)) {
                msg = kernel.leafKernel(tipData[r][child.getNumber()], sigma, bl);
            } else {
                final double[] childValues = nodeValues(r, child, sigma, logScale);
                msg = kernel.propagate(childValues, sigma, bl);
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

    /** Invert a dim x dim precision matrix to a covariance, by Gauss-Jordan elimination. */
    private double[][] invertPrecision(MatrixParameter p) {
        final int d = dim;
        final double[][] a = new double[d][2 * d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) a[i][j] = p.getParameterValue(i, j);
            a[i][d + i] = 1.0;
        }
        for (int col = 0; col < d; col++) {
            int piv = col;
            for (int rr = col + 1; rr < d; rr++) {
                if (Math.abs(a[rr][col]) > Math.abs(a[piv][col])) piv = rr;
            }
            if (Math.abs(a[piv][col]) < 1.0e-300) {
                throw new RuntimeException("Singular precision matrix");
            }
            final double[] tmp = a[col]; a[col] = a[piv]; a[piv] = tmp;
            final double diag = a[col][col];
            for (int j = 0; j < 2 * d; j++) a[col][j] /= diag;
            for (int rr = 0; rr < d; rr++) {
                if (rr != col) {
                    final double f = a[rr][col];
                    if (f != 0.0) for (int j = 0; j < 2 * d; j++) a[rr][j] -= f * a[col][j];
                }
            }
        }
        final double[][] inv = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) inv[i][j] = a[i][d + j];
        }
        return inv;
    }

    // ---- Model bookkeeping (v1: any change invalidates the whole likelihood) ----

    protected void handleModelChangedEvent(Model model, Object object, int index) { likelihoodKnown = false; }

    protected void handleVariableChangedEvent(Variable variable, int index, Parameter.ChangeType type) { likelihoodKnown = false; }

    protected void storeState() { storedLogLikelihood = logLikelihood; storedLikelihoodKnown = likelihoodKnown; }

    protected void restoreState() { logLikelihood = storedLogLikelihood; likelihoodKnown = storedLikelihoodKnown; }

    protected void acceptState() { }
}
