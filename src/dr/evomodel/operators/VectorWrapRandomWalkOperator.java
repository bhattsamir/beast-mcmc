/*
 * VectorWrapRandomWalkOperator.java
 *
 * MCMC operator: pick K distinct non-pinned wrap indices uniformly at random
 * (Fisher-Yates partial shuffle) and flip each by an independent +/-1. K is
 * a constant (blockSize). Pinned wraps (taxon 0 gauge fix and missing-data
 * coords) are excluded from the index pool, so no proposal mass is wasted on
 * indices that cannot affect the likelihood.
 *
 * Per-coordinate +/-1 picks are independent and uniform, and index selection
 * is uniform without replacement -- so the proposal density is the same for
 * the forward and reverse moves, and the log Hastings ratio is 0 for any
 * accepted proposal. If any of the K proposed flips would exit the parameter
 * bounds, the entire proposal is rejected (logq = -infinity) to preserve
 * detailed balance at the boundary.
 *
 * Faster mixing than the per-coordinate randomWalkIntegerOperator at the cost
 * of K likelihood-affecting changes per proposal instead of 1.
 *
 * @author Samir Bhatt
 */

package dr.evomodel.operators;

import dr.evomodel.treedatalikelihood.continuous.WrappedContinuousTraitDataModel;
import dr.inference.model.Parameter;
import dr.inference.operators.SimpleMCMCOperator;
import dr.math.MathUtils;

public class VectorWrapRandomWalkOperator extends SimpleMCMCOperator {

    private final WrappedContinuousTraitDataModel dataModel;
    private final Parameter wraps;
    private final int blockSize;
    private final int[] perm;     // working buffer; permuted in place each fire

    public VectorWrapRandomWalkOperator(WrappedContinuousTraitDataModel dataModel,
                                        int blockSize,
                                        double weight) {
        this.dataModel = dataModel;
        this.wraps = dataModel.getWrapsParameter();
        // Build the list of non-pinned indices once at construction.
        int totalDim = wraps.getDimension();
        int[] tmp = new int[totalDim];
        int n = 0;
        for (int i = 0; i < totalDim; i++) {
            if (!dataModel.isWrapPinned(i)) tmp[n++] = i;
        }
        this.perm = new int[n];
        System.arraycopy(tmp, 0, this.perm, 0, n);
        this.blockSize = blockSize;
        setWeight(weight);
    }

    @Override
    public double doOperation() {
        final int n = perm.length;
        final int k = Math.min(blockSize, n);
        if (k == 0) return Double.NEGATIVE_INFINITY;
        final double lower = wraps.getBounds().getLowerLimit(0);
        final double upper = wraps.getBounds().getUpperLimit(0);

        // Fisher-Yates partial shuffle: perm[0..k-1] become k distinct uniformly
        // random elements drawn from the n non-pinned indices.
        final int[] steps = new int[k];
        for (int i = 0; i < k; i++) {
            int j = i + MathUtils.nextInt(n - i);
            int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
            int idx = perm[i];
            int step = MathUtils.nextBoolean() ? 1 : -1;
            double proposed = wraps.getParameterValue(idx) + step;
            if (proposed < lower || proposed > upper) {
                return Double.NEGATIVE_INFINITY;
            }
            steps[i] = step;
        }

        for (int i = 0; i < k; i++) {
            int idx = perm[i];
            wraps.setParameterValue(idx, wraps.getParameterValue(idx) + steps[i]);
        }
        return 0.0;
    }

    @Override
    public String getOperatorName() {
        return "vectorWrapRandomWalk(" + wraps.getId() + ", k=" + blockSize + ")";
    }

    public String getPerformanceSuggestion() { return ""; }
}
