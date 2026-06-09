/*
 * TaxonBlockWrapOperator.java
 *
 * MCMC operator for the integer wraps of a WrappedContinuousTraitDataModel.
 * Picks one taxon, picks a step in {+1, -1}, and adds 'step' to every wrap
 * entry belonging to that taxon (across all trait dimensions). Pinned wraps
 * (taxon 0 gauge fix and missing-data coords) are skipped. If any non-pinned
 * entry of the chosen taxon would leave the parameter's bounds, the proposal
 * is rejected (logq = -infinity).
 *
 * Proposal is symmetric (taxon uniform; +1/-1 equiprobable), so the log
 * Hastings ratio is 0 for any accepted proposal. The reverse proposal flips
 * the same taxon's wraps by -step and reaches the original state with the
 * same probability density.
 *
 * Complements SubtreeWrapOperator (which couples across leaves of a clade
 * within one (residue, dim)) by coupling across (residue, dim) within one
 * leaf, which is informed by the trait covariance Sigma's off-diagonal.
 *
 * @author Samir Bhatt
 */

package dr.evomodel.operators;

import dr.evomodel.treedatalikelihood.continuous.WrappedContinuousTraitDataModel;
import dr.inference.model.Parameter;
import dr.inference.operators.SimpleMCMCOperator;
import dr.math.MathUtils;

public class TaxonBlockWrapOperator extends SimpleMCMCOperator {

    private final WrappedContinuousTraitDataModel dataModel;
    private final Parameter wraps;
    private final int numTraits;
    private final int dimTrait;
    private final int numTaxa;

    public TaxonBlockWrapOperator(WrappedContinuousTraitDataModel dataModel,
                                  double weight) {
        this.dataModel = dataModel;
        this.wraps = dataModel.getWrapsParameter();
        this.numTraits = dataModel.getTraitCount();
        this.dimTrait = dataModel.getTraitDimension();
        int stride = numTraits * dimTrait;
        this.numTaxa = wraps.getDimension() / stride;
        setWeight(weight);
    }

    @Override
    public double doOperation() {
        final int taxon = MathUtils.nextInt(numTaxa);
        final int step = MathUtils.nextBoolean() ? 1 : -1;
        final int base = taxon * numTraits * dimTrait;
        final double lower = wraps.getBounds().getLowerLimit(0);
        final double upper = wraps.getBounds().getUpperLimit(0);

        int[] toFlip = new int[numTraits * dimTrait];
        int nFlip = 0;
        for (int i = 0; i < numTraits; i++) {
            for (int j = 0; j < dimTrait; j++) {
                int idx = base + i * dimTrait + j;
                if (dataModel.isWrapPinned(idx)) continue;
                double proposed = wraps.getParameterValue(idx) + step;
                if (proposed < lower || proposed > upper) {
                    return Double.NEGATIVE_INFINITY;
                }
                toFlip[nFlip++] = idx;
            }
        }
        if (nFlip == 0) return Double.NEGATIVE_INFINITY;  // all pinned (taxon 0)

        for (int i = 0; i < nFlip; i++) {
            int idx = toFlip[i];
            wraps.setParameterValue(idx, wraps.getParameterValue(idx) + step);
        }
        return 0.0;
    }

    @Override
    public String getOperatorName() {
        return "taxonBlockWrap(" + wraps.getId() + ")";
    }

    public String getPerformanceSuggestion() { return ""; }
}
