/*
 * WrappedContinuousTraitDataModel.java
 *
 * Wrapped Brownian motion via integer-valued data augmentation. Each tip's
 * observed angle theta_obs in (-pi, pi]^d is treated as a wrapped version of
 * a latent x = theta_obs + 2*pi*k in R^d, where k is an integer-valued
 * Parameter sampled by MCMC (use randomWalkIntegerOperator with windowSize=1).
 *
 * Pinned wrap entries (taxon 0 -> gauge fix; missing-data coords -> 0)
 * are ignored by getTipPartial: the wraps parameter may still hold any value
 * at those indices, but the likelihood treats them as zero.
 *
 * From the rest of the new-path pipeline's perspective the trait values are
 * standard real-valued observations, so HomogeneousDiffusionModelDelegate,
 * Wishart-Gibbs, tree operators all work unchanged.
 *
 * @author Samir Bhatt
 */

package dr.evomodel.treedatalikelihood.continuous;

import dr.evomodel.treedatalikelihood.continuous.cdi.PrecisionType;
import dr.inference.model.CompoundParameter;
import dr.inference.model.Parameter;
import dr.inference.model.Variable;
import dr.inference.model.Model;

public class WrappedContinuousTraitDataModel extends ContinuousTraitDataModel {

    private static final double TWO_PI = 2.0 * Math.PI;

    private final Parameter wraps;

    public WrappedContinuousTraitDataModel(String name,
                                           CompoundParameter parameter,
                                           boolean[] missingIndicators,
                                           boolean useMissingIndices,
                                           int dimTrait,
                                           int numTraits,
                                           PrecisionType precisionType,
                                           Parameter wraps) {
        super(name, parameter, missingIndicators, useMissingIndices, dimTrait, numTraits, precisionType);

        int expected = parameter.getParameterCount() * numTraits * dimTrait;
        if (wraps.getDimension() != expected) {
            throw new IllegalArgumentException(
                    "wraps parameter dimension " + wraps.getDimension()
                    + " does not match nTaxa(" + parameter.getParameterCount()
                    + ") * numTraits(" + numTraits + ") * dimTrait(" + dimTrait + ") = " + expected);
        }

        this.wraps = wraps;
        addVariable(wraps);
    }

    public Parameter getWrapsParameter() {
        return wraps;
    }

    /**
     * Returns true if the wrap entry at this flat index is pinned to zero by
     * gauge or missing-data choice and should NOT contribute to the partial.
     *   - taxon 0 is the gauge reference (one wrap per (residue, dim) is
     *     redundant under the conjugate-root translation symmetry).
     *   - missing-data coordinates have no observed angle to unwrap.
     * Operators may still propose changes to pinned indices; getTipPartial
     * ignores them, so the proposal becomes a no-op on the likelihood.
     */
    public boolean isWrapPinned(int wrapIndex) {
        if (wrapIndex < numTraits * dimTrait) return true;       // taxon 0 -> gauge
        boolean[] missing = getDataMissingIndicators();
        return missing != null && missing[wrapIndex];             // missing -> pinned
    }

    @Override
    public double[] getTipPartial(int taxonIndex, boolean fullyObserved) {
        double[] partial = super.getTipPartial(taxonIndex, fullyObserved);

        // The first dimTrait entries of each trait's partial are the mean.
        // offsetInc depends on the actual PrecisionType used internally:
        // - fullyObserved=true  : SCALAR
        // - fullyObserved=false : this.precisionType
        PrecisionType effective = fullyObserved ? PrecisionType.SCALAR : precisionType;
        int offsetInc = effective.getPartialsDimension(dimTrait);

        int wrapBase = taxonIndex * numTraits * dimTrait;
        for (int i = 0; i < numTraits; ++i) {
            for (int j = 0; j < dimTrait; ++j) {
                int wrapIdx = wrapBase + i * dimTrait + j;
                if (isWrapPinned(wrapIdx)) continue;
                double k = wraps.getParameterValue(wrapIdx);
                if (k != 0.0) {
                    partial[i * offsetInc + j] += TWO_PI * k;
                }
            }
        }
        return partial;
    }

    @Override
    protected void handleVariableChangedEvent(Variable variable, int index, Parameter.ChangeType type) {
        if (variable == wraps) {
            if (type == Parameter.ChangeType.VALUE_CHANGED) {
                // wraps.dimension = nTaxa * numTraits * dimTrait; same per-taxon stride as `parameter`.
                int taxonIndex = index / (numTraits * dimTrait);
                fireModelChanged(this, taxonIndex);
            } else if (type == Parameter.ChangeType.ALL_VALUES_CHANGED) {
                fireModelChanged(this);
            } else {
                throw new RuntimeException("Unhandled wraps-parameter change type: " + type);
            }
        } else {
            super.handleVariableChangedEvent(variable, index, type);
        }
    }
}
