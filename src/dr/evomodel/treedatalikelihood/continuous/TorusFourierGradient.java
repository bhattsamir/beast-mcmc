package dr.evomodel.treedatalikelihood.continuous;

import dr.inference.hmc.GradientWrtParameterProvider;
import dr.inference.model.CompoundParameter;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.Likelihood;
import dr.inference.model.MaskedParameter;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import dr.inference.model.TransformedParameter;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

/**
 * One residue's FFT likelihood score in raw coordinates
 * [precision diagonal 1, precision diagonal 2, precision correlation, jump rate].
 * Prior scores and the coordinate-transform Jacobian are supplied separately by
 * BEAST's joint gradient and HMC transform. No copy of a sampled value is made.
 */
public final class TorusFourierGradient implements GradientWrtParameterProvider {
    private final TorusFourierTreeLikelihood likelihood;
    private final int residue;
    private final CompoundParameter parameter;

    /** Java residue indices are zero based; the XML parser uses one based indices. */
    public TorusFourierGradient(TorusFourierTreeLikelihood likelihood, int residue) {
        if (likelihood == null) throw new IllegalArgumentException("A torus likelihood is required");
        if (TorusFourierTreeLikelihood.isValidateEveryEvaluation()) {
            throw new IllegalArgumentException("Torus HMC requires torus.validate=false: the gradient "
                    + "differentiates the FFT likelihood, not the adaptive reference");
        }
        if (residue < 0 || residue >= likelihood.getResidueCount()) {
            throw new IllegalArgumentException("Torus gradient residue is outside the likelihood's residue range");
        }
        MatrixParameter matrix = likelihood.getResiduePrecision(residue);
        if (!(matrix instanceof CompoundSymmetricMatrix)
                || matrix.getRowDimension() != 2 || matrix.getColumnDimension() != 2) {
            throw new IllegalArgumentException("Torus HMC requires a 2x2 compoundSymmetricMatrix precision");
        }
        CompoundSymmetricMatrix precision = (CompoundSymmetricMatrix) matrix;
        if (!precision.asCorrelation() || precision.isCholesky()) {
            throw new IllegalArgumentException("Torus HMC requires asCorrelation=true and isCholesky=false");
        }
        Parameter diagonal = precision.getDiagonalParameter();
        Parameter correlation = precision.getOffDiagonalParameter();
        if (diagonal.getDimension() != 2 || correlation.getDimension() != 1) {
            throw new IllegalArgumentException("Torus HMC requires two precision diagonals and one correlation");
        }
        Set<Parameter> dependencies = dependencySet(diagonal);
        dependencies.addAll(dependencySet(correlation));
        for (int other = 0; other < likelihood.getResidueCount(); other++) {
            if (other != residue && intersects(dependencies,
                    dependencySet(likelihood.getResiduePrecision(other)))) {
                throw new IllegalArgumentException("Torus HMC requires independent precision parameters across residues; "
                        + "shared diagonals or correlations need summed residue gradients");
            }
        }
        Parameter rates = likelihood.getJumpRateParameter();
        if (rates == null || rates.getDimension() != likelihood.getResidueCount()) {
            throw new IllegalArgumentException("Torus HMC requires one jump-rate parameter per residue; "
                    + "a shared rate across multiple residues needs a summed gradient");
        }
        if (intersects(dependencies, dependencySet(rates))) {
            throw new IllegalArgumentException("Torus HMC precision and jump-rate parameters must be independent");
        }
        double q1 = diagonal.getParameterValue(0), q2 = diagonal.getParameterValue(1);
        double rho = correlation.getParameterValue(0), rateValue = rates.getParameterValue(residue);
        if (!(q1 > 0.0) || !(q2 > 0.0) || !(rateValue > 0.0)
                || !Double.isFinite(q1) || !Double.isFinite(q2) || !Double.isFinite(rateValue)
                || !Double.isFinite(rho) || !(Math.abs(rho) < 1.0)) {
            throw new IllegalArgumentException("Torus HMC requires finite positive precision diagonals and jump rate, "
                    + "and a finite correlation strictly between -1 and 1");
        }
        this.likelihood = likelihood;
        this.residue = residue;
        ScalarRateParameter rate = new ScalarRateParameter(rates, residue);
        rate.setId("torus.rateView." + (residue + 1));
        this.parameter = new CompoundParameter("torus.residue." + (residue + 1)) {
            @Override
            public void fireParameterChangedEvent() {
                // CompoundParameter normally visits a 2D child twice. Visit each
                // underlying parameter once, then notify listeners of this block.
                doNotPropagateChangeUp = true;
                try {
                    for (Parameter child : uniqueParameters) child.fireParameterChangedEvent();
                } finally {
                    doNotPropagateChangeUp = false;
                }
                fireParameterChangedEvent(-1, ChangeType.ALL_VALUES_CHANGED);
            }
        };
        parameter.addParameter(diagonal);
        parameter.addParameter(correlation);
        parameter.addParameter(rate);
    }

    private static Set<Parameter> dependencySet(Parameter parameter) {
        Set<Parameter> seen = Collections.newSetFromMap(new IdentityHashMap<Parameter, Boolean>());
        collectDependencies(parameter, seen);
        return seen;
    }

    private static void collectDependencies(Parameter parameter, Set<Parameter> seen) {
        if (!seen.add(parameter)) return;
        if (parameter instanceof CompoundSymmetricMatrix) {
            CompoundSymmetricMatrix matrix = (CompoundSymmetricMatrix) parameter;
            collectDependencies(matrix.getDiagonalParameter(), seen);
            collectDependencies(matrix.getUntransformedOffDiagonalParameter(), seen);
        } else if (parameter instanceof CompoundParameter) {
            CompoundParameter compound = (CompoundParameter) parameter;
            for (int i = 0; i < compound.getParameterCount(); i++)
                collectDependencies(compound.getParameter(i), seen);
        } else if (parameter instanceof MaskedParameter) {
            collectDependencies(((MaskedParameter) parameter).getUnmaskedParameter(), seen);
        } else if (parameter instanceof TransformedParameter) {
            collectDependencies(((TransformedParameter) parameter).getUntransformedParameter(), seen);
        }
    }

    private static boolean intersects(Set<Parameter> a, Set<Parameter> b) {
        for (Parameter parameter : a) if (b.contains(parameter)) return true;
        return false;
    }

    @Override
    public Likelihood getLikelihood() { return likelihood; }

    @Override
    public CompoundParameter getParameter() { return parameter; }

    @Override
    public int getDimension() { return 4; }

    @Override
    public double[] getGradientLogDensity() {
        return likelihood.getResidueGradient(residue);
    }

    /**
     * A scalar alias whose bulk notification names the original vector index.
     * Stock MaskedParameter forwards a bulk notification to every rate, which
     * would dirty every residue at every HMC leapfrog position.
     */
    private static final class ScalarRateParameter extends MaskedParameter {
        private final Parameter rates;
        private final int rateIndex;

        ScalarRateParameter(Parameter rates, int rateIndex) {
            super(rates, Signaling.NO_DEPENDENT);
            this.rates = rates;
            this.rateIndex = rateIndex;
            Parameter mask = new Parameter.Default(rates.getDimension(), 0.0);
            mask.setParameterValueQuietly(rateIndex, 1.0);
            addMask(mask, true);
        }

        @Override
        public void fireParameterChangedEvent() {
            // The base vector's event comes back through MaskedParameter's
            // listener and updates this view's listeners as well.
            rates.fireParameterChangedEvent(rateIndex, ChangeType.VALUE_CHANGED);
        }

        @Override
        public void fireParameterChangedEvent(int index, ChangeType type) {
            if (index != -1 && index != 0) throw new IndexOutOfBoundsException("Scalar rate index");
            rates.fireParameterChangedEvent(rateIndex, ChangeType.VALUE_CHANGED);
        }
    }
}
