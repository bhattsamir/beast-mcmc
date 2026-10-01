package dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.math.MathUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Positive, log-space reference for torus pruning. Unlike an FFT followed by a
 * positive floor, this retains Gaussian tails below floating-point density range. Quadrature is
 * checked on two fixed-seed independently shifted meshes (per node and axis),
 * and refined until the log likelihood agrees
 * and the mesh resolves every conditional Gaussian component. Unsupported costs
 * raise AccuracyException; numerical failure is never a likelihood of -infinity.
 */
public final class TorusLogSpacePruning {
    public static final double LOG_TOLERANCE = 1.0e-7;
    private static final double SERIES_TOLERANCE = 1.0e-12;
    private static final double TWO_PI = 2.0 * Math.PI;
    private static final int MAX_CELLS = 16384;
    private static final long MAX_CONVOLUTION_PAIRS = 80000000L;
    private static final int MAX_COMPONENTS = 50000;
    private static final int MAX_COUNTS = 10000;
    private static final int MAX_IMAGES = 200000;
    private static final class RefineGrid extends RuntimeException { }

    public static final class AccuracyException extends IllegalStateException {
        public AccuracyException(String message) {
            super("Torus likelihood accuracy could not be established: " + message);
        }
    }

    /** Effective branch length, including any branch-rate model. */
    public interface BranchLengths { double get(NodeRef child); }

    /** Immutable accepted quadrature state; the likelihood cache snapshots references. */
    public static final class Result {
        public final double logLikelihood;
        public final int gridSize;
        private final int dim;
        private final int cells;
        private final int[][] coordinates;
        private final double[][] partials;
        private final Map<Double, double[]> transitions;
        private final Tree tree;
        private final BranchLengths lengths;

        private Result(double logLikelihood, int gridSize, int dim, int[][] coordinates,
                       double[][] partials, Map<Double, double[]> transitions,
                       Tree tree, BranchLengths lengths) {
            this.logLikelihood = logLikelihood;
            this.gridSize = gridSize;
            this.dim = dim;
            this.cells = coordinates.length;
            this.coordinates = coordinates;
            this.partials = partials;
            this.transitions = transitions;
            this.tree = tree;
            this.lengths = lengths;
        }

        /** Joint ancestral draw from exactly the log partials used for integration. */
        public void sample(double[][] output) {
            NodeRef root = tree.getRoot();
            int rootCell = sampleLogWeights(partials[root.getNumber()]);
            output[root.getNumber()] = angles(rootCell);
            sampleChildren(root, rootCell, output);
        }

        private void sampleChildren(NodeRef parent, int parentCell, double[][] output) {
            for (int k = 0; k < tree.getChildCount(parent); k++) {
                NodeRef child = tree.getChild(parent, k);
                if (tree.isExternal(child)) {
                    output[child.getNumber()] = null;
                    continue;
                }
                double[] transition = transitions.get(lengths.get(child));
                double[] weights = partials[child.getNumber()].clone();
                for (int cell = 0; cell < cells; cell++) {
                    weights[cell] += transition[displacement(coordinates[cell],
                            coordinates[parentCell], gridSize)];
                }
                int childCell = sampleLogWeights(weights);
                output[child.getNumber()] = angles(childCell);
                sampleChildren(child, childCell, output);
            }
        }

        private double[] angles(int cell) {
            double[] out = new double[dim];
            for (int ax = 0; ax < dim; ax++)
                out[ax] = wrap(TWO_PI * coordinates[cell][ax] / gridSize);
            return out;
        }
    }

    public static Result calculate(Tree tree, BranchLengths lengths, double[][] observations,
                                   double[][] sigma, double rate, double[] weights,
                                   double[][] displacements, double[][][] jumpCovariances) {
        return calculate(tree, lengths, observations, sigma, rate, weights, displacements,
                jumpCovariances, null);
    }

    /**
     * As above, with a wrapped-Gaussian tip observation error of covariance sigmaEps
     * (radians^2; null = the historical Dirac tip, taken bit-for-bit).
     *
     * A tip message becomes the branch transition convolved with N(0, sigmaEps), and on
     * the torus that convolution just adds sigmaEps to the covariance of every Poisson
     * jump-count component -- i.e. to the base covariance time*Sigma. Nothing else in the
     * reference changes.
     *
     * The grid-refinement bound below uses the SAME covariance: a tip branch is bounded on
     * time*Sigma + sigmaEps, not on time*Sigma alone. Bounding it on the Dirac assumption was
     * safe in the sense of never being too coarse, but it made the reference demand a grid fine
     * enough to resolve a delta function -- so -Dtorus.validate=true and ancestral-state
     * reconstruction threw AccuracyException at exactly the sharp terminal branches that
     * Sigma_eps exists to make tractable.
     */
    public static Result calculate(Tree tree, BranchLengths lengths, double[][] observations,
                                   double[][] sigma, double rate, double[] weights,
                                   double[][] displacements, double[][][] jumpCovariances,
                                   double[][] sigmaEps) {
        final int dim = sigma.length;
        if (dim < 1 || dim > 3) throw new AccuracyException("reference supports dimensions 1 to 3");
        final Gaussian diffusion = new Gaussian(new double[dim], sigma);
        double minimumJumpVariance = 0.0;
        if (rate > 0.0 && jumpCovariances != null && weights != null
                && weights.length == jumpCovariances.length) {
            minimumJumpVariance = Double.POSITIVE_INFINITY;
            for (int k = 0; k < weights.length; k++)
                if (weights[k] > 0.0) minimumJumpVariance = Math.min(minimumJumpVariance,
                        minimumJumpVariance(jumpCovariances[k], dim));
        }
        boolean[] informative = new boolean[tree.getNodeCount()];
        markInformative(tree.getRoot(), tree, observations, informative);
        double maxConditionalPrecision = 0.0;
        for (int i = 0; i < tree.getNodeCount(); i++) {
            NodeRef node = tree.getNode(i);
            if (tree.isExternal(node) || !informative[node.getNumber()]) continue;
            double inverseTime = node == tree.getRoot() ? 0.0 : inverseSharpTime(
                    lengths.get(node), diffusion.precisionNorm, rate, minimumJumpVariance, dim);
            // Tip branches only. sigmaEps enters the message at exactly one place (see prune),
            // so it enters the bound at exactly one place too; their precisions are already in
            // absolute units and are accumulated separately from the 1/time terms.
            double tipPrecision = 0.0;
            for (int c = 0; c < tree.getChildCount(node); c++) {
                NodeRef child = tree.getChild(node, c);
                if (!informative[child.getNumber()]) continue;
                if (sigmaEps != null && tree.isExternal(child)) {
                    tipPrecision += sharpPrecision(lengths.get(child), sigma, sigmaEps,
                            rate, minimumJumpVariance, dim);
                } else {
                    inverseTime += inverseSharpTime(lengths.get(child),
                            diffusion.precisionNorm, rate, minimumJumpVariance, dim);
                }
            }
            double conditionalPrecision = inverseTime * diffusion.precisionNorm;
            // Adding nothing must not perturb the Dirac arithmetic by even one ulp: with
            // sigmaEps == null this is the original expression, unchanged.
            if (tipPrecision > 0.0) conditionalPrecision += tipPrecision;
            maxConditionalPrecision = Math.max(maxConditionalPrecision, conditionalPrecision);
        }
        final double conditionalWidth = 1.0 / Math.sqrt(maxConditionalPrecision);
        int grid = 8;
        while (TWO_PI / grid > 1.25 * conditionalWidth) {
            grid *= 2;
            checkCost(tree, dim, grid);
        }
        for (;;) {
            checkCost(tree, dim, grid);
            try {
                Result a = evaluate(tree, lengths, observations, sigma, rate, weights,
                        displacements, jumpCovariances, grid, 0.0, informative, sigmaEps);
                Result b = evaluate(tree, lengths, observations, sigma, rate, weights,
                        displacements, jumpCovariances, grid, 0.5, informative, sigmaEps);
                Result c = evaluate(tree, lengths, observations, sigma, rate, weights,
                        displacements, jumpCovariances, grid, 0.3819660112501051, informative, sigmaEps);
                if (Double.isFinite(a.logLikelihood) && Double.isFinite(b.logLikelihood)
                        && Double.isFinite(c.logLikelihood)
                        && Math.max(Math.abs(a.logLikelihood - b.logLikelihood),
                        Math.abs(a.logLikelihood - c.logLikelihood)) <= LOG_TOLERANCE) return a;
            } catch (RefineGrid needsRefinement) {
                // A sampled internal transition has not yet integrated to one.
            }
            grid *= 2;
        }
    }

    private static boolean markInformative(NodeRef node, Tree tree, double[][] observations,
                                           boolean[] informative) {
        if (tree.isExternal(node)) return informative[node.getNumber()] = observations[node.getNumber()] != null;
        boolean any = false;
        for (int c = 0; c < tree.getChildCount(node); c++)
            any |= markInformative(tree.getChild(node, c), tree, observations, informative);
        return informative[node.getNumber()] = any;
    }

    /**
     * Precision scale of a branch whose message covariance is time*Sigma + sigmaEps, or 0 when
     * that branch is provably flat -- the same two uniformity tests inverseSharpTime applies,
     * evaluated on the actual branch covariance.
     *
     * inverseSharpTime is this function specialised to sigmaEps == 0: there the branch precision
     * norm is exactly precisionNorm(Sigma)/time, which is why it can return 1/time and let the
     * caller multiply through. With sigmaEps added that factorisation is gone, so this returns
     * the branch precision norm itself.
     */
    private static double sharpPrecision(double time, double[][] sigma, double[][] sigmaEps,
                                         double rate, double minimumJumpVariance, int dim) {
        double[][] base = new double[dim][dim];
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                base[i][j] = time * sigma[i][j] + (sigmaEps == null ? 0.0 : sigmaEps[i][j]);
        double branchPrecisionNorm = new Gaussian(new double[dim], base).precisionNorm;
        double logModeBound = dim * Math.log1p(Math.sqrt(TWO_PI * branchPrecisionNorm));
        if (minimumJumpVariance > 0.0 && logModeBound
                + rate * time * Math.expm1(-0.5 * minimumJumpVariance) < Math.log(SERIES_TOLERANCE)) return 0.0;
        double q = Math.exp(-0.5 / branchPrecisionNorm);
        if (Math.expm1(dim * Math.log1p(2.0 * q / (1.0 - q * q * q))) < SERIES_TOLERANCE) return 0.0;
        return branchPrecisionNorm;
    }

    private static double inverseSharpTime(double time, double precisionNorm, double rate,
                                            double minimumJumpVariance, int dim) {
        // A proved uniform transition has no narrow conditional peak even when
        // the diffusion coefficient alone is tiny (e.g. a large jump rate).
        double logModeBound = dim * Math.log1p(Math.sqrt(TWO_PI * precisionNorm / time));
        if (minimumJumpVariance > 0.0 && logModeBound
                + rate * time * Math.expm1(-0.5 * minimumJumpVariance) < Math.log(SERIES_TOLERANCE)) return 0.0;
        double q = Math.exp(-0.5 * time / precisionNorm);
        if (Math.expm1(dim * Math.log1p(2.0 * q / (1.0 - q * q * q))) < SERIES_TOLERANCE) return 0.0;
        return 1.0 / time;
    }

    private static void checkCost(Tree tree, int dim, int grid) {
        long cells = 1;
        for (int i = 0; i < dim; i++) cells *= grid;
        if (cells > MAX_CELLS || (tree.getInternalNodeCount() > 1
                && cells * cells > MAX_CONVOLUTION_PAIRS)) {
            throw new AccuracyException("required grid " + grid + "^" + dim
                    + " exceeds the positive reference's resource cap; use a more capable quadrature backend");
        }
    }

    private static Result evaluate(Tree tree, BranchLengths lengths, double[][] observations,
                                   double[][] sigma, double rate, double[] weights,
                                   double[][] displacements, double[][][] jumpCovariances,
                                   int grid, double offset, boolean[] informative,
                                   double[][] sigmaEps) {
        int dim = sigma.length, cells = 1;
        for (int i = 0; i < dim; i++) cells *= grid;
        int[][] coordinates = new int[cells][dim];
        double[][] points = new double[cells][dim];
        for (int cell = 0; cell < cells; cell++) {
            int index = cell;
            for (int ax = dim - 1; ax >= 0; ax--) {
                coordinates[cell][ax] = index % grid;
                points[cell][ax] = TWO_PI * coordinates[cell][ax] / grid;
                index /= grid;
            }
        }
        double[][] offsets = new double[tree.getNodeCount()][dim];
        if (offset != 0.0) {
            java.util.Random random = new java.util.Random(0x6a09e667f3bcc909L ^ Double.doubleToLongBits(offset));
            for (int node = 0; node < offsets.length; node++)
                for (int ax = 0; ax < dim; ax++) offsets[node][ax] = random.nextDouble();
        }
        double[][] partials = new double[tree.getNodeCount()][];
        Map<Double, double[]> transitions = new HashMap<Double, double[]>();
        prune(tree.getRoot(), tree, lengths, observations, sigma, rate, weights,
                displacements, jumpCovariances, grid, coordinates, points, partials, transitions, offsets, offset != 0.0, informative, sigmaEps);
        double ll = logSum(partials[tree.getRoot().getNumber()]) - Math.log(cells);
        return new Result(ll, grid, dim, coordinates, partials, transitions, tree, lengths);
    }

    private static void prune(NodeRef node, Tree tree, BranchLengths lengths,
                              double[][] observations, double[][] sigma, double rate,
                              double[] weights, double[][] displacements, double[][][] jumpCovariances,
                              int grid, int[][] coordinates, double[][] points,
                              double[][] partials, Map<Double, double[]> transitions,
                              double[][] offsets, boolean shifted, boolean[] informative,
                              double[][] sigmaEps) {
        int cells = points.length, dim = sigma.length;
        double[] partial = new double[cells];
        for (int c = 0; c < tree.getChildCount(node); c++) {
            NodeRef child = tree.getChild(node, c);
            double time = lengths.get(child);
            double[] message;
            if (tree.isExternal(child)) {
                double[] y = observations[child.getNumber()];
                if (y == null) continue;
                double[][] delta = new double[cells][dim];
                for (int cell = 0; cell < cells; cell++)
                    for (int ax = 0; ax < dim; ax++) delta[cell][ax] = y[ax] - points[cell][ax]
                            - TWO_PI * offsets[node.getNumber()][ax] / grid;
                // TIP: the observation error convolves into the branch transition here, and
                // ONLY here. Internal branches carry no tip error.
                message = logTransition(delta, sigma, time, rate, weights, displacements,
                        jumpCovariances, sigmaEps);
            } else {
                prune(child, tree, lengths, observations, sigma, rate, weights, displacements,
                        jumpCovariances, grid, coordinates, points, partials, transitions, offsets, shifted, informative, sigmaEps);
                // Different node shifts change the branch displacement mesh. A
                // shared origin shift would leave internal aliasing invisible.
                double[] transition = shifted ? null : transitions.get(time);
                if (transition == null) {
                    double[][] delta = new double[cells][dim];
                    for (int cell = 0; cell < cells; cell++)
                        for (int ax = 0; ax < dim; ax++)
                            delta[cell][ax] = TWO_PI * (coordinates[cell][ax]
                                    + offsets[child.getNumber()][ax] - offsets[node.getNumber()][ax]) / grid;
                    transition = logTransition(delta, sigma, time, rate, weights, displacements, jumpCovariances);
                    transitions.put(time, transition);
                }
                // An unobserved subtree integrates to exactly one. Keep its flat
                // partial and transition for ancestral draws, but do not introduce
                // transition-grid mass error into the observed-data likelihood.
                if (!informative[child.getNumber()]) continue;
                double logMass = logSum(transition) + dim * Math.log(TWO_PI / grid);
                if (Math.abs(logMass) > LOG_TOLERANCE / Math.max(1, tree.getNodeCount())) throw new RefineGrid();
                message = propagate(partials[child.getNumber()], transition, coordinates, grid, dim);
            }
            for (int cell = 0; cell < cells; cell++) partial[cell] += message[cell];
        }
        partials[node.getNumber()] = partial;
    }

    private static double[] propagate(double[] child, double[] transition,
                                      int[][] coordinates, int grid, int dim) {
        int cells = child.length;
        double[] output = new double[cells];
        double logCellVolume = dim * Math.log(TWO_PI / grid);
        // Two passes per parent avoid an exp/log pair for every summand. No value is
        // floored; exponentiation occurs only after subtracting that parent's maximum.
        for (int parent = 0; parent < cells; parent++) {
            double maximum = Double.NEGATIVE_INFINITY;
            for (int cell = 0; cell < cells; cell++) {
                double value = child[cell] + transition[displacement(coordinates[cell], coordinates[parent], grid)];
                maximum = Math.max(maximum, value);
            }
            double sum = 0.0;
            for (int cell = 0; cell < cells; cell++) {
                double value = child[cell] + transition[displacement(coordinates[cell], coordinates[parent], grid)];
                sum += Math.exp(value - maximum);
            }
            output[parent] = maximum + Math.log(sum) + logCellVolume;
        }
        return output;
    }

    private static int displacement(int[] child, int[] parent, int grid) {
        int index = 0;
        for (int ax = 0; ax < child.length; ax++) {
            int delta = child[ax] - parent[ax];
            if (delta < 0) delta += grid;
            index = grid * index + delta;
        }
        return index;
    }

    /** Log density of child-parent under BM plus directed Gaussian-mixture jumps. */
    public static double[] logTransition(double[][] delta, double[][] sigma, double time,
                                         double rate, double[] weights, double[][] displacements,
                                         double[][][] jumpCovariances) {
        return logTransition(delta, sigma, time, rate, weights, displacements, jumpCovariances, null);
    }

    /**
     * As above, additionally convolved with a wrapped Gaussian of covariance sigmaEps
     * (null = no convolution, bit-for-bit the original). Convolution adds sigmaEps to the
     * base covariance time*Sigma, which every jump-count component of the Poisson mixture
     * inherits, so this single addition is the whole change.
     */
    public static double[] logTransition(double[][] delta, double[][] sigma, double time,
                                         double rate, double[] weights, double[][] displacements,
                                         double[][][] jumpCovariances, double[][] sigmaEps) {
        if (!(time > 0.0) || !Double.isFinite(time) || rate < 0.0 || !Double.isFinite(rate))
            throw new IllegalArgumentException("branch time must be positive and jump rate nonnegative, both finite");
        int dim = sigma.length;
        double[][] base = new double[dim][dim];
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++) base[i][j] = time * sigma[i][j];
        if (sigmaEps != null)
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++) base[i][j] += sigmaEps[i][j];
        Gaussian baseGaussian = new Gaussian(new double[dim], base);
        boolean hasJump = rate > 0.0 && jumpCovariances != null;
        if (!hasJump) return baseGaussian.logDensity(delta);
        if (weights == null || displacements == null || weights.length == 0
                || weights.length != displacements.length || weights.length != jumpCovariances.length)
            throw new IllegalArgumentException("jump mixture dimensions disagree");
        double weightSum = 0.0, minJumpVariance = Double.POSITIVE_INFINITY;
        for (int k = 0; k < weights.length; k++) {
            if (weights[k] < 0.0 || !Double.isFinite(weights[k]))
                throw new IllegalArgumentException("jump weights must be finite and nonnegative");
            weightSum += weights[k];
            if (displacements[k].length != dim) throw new IllegalArgumentException("jump displacement dimension mismatch");
            for (double angle : displacements[k])
                if (!Double.isFinite(angle)) throw new IllegalArgumentException("jump displacements must be finite");
            double minimumVariance = minimumJumpVariance(jumpCovariances[k], dim);
            if (weights[k] > 0.0) minJumpVariance = Math.min(minJumpVariance, minimumVariance);
        }
        if (Math.abs(weightSum - 1.0) > 1.0e-10)
            throw new IllegalArgumentException("jump mixture weights must sum to one");
        double mean = rate * time;
        double logMean = Math.log(rate) + Math.log(time);
        double logVolume = dim * Math.log(TWO_PI);
        // A certified Fourier bound can establish uniformity without enumerating a
        // huge Poisson mixture. It bounds all nonzero modes, independent of phases.
        double logModeBound = dim * Math.log1p(Math.sqrt(TWO_PI * baseGaussian.precisionNorm));
        double logUniformError = logModeBound + mean * Math.expm1(-0.5 * minJumpVariance);
        if (logUniformError < Math.log(SERIES_TOLERANCE)) {
            double[] uniform = new double[delta.length];
            Arrays.fill(uniform, -logVolume);
            return uniform;
        }
        MixtureAccumulator accumulator = new MixtureAccumulator(delta, base, weights,
                displacements, jumpCovariances, mean, logMean);
        double logPoisson = -mean;
        double logUpperDensity = logModeBound - logVolume;
        for (int n = 0; n <= MAX_COUNTS; n++) {
            accumulator.enumerate(0, n, new int[weights.length], -mean);
            double minimum = Double.POSITIVE_INFINITY;
            for (double value : accumulator.values) minimum = Math.min(minimum, value);
            double nextLogPoisson = logPoisson + logMean - Math.log(n + 1.0);
            if (n + 2.0 > mean) {
                double logTail = nextLogPoisson - Math.log1p(-mean / (n + 2.0));
                if (logTail + logUpperDensity <= minimum + Math.log(SERIES_TOLERANCE))
                    return accumulator.values;
            }
            logPoisson = nextLogPoisson;
        }
        throw new AccuracyException("Poisson count truncation needs more than " + MAX_COUNTS + " terms");
    }

    private static final class MixtureAccumulator {
        final double[][] delta, base, displacements;
        final double[][][] covariances;
        final double[] weights, values;
        final double mean, logMean;
        int components = 0;

        MixtureAccumulator(double[][] delta, double[][] base, double[] weights,
                           double[][] displacements, double[][][] covariances, double mean, double logMean) {
            this.delta = delta; this.base = base; this.weights = weights;
            this.displacements = displacements; this.covariances = covariances; this.mean = mean;
            this.logMean = logMean;
            this.values = new double[delta.length];
            Arrays.fill(values, Double.NEGATIVE_INFINITY);
        }

        void enumerate(int k, int remaining, int[] counts, double logWeight) {
            if (k == weights.length) {
                if (remaining != 0) return;
                if (++components > MAX_COMPONENTS)
                    throw new AccuracyException("Gaussian jump mixture exceeds " + MAX_COMPONENTS + " components");
                int dim = base.length;
                double[] shift = new double[dim];
                double[][] covariance = new double[dim][dim];
                for (int i = 0; i < dim; i++) {
                    System.arraycopy(base[i], 0, covariance[i], 0, dim);
                    for (int component = 0; component < counts.length; component++) {
                        shift[i] += counts[component] * displacements[component][i];
                        for (int j = 0; j < dim; j++)
                            covariance[i][j] += counts[component] * covariances[component][i][j];
                    }
                }
                double[] density = new Gaussian(shift, covariance).logDensity(delta);
                for (int i = 0; i < values.length; i++) values[i] = logAdd(values[i], logWeight + density[i]);
                return;
            }
            double componentLogWeight = 0.0;
            int first = k == weights.length - 1 ? remaining : 0;
            for (int count = 0; count <= remaining; count++) {
                if (count > 0) {
                    if (weights[k] == 0.0) break;
                    componentLogWeight += logMean + Math.log(weights[k]) - Math.log(count);
                }
                if (count < first) continue;
                counts[k] = count;
                enumerate(k + 1, remaining - count, counts, logWeight + componentLogWeight);
            }
            counts[k] = 0;
        }
    }

    // Jacobi diagonalisation validates semidefinite jump covariances too (a
    // deterministic jump is legitimate). The diffusion covariance stays strictly
    // positive, so every Gaussian in the Poisson sum has a proper density.
    private static double minimumJumpVariance(double[][] covariance, int dim) {
        if (covariance.length != dim) throw new IllegalArgumentException("jump covariance dimension mismatch");
        for (double[] row : covariance)
            if (row.length != dim) throw new IllegalArgumentException("jump covariance dimension mismatch");
        double[][] a = new double[dim][dim];
        double norm = 0.0;
        for (int i = 0; i < dim; i++) {
            double rowSum = 0.0;
            for (int j = 0; j < dim; j++) {
                double value = covariance[i][j];
                if (!Double.isFinite(value) || Math.abs(value - covariance[j][i]) > 1e-12 * Math.max(1.0, Math.abs(value)))
                    throw new IllegalArgumentException("jump covariance must be finite and symmetric");
                a[i][j] = value;
                rowSum += Math.abs(value);
            }
            norm = Math.max(norm, rowSum);
        }
        for (int iteration = 0; iteration < 32; iteration++) {
            int p = 0, q = 0;
            double largest = 0.0;
            for (int i = 0; i < dim; i++)
                for (int j = i + 1; j < dim; j++)
                    if (Math.abs(a[i][j]) > largest) { largest = Math.abs(a[i][j]); p = i; q = j; }
            if (largest <= 1e-15 * Math.max(norm, Double.MIN_NORMAL)) break;
            double angle = 0.5 * Math.atan2(2.0 * a[p][q], a[q][q] - a[p][p]);
            double c = Math.cos(angle), ss = Math.sin(angle);
            double app = a[p][p], aqq = a[q][q], apq = a[p][q];
            for (int k = 0; k < dim; k++) {
                if (k == p || k == q) continue;
                double akp = a[k][p], akq = a[k][q];
                a[k][p] = a[p][k] = c * akp - ss * akq;
                a[k][q] = a[q][k] = ss * akp + c * akq;
            }
            a[p][p] = c * c * app - 2.0 * c * ss * apq + ss * ss * aqq;
            a[q][q] = ss * ss * app + 2.0 * c * ss * apq + c * c * aqq;
            a[p][q] = a[q][p] = 0.0;
        }
        double minimum = Double.POSITIVE_INFINITY;
        for (int i = 0; i < dim; i++) {
            double bound = a[i][i];
            for (int j = 0; j < dim; j++) if (j != i) bound -= Math.abs(a[i][j]);
            minimum = Math.min(minimum, bound);
        }
        if (minimum < -1e-12 * Math.max(norm, Double.MIN_NORMAL))
            throw new IllegalArgumentException("jump covariance must be positive semidefinite");
        return Math.max(0.0, minimum - 32.0 * Math.ulp(norm));
    }

    /** Winding sum with a conservative bound for all images outside a growing box. */
    private static final class Gaussian {
        final int dim;
        final double[] mean;
        final double[][] precision;
        final double logNormalization, covarianceNorm, precisionNorm;

        Gaussian(double[] mean, double[][] covariance) {
            this.dim = covariance.length;
            if (mean.length != dim) throw new IllegalArgumentException("Gaussian mean dimension mismatch");
            this.mean = mean.clone();
            double[][] chol = new double[dim][dim];
            double logDet = 0.0, covarianceBound = 0.0;
            for (int i = 0; i < dim; i++) {
                if (covariance[i].length != dim || !Double.isFinite(mean[i]))
                    throw new IllegalArgumentException("invalid Gaussian dimensions or mean");
                double row = 0.0;
                for (int j = 0; j < dim; j++) {
                    if (!Double.isFinite(covariance[i][j])
                            || Math.abs(covariance[i][j] - covariance[j][i]) > 1e-10 * Math.max(1.0, Math.abs(covariance[i][j])))
                        throw new IllegalArgumentException("covariance must be finite and symmetric");
                    row += Math.abs(covariance[i][j]);
                }
                covarianceBound = Math.max(covarianceBound, row);
                for (int j = 0; j <= i; j++) {
                    double value = covariance[i][j];
                    for (int k = 0; k < j; k++) value -= chol[i][k] * chol[j][k];
                    if (i == j) {
                        if (!(value > 0.0)) throw new IllegalArgumentException("covariance must be positive definite");
                        chol[i][j] = Math.sqrt(value);
                        logDet += Math.log(value);
                    } else chol[i][j] = value / chol[j][j];
                }
            }
            double[][] inverseChol = new double[dim][dim];
            for (int column = 0; column < dim; column++) {
                for (int i = 0; i < dim; i++) {
                    double value = i == column ? 1.0 : 0.0;
                    for (int j = 0; j < i; j++) value -= chol[i][j] * inverseChol[j][column];
                    inverseChol[i][column] = value / chol[i][i];
                }
            }
            this.precision = new double[dim][dim];
            double precisionBound = 0.0;
            for (int i = 0; i < dim; i++) {
                double row = 0.0;
                for (int j = 0; j < dim; j++) {
                    for (int k = 0; k < dim; k++) precision[i][j] += inverseChol[k][i] * inverseChol[k][j];
                    row += Math.abs(precision[i][j]);
                }
                precisionBound = Math.max(precisionBound, row);
            }
            this.precisionNorm = precisionBound;
            this.covarianceNorm = covarianceBound;
            this.logNormalization = -0.5 * (dim * Math.log(TWO_PI) + logDet);
        }

        double[] logDensity(double[][] points) {
            double[] sum = new double[points.length];
            Arrays.fill(sum, Double.NEGATIVE_INFINITY);
            // Uniform approximation with a relative bound on every nonzero mode.
            double q = Math.exp(-0.5 / precisionNorm);
            double relativeBound = Math.expm1(dim * Math.log1p(2.0 * q / (1.0 - q * q * q)));
            if (relativeBound < SERIES_TOLERANCE) {
                Arrays.fill(sum, -dim * Math.log(TWO_PI));
                return sum;
            }
            double[][] residuals = new double[points.length][dim];
            for (int p = 0; p < points.length; p++) {
                if (points[p].length != dim) throw new IllegalArgumentException("point dimension mismatch");
                for (int ax = 0; ax < dim; ax++) {
                    if (!Double.isFinite(points[p][ax])) throw new IllegalArgumentException("angles must be finite");
                    residuals[p][ax] = wrap(points[p][ax] - mean[ax]);
                }
            }
            double a = 2.0 * Math.PI * Math.PI / covarianceNorm;
            double logAxisTotal = Math.log1p(Math.exp(logAxisTail(a, 0)));
            for (int radius = 0; ; radius++) {
                long imageCount = 1;
                for (int ax = 0; ax < dim; ax++) imageCount *= 2L * radius + 1L;
                if (imageCount > MAX_IMAGES)
                    throw new AccuracyException("wrapped-normal winding sum exceeds " + MAX_IMAGES + " images");
                addShell(0, radius, false, new int[dim], residuals, sum);
                double minimum = Double.POSITIVE_INFINITY;
                for (double value : sum) minimum = Math.min(minimum, value);
                double logTail = logNormalization + Math.log(dim) + logAxisTail(a, radius)
                        + (dim - 1) * logAxisTotal;
                if (logTail <= minimum + Math.log(SERIES_TOLERANCE)) return sum;
            }
        }

        private static double logAxisTail(double a, int radius) {
            return Math.log(2.0) - a * (radius + 0.5) * (radius + 0.5)
                    - Math.log(-Math.expm1(-2.0 * a * (radius + 1.0)));
        }

        private void addShell(int axis, int radius, boolean shell, int[] winding,
                              double[][] residuals, double[] sum) {
            if (axis == dim) {
                if (!shell && radius != 0) return;
                double[] residual = new double[dim];
                for (int p = 0; p < residuals.length; p++) {
                    for (int ax = 0; ax < dim; ax++) residual[ax] = residuals[p][ax] + TWO_PI * winding[ax];
                    double quadratic = 0.0;
                    for (int i = 0; i < dim; i++)
                        for (int j = 0; j < dim; j++) quadratic += residual[i] * precision[i][j] * residual[j];
                    sum[p] = logAdd(sum[p], logNormalization - 0.5 * quadratic);
                }
                return;
            }
            for (int image = -radius; image <= radius; image++) {
                winding[axis] = image;
                addShell(axis + 1, radius, shell || Math.abs(image) == radius, winding, residuals, sum);
            }
        }
    }

    private static double wrap(double angle) {
        return angle - TWO_PI * Math.floor((angle + Math.PI) / TWO_PI);
    }

    private static double logAdd(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) return b;
        if (b == Double.NEGATIVE_INFINITY) return a;
        double maximum = Math.max(a, b);
        return maximum + Math.log1p(Math.exp(Math.min(a, b) - maximum));
    }

    private static double logSum(double[] values) {
        double maximum = Double.NEGATIVE_INFINITY;
        for (double value : values) maximum = Math.max(maximum, value);
        double sum = 0.0;
        for (double value : values) sum += Math.exp(value - maximum);
        return maximum + Math.log(sum);
    }

    private static int sampleLogWeights(double[] values) {
        double maximum = Double.NEGATIVE_INFINITY;
        for (double value : values) maximum = Math.max(maximum, value);
        double total = 0.0;
        for (double value : values) total += Math.exp(value - maximum);
        if (!(total > 0.0) || !Double.isFinite(total)) throw new AccuracyException("invalid ancestral weights");
        double target = MathUtils.nextDouble() * total;
        double cumulative = 0.0;
        for (int i = 0; i < values.length; i++) {
            cumulative += Math.exp(values[i] - maximum);
            if (cumulative > target) return i;
        }
        return values.length - 1;
    }
}
