package dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.math.MathUtils;

import java.util.Arrays;
import java.util.Random;

/**
 * Independent positive quadrature for reset (landing) jumps. Densities are
 * Gaussian winding sums; the last-reset age is integrated in real time. No
 * Fourier transform or shift-jump transition is used. An accepted result owns
 * the same log partials and transitions used by its joint ancestral sampler.
 *
 * This deliberately bounded reference can reject sharp internal transitions
 * even when the Fourier likelihood is resolved. It never renormalizes an
 * unresolved transition grid to make it appear accurate.
 */
public final class TorusLandingPruning {
    private static final double TWO_PI = 2.0 * Math.PI;
    public static final double LOG_TOLERANCE = TorusLogSpacePruning.LOG_TOLERANCE;
    private static final double TOL = LOG_TOLERANCE;
    private static final double TIME_TOL = 1.0e-11;
    private static final int MAX_CELLS = 16384;
    private static final long MAX_PAIRS = 300000000L;
    private static final long MAX_TIME_WORK = 200000000L;
    private static final int MAX_TIME_ORDER = 128;
    private static final class RefineGrid extends RuntimeException { }

    private TorusLandingPruning() { }

    /** Accepted unshifted quadrature, with immutable topology and branch lengths. */
    public static final class Result {
        public final double logLikelihood;
        public final int gridSize;
        private final int dim, cells, root;
        private final int[][] coordinates, children;
        private final double[][] partials, noReset, reset;
        private final double[] stationary;

        private Result(double ll, int grid, int dim, int root, int[][] coordinates,
                       int[][] children, double[][] partials, double[][] noReset,
                       double[][] reset, double[] stationary) {
            this.logLikelihood = ll;
            this.gridSize = grid;
            this.dim = dim;
            this.cells = coordinates.length;
            this.root = root;
            this.coordinates = coordinates;
            this.children = children;
            this.partials = partials;
            this.noReset = noReset;
            this.reset = reset;
            this.stationary = stationary;
        }

        /** Joint grid draw; external nodes are represented by null. */
        public void sample(double[][] output) {
            if (output.length != children.length)
                throw new IllegalArgumentException("ancestral output node count mismatch");
            Arrays.fill(output, null);
            double[] weights = partials[root].clone();
            for (int cell = 0; cell < cells; cell++) weights[cell] += stationary[cell];
            int selected = sampleLogWeights(weights);
            output[root] = angles(selected);
            sampleChildren(root, selected, output);
        }

        private void sampleChildren(int parent, int parentCell, double[][] output) {
            for (int child : children[parent]) {
                if (children[child].length == 0) continue;
                double[] weights = partials[child].clone();
                for (int cell = 0; cell < cells; cell++) {
                    int delta = displacement(coordinates[cell], coordinates[parentCell], gridSize);
                    weights[cell] += logAdd(noReset[child][delta], reset[child][cell]);
                }
                int selected = sampleLogWeights(weights);
                output[child] = angles(selected);
                sampleChildren(child, selected, output);
            }
        }

        private double[] angles(int cell) {
            double[] angles = new double[dim];
            for (int ax = 0; ax < dim; ax++) angles[ax] = wrap(TWO_PI * coordinates[cell][ax] / gridSize);
            return angles;
        }
    }

    public static Result calculate(Tree tree, TorusLogSpacePruning.BranchLengths lengths,
                                   double[][] observations, double[][] sigma, double rate,
                                   double[] weights, double[][] means, double[][][] covariances,
                                   double[][] sigmaEps, int maxGrid) {
        Model model = new Model(tree, lengths, observations, sigma, rate, weights, means,
                covariances, sigmaEps, maxGrid);
        for (int grid = 8; grid <= maxGrid; grid *= 2) {
            model.checkCost(grid);
            try {
                Result a = model.evaluate(grid, 0);
                Result b = model.evaluate(grid, 0x6a09e667f3bcc909L);
                Result c = model.evaluate(grid, 0xbb67ae8584caa73bL);
                if (Double.isFinite(a.logLikelihood) && Double.isFinite(b.logLikelihood)
                        && Double.isFinite(c.logLikelihood)
                        && Math.max(Math.abs(a.logLikelihood - b.logLikelihood),
                        Math.abs(a.logLikelihood - c.logLikelihood)) <= TOL) return a;
            } catch (RefineGrid underResolved) {
                // Refinement remains bounded by the caller's actual Fourier grid.
            }
        }
        throw accuracy("landing quadrature did not converge by the supplied grid cap " + maxGrid);
    }

    private static final class Model {
        final int dim, nodes, root, internalEdges;
        final int[][] children;
        final double[][] observations, sigma, eps, means;
        final double[][][] covariances;
        final double[] times, weights;
        final double rate, sigmaLower, timeScale;

        Model(Tree tree, TorusLogSpacePruning.BranchLengths lengths, double[][] observations,
              double[][] sigma, double rate, double[] weights, double[][] means,
              double[][][] covariances, double[][] eps, int maxGrid) {
            if (sigma == null || sigma.length < 1 || sigma.length > 3)
                throw accuracy("landing reference supports dimensions 1 to 3");
            if (maxGrid < 8 || maxGrid > 128 || (maxGrid & (maxGrid - 1)) != 0)
                throw new IllegalArgumentException("landing reference grid cap must be a power of two from 8 to 128");
            if (!(rate >= 0.0) || !Double.isFinite(rate))
                throw new IllegalArgumentException("landing rate must be finite and nonnegative");
            this.dim = sigma.length;
            this.sigma = copy(sigma);
            this.sigmaLower = positiveLowerBound(this.sigma, dim);
            if (eps == null) throw new IllegalArgumentException("landing reference requires positive tip error covariance");
            this.eps = copy(eps);
            positiveLowerBound(this.eps, dim);
            if (weights == null || means == null || covariances == null || weights.length == 0
                    || weights.length != means.length || weights.length != covariances.length)
                throw new IllegalArgumentException("landing mixture dimensions disagree");
            this.weights = weights.clone();
            this.means = copy(means);
            this.covariances = new double[covariances.length][][];
            double total = 0.0, minVariance = Double.POSITIVE_INFINITY;
            for (int k = 0; k < weights.length; k++) {
                if (weights[k] < 0.0 || !Double.isFinite(weights[k]))
                    throw new IllegalArgumentException("landing weights must be finite and nonnegative");
                if (means[k].length != dim) throw new IllegalArgumentException("landing mean dimension mismatch");
                for (double value : means[k]) if (!Double.isFinite(value))
                    throw new IllegalArgumentException("landing means must be finite");
                this.covariances[k] = copy(covariances[k]);
                // A singular landing basin can have an unbounded stationary density.
                // Such cases require a different quadrature backend, not a hidden floor.
                double lower;
                try { lower = positiveLowerBound(this.covariances[k], dim); }
                catch (IllegalArgumentException invalid) {
                    throw accuracy("positive landing reference requires positive definite basin covariances: "
                            + invalid.getMessage());
                }
                if (weights[k] > 0.0) minVariance = Math.min(minVariance, lower);
                total += weights[k];
            }
            if (Math.abs(total - 1.0) > 1.0e-10)
                throw new IllegalArgumentException("landing mixture weights must sum to one");
            // Match the kernel's normalization for a sum within the input tolerance.
            for (int k = 0; k < this.weights.length; k++) this.weights[k] /= total;
            this.rate = rate;
            this.timeScale = minVariance / infinityNorm(sigma);
            this.nodes = tree.getNodeCount();
            this.root = tree.getRoot().getNumber();
            if (tree.isExternal(tree.getRoot())) throw accuracy("landing reference requires an internal root");
            if (observations.length < nodes && observations.length != tree.getExternalNodeCount())
                throw new IllegalArgumentException("landing observation node count mismatch");
            this.observations = new double[nodes][];
            this.children = new int[nodes][];
            this.times = new double[nodes];
            int edges = 0;
            for (int i = 0; i < nodes; i++) {
                NodeRef node = tree.getNode(i);
                int number = node.getNumber();
                children[number] = new int[tree.getChildCount(node)];
                for (int c = 0; c < children[number].length; c++)
                    children[number][c] = tree.getChild(node, c).getNumber();
                if (node != tree.getRoot()) {
                    times[number] = lengths.get(node);
                    if (!(times[number] > 0.0) || !Double.isFinite(times[number]))
                        throw new IllegalArgumentException("landing branch time must be positive and finite");
                    if (!tree.isExternal(node)) edges++;
                }
                if (tree.isExternal(node) && number < observations.length && observations[number] != null) {
                    if (observations[number].length != dim)
                        throw new IllegalArgumentException("landing observation dimension mismatch");
                    this.observations[number] = observations[number].clone();
                    for (double angle : observations[number]) if (!Double.isFinite(angle))
                        throw new IllegalArgumentException("landing observations must be finite or missing");
                }
            }
            this.internalEdges = edges;
        }

        void checkCost(int grid) {
            long cells = 1;
            for (int ax = 0; ax < dim; ax++) cells *= grid;
            if (cells > MAX_CELLS || cells * cells * internalEdges > MAX_PAIRS)
                throw accuracy("landing grid " + grid + "^" + dim
                        + " exceeds the bounded positive reference's convolution resource cap");
        }

        Result evaluate(int grid, long seed) {
            int cells = 1;
            for (int ax = 0; ax < dim; ax++) cells *= grid;
            int[][] coordinates = new int[cells][dim];
            for (int cell = 0; cell < cells; cell++) {
                int index = cell;
                for (int ax = dim - 1; ax >= 0; ax--) {
                    coordinates[cell][ax] = index % grid;
                    index /= grid;
                }
            }
            double[][] offsets = new double[nodes][dim];
            if (seed != 0) {
                Random random = new Random(seed);
                for (double[] offset : offsets) for (int ax = 0; ax < dim; ax++) offset[ax] = random.nextDouble();
            }
            Evaluation e = new Evaluation(this, grid, coordinates, offsets);
            e.prune(root);
            double[] stationary = rate == 0.0 ? constant(cells, -dim * Math.log(TWO_PI))
                    : timeDensity(e.points(root), Double.POSITIVE_INFINITY, null);
            e.checkMass(stationary, 0.0);
            double[] integrated = e.partials[root].clone();
            for (int cell = 0; cell < cells; cell++) integrated[cell] += stationary[cell];
            double ll = logSum(integrated) + e.logCell;
            return new Result(ll, grid, dim, root, coordinates, children, e.partials,
                    e.noReset, e.reset, stationary);
        }

        /** Positive last-reset-age integral with separately refined Gauss orders. */
        double[] timeDensity(double[][] points, double time, double[][] error) {
            if (rate == 0.0) return constant(points.length, Double.NEGATIVE_INFINITY);
            double limit = rate * time;
            boolean infinite = Double.isInfinite(limit);
            double end = infinite ? 40.0 : Math.min(limit, 40.0);
            if (!(end > 0.0)) return constant(points.length, Double.NEGATIVE_INFINITY);
            for (;;) {
                double[] previous = null;
                double[] accepted = null;
                for (int order = 8; order <= MAX_TIME_ORDER; order *= 2) {
                    double[] current = timeIntegral(points, end, error, order);
                    if (previous != null && maxDifference(previous, current) <= TIME_TOL) {
                        accepted = current;
                        break;
                    }
                    previous = current;
                }
                if (accepted == null) throw accuracy("landing reset-age quadrature did not converge");
                if (!infinite && end >= limit) return accepted;
                // Convolution cannot increase the supremum of the wrapped Gaussian.
                // Sigma*u >= sigmaLower*u*I. A decreasing winding-sum integral
                // bounds its supremum by (1/(2pi)+1/sqrt(2pi*v))^dim.
                double variance = sigmaLower * (end / rate);
                double logUpper = dim * Math.log(1.0 / TWO_PI + 1.0 / Math.sqrt(TWO_PI * variance));
                double minimum = Double.POSITIVE_INFINITY;
                for (double value : accepted) minimum = Math.min(minimum, value);
                if (-end + logUpper <= minimum + Math.log(TIME_TOL)) return accepted;
                double next = Math.min(limit, end + 20.0);
                if (!(next > end) || next > 1000.0)
                    throw accuracy("landing stationary age tail exceeds the reference's resource cap");
                end = next;
            }
        }

        double[] timeIntegral(double[][] points, double end, double[][] error, int order) {
            double[] result = constant(points.length, Double.NEGATIVE_INFINITY);
            double[][] rule = gaussLegendre(order);
            double first = Math.min(end, Math.min(1.0, rate * timeScale));
            if (!(first > 0.0) || !Double.isFinite(first))
                throw accuracy("landing covariance/time scales exceed floating-point reference resolution");
            double lo = 0.0, hi = first;
            long work = 0;
            int segments = 0;
            while (lo < end) {
                if (++segments > 1024) throw accuracy("landing age integration needs too many scale intervals");
                work += (long) order * points.length * weights.length;
                if (work > MAX_TIME_WORK) throw accuracy("landing age quadrature exceeds the reference's resource cap");
                double middle = 0.5 * (lo + hi), half = 0.5 * (hi - lo);
                for (int q = 0; q < order; q++) {
                    double s = middle + half * rule[0][q];
                    double u = s / rate;
                    if (!Double.isFinite(u)) throw accuracy("landing age is outside floating-point range");
                    double logWeight = Math.log(half * rule[1][q]) - s;
                    for (int k = 0; k < weights.length; k++) {
                        if (weights[k] == 0.0) continue;
                        double[][] covariance = covariance(covariances[k], sigma, u, error);
                        double[][] residual = new double[points.length][dim];
                        for (int p = 0; p < points.length; p++)
                            for (int ax = 0; ax < dim; ax++) residual[p][ax] = points[p][ax] - means[k][ax];
                        double[] density = gaussian(residual, covariance);
                        double componentWeight = logWeight + Math.log(weights[k]);
                        for (int p = 0; p < result.length; p++) result[p] = logAdd(result[p], componentWeight + density[p]);
                    }
                }
                lo = hi;
                hi = Math.min(end, 2.0 * hi);
            }
            return result;
        }
    }

    private static final class Evaluation {
        final Model model;
        final int grid, cells;
        final int[][] coordinates;
        final double[][] offsets;
        final double[][] partials, noReset, reset;
        final double logCell, massTolerance;

        Evaluation(Model model, int grid, int[][] coordinates, double[][] offsets) {
            this.model = model;
            this.grid = grid;
            this.cells = coordinates.length;
            this.coordinates = coordinates;
            this.offsets = offsets;
            this.partials = new double[model.nodes][];
            this.noReset = new double[model.nodes][];
            this.reset = new double[model.nodes][];
            this.logCell = model.dim * Math.log(TWO_PI / grid);
            this.massTolerance = TOL / Math.max(1, model.nodes);
        }

        double[][] points(int node) {
            double[][] points = new double[cells][model.dim];
            for (int cell = 0; cell < cells; cell++)
                for (int ax = 0; ax < model.dim; ax++)
                    points[cell][ax] = TWO_PI * (coordinates[cell][ax] + offsets[node][ax]) / grid;
            return points;
        }

        void checkMass(double[] density, double expectedLogMass) {
            if (expectedLogMass == Double.NEGATIVE_INFINITY) return;
            double measured = logSum(density) + logCell;
            if (!Double.isFinite(measured) || Math.abs(measured - expectedLogMass) > massTolerance)
                throw new RefineGrid();
        }

        void prune(int node) {
            double[] partial = new double[cells];
            for (int child : model.children[node]) {
                double t = model.times[child];
                double[] message;
                if (model.children[child].length == 0) {
                    double[] theta = model.observations[child];
                    if (theta == null) continue;
                    double[][] delta = points(node);
                    for (int cell = 0; cell < cells; cell++)
                        for (int ax = 0; ax < model.dim; ax++) delta[cell][ax] = theta[ax] - delta[cell][ax];
                    message = gaussian(delta, covariance(null, model.sigma, t, model.eps));
                    double landing = model.timeDensity(new double[][]{theta}, t, model.eps)[0];
                    for (int cell = 0; cell < cells; cell++) message[cell] = logAdd(message[cell] - model.rate * t, landing);
                } else {
                    prune(child);
                    double[][] delta = new double[cells][model.dim];
                    for (int cell = 0; cell < cells; cell++)
                        for (int ax = 0; ax < model.dim; ax++)
                            delta[cell][ax] = TWO_PI * (coordinates[cell][ax]
                                    + offsets[child][ax] - offsets[node][ax]) / grid;
                    noReset[child] = gaussian(delta, covariance(null, model.sigma, t, null));
                    checkMass(noReset[child], 0.0);
                    for (int cell = 0; cell < cells; cell++) noReset[child][cell] -= model.rate * t;
                    reset[child] = model.timeDensity(points(child), t, null);
                    if (model.rate > 0.0) checkMass(reset[child], Math.log(-Math.expm1(-model.rate * t)));
                    double[] landed = partials[child].clone();
                    for (int cell = 0; cell < cells; cell++) landed[cell] += reset[child][cell];
                    double resetIntegral = logSum(landed) + logCell;
                    message = propagate(partials[child], noReset[child], coordinates, grid, logCell);
                    for (int cell = 0; cell < cells; cell++) message[cell] = logAdd(message[cell], resetIntegral);
                }
                for (int cell = 0; cell < cells; cell++) partial[cell] += message[cell];
            }
            partials[node] = partial;
        }
    }

    private static double[] propagate(double[] child, double[] transition, int[][] coordinates,
                                      int grid, double logCell) {
        int cells = child.length;
        double[] result = new double[cells];
        for (int parent = 0; parent < cells; parent++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int cell = 0; cell < cells; cell++)
                max = Math.max(max, child[cell] + transition[displacement(coordinates[cell], coordinates[parent], grid)]);
            double sum = 0.0;
            for (int cell = 0; cell < cells; cell++)
                sum += Math.exp(child[cell] + transition[displacement(coordinates[cell], coordinates[parent], grid)] - max);
            result[parent] = max + Math.log(sum) + logCell;
        }
        return result;
    }

    private static double[] gaussian(double[][] points, double[][] covariance) {
        return TorusLogSpacePruning.logTransition(points, covariance, 1.0, 0.0, null, null, null);
    }

    private static double[][] covariance(double[][] base, double[][] sigma, double time, double[][] eps) {
        int dim = sigma.length;
        double[][] covariance = new double[dim][dim];
        for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++)
            covariance[i][j] = time * sigma[i][j] + (base == null ? 0.0 : base[i][j]) + (eps == null ? 0.0 : eps[i][j]);
        return covariance;
    }

    /** Lower eigenvalue bound from the inverse's infinity norm, computed by Cholesky. */
    private static double positiveLowerBound(double[][] matrix, int dim) {
        if (matrix.length != dim) throw new IllegalArgumentException("covariance dimension mismatch");
        double[][] chol = new double[dim][dim];
        for (double[] row : matrix) if (row.length != dim) throw new IllegalArgumentException("covariance dimension mismatch");
        for (int i = 0; i < dim; i++) for (int j = 0; j <= i; j++) {
            double value = matrix[i][j];
            if (!Double.isFinite(value) || !Double.isFinite(matrix[j][i])
                    || Math.abs(value - matrix[j][i]) > 1.0e-12 * Math.max(1.0, Math.abs(value)))
                throw new IllegalArgumentException("covariance must be finite and symmetric");
            for (int k = 0; k < j; k++) value -= chol[i][k] * chol[j][k];
            if (i == j) {
                if (!(value > 0.0)) throw new IllegalArgumentException("covariance must be positive definite");
                chol[i][j] = Math.sqrt(value);
            } else chol[i][j] = value / chol[j][j];
        }
        double[][] inverseChol = new double[dim][dim];
        for (int j = 0; j < dim; j++) for (int i = 0; i < dim; i++) {
            double value = i == j ? 1.0 : 0.0;
            for (int k = 0; k < i; k++) value -= chol[i][k] * inverseChol[k][j];
            inverseChol[i][j] = value / chol[i][i];
        }
        double norm = 0.0;
        for (int i = 0; i < dim; i++) {
            double row = 0.0;
            for (int j = 0; j < dim; j++) {
                double value = 0.0;
                for (int k = 0; k < dim; k++) value += inverseChol[k][i] * inverseChol[k][j];
                row += Math.abs(value);
            }
            norm = Math.max(norm, row);
        }
        return 1.0 / norm;
    }

    private static double infinityNorm(double[][] matrix) {
        double norm = 0.0;
        for (double[] row : matrix) {
            double sum = 0.0;
            for (double value : row) sum += Math.abs(value);
            norm = Math.max(norm, sum);
        }
        return norm;
    }

    private static double[][] gaussLegendre(int order) {
        double[][] rule = new double[2][order];
        for (int i = 0; i < (order + 1) / 2; i++) {
            double x = Math.cos(Math.PI * (i + 0.75) / (order + 0.5)), derivative = 0.0;
            for (int iteration = 0; iteration < 30; iteration++) {
                double p = 1.0, previous = 0.0;
                for (int j = 1; j <= order; j++) {
                    double older = previous;
                    previous = p;
                    p = ((2.0 * j - 1.0) * x * previous - (j - 1.0) * older) / j;
                }
                derivative = order * (x * p - previous) / (x * x - 1.0);
                double next = x - p / derivative;
                if (Math.abs(next - x) < 2.0e-15) { x = next; break; }
                x = next;
            }
            double weight = 2.0 / ((1.0 - x * x) * derivative * derivative);
            rule[0][i] = -x;
            rule[0][order - 1 - i] = x;
            rule[1][i] = weight;
            rule[1][order - 1 - i] = weight;
        }
        return rule;
    }

    private static double maxDifference(double[] a, double[] b) {
        double difference = 0.0;
        for (int i = 0; i < a.length; i++) {
            if (!Double.isFinite(a[i]) || !Double.isFinite(b[i])) return Double.POSITIVE_INFINITY;
            difference = Math.max(difference, Math.abs(a[i] - b[i]));
        }
        return difference;
    }

    private static int displacement(int[] child, int[] parent, int grid) {
        int index = 0;
        for (int ax = 0; ax < child.length; ax++) index = index * grid + (child[ax] - parent[ax] + grid) % grid;
        return index;
    }

    private static double logAdd(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) return b;
        if (b == Double.NEGATIVE_INFINITY) return a;
        double max = Math.max(a, b);
        return max + Math.log1p(Math.exp(Math.min(a, b) - max));
    }

    private static double logSum(double[] values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) max = Math.max(max, value);
        if (max == Double.NEGATIVE_INFINITY) return max;
        double sum = 0.0;
        for (double value : values) sum += Math.exp(value - max);
        return max + Math.log(sum);
    }

    private static int sampleLogWeights(double[] weights) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : weights) max = Math.max(max, value);
        double total = 0.0;
        for (double value : weights) total += Math.exp(value - max);
        if (!(total > 0.0) || !Double.isFinite(total)) throw accuracy("invalid landing ancestral weights");
        double draw = MathUtils.nextDouble() * total;
        int lastPositive = -1;
        for (int i = 0; i < weights.length; i++) {
            double weight = Math.exp(weights[i] - max);
            if (weight > 0.0) lastPositive = i;
            draw -= weight;
            if (draw < 0.0) return i;
        }
        // Rounding in the cumulative sum must never select a zero-probability cell.
        if (lastPositive < 0) throw accuracy("invalid landing ancestral weights");
        return lastPositive;
    }

    private static double wrap(double angle) { return angle - TWO_PI * Math.floor((angle + Math.PI) / TWO_PI); }
    private static double[][] copy(double[][] values) {
        double[][] copy = new double[values.length][];
        for (int i = 0; i < values.length; i++) copy[i] = values[i].clone();
        return copy;
    }
    private static double[] constant(int length, double value) {
        double[] values = new double[length];
        Arrays.fill(values, value);
        return values;
    }
    private static TorusLogSpacePruning.AccuracyException accuracy(String message) {
        return new TorusLogSpacePruning.AccuracyException(message);
    }
}
