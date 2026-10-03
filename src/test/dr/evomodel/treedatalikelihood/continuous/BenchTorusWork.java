package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evomodel.tree.DefaultTreeModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Bounded component benchmark for one residue on a supplied production tree.
 *
 * Required inputs:
 * --tree NEWICK --tips TSV --mixture TSV
 * Tip rows are taxon, phi, psi; NaN preserves partial observations. Mixture rows
 * are weight, d0, d1, S00, S01, S11. Tabs or spaces delimit fields. Headers
 * beginning with "taxon" or "weight", blank lines and # comments are skipped.
 *
 * Optional controls: --grids 64,128 --variances .0001,.001,.01 --rhos 0,.8
 * --rates .001,.01 --reps 5 --warmup 2. Sigma = variance * [[1,rho],[rho,1]].
 * Output is CSV, with median/min/max milliseconds across repetitions.
 *
 * "tips" measures complete production leafKernelFromPsi calls: winding sums,
 * spectral remainder, an inverse FFT, multipliers, and allocation are all included.
 * It must not be interpreted as a non-FFT fraction. "internal" measures complete
 * propagateFromPsi calls with a fixed positive dense partial; it includes two FFTs,
 * multipliers and allocation. The whole-tree timing additionally includes pruning
 * products, normalization, precision inversion, bookkeeping and allocation.
 * Component sum versus whole timing is an empirical consistency check, not an
 * exact accounting identity: allocation/GC/JIT and partial values can differ.
 * No production source or private method is instrumented.
 */
public final class BenchTorusWork {
    private static volatile double blackhole;
    private static final double MIN_BRANCH = 1e-8;
    private static final String[] PHASES = {"psi", "tips", "internal", "whole"};

    private static final class Mixture {
        final double[] weights;
        final double[][] means;
        final double[][][] covariances;

        Mixture(List<double[]> rows) {
            weights = new double[rows.size()];
            means = new double[rows.size()][2];
            covariances = new double[rows.size()][2][2];
            double sum = 0;
            for (int k = 0; k < rows.size(); k++) {
                double[] row = rows.get(k);
                weights[k] = row[0]; sum += row[0];
                means[k][0] = row[1]; means[k][1] = row[2];
                covariances[k][0][0] = row[3];
                covariances[k][0][1] = covariances[k][1][0] = row[4];
                covariances[k][1][1] = row[5];
                for (double value : row) if (!Double.isFinite(value))
                    throw new IllegalArgumentException("Mixture fields must be finite");
                if (row[0] < 0 || row[3] < 0 || row[5] < 0 || row[3] * row[5] < row[4] * row[4])
                    throw new IllegalArgumentException("Invalid mixture weight or covariance");
            }
            if (rows.isEmpty() || Math.abs(sum - 1) > 1e-10)
                throw new IllegalArgumentException("Mixture weights must sum to one");
        }
    }

    private static final class Work {
        final TorusFourierKernel kernel;
        final DefaultTreeModel tree;
        final double[][] tips;
        final Mixture mixture;
        final double[][] sigma;
        final double rate;
        final double[] psiRe, psiIm, partial;
        final TorusFourierTreeLikelihood likelihood;

        Work(DefaultTreeModel tree, double[][] tips, Mixture mixture,
             int grid, double variance, double rho, double rate) {
            this.tree = tree; this.tips = tips; this.mixture = mixture; this.rate = rate;
            kernel = new TorusFourierKernel(2, grid);
            sigma = new double[][]{{variance, variance * rho}, {variance * rho, variance}};
            psiRe = new double[kernel.getGridCount()];
            psiIm = new double[psiRe.length];
            partial = new double[psiRe.length];
            for (int i = 0; i < partial.length; i++) {
                partial[i] = .5 + .2 * Math.cos(2 * Math.PI * (i / grid) / grid)
                        + .2 * Math.sin(2 * Math.PI * (i % grid) / grid);
            }
            double precisionDiagonal = 1.0 / (variance * (1 - rho * rho));
            MatrixParameter precision = new CompoundSymmetricMatrix(
                    new Parameter.Default(new double[]{precisionDiagonal, precisionDiagonal}),
                    new Parameter.Default(new double[]{-rho}), true, false);
            likelihood = new TorusFourierTreeLikelihood("work", tree, null,
                    Collections.singletonList(precision), new double[][][]{tips}, kernel,
                    new Parameter.Default(new double[]{rate}), mixture.weights,
                    mixture.means, mixture.covariances);
            psi();
        }

        double branchLength(NodeRef node) {
            return Math.max(MIN_BRANCH, tree.getNodeHeight(tree.getParent(node)) - tree.getNodeHeight(node));
        }

        void psi() {
            kernel.computePsiComplex(sigma, rate, mixture.weights, mixture.means,
                    mixture.covariances, psiRe, psiIm);
            blackhole = psiRe[psiRe.length - 1];
        }

        void tips() {
            double sum = 0;
            for (int i = 0; i < tree.getExternalNodeCount(); i++) {
                NodeRef node = tree.getExternalNode(i);
                double[] point = tips[node.getNumber()];
                if (point != null) {
                    double[] result = kernel.leafKernelFromPsi(point, branchLength(node), psiRe, psiIm, sigma, rate);
                    sum += result[0];
                }
            }
            blackhole = sum;
        }

        void internal() {
            double sum = 0;
            for (int i = 0; i < tree.getInternalNodeCount(); i++) {
                NodeRef node = tree.getInternalNode(i);
                if (!tree.isRoot(node)) {
                    double[] result = kernel.propagateFromPsi(partial, branchLength(node), psiRe, psiIm);
                    sum += result[0];
                }
            }
            blackhole = sum;
        }

        void whole() {
            likelihood.makeDirty();
            blackhole = likelihood.getLogLikelihood();
        }

        void run(int phase) {
            switch (phase) {
                case 0: psi(); break;
                case 1: tips(); break;
                case 2: internal(); break;
                case 3: whole(); break;
                default: throw new IllegalArgumentException("Unknown phase");
            }
        }
    }

    private static List<String[]> rows(String path, String header, int columns) throws Exception {
        List<String[]> rows = new ArrayList<>();
        for (String raw : Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8)) {
            String line = raw.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] fields = line.split("\\s+");
            if (fields[0].equalsIgnoreCase(header)) continue;
            if (fields.length != columns)
                throw new IllegalArgumentException(path + ": expected " + columns + " fields: " + line);
            rows.add(fields);
        }
        return rows;
    }

    private static double angle(String text) {
        double value = text.equalsIgnoreCase("nan") ? Double.NaN : Double.parseDouble(text);
        if (Double.isInfinite(value)) throw new IllegalArgumentException("Infinite tip angle");
        return value;
    }

    private static double[] numbers(Map<String, String> options, String key, String fallback) {
        String[] fields = options.getOrDefault(key, fallback).split(",");
        double[] values = new double[fields.length];
        for (int i = 0; i < fields.length; i++) values[i] = Double.parseDouble(fields[i]);
        return values;
    }

    private static double median(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        return sorted.length % 2 == 1 ? sorted[sorted.length / 2]
                : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        Map<String, String> options = new HashMap<>();
        if (args.length % 2 != 0) throw new IllegalArgumentException("Arguments are --name value pairs");
        for (int i = 0; i < args.length; i += 2) options.put(args[i], args[i + 1]);
        for (String key : new String[]{"--tree", "--tips", "--mixture"})
            if (!options.containsKey(key)) throw new IllegalArgumentException("Required: " + key);
        int reps = Integer.parseInt(options.getOrDefault("--reps", "5"));
        int warmup = Integer.parseInt(options.getOrDefault("--warmup", "2"));
        if (reps < 1 || warmup < 0) throw new IllegalArgumentException("Invalid repetition count");
        String newick = new String(Files.readAllBytes(Paths.get(options.get("--tree"))), StandardCharsets.UTF_8);
        DefaultTreeModel tree = new DefaultTreeModel("workTree", new NewickImporter(newick).importTree(null));
        Map<String, double[]> byTaxon = new HashMap<>();
        for (String[] fields : rows(options.get("--tips"), "taxon", 3)) {
            double[] point = {angle(fields[1]), angle(fields[2])};
            if (byTaxon.containsKey(fields[0])) throw new IllegalArgumentException("Duplicate taxon " + fields[0]);
            byTaxon.put(fields[0], Double.isNaN(point[0]) && Double.isNaN(point[1]) ? null : point);
        }
        double[][] tips = new double[tree.getExternalNodeCount()][];
        int observed = 0, partial = 0;
        for (int i = 0; i < tips.length; i++) {
            NodeRef node = tree.getExternalNode(i);
            String name = tree.getNodeTaxon(node).getId();
            if (!byTaxon.containsKey(name)) throw new IllegalArgumentException("Missing taxon " + name);
            tips[node.getNumber()] = byTaxon.get(name);
            if (tips[node.getNumber()] != null) {
                observed++;
                for (double value : tips[node.getNumber()]) if (Double.isNaN(value)) partial++;
            }
        }
        List<double[]> mixtureRows = new ArrayList<>();
        for (String[] fields : rows(options.get("--mixture"), "weight", 6)) {
            double[] row = new double[6];
            for (int i = 0; i < row.length; i++) row[i] = Double.parseDouble(fields[i]);
            mixtureRows.add(row);
        }
        Mixture mixture = new Mixture(mixtureRows);
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(false);
        System.err.printf("Production component benchmark: tips=%d observed=%d partial=%d internalBranches=%d mixture=%d reps=%d warmup=%d%n",
                tips.length, observed, partial, tree.getInternalNodeCount() - 1, mixture.weights.length, reps, warmup);
        System.err.println("Tip timings include lattice work AND inverse FFTs; internal timings include two FFTs and multipliers.");
        System.out.print("grid,variance,rho,rate,reps,observed_tips,partial_tips,internal_branches");
        for (String phase : PHASES) System.out.print("," + phase + "_median_ms," + phase + "_min_ms," + phase + "_max_ms");
        System.out.println(",component_sum_ms,tips_fraction,internal_fraction,psi_fraction,whole_minus_components_ms,log_likelihood");
        for (double gridValue : numbers(options, "--grids", "64,128")) {
            if (gridValue != (int) gridValue) throw new IllegalArgumentException("Grid must be an integer");
            for (double variance : numbers(options, "--variances", ".0001,.001,.01"))
                for (double rho : numbers(options, "--rhos", "0,.8"))
                    for (double rate : numbers(options, "--rates", ".001,.01")) {
                        if (!(variance > 0) || Math.abs(rho) >= 1 || !(rate > 0))
                            throw new IllegalArgumentException("Need positive variance/rate and |rho| < 1");
                        Work work = new Work(tree, tips, mixture, (int) gridValue, variance, rho, rate);
                        for (int repeat = 0; repeat < warmup; repeat++)
                            for (int phase = 0; phase < PHASES.length; phase++) work.run(phase);
                        double[][] timing = new double[PHASES.length][reps];
                        for (int repeat = 0; repeat < reps; repeat++) {
                            // Rotate order to reduce systematic effects of heap/CPU state.
                            for (int step = 0; step < PHASES.length; step++) {
                                int phase = (step + repeat) % PHASES.length;
                                long start = System.nanoTime();
                                work.run(phase);
                                timing[phase][repeat] = (System.nanoTime() - start) / 1e6;
                            }
                        }
                        double[] medians = new double[PHASES.length];
                        System.out.printf("%d,%.8g,%.8g,%.8g,%d,%d,%d,%d", (int) gridValue,
                                variance, rho, rate, reps, observed, partial, tree.getInternalNodeCount() - 1);
                        for (int phase = 0; phase < PHASES.length; phase++) {
                            medians[phase] = median(timing[phase]);
                            Arrays.sort(timing[phase]);
                            System.out.printf(",%.6f,%.6f,%.6f", medians[phase], timing[phase][0], timing[phase][reps - 1]);
                        }
                        double sum = medians[0] + medians[1] + medians[2];
                        System.out.printf(",%.6f,%.6f,%.6f,%.6f,%.6f,%.12g%n", sum,
                                medians[1] / sum, medians[2] / sum, medians[0] / sum,
                                medians[3] - sum, work.likelihood.getLogLikelihood());
                    }
        }
    }
}
