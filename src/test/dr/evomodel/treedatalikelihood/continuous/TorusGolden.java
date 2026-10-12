package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evolution.tree.TreeTrait;
import dr.evomodel.branchratemodel.ArbitraryBranchRates;
import dr.evomodel.operators.ExchangeOperator;
import dr.evomodel.operators.SubtreeSlideOperator;
import dr.evomodel.operators.WilsonBalding;
import dr.evomodel.tree.DefaultTreeModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood.JumpLaw;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import dr.inference.operators.AdaptationMode;
import dr.math.MathUtils;

import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Golden-output driver for the engine speed-up work. Prints, as exact bit patterns (arrays as SHA-256
 * digests of their bits), every value a deterministic set of scenarios produces: log-likelihoods along
 * chains of tree and parameter proposals (one evaluation per proposal, the MCMC cycle), ancestral draws
 * under fixed seeds, the landing diagnostics, and the output of the public kernel operations the
 * likelihood relies on, in 1, 2 and 3 dimensions over a range of grid sizes. Run it on two builds and
 * diff the outputs: identical files mean identical results to the last bit.
 *
 * Usage: TorusGolden [likelihood|kernel|all]
 */
public class TorusGolden {

    private static final StringBuilder OUT = new StringBuilder();

    private static void emit(String line) {
        OUT.append(line).append('\n');
    }

    private static String hex(double v) {
        return Long.toHexString(Double.doubleToRawLongBits(v));
    }

    private static String digest(double[] values) {
        if (values == null) return "null";
        try {
            final MessageDigest md = MessageDigest.getInstance("SHA-256");
            final byte[] buf = new byte[8];
            for (double v : values) {
                long bits = Double.doubleToRawLongBits(v);
                for (int i = 0; i < 8; i++) {
                    buf[i] = (byte) (bits >>> (8 * i));
                }
                md.update(buf);
            }
            final byte[] h = md.digest();
            final StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 12; i++) sb.append(String.format("%02x", h[i]));
            return sb.toString() + " n=" + values.length + " first=" + hex(values[0]) + " last=" + hex(values[values.length - 1]);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // ------------------------------------------------------------------ likelihood chains

    private static final class Scenario {
        String name;
        JumpLaw law = JumpLaw.LANDING;
        boolean jumps = true, tipError = true, perResidueRate = true, branchRates = false, heterochronous = true;
        int tips = 10, residues = 3, nGrid = 32, steps = 60;
        long seed = 1L;
        double tipErrorValue = Double.NaN;
    }

    private static void chain(Scenario sc) throws Exception {
        emit("== chain " + sc.name);
        final Random rng = new Random(sc.seed);
        final DefaultTreeModel tree = new DefaultTreeModel("goldenTree",
                new NewickImporter(TorusCacheFuzzTest.randomNewick(sc.tips, rng, sc.heterochronous)).importTree(null));
        final List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        final List<Parameter> diagonals = new ArrayList<Parameter>();
        for (int r = 0; r < sc.residues; r++) {
            final Parameter diagonal = new Parameter.Default(new double[]{0.5 + 1.5 * rng.nextDouble(),
                    0.5 + 1.5 * rng.nextDouble()});
            final Parameter correlation = new Parameter.Default(new double[]{1.2 * (rng.nextDouble() - 0.5)});
            precisions.add(new CompoundSymmetricMatrix(diagonal, correlation, true, false));
            diagonals.add(diagonal);
        }
        final int external = tree.getExternalNodeCount();
        final double[][][] data = new double[sc.residues][external][];
        for (int r = 0; r < sc.residues; r++) {
            for (int e = 0; e < external; e++) {
                data[r][e] = e >= 2 && rng.nextInt(12) == 0 ? null
                        : new double[]{Math.PI * (2.0 * rng.nextDouble() - 1.0), Math.PI * (2.0 * rng.nextDouble() - 1.0)};
            }
        }
        final int k = sc.law == JumpLaw.LANDING ? 4 : 3;
        final double[] w = k == 4 ? new double[]{0.4, 0.3, 0.2, 0.1} : new double[]{0.5, 0.3, 0.2};
        final double[][] d = new double[k][2];
        final double[][][] s = new double[k][2][2];
        for (int i = 0; i < k; i++) {
            d[i][0] = Math.PI * (2.0 * rng.nextDouble() - 1.0);
            d[i][1] = Math.PI * (2.0 * rng.nextDouble() - 1.0);
            final double a = 0.05 + 0.3 * rng.nextDouble(), b = 0.05 + 0.3 * rng.nextDouble();
            final double c = 0.8 * Math.sqrt(a * b) * (2.0 * rng.nextDouble() - 1.0);
            s[i][0][0] = a;
            s[i][1][1] = b;
            s[i][0][1] = c;
            s[i][1][0] = c;
        }
        Parameter jumpRate = null;
        if (sc.jumps) {
            final double[] rates = new double[sc.perResidueRate ? sc.residues : 1];
            for (int i = 0; i < rates.length; i++) rates[i] = 0.1 + 1.5 * rng.nextDouble();
            jumpRate = new Parameter.Default(rates);
        }
        final Parameter tipError = sc.tipError ? new Parameter.Default(
                Double.isNaN(sc.tipErrorValue) ? 0.03 + 0.03 * rng.nextDouble() : sc.tipErrorValue) : null;
        Parameter rateValues = null;
        ArbitraryBranchRates branchRates = null;
        if (sc.branchRates) {
            final double[] values = new double[tree.getNodeCount() - 1];
            for (int i = 0; i < values.length; i++) values[i] = 0.5 + rng.nextDouble();
            rateValues = new Parameter.Default(values);
            branchRates = new ArbitraryBranchRates(tree, rateValues, new ArbitraryBranchRates.BranchRateTransform.None(), false);
        }
        final TorusFourierKernel kernel = new TorusFourierKernel(2, sc.nGrid);
        final TorusFourierTreeLikelihood like = new TorusFourierTreeLikelihood("golden", tree, branchRates,
                precisions, data, kernel, jumpRate, sc.jumps ? w : null, sc.jumps ? d : null, sc.jumps ? s : null,
                tipError, sc.law);
        final SubtreeSlideOperator slide = new SubtreeSlideOperator(tree, 1.0, 0.1, true, false, false, false,
                AdaptationMode.ADAPTATION_OFF, 0.234);
        final ExchangeOperator narrow = new ExchangeOperator(ExchangeOperator.NARROW, tree, 1.0);
        final ExchangeOperator wide = new ExchangeOperator(ExchangeOperator.WIDE, tree, 1.0);
        final WilsonBalding wb = new WilsonBalding(tree, 1.0);
        MathUtils.setSeed(3000L + sc.seed);
        emit("start " + hex(like.getLogLikelihood()));
        for (int step = 0; step < sc.steps; step++) {
            like.storeModelState();
            final int move = rng.nextInt(sc.branchRates ? 10 : 8);
            boolean ok = true;
            switch (move) {
                case 0: {
                    NodeRef v;
                    do {
                        v = tree.getNode(rng.nextInt(tree.getNodeCount()));
                    } while (tree.isExternal(v));
                    double low = 0.0;
                    for (int c = 0; c < tree.getChildCount(v); c++) low = Math.max(low, tree.getNodeHeight(tree.getChild(v, c)));
                    final double high = tree.isRoot(v) ? low + 1.5 * (tree.getNodeHeight(v) - low)
                            : tree.getNodeHeight(tree.getParent(v));
                    tree.setNodeHeight(v, low + (high - low) * (0.05 + 0.9 * rng.nextDouble()));
                    break;
                }
                case 1: ok = slide.doOperation() != Double.NEGATIVE_INFINITY; break;
                case 2: ok = narrow.doOperation() != Double.NEGATIVE_INFINITY; break;
                case 3: ok = wide.doOperation() != Double.NEGATIVE_INFINITY; break;
                case 4: ok = wb.doOperation() != Double.NEGATIVE_INFINITY; break;
                case 5: {
                    final Parameter p = diagonals.get(rng.nextInt(diagonals.size()));
                    final int i = rng.nextInt(2);
                    p.setParameterValue(i, p.getParameterValue(i) * Math.exp(0.6 * (rng.nextDouble() - 0.5)));
                    break;
                }
                case 6: {
                    if (jumpRate != null) {
                        final int i = rng.nextInt(jumpRate.getDimension());
                        jumpRate.setParameterValue(i, jumpRate.getParameterValue(i) * Math.exp(rng.nextDouble() - 0.5));
                    }
                    break;
                }
                case 7: {
                    if (tipError != null) tipError.setParameterValue(0, tipError.getParameterValue(0) * Math.exp(0.4 * (rng.nextDouble() - 0.5)));
                    break;
                }
                case 8: {
                    final int i = rng.nextInt(rateValues.getDimension());
                    rateValues.setParameterValue(i, 0.5 + rng.nextDouble());
                    break;
                }
                default: {
                    final int i = rng.nextInt(rateValues.getDimension());
                    rateValues.setParameterValueNotifyChangedAll(i, 0.5 + rng.nextDouble());
                }
            }
            final double value = ok ? like.getLogLikelihood() : Double.NaN;
            final boolean accept = ok && rng.nextInt(3) == 0;
            if (accept) like.acceptModelState();
            else like.restoreModelState();
            emit("step " + step + " move " + move + " ok " + ok + " logL " + hex(value)
                    + (accept ? " accept" : " restore " + hex(like.getLogLikelihood())));
            if (sc.law == JumpLaw.LANDING) {
                final long[][] branch = like.getLandingBranchNonpositiveCounts();
                long h = 1469598103934665603L;
                for (long[] row : branch) for (long v : row) h = (h ^ v) * 1099511628211L;
                emit("  branchNonpositive " + Long.toHexString(h));
            }
            if (step % 15 == 14) {
                MathUtils.setSeed(7000L + step);
                final StringBuilder anc = new StringBuilder("ancestral");
                try {
                    like.forceAncestralResample();
                    for (int r = 0; r < sc.residues; r++) {
                        final TreeTrait trait = like.getTreeTrait("theta_" + r);
                        final double[] all = new double[2 * tree.getNodeCount()];
                        int n = 0;
                        for (int i = 0; i < tree.getNodeCount(); i++) {
                            final double[] v = (double[]) trait.getTrait(tree, tree.getNode(i));
                            if (v != null) {
                                all[n++] = v[0];
                                all[n++] = v[1];
                            }
                        }
                        anc.append(" r").append(r).append(' ').append(digest(java.util.Arrays.copyOf(all, n)));
                    }
                } catch (RuntimeException e) {
                    anc.append(" threw ").append(e.getClass().getSimpleName()).append(": ").append(e.getMessage());
                }
                emit(anc.toString());
            }
        }
        if (sc.law == JumpLaw.LANDING) {
            emit("diagnostics nonpositive " + kernel.getLandingNonpositiveCount() + " negative " + kernel.getLandingNegativeCount()
                    + " min " + hex(kernel.getLandingMinimumValue()) + " negativeMass " + hex(kernel.getLandingNegativeMass())
                    + " densityNonpositive " + kernel.getLandingDensityNonpositiveCount()
                    + " undefined " + like.getLandingUndefinedEvaluations());
        }
        emit("ancestralClipped " + hex(kernel.getAncestralClippedMass()) + " unresolved " + kernel.getAncestralUnresolvedDraws());
        emit("recomputes " + like.getNodeRecomputeCount());
    }

    private static void likelihoodScenarios() throws Exception {
        Scenario sc = new Scenario();
        sc.name = "landing per-residue rates n32";
        sc.seed = 21L;
        chain(sc);

        sc = new Scenario();
        sc.name = "landing 9 residues parallel, branch rates";
        sc.residues = 9;
        sc.perResidueRate = false;
        sc.branchRates = true;
        sc.seed = 22L;
        chain(sc);

        sc = new Scenario();
        sc.name = "shift jumps tip error branch rates";
        sc.law = JumpLaw.SHIFT;
        sc.branchRates = true;
        sc.seed = 23L;
        chain(sc);

        sc = new Scenario();
        sc.name = "shift brownian dirac";
        sc.law = JumpLaw.SHIFT;
        sc.jumps = false;
        sc.tipError = false;
        sc.seed = 24L;
        chain(sc);

        sc = new Scenario();
        sc.name = "shift jumps dirac n64";
        sc.law = JumpLaw.SHIFT;
        sc.tipError = false;
        sc.nGrid = 64;
        sc.tips = 8;
        sc.steps = 30;
        sc.seed = 25L;
        chain(sc);

        sc = new Scenario();
        sc.name = "landing n128 sharp tips";
        sc.nGrid = 128;
        sc.tips = 7;
        sc.residues = 1;
        sc.steps = 30;
        sc.tipErrorValue = 0.0129;
        sc.seed = 26L;
        chain(sc);

        sc = new Scenario();
        sc.name = "landing n256 production-like";
        sc.nGrid = 256;
        sc.tips = 6;
        sc.residues = 1;
        sc.steps = 12;
        sc.tipErrorValue = 0.0129;
        sc.seed = 27L;
        chain(sc);
    }

    // ------------------------------------------------------------------ kernel operations

    private static double[][] spd(Random rng, int dim, double scale) {
        final double[][] a = new double[dim][dim];
        for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) a[i][j] = rng.nextGaussian();
        final double[][] s = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                double v = 0.0;
                for (int k = 0; k < dim; k++) v += a[i][k] * a[j][k];
                s[i][j] = scale * v / dim;
            }
            s[i][i] += 0.2 * scale;
        }
        for (int i = 0; i < dim; i++) for (int j = 0; j < i; j++) s[i][j] = s[j][i];
        return s;
    }

    private static void kernelScenario(int dim, int nGrid, long seed) {
        final Random rng = new Random(seed);
        final TorusFourierKernel kernel = new TorusFourierKernel(dim, nGrid);
        final int n = kernel.getGridCount();
        emit("== kernel dim " + dim + " nGrid " + nGrid);
        final double[] partial = new double[n];
        for (int i = 0; i < n; i++) partial[i] = 0.05 + rng.nextDouble();
        final double[][] sigma = spd(rng, dim, 0.4);
        final double[][] eps = spd(rng, dim, 0.02);
        final double[][] sharp = spd(rng, dim, 0.002);
        final double[] y = new double[dim], theta = new double[dim];
        for (int i = 0; i < dim; i++) {
            y[i] = Math.PI * (2.0 * rng.nextDouble() - 1.0);
            theta[i] = 7.0 * (2.0 * rng.nextDouble() - 1.0);
        }
        final int kc = 3;
        final double[] w = {0.5, 0.3, 0.2};
        final double[][] d = new double[kc][dim];
        final double[][][] s = new double[kc][][];
        for (int c = 0; c < kc; c++) {
            for (int i = 0; i < dim; i++) d[c][i] = Math.PI * (2.0 * rng.nextDouble() - 1.0);
            s[c] = spd(rng, dim, 0.3);
        }
        final double lambda = 0.7;
        for (double t : new double[]{1.0e-8, 1.0e-3, 0.05, 0.7, 4.0}) {
            final String tag = " t=" + t;
            emit("propagateBM" + tag + " " + digest(kernel.propagate(partial, sigma, t)));
            emit("propagateGaussJump" + tag + " " + digest(kernel.propagate(partial, sigma, t, lambda, s[0])));
            emit("propagateMixture" + tag + " " + digest(kernel.propagate(partial, sigma, t, lambda, w, d, s)));
            emit("leafBM" + tag + " " + digest(kernel.leafKernel(y, sigma, t)));
            emit("leafBMeps" + tag + " " + digest(kernel.leafKernel(y, sigma, t, eps)));
            emit("leafBMsharpEps" + tag + " " + digest(kernel.leafKernel(y, sigma, t, sharp)));
            emit("leafGaussJump" + tag + " " + digest(kernel.leafKernel(y, sigma, t, lambda, s[0])));
            emit("leafMixture" + tag + " " + digest(kernel.leafKernel(y, sigma, t, lambda, w, d, s)));
            emit("leafMixtureEps" + tag + " " + digest(kernel.leafKernel(y, sigma, t, lambda, w, d, s, eps)));
            final double[] psiRe = new double[n], psiIm = new double[n];
            kernel.computePsiComplex(sigma, lambda, w, d, s, psiRe, psiIm);
            emit("psi" + tag + " " + digest(psiRe) + " | " + digest(psiIm));
            emit("propagateFromPsi" + tag + " " + digest(kernel.propagateFromPsi(partial, t, psiRe, psiIm)));
            emit("leafFromPsi" + tag + " " + digest(kernel.leafKernelFromPsi(y, t, psiRe, psiIm, sigma, lambda)));
            emit("leafFromPsiEps" + tag + " " + digest(kernel.leafKernelFromPsi(y, t, psiRe, psiIm, sigma, lambda, eps)));
            emit("transitionKernel" + tag + " " + digest(kernel.transitionKernel(t, psiRe, psiIm)));
            emit("branchDiffusionKernel" + tag + " " + digest(kernel.branchDiffusionKernel(sigma, t)));
            emit("shiftJumpKernel" + tag + " " + digest(kernel.shiftJumpKernel(sigma, t, lambda, psiRe, psiIm)));
            final double[][] gh = kernel.computeGhat(w, d, s);
            emit("ghat" + tag + " " + digest(gh[0]) + " | " + digest(gh[1]));
            emit("propagateLanding" + tag + " " + digest(kernel.propagateLanding(partial, sigma, t, lambda, gh[0], gh[1])));
            emit("propagateLandingZeroRate" + tag + " " + digest(kernel.propagateLanding(partial, sigma, t, 0.0, gh[0], gh[1])));
            emit("leafLanding" + tag + " " + digest(kernel.leafKernelLanding(theta, eps, sigma, t, lambda, gh[0], gh[1])));
            emit("leafLandingZeroRate" + tag + " " + digest(kernel.leafKernelLanding(theta, eps, sigma, t, 0.0, gh[0], gh[1])));
            emit("landingResetDensity" + tag + " " + digest(kernel.landingResetDensity(sigma, t, lambda, gh[0], gh[1])));
            emit("transitionDensityGridLanding" + tag + " "
                    + digest(kernel.transitionDensityGridLanding(theta, sigma, t, lambda, gh[0], gh[1])));
        }
        final double[][] gh = kernel.computeGhat(w, d, s);
        emit("leafLandingTipOnly " + digest(kernel.leafKernelLanding(theta, eps)));
        emit("rootIntegral " + hex(kernel.rootIntegral(partial)));
        emit("rootIntegralLanding " + hex(kernel.rootIntegralLanding(partial, sigma, lambda, gh[0], gh[1])));
        emit("rootIntegralLandingZeroRate " + hex(kernel.rootIntegralLanding(partial, sigma, 0.0, gh[0], gh[1])));
        emit("landingStationaryDensity " + digest(kernel.landingStationaryDensity(sigma, lambda, gh[0], gh[1])));
        emit("rootDensityGridLanding " + digest(kernel.rootDensityGridLanding(sigma, lambda, gh[0], gh[1])));
        emit("dampingFourier " + digest(kernel.dampingFourier(sigma, 0.3)));
        emit("diagnostics " + kernel.getLandingNonpositiveCount() + " " + kernel.getLandingNegativeCount() + " "
                + hex(kernel.getLandingMinimumValue()) + " " + hex(kernel.getLandingNegativeMass()) + " "
                + kernel.getLandingDensityNonpositiveCount() + " " + kernel.getLandingDensityNegativeCount() + " "
                + hex(kernel.getAncestralClippedMass()));
        final double[] density = kernel.landingStationaryDensity(sigma, lambda, gh[0], gh[1]);
        final StringBuilder cells = new StringBuilder("samples");
        for (int i = 0; i < 20; i++) {
            final double u = (i + 0.5) / 20.0;
            cells.append(' ').append(kernel.sampleFromProduct(density, partial, u));
            cells.append(':').append(kernel.sampleChildCell(kernel.branchDiffusionKernel(sigma, 0.05), density,
                    (int) (u * n), partial, 1.0 - u));
        }
        emit(cells.toString());
        emit("requiredGrid " + TorusFourierKernel.requiredGrid(eps) + " " + TorusFourierKernel.requiredGrid(sharp));
    }

    private static void kernelScenarios() {
        long seed = 100L;
        for (int nGrid : new int[]{4, 8, 16, 32, 64, 128, 256, 512}) kernelScenario(1, nGrid, seed++);
        for (int nGrid : new int[]{4, 8, 16, 32, 64, 128}) kernelScenario(2, nGrid, seed++);
        for (int nGrid : new int[]{4, 8, 16}) kernelScenario(3, nGrid, seed++);
    }

    public static void main(String[] args) throws Exception {
        final String what = args.length > 0 ? args[0] : "all";
        java.util.logging.Logger.getLogger("dr.evomodel").setLevel(java.util.logging.Level.OFF);
        final long t0 = System.nanoTime();
        if (what.equals("kernel") || what.equals("all")) kernelScenarios();
        if (what.equals("likelihood") || what.equals("all")) likelihoodScenarios();
        System.out.print(OUT);
        System.err.printf("golden output: %d lines in %.1f s%n", OUT.toString().split("\n").length,
                (System.nanoTime() - t0) * 1e-9);
    }
}
