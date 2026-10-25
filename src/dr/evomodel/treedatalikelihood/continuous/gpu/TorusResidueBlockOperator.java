/*
 * TorusResidueBlockOperator.java
 *
 * @author Samir Bhatt
 */

package dr.evomodel.treedatalikelihood.continuous.gpu;

import dr.inference.model.Bounds;
import dr.inference.model.CompoundLikelihood;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.Likelihood;
import dr.inference.model.MatrixParameter;
import dr.inference.model.Parameter;
import dr.inference.operators.GibbsOperator;
import dr.inference.operators.SimpleMCMCOperator;
import dr.math.MathUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * One Metropolis-Hastings update of EVERY residue of a {@link TorusLandingGpuLikelihood}, with all residues'
 * proposals evaluated in one GPU pass. BEAST sees a Gibbs operator (always accepted); the accept/reject
 * decisions are made here, residue by residue.
 *
 * <p><b>Target.</b> Residue r's parameters x_r = (q1, q2, rho, lambda_r) are its precision matrix's diagonal and
 * correlation and its element of the jump-rate vector. Residues are updated in turn (a Metropolis-within-Gibbs
 * sweep), each targeting its full conditional under BEAST's own posterior: the likelihood change is the
 * residue's own term (the likelihood is a sum over residues), and the prior change is BEAST's own prior
 * components evaluated before and after setting the residue's proposal, with every other residue at its
 * current value. This is exact whatever the prior is, including priors that couple residues (a hierarchical
 * prior on the jump rates), and uses BEAST's prior code, not a copy. Which of the components of {@code prior}
 * depend on residue r is found once, at the first step, by moving all four of r's parameters and seeing which
 * component values change; components that do not change there (a flat LKJ(1) prior on the correlation) are
 * skipped, and the posterior guard below would catch any dependence this misses.
 * If a {@code posterior} is given, every step checks that the sum of the accepted changes equals the change
 * in BEAST's posterior, which fails loudly if anything outside {@code prior} and this likelihood depends on
 * the residues' parameters.
 *
 * <p><b>Proposal.</b> A Gaussian random walk on u = (log q1, log q2, atanh(rho / b), log lambda), where b is the
 * correlation bound (|rho| <= b), with the log-Jacobian of x(u) in the acceptance ratio. Each residue has its
 * own proposal covariance: (2.38^2 / 4) s_r (C_r + eps I) after {@code adaptStart} steps, where C_r is the
 * running covariance of the residue's u, and a diagonal (initialScale^2 s_r I) before that. The scale s_r
 * follows Robbins-Monro towards acceptance 0.234. All adaptation stops after {@code adaptUntil} steps of this
 * operator; from then on the kernel is fixed, so draws after that point are a valid MCMC sample. Discard at
 * least the adaptation period as burn-in.
 *
 * <p><b>Combined model.</b> If the likelihood has additive jumps ({@link TorusLandingGpuLikelihood#isCombined}),
 * each residue's block is six-dimensional: x_r also holds its additive jump rate lambda_A,r and jump size s_J,r
 * (one value per residue each), proposed as u = (..., log lambda_A, log s_J) with the matching Jacobian terms. A
 * finite upper bound on s_J is honoured by rejecting proposals above it. Without additive jumps the block is the
 * four-dimensional one above, unchanged.
 *
 * @author Samir Bhatt
 */
public class TorusResidueBlockOperator extends SimpleMCMCOperator implements GibbsOperator {

    public static final String OPERATOR_NAME = "torusResidueBlockOperator";
    private final int D;                         // 4, or 6 with additive jumps
    private static final double TARGET = 0.234;
    private static final double EPS = 1e-8;

    private final TorusLandingGpuLikelihood likelihood;
    private final Likelihood prior;
    private final Likelihood posterior;          // nullable: the consistency guard
    private final boolean ignoreLikelihood;
    private final int R;
    private final Parameter[] diag;
    private final Parameter[] off;
    private final Parameter lambda;
    private final Parameter lamA;                // combined model only (else null)
    private final Parameter sJ;
    private final double sJMax;                  // s_J upper bound (infinite if none)
    private final double bound;
    private final long adaptStart;
    private final long adaptUntil;
    private final double initialScale;
    private final double guardTolerance;

    // per-residue adaptation state
    private final double[] logScale;
    private final double[][] mean;               // running mean of u
    private final double[][] m2;                 // running co-moments, D*D
    private final double[][] chol;               // lower Cholesky factor of the proposal covariance, D*D
    private long n = 0L;                         // steps of this operator so far
    private boolean frozenLogged = false;
    private Likelihood[][] residuePriors;        // per residue: the leaf prior components that depend on it

    // diagnostics
    private final long[] accepted;
    private long proposedTotal = 0L, acceptedTotal = 0L, numericalRejects = 0L;
    private double lastGuardError = 0.0;
    private double pathParameter = 1.0;          // power posterior: the likelihood change is multiplied by it

    public TorusResidueBlockOperator(TorusLandingGpuLikelihood likelihood, Likelihood prior, Likelihood posterior,
                                     boolean ignoreLikelihood, long adaptStart, long adaptUntil, double initialScale,
                                     double guardTolerance, double weight) {
        this.likelihood = likelihood;
        this.prior = prior;
        this.posterior = posterior;
        this.ignoreLikelihood = ignoreLikelihood;
        this.R = likelihood.getResidueCount();
        this.adaptStart = adaptStart;
        this.adaptUntil = adaptUntil;
        this.initialScale = initialScale;
        this.guardTolerance = guardTolerance;
        setWeight(weight);

        this.lambda = likelihood.getJumpRate();
        if (lambda.getDimension() != R) {
            throw new IllegalArgumentException(OPERATOR_NAME + " needs one jump rate per residue (dimension " + R
                    + "), got dimension " + lambda.getDimension());
        }
        this.lamA = likelihood.isCombined() ? likelihood.getAdditiveRate() : null;
        this.sJ = likelihood.isCombined() ? likelihood.getAdditiveSize() : null;
        this.D = lamA == null ? 4 : 6;
        double smax = Double.POSITIVE_INFINITY;
        if (lamA != null) {
            if (lamA.getDimension() != R || sJ.getDimension() != R) {
                throw new IllegalArgumentException(OPERATOR_NAME + " needs one additive rate and one jump size per residue"
                        + " (dimension " + R + ")");
            }
            final Bounds<Double> sb = sJ.getBounds();
            if (sb != null) {
                smax = sb.getUpperLimit(0);
                for (int r = 0; r < R; r++) {
                    if (sb.getLowerLimit(r) > 0.0 || sb.getUpperLimit(r) != smax) {
                        throw new IllegalArgumentException(OPERATOR_NAME + ": jump size bounds must be [0, U] with one U"
                                + " for every residue");
                    }
                }
            }
        }
        this.sJMax = smax;
        this.diag = new Parameter[R];
        this.off = new Parameter[R];
        double b = Double.NaN;
        for (int r = 0; r < R; r++) {
            final MatrixParameter p = likelihood.getPrecision(r);
            if (!(p instanceof CompoundSymmetricMatrix) || !((CompoundSymmetricMatrix) p).asCorrelation()) {
                throw new IllegalArgumentException(OPERATOR_NAME + ": residue " + r
                        + " precision must be a compoundSymmetricMatrix with asCorrelation=\"true\"");
            }
            final CompoundSymmetricMatrix c = (CompoundSymmetricMatrix) p;
            diag[r] = c.getDiagonalParameter();
            off[r] = c.getOffDiagonalParameter();
            if (diag[r].getDimension() != 2 || off[r].getDimension() != 1) {
                throw new IllegalArgumentException(OPERATOR_NAME + ": residue " + r + " needs 2 diagonal and 1 correlation value");
            }
            requireBounds(diag[r], 0, 0.0, Double.POSITIVE_INFINITY, "diagonal");
            requireBounds(diag[r], 1, 0.0, Double.POSITIVE_INFINITY, "diagonal");
            final Bounds<Double> ob = off[r].getBounds();
            final double lo = ob.getLowerLimit(0), hi = ob.getUpperLimit(0);
            if (!(hi > 0.0) || hi >= 1.0 || lo != -hi) {
                throw new IllegalArgumentException(OPERATOR_NAME + ": residue " + r + " correlation bounds must be "
                        + "symmetric (-b, b) with 0 < b < 1, got [" + lo + ", " + hi + "]");
            }
            if (r == 0) b = hi;
            else if (hi != b) throw new IllegalArgumentException(OPERATOR_NAME + ": correlation bounds differ across residues");
            requireBounds(lambda, r, 0.0, Double.POSITIVE_INFINITY, "jump rate");
            if (lamA != null) requireBounds(lamA, r, 0.0, Double.POSITIVE_INFINITY, "additive jump rate");
        }
        this.bound = b;

        this.logScale = new double[R];
        this.mean = new double[R][D];
        this.m2 = new double[R][D * D];
        this.chol = new double[R][D * D];
        for (int r = 0; r < R; r++) for (int i = 0; i < D; i++) chol[r][i * D + i] = initialScale;
        this.accepted = new long[R];
    }

    private static void requireBounds(Parameter p, int i, double lo, double hi, String what) {
        final Bounds<Double> bd = p.getBounds();
        if (bd == null) return;
        if (bd.getLowerLimit(i) > lo || bd.getUpperLimit(i) < hi) {
            throw new IllegalArgumentException(OPERATOR_NAME + ": " + what + " " + p.getId() + "[" + i + "] has bounds ["
                    + bd.getLowerLimit(i) + ", " + bd.getUpperLimit(i) + "]; the transform assumes [" + lo + ", " + hi + ")");
        }
    }

    public String getOperatorName() { return OPERATOR_NAME + "(" + likelihood.getId() + ")"; }

    public long getAcceptedTotal() { return acceptedTotal; }
    public long getProposedTotal() { return proposedTotal; }
    public long getStepCount() { return n; }
    public double getLastGuardError() { return lastGuardError; }

    public double doOperation() {
        if (residuePriors == null) discoverPriorDependencies();
        final double postBefore = posterior == null ? Double.NaN : posterior.getLogLikelihood();
        final double[][] x = new double[R][D];
        final double[][] u = new double[R][D];
        final double[][] x2 = new double[R][D];
        final double[][] u2 = new double[R][D];
        final boolean[] valid = new boolean[R];
        final double[] oldLogL = new double[R];
        for (int r = 0; r < R; r++) {
            read(r, x[r]);
            toU(x[r], u[r]);
            if (!ignoreLikelihood) oldLogL[r] = likelihood.getResidueLogLikelihood(r);
        }

        // propose every residue
        final double[] z = new double[D];
        int nValid = 0;
        for (int r = 0; r < R; r++) {
            for (int i = 0; i < D; i++) z[i] = MathUtils.nextGaussian();
            final double s = Math.exp(logScale[r]);
            for (int i = 0; i < D; i++) {
                double v = 0.0;
                for (int j = 0; j <= i; j++) v += chol[r][i * D + j] * z[j];
                u2[r][i] = u[r][i] + s * v;
            }
            fromU(u2[r], x2[r]);
            valid[r] = x2[r][0] > 0.0 && x2[r][1] > 0.0 && Double.isFinite(x2[r][0]) && Double.isFinite(x2[r][1])
                    && Math.abs(x2[r][2]) <= bound && x2[r][3] > 0.0 && Double.isFinite(x2[r][3]);
            if (D == 6) {
                valid[r] &= x2[r][4] > 0.0 && Double.isFinite(x2[r][4]) && x2[r][5] > 0.0 && x2[r][5] <= sJMax;
            }
            if (valid[r]) nValid++;
            else numericalRejects++;
        }

        // one GPU pass for every valid proposal: set quietly, evaluate, restore quietly (no events: the state
        // seen by every other model is unchanged when this block ends)
        final double[] newLogL = new double[R];
        if (!ignoreLikelihood && nValid > 0) {
            final int[] idx = new int[nValid];
            int k = 0;
            for (int r = 0; r < R; r++) if (valid[r]) { idx[k++] = r; writeQuietly(r, x2[r]); }
            final double[] ll = new double[nValid];
            final int[] status = new int[nValid];
            try {
                likelihood.evaluateProposals(idx, ll, status);
            } finally {
                for (int i = 0; i < nValid; i++) writeQuietly(idx[i], x[idx[i]]);
            }
            for (int i = 0; i < nValid; i++) {
                final boolean ok = status[i] == TorusLandingGpuEngine.STATUS_OK
                        || status[i] == TorusLandingGpuEngine.STATUS_ROOT_UNDEFINED;
                newLogL[idx[i]] = ok ? ll[i] : Double.NEGATIVE_INFINITY;   // production would have thrown: reject
            }
        }

        // the sweep: each residue against BEAST's prior with every other residue at its current value
        double predicted = 0.0;
        final double[] alpha = new double[R];
        for (int r = 0; r < R; r++) {
            proposedTotal++;
            if (!valid[r]) continue;
            final double dLike = ignoreLikelihood ? 0.0 : pathParameter * (newLogL[r] - oldLogL[r]);
            if (dLike == Double.NEGATIVE_INFINITY) continue;
            final double priorBefore = sum(residuePriors[r]);
            write(r, x2[r]);
            final double priorAfter = sum(residuePriors[r]);
            final double dPrior = priorAfter - priorBefore;
            final double logRatio = dLike + dPrior + logJacobian(u2[r]) - logJacobian(u[r]);
            alpha[r] = logRatio >= 0.0 ? 1.0 : (Double.isNaN(logRatio) ? 0.0 : Math.exp(logRatio));
            if (MathUtils.nextDouble() < alpha[r]) {
                accepted[r]++;
                acceptedTotal++;
                predicted += dLike + dPrior;
                System.arraycopy(u2[r], 0, u[r], 0, D);
            } else {
                write(r, x[r]);
            }
        }

        if (posterior != null) {
            final double postAfter = posterior.getLogLikelihood();
            lastGuardError = Math.abs((postAfter - postBefore) - predicted);
            if (!(lastGuardError <= guardTolerance * Math.max(1.0, Math.abs(postBefore)))) {
                throw new RuntimeException(getOperatorName() + ": the posterior changed by " + (postAfter - postBefore)
                        + " but the accepted residue updates account for " + predicted + ". Something other than '"
                        + prior.getId() + "' and the likelihood depends on the residue parameters, so this operator "
                        + "is not targeting BEAST's posterior.");
            }
        }

        adapt(u, alpha);
        n++;
        return 0.0;
    }

    private static double sum(Likelihood[] ls) {
        double s = 0.0;
        for (Likelihood l : ls) s += l.getLogLikelihood();
        return s;
    }

    /** The leaf components of 'prior' whose value changes when all of residue r's parameters change. */
    private void discoverPriorDependencies() {
        final List<Likelihood> leaves = new ArrayList<Likelihood>();
        flatten(prior, leaves);
        residuePriors = new Likelihood[R][];
        final double[] x = new double[D];
        final double[] y = new double[D];
        int links = 0;
        for (int r = 0; r < R; r++) {
            read(r, x);
            y[0] = 1.7 * x[0];
            y[1] = 0.6 * x[1];
            y[2] = Math.abs(x[2]) < 0.5 * bound ? x[2] + 0.4 * bound : x[2] - Math.signum(x[2]) * 0.4 * bound;
            y[3] = 1.7 * x[3];
            if (D == 6) {
                y[4] = 1.7 * x[4];
                y[5] = 0.6 * x[5];
            }
            final double[] before = new double[leaves.size()];
            for (int c = 0; c < before.length; c++) before[c] = leaves.get(c).getLogLikelihood();
            write(r, y);
            final List<Likelihood> dep = new ArrayList<Likelihood>();
            for (int c = 0; c < before.length; c++) {
                final double after = leaves.get(c).getLogLikelihood();
                if (!(after == before[c])) dep.add(leaves.get(c));   // NaN or a change counts as dependence
            }
            write(r, x);
            residuePriors[r] = dep.toArray(new Likelihood[0]);
            links += dep.size();
        }
        Logger.getLogger("dr.inference").info(getOperatorName() + ": " + leaves.size() + " prior components; "
                + links + " residue-component dependencies found (" + String.format("%.2f", links / (double) R)
                + " per residue); the posterior guard is " + (posterior == null ? "OFF" : "on"));
    }

    private static void flatten(Likelihood l, List<Likelihood> out) {
        if (l instanceof CompoundLikelihood) {
            for (Likelihood c : ((CompoundLikelihood) l).getLikelihoods()) flatten(c, out);
        } else {
            out.add(l);
        }
    }

    // ---- adaptation ------------------------------------------------------------------------------------

    private void adapt(double[][] u, double[] alpha) {
        if (n >= adaptUntil) {
            if (!frozenLogged) {
                frozenLogged = true;
                Logger.getLogger("dr.inference").info(getOperatorName() + ": adaptation stopped after " + n
                        + " steps; the proposal is fixed from here on (overall acceptance so far "
                        + String.format("%.3f", acceptedTotal / (double) Math.max(1L, proposedTotal)) + ")");
            }
            return;
        }
        final double gamma = Math.pow(n + 1.0, -0.6);
        final double t = n + 1.0;
        for (int r = 0; r < R; r++) {
            logScale[r] += gamma * (alpha[r] - TARGET);
            // Welford update of mean and co-moments of u
            final double[] d = new double[D];
            for (int i = 0; i < D; i++) d[i] = u[r][i] - mean[r][i];
            for (int i = 0; i < D; i++) mean[r][i] += d[i] / t;
            for (int i = 0; i < D; i++) {
                for (int j = 0; j < D; j++) m2[r][i * D + j] += d[i] * (u[r][j] - mean[r][j]);
            }
        }
        if (n + 1 >= adaptStart && (n + 1) % 50 == 0) {
            for (int r = 0; r < R; r++) {
                final double[] c = new double[D * D];
                for (int i = 0; i < D * D; i++) c[i] = m2[r][i] / Math.max(1.0, t - 1.0) * (2.38 * 2.38 / D);
                for (int i = 0; i < D; i++) c[i * D + i] += EPS;
                if (cholesky(c, D)) {
                    System.arraycopy(c, 0, chol[r], 0, D * D);
                }
            }
        }
    }

    /** In-place lower Cholesky of a D x D matrix; false (input undefined) if not positive definite. */
    private static boolean cholesky(double[] a, int D) {
        for (int j = 0; j < D; j++) {
            double s = a[j * D + j];
            for (int k = 0; k < j; k++) s -= a[j * D + k] * a[j * D + k];
            if (!(s > 0.0)) return false;
            final double ljj = Math.sqrt(s);
            a[j * D + j] = ljj;
            for (int i = j + 1; i < D; i++) {
                double v = a[i * D + j];
                for (int k = 0; k < j; k++) v -= a[i * D + k] * a[j * D + k];
                a[i * D + j] = v / ljj;
            }
            for (int i = 0; i < j; i++) a[i * D + j] = 0.0;
        }
        return true;
    }

    // ---- parameters and transforms -----------------------------------------------------------------------

    private void read(int r, double[] x) {
        x[0] = diag[r].getParameterValue(0);
        x[1] = diag[r].getParameterValue(1);
        x[2] = off[r].getParameterValue(0);
        x[3] = lambda.getParameterValue(r);
        if (D == 6) {
            x[4] = lamA.getParameterValue(r);
            x[5] = sJ.getParameterValue(r);
        }
    }

    private void write(int r, double[] x) {
        diag[r].setParameterValue(0, x[0]);
        diag[r].setParameterValue(1, x[1]);
        off[r].setParameterValue(0, x[2]);
        lambda.setParameterValue(r, x[3]);
        if (D == 6) {
            lamA.setParameterValue(r, x[4]);
            sJ.setParameterValue(r, x[5]);
        }
    }

    private void writeQuietly(int r, double[] x) {
        diag[r].setParameterValueQuietly(0, x[0]);
        diag[r].setParameterValueQuietly(1, x[1]);
        off[r].setParameterValueQuietly(0, x[2]);
        lambda.setParameterValueQuietly(r, x[3]);
        if (D == 6) {
            lamA.setParameterValueQuietly(r, x[4]);
            sJ.setParameterValueQuietly(r, x[5]);
        }
    }

    private void toU(double[] x, double[] u) {
        u[0] = Math.log(x[0]);
        u[1] = Math.log(x[1]);
        final double c = x[2] / bound;
        u[2] = 0.5 * Math.log1p(2.0 * c / (1.0 - c));   // atanh(c), stable
        u[3] = Math.log(x[3]);
        if (D == 6) {
            u[4] = Math.log(x[4]);
            u[5] = Math.log(x[5]);
        }
    }

    private void fromU(double[] u, double[] x) {
        x[0] = Math.exp(u[0]);
        x[1] = Math.exp(u[1]);
        x[2] = bound * Math.tanh(u[2]);
        x[3] = Math.exp(u[3]);
        if (D == 6) {
            x[4] = Math.exp(u[4]);
            x[5] = Math.exp(u[5]);
        }
    }

    /** log |dx/du| = u0 + u1 + log(b (1 - tanh^2 u2)) + u3 [+ u4 + u5], with log(1 - tanh^2 v) = -2 log cosh v */
    private double logJacobian(double[] u) {
        final double a = Math.abs(u[2]);
        final double logCosh = a + Math.log1p(Math.exp(-2.0 * a)) - Math.log(2.0);
        final double j = u[0] + u[1] + Math.log(bound) - 2.0 * logCosh + u[3];
        return D == 6 ? j + u[4] + u[5] : j;
    }

    public void setPathParameter(double beta) {
        if (!(beta >= 0.0 && beta <= 1.0)) throw new IllegalArgumentException("path parameter must be in [0, 1]");
        this.pathParameter = beta;
    }

    public String getPerformanceSuggestion() {
        return String.format("per-residue acceptance %.3f over %d proposals; %d numerically invalid proposals; "
                        + "adaptation %s after %d steps", acceptedTotal / (double) Math.max(1L, proposedTotal), proposedTotal,
                numericalRejects, n >= adaptUntil ? "stopped" : "running", n);
    }
}
