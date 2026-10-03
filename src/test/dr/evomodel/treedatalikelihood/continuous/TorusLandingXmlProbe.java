package test.dr.evomodel.treedatalikelihood.continuous;

import dr.app.beast.BeastParser;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.Parameter;
import dr.xml.XMLObject;

import java.io.FileReader;
import java.io.Reader;
import java.util.Locale;
import java.util.Map;

/**
 * Command-line probe (not a JUnit test): parse a BEAST XML containing a single-residue
 * landing torusFourierTreeLikelihood, set (q1, q2, rho, lambda) and print one JSON line per
 * point with the likelihood's log L, the landing gradient sweep's own log L (and whether the
 * two are bit-identical), and the analytic gradient d log L / d(q1, q2, rho, lambda).
 *
 * usage: java -cp ... test.dr.evomodel.treedatalikelihood.continuous.TorusLandingXmlProbe
 *            file.xml likelihoodId [q1 q2 rho lambda]...
 * With no parameter quadruple the XML's start values are used. Parsing constructs the whole
 * model, including loggers, so run it in a scratch directory on a scratch copy of the XML.
 */
public final class TorusLandingXmlProbe {
    private TorusLandingXmlProbe() { }

    public static void main(String[] args) throws Exception {
        if (args.length < 2 || (args.length - 2) % 4 != 0) {
            System.err.println("usage: TorusLandingXmlProbe file.xml likelihoodId [q1 q2 rho lambda]...");
            System.exit(2);
        }
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(false);
        final BeastParser parser = new BeastParser(new String[]{args[0]}, null, false, false, false, null);
        final Map<String, XMLObject> store;
        try (Reader reader = new FileReader(args[0])) {
            store = parser.parse(reader, false);
        }
        final TorusFourierTreeLikelihood like = (TorusFourierTreeLikelihood) store.get(args[1]).getNativeObject();
        final CompoundSymmetricMatrix precision = (CompoundSymmetricMatrix) like.getResiduePrecision(0);
        final Parameter diagonal = precision.getDiagonalParameter();
        final Parameter correlation = precision.getOffDiagonalParameter();
        final Parameter rate = like.getJumpRateParameter();
        final int points = Math.max(1, (args.length - 2) / 4);
        for (int i = 0; i < points; i++) {
            if (args.length > 2) {
                diagonal.setParameterValue(0, Double.parseDouble(args[2 + 4 * i]));
                diagonal.setParameterValue(1, Double.parseDouble(args[3 + 4 * i]));
                correlation.setParameterValue(0, Double.parseDouble(args[4 + 4 * i]));
                rate.setParameterValue(0, Double.parseDouble(args[5 + 4 * i]));
            }
            final double logL = like.getLogLikelihood();
            final StringBuilder sb = new StringBuilder("{");
            sb.append("\"q1\": ").append(diagonal.getParameterValue(0))
              .append(", \"q2\": ").append(diagonal.getParameterValue(1))
              .append(", \"rho\": ").append(correlation.getParameterValue(0))
              .append(", \"lambda\": ").append(rate.getParameterValue(0))
              .append(", \"logL\": ").append(json(logL));
            try {
                final long t0 = System.nanoTime();
                final double[] g = like.getLandingResidueGradientWithLogL(0);
                final double seconds = (System.nanoTime() - t0) * 1e-9;
                sb.append(", \"sweepLogL\": ").append(json(g[0]))
                  .append(", \"bitIdentical\": ").append(Double.doubleToRawLongBits(g[0]) == Double.doubleToRawLongBits(logL))
                  .append(", \"gradient\": [").append(json(g[1])).append(", ").append(json(g[2])).append(", ")
                  .append(json(g[3])).append(", ").append(json(g[4])).append("]")
                  .append(String.format(Locale.ROOT, ", \"gradientSeconds\": %.4f", seconds));
            } catch (ArithmeticException e) {
                sb.append(", \"gradientError\": \"").append(e.getMessage().replace('"', '\'')).append("\"");
            }
            sb.append("}");
            System.out.println(sb);
        }
    }

    private static String json(double x) {
        if (Double.isNaN(x)) return "NaN";
        if (Double.isInfinite(x)) return x > 0 ? "Infinity" : "-Infinity";
        return Double.toString(x);
    }
}
