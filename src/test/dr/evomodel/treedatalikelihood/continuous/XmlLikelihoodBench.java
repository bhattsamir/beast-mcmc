package test.dr.evomodel.treedatalikelihood.continuous;

import dr.app.beast.BeastParser;
import dr.app.beast.BeastVersion;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.inference.model.Parameter;
import dr.xml.XMLObject;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

/**
 * Times the torus likelihood of a production trait XML, parsed by BEAST's own parser, through a
 * fixed, seeded sequence of the moves the production chain makes (scale the jump rate, scale a
 * precision diagonal, random-walk the correlation), each followed by an evaluation and then an
 * accept or a restore, as in MCMC. Every evaluation recomputes the residue, so its time is the
 * cost of an evaluated state of the chain. Prints each log-likelihood as raw bits (for diffing two engines on
 * real data) and the timing summary on stderr. The XML must use the ids tfl, lambda, diag_0 and
 * off_0, and BeastParser needs beast.properties on the classpath (src/, or the dist jar).
 *
 * Usage: XmlLikelihoodBench file.xml [evaluations=60] [warmup=5] [seed=1]
 */
public class XmlLikelihoodBench {

    public static void main(String[] args) throws Exception {
        final String xml = args[0];
        final int evaluations = args.length > 1 ? Integer.parseInt(args[1]) : 60;
        final int warmup = args.length > 2 ? Integer.parseInt(args[2]) : 5;
        final long seed = args.length > 3 ? Long.parseLong(args[3]) : 1L;
        java.util.logging.Logger.getLogger("dr.evomodel").setLevel(java.util.logging.Level.WARNING);
        final long tParse = System.nanoTime();
        final BeastParser parser = new BeastParser(new String[]{xml}, null, false, false, false, new BeastVersion());
        final Map<String, XMLObject> store = parser.parse(new FileReader(xml), false);
        final TorusFourierTreeLikelihood like = (TorusFourierTreeLikelihood) store.get("tfl").getNativeObject();
        final Parameter lambda = (Parameter) store.get("lambda").getNativeObject();
        final Parameter diag = (Parameter) store.get("diag_0").getNativeObject();
        final Parameter corr = (Parameter) store.get("off_0").getNativeObject();
        like.makeDirty();   // parsing the <mcmc> element has already evaluated it: time a fresh full evaluation
        final long tFirst = System.nanoTime();
        System.out.println("initial " + Long.toHexString(Double.doubleToRawLongBits(like.getLogLikelihood())));
        final long tDone = System.nanoTime();
        final Random rng = new Random(seed);
        final double[] ms = new double[evaluations];
        for (int i = 0; i < evaluations; i++) {
            like.storeModelState();
            final int move = rng.nextInt(3);
            if (move == 0) {
                lambda.setParameterValue(0, lambda.getParameterValue(0) * Math.exp(0.5 * (rng.nextDouble() - 0.5)));
            } else if (move == 1) {
                final int j = rng.nextInt(2);
                diag.setParameterValue(j, diag.getParameterValue(j) * Math.exp(0.5 * (rng.nextDouble() - 0.5)));
            } else {
                corr.setParameterValue(0, Math.max(-0.9, Math.min(0.9, corr.getParameterValue(0) + 0.1 * (rng.nextDouble() - 0.5))));
            }
            final long t0 = System.nanoTime();
            final double value = like.getLogLikelihood();
            ms[i] = (System.nanoTime() - t0) * 1e-6;
            final boolean accept = rng.nextInt(3) == 0;
            if (accept) like.acceptModelState();
            else like.restoreModelState();
            System.out.println("eval " + i + " move " + move + " " + Long.toHexString(Double.doubleToRawLongBits(value))
                    + (accept ? " accept" : " restore " + Long.toHexString(Double.doubleToRawLongBits(like.getLogLikelihood()))));
        }
        final double[] timed = Arrays.copyOfRange(ms, Math.min(warmup, evaluations - 1), evaluations);
        Arrays.sort(timed);
        double mean = 0.0;
        for (double t : timed) mean += t;
        mean /= timed.length;
        System.err.printf("%s: parse %.1f s (includes one evaluation), fresh full evaluation %.0f ms; %d timed evaluations: median %.1f ms, mean %.1f ms, "
                        + "min %.1f, max %.1f%n", xml, (tFirst - tParse) * 1e-9, (tDone - tFirst) * 1e-6, timed.length,
                timed[timed.length / 2], mean, timed[0], timed[timed.length - 1]);
    }
}
