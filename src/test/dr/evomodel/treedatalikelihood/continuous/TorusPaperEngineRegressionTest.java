package test.dr.evomodel.treedatalikelihood.continuous;

import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood.JumpLaw;
import junit.framework.TestCase;

/**
 * Pins the likelihood to the engine behind the published HA analysis (commit 838e4ebfb, the classes
 * of beast-torus-landing.jar, md5 29423c0535b945f109b845770385378d). Each scenario is a seeded chain
 * of one-evaluation proposals with store, accept and restore (TorusCacheFuzzTest.chainDigest), and its
 * expected value is the hash of every log-likelihood and every ancestral draw that engine produced on
 * the same chain. Any change to a single bit of any of them changes the hash.
 *
 * Run main() on a build to print the digests it produces. The scenarios avoid the configurations
 * where the published engine was wrong and this one deliberately differs: two evaluations between
 * store and restore, residues sharing a precision matrix, branch lengths that change without an
 * event, and non-positive-definite covariances.
 */
public class TorusPaperEngineRegressionTest extends TestCase {

    static TorusCacheFuzzTest.Config[] scenarios() {
        final TorusCacheFuzzTest.Config a = new TorusCacheFuzzTest.Config();       // landing, root moves
        a.slideSize = 1.0;
        a.tips = 8;
        a.steps = 60;
        a.seed = 101L;
        final TorusCacheFuzzTest.Config b = new TorusCacheFuzzTest.Config();       // parallel residues, rates
        b.residues = 9;
        b.perResidueRate = false;
        b.branchRates = true;
        b.nGrid = 16;
        b.steps = 40;
        b.seed = 102L;
        final TorusCacheFuzzTest.Config c = new TorusCacheFuzzTest.Config();       // shift, jumps, tip error
        c.law = JumpLaw.SHIFT;
        c.branchRates = true;
        c.steps = 60;
        c.seed = 103L;
        final TorusCacheFuzzTest.Config d = new TorusCacheFuzzTest.Config();       // shift, Brownian, Dirac
        d.law = JumpLaw.SHIFT;
        d.jumps = false;
        d.tipError = false;
        d.steps = 60;
        d.seed = 104L;
        final TorusCacheFuzzTest.Config e = new TorusCacheFuzzTest.Config();       // shift, Brownian, tip error
        e.law = JumpLaw.SHIFT;
        e.jumps = false;
        e.steps = 60;
        e.seed = 105L;
        final TorusCacheFuzzTest.Config f = new TorusCacheFuzzTest.Config();       // landing, one dimension
        f.dim = 1;
        f.nGrid = 64;
        f.steps = 60;
        f.seed = 106L;
        final TorusCacheFuzzTest.Config g = new TorusCacheFuzzTest.Config();       // shift, three dimensions
        g.dim = 3;
        g.nGrid = 8;
        g.tips = 7;
        g.law = JumpLaw.SHIFT;
        g.steps = 40;
        g.seed = 107L;
        final TorusCacheFuzzTest.Config h = new TorusCacheFuzzTest.Config();       // landing at nGrid 256
        h.nGrid = 256;
        h.tips = 6;
        h.residues = 1;
        h.steps = 12;
        h.seed = 108L;
        return new TorusCacheFuzzTest.Config[]{a, b, c, d, e, f, g, h};
    }

    /** Digests of the published engine on scenarios() a..h, in order (main() run against
     *  beast-torus-landing.jar, md5 29423c0535b945f109b845770385378d, on 12 Oct 2026). */
    private static final long[] PUBLISHED = {
            0xcd083f9667f029e9L,   // a  landing, root moves
            0xa426c456b7f72176L,   // b  landing, 9 residues in parallel, branch rates
            0x75415b4e5f99c5bfL,   // c  shift, jumps, tip error, branch rates
            0xc52c56960fc5c3c4L,   // d  shift, Brownian, Dirac tips
            0x38384816e9033b57L,   // e  shift, Brownian, tip error
            0xf1fca91000534993L,   // f  landing, one dimension
            0x44accfc1f98a1642L,   // g  shift, three dimensions
            0x1ed90257a661c6ffL,   // h  landing, nGrid 256
    };

    public void testBitIdenticalToPublishedEngine() throws Exception {
        final TorusCacheFuzzTest.Config[] scenarios = scenarios();
        int differing = 0;
        final StringBuilder report = new StringBuilder();
        for (int i = 0; i < scenarios.length; i++) {
            final long digest = TorusCacheFuzzTest.chainDigest(scenarios[i]);
            if (digest != PUBLISHED[i]) {
                differing++;
                report.append(String.format(" scenario %c: %016x, published engine %016x;", 'a' + i, digest, PUBLISHED[i]));
            }
        }
        assertEquals("differs from the published engine:" + report, 0, differing);
    }

    public static void main(String[] args) throws Exception {
        java.util.logging.Logger.getLogger("dr.evomodel").setLevel(java.util.logging.Level.OFF);
        final TorusCacheFuzzTest.Config[] scenarios = scenarios();
        for (int i = 0; i < scenarios.length; i++) {
            System.out.printf("            0x%016xL,   // %c%n", TorusCacheFuzzTest.chainDigest(scenarios[i]), 'a' + i);
        }
    }
}
