package test.dr.evomodel.treedatalikelihood.continuous;

import dr.app.beast.BeastParser;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.inference.model.CompoundSymmetricMatrix;
import dr.inference.model.Parameter;
import dr.xml.XMLObject;
import junit.framework.TestCase;

import java.io.File;
import java.io.FileReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Map;

/**
 * Landing gradient on real HA production XMLs (job 846999, K6, nGrid 128) at logged posterior
 * draws, against the independent numpy port of the as-ran engine (reference values from the
 * derivation spec, section 10; that port reproduces the 846999 logged likelihoods to <= 3e-10).
 * Skipped when the production XMLs are not on this machine. The XML is parsed from a temporary
 * copy whose logger file names point into a temporary directory.
 */
public class TorusLandingProductionGradientTest extends TestCase {
    private static final String DIR = "/home/sjbhatt/influenza/hx2/trait/release_20261002_K6/prod316_K6/";

    private boolean previousValidation;

    protected void setUp() {
        previousValidation = TorusFourierTreeLikelihood.isValidateEveryEvaluation();
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(false);
    }

    protected void tearDown() {
        TorusFourierTreeLikelihood.setValidateEveryEvaluation(previousValidation);
    }

    // {q1, q2, rho, lambda, logL, dq1, dq2, drho, dlambda}
    private static final double[][] RES49 = {
            {681.2920690579936, 681.2920690579936, 0.0, 0.0068129207, 214.940043600364,
                    -0.003462115704377113, 0.005143717204616309, 20.25448769732197, -26.814431236462767},
            {1579.2023666484952, 3222.0407938824846, 0.8880318829989682, 0.0070167834313880715, 229.5152923872233,
                    0.0005327173155944594, -5.325407196572894e-05, -5.56948037921183, -53.68269844459295}};
    private static final double[][] RES139 = {
            {681.2920690579936, 681.2920690579936, 0.0, 0.0068129207, 107.09297241969101,
                    0.016766941669305816, -0.007085143258509351, -12.915384184647268, 587.5902371141848},
            {15644.213081488671, 1112.752862968525, -0.5836047598414045, 0.013356675811821383, 116.57917610196914,
                    -6.172088267801146e-05, -0.0007777310993333387, -7.833903191579312, 198.76482476234898}};

    public void testResidue49HalfMissingTips() throws Exception {
        check("res_49.xml", RES49);
    }

    public void testResidue139SmallDiffusion() throws Exception {
        check("res_139.xml", RES139);
    }

    private void check(String name, double[][] points) throws Exception {
        File xml = new File(DIR + name);
        if (!xml.isFile()) {
            System.err.println(getName() + ": " + xml + " not found; skipped");
            return;
        }
        File tmp = Files.createTempDirectory("landing_prod_grad").toFile();
        try {
            String text = new String(Files.readAllBytes(xml.toPath()), StandardCharsets.UTF_8)
                    .replaceAll("fileName=\"[^\"]*\"", "fileName=\"" + new File(tmp, "probe.log").getPath() + "\"")
                    .replaceAll("operatorAnalysis=\"[^\"]*\"", "");
            File copy = new File(tmp, name);
            Files.write(copy.toPath(), text.getBytes(StandardCharsets.UTF_8));
            BeastParser parser = new BeastParser(new String[]{copy.getPath()}, null, false, false, false, null);
            Map<String, XMLObject> store;
            try (Reader reader = new FileReader(copy)) {
                store = parser.parse(reader, false);
            }
            TorusFourierTreeLikelihood like = (TorusFourierTreeLikelihood) store.get("tfl").getNativeObject();
            assertEquals(TorusFourierTreeLikelihood.JumpLaw.LANDING, like.getJumpLaw());
            CompoundSymmetricMatrix precision = (CompoundSymmetricMatrix) like.getResiduePrecision(0);
            Parameter diagonal = precision.getDiagonalParameter();
            Parameter correlation = precision.getOffDiagonalParameter();
            Parameter rate = like.getJumpRateParameter();
            for (double[] pt : points) {
                diagonal.setParameterValue(0, pt[0]);
                diagonal.setParameterValue(1, pt[1]);
                correlation.setParameterValue(0, pt[2]);
                rate.setParameterValue(0, pt[3]);
                double logL = like.getLogLikelihood();
                assertEquals(name + " log L vs reference", pt[4], logL, 1e-9);
                double[] g = like.getLandingResidueGradientWithLogL(0);
                assertEquals(name + " sweep log L must be bit-identical",
                        Double.doubleToRawLongBits(logL), Double.doubleToRawLongBits(g[0]));
                for (int p = 0; p < 4; p++) {
                    assertEquals(name + " gradient p=" + p, pt[5 + p], g[1 + p], 1e-8 * Math.abs(pt[5 + p]));
                }
            }
        } finally {
            for (File f : tmp.listFiles()) f.delete();
            tmp.delete();
        }
    }
}
