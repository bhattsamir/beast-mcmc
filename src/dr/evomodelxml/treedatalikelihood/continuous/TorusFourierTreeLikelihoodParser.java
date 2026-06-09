/*
 * TorusFourierTreeLikelihoodParser.java
 *
 * @author Samir Bhatt
 */

package dr.evomodelxml.treedatalikelihood.continuous;

import dr.evolution.tree.NodeRef;
import dr.evolution.tree.Tree;
import dr.evolution.util.Taxon;
import dr.evomodel.branchratemodel.BranchRateModel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierKernel;
import dr.evomodel.treedatalikelihood.continuous.TorusFourierTreeLikelihood;
import dr.inference.model.MatrixParameter;
import dr.xml.*;

import java.util.ArrayList;
import java.util.List;

/**
 * Parses a {@link TorusFourierTreeLikelihood}: a spectral (FFT) Felsenstein pruning
 * likelihood for independent 2D wrapped-Brownian diffusions of backbone dihedrals.
 * Each &lt;residue&gt; block names a 2-dimensional tip trait (phi, psi) read from the
 * taxon attributes and supplies a 2x2 precision matrix for that residue.
 *
 * @author Samir Bhatt
 */
public class TorusFourierTreeLikelihoodParser extends AbstractXMLObjectParser {

    public static final String TORUS_FOURIER_LIKELIHOOD = TorusFourierTreeLikelihood.TORUS_FOURIER_LIKELIHOOD;
    public static final String RESIDUE = "residue";
    public static final String TRAIT_NAME = "traitName";
    public static final String N_GRID = "nGrid";

    public String getParserName() {
        return TORUS_FOURIER_LIKELIHOOD;
    }

    public Object parseXMLObject(XMLObject xo) throws XMLParseException {

        final Tree tree = (Tree) xo.getChild(Tree.class);
        final BranchRateModel branchRateModel = (BranchRateModel) xo.getChild(BranchRateModel.class);
        final int nGrid = xo.getAttribute(N_GRID, 64);
        final TorusFourierKernel kernel = new TorusFourierKernel(nGrid);
        final int nExt = tree.getExternalNodeCount();

        final List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        final List<double[][]> tip = new ArrayList<double[][]>();

        for (int i = 0; i < xo.getChildCount(); i++) {
            if (xo.getChild(i) instanceof XMLObject) {
                final XMLObject cxo = (XMLObject) xo.getChild(i);
                if (cxo.getName().equals(RESIDUE)) {
                    final String traitName = cxo.getStringAttribute(TRAIT_NAME);
                    final MatrixParameter precision = (MatrixParameter) cxo.getChild(MatrixParameter.class);
                    if (precision == null) {
                        throw new XMLParseException("residue '" + traitName
                                + "' requires a 2x2 matrixParameter precision");
                    }
                    final double[][] data = new double[nExt][];
                    for (int e = 0; e < nExt; e++) {
                        final NodeRef ext = tree.getExternalNode(e);
                        final Taxon taxon = tree.getNodeTaxon(ext);
                        final Object attr = (taxon == null) ? null : taxon.getAttribute(traitName);
                        data[ext.getNumber()] = parsePair(attr, traitName, taxon);
                    }
                    precisions.add(precision);
                    tip.add(data);
                }
            }
        }
        if (precisions.isEmpty()) {
            throw new XMLParseException(TORUS_FOURIER_LIKELIHOOD + " requires at least one <" + RESIDUE + ">");
        }

        final double[][][] tipData = tip.toArray(new double[0][][]);
        String id = xo.getId();
        if (id == null) id = TORUS_FOURIER_LIKELIHOOD;
        return new TorusFourierTreeLikelihood(id, tree, branchRateModel, precisions, tipData, kernel);
    }

    /** Parse a taxon attribute holding two angles (radians), as Object[]/double[]/string. */
    private double[] parsePair(Object attr, String traitName, Taxon taxon) throws XMLParseException {
        final String who = (taxon == null) ? "?" : taxon.getId();
        if (attr == null) {
            throw new XMLParseException("taxon '" + who + "' lacks attribute '" + traitName + "'");
        }
        if (attr instanceof Object[]) {
            final Object[] a = (Object[]) attr;
            if (a.length != 2) throw new XMLParseException("attribute '" + traitName + "' needs 2 values for " + who);
            return new double[]{((Number) a[0]).doubleValue(), ((Number) a[1]).doubleValue()};
        }
        if (attr instanceof double[]) {
            final double[] a = (double[]) attr;
            if (a.length != 2) throw new XMLParseException("attribute '" + traitName + "' needs 2 values for " + who);
            return new double[]{a[0], a[1]};
        }
        final String[] parts = attr.toString().trim().split("\\s+");
        if (parts.length != 2) throw new XMLParseException("attribute '" + traitName + "' needs 2 values for " + who);
        return new double[]{Double.parseDouble(parts[0]), Double.parseDouble(parts[1])};
    }

    public XMLSyntaxRule[] getSyntaxRules() {
        return new XMLSyntaxRule[]{
                AttributeRule.newIntegerRule(N_GRID, true),
                new ElementRule(Tree.class),
                new ElementRule(BranchRateModel.class, true),
                new ElementRule(RESIDUE, new XMLSyntaxRule[]{
                        AttributeRule.newStringRule(TRAIT_NAME),
                        new ElementRule(MatrixParameter.class)
                }, 1, Integer.MAX_VALUE),
        };
    }

    public String getParserDescription() {
        return "Spectral (FFT) Felsenstein pruning likelihood for independent 2D wrapped-Brownian "
                + "diffusions of protein backbone dihedrals on a flat torus.";
    }

    public Class getReturnType() {
        return TorusFourierTreeLikelihood.class;
    }
}
