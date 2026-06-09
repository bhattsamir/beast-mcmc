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
 * likelihood for independent d-dimensional wrapped-Brownian diffusions of backbone
 * dihedrals. Each &lt;residue&gt; block names a d-dimensional tip trait (e.g. phi,
 * psi) read from the taxon attributes and supplies a dim x dim precision matrix; the
 * common dimension d is inferred from the precision matrices.
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

        // collect residue specifications and infer the common dimension
        final List<String> traitNames = new ArrayList<String>();
        final List<MatrixParameter> precisions = new ArrayList<MatrixParameter>();
        for (int i = 0; i < xo.getChildCount(); i++) {
            if (xo.getChild(i) instanceof XMLObject) {
                final XMLObject cxo = (XMLObject) xo.getChild(i);
                if (cxo.getName().equals(RESIDUE)) {
                    final MatrixParameter precision = (MatrixParameter) cxo.getChild(MatrixParameter.class);
                    if (precision == null) {
                        throw new XMLParseException("a <" + RESIDUE + "> requires a square precision matrixParameter");
                    }
                    traitNames.add(cxo.getStringAttribute(TRAIT_NAME));
                    precisions.add(precision);
                }
            }
        }
        if (precisions.isEmpty()) {
            throw new XMLParseException(TORUS_FOURIER_LIKELIHOOD + " requires at least one <" + RESIDUE + ">");
        }

        final int dim = precisions.get(0).getRowDimension();
        for (MatrixParameter p : precisions) {
            if (p.getRowDimension() != dim || p.getColumnDimension() != dim) {
                throw new XMLParseException("all <" + RESIDUE + "> precision matrices must be square of the same dimension ("
                        + dim + ")");
            }
        }
        final TorusFourierKernel kernel = new TorusFourierKernel(dim, nGrid);

        // read the per-taxon d-vectors for each residue
        final int nExt = tree.getExternalNodeCount();
        final double[][][] tipData = new double[precisions.size()][][];
        for (int r = 0; r < traitNames.size(); r++) {
            final String traitName = traitNames.get(r);
            final double[][] data = new double[nExt][];
            for (int e = 0; e < nExt; e++) {
                final NodeRef ext = tree.getExternalNode(e);
                final Taxon taxon = tree.getNodeTaxon(ext);
                final Object attr = (taxon == null) ? null : taxon.getAttribute(traitName);
                data[ext.getNumber()] = parseVector(attr, dim, traitName, taxon);
            }
            tipData[r] = data;
        }

        String id = xo.getId();
        if (id == null) id = TORUS_FOURIER_LIKELIHOOD;
        return new TorusFourierTreeLikelihood(id, tree, branchRateModel, precisions, tipData, kernel);
    }

    /** Parse a taxon attribute holding dim angles (radians), as Object[]/double[]/string. */
    private double[] parseVector(Object attr, int dim, String traitName, Taxon taxon) throws XMLParseException {
        final String who = (taxon == null) ? "?" : taxon.getId();
        if (attr == null) {
            throw new XMLParseException("taxon '" + who + "' lacks attribute '" + traitName + "'");
        }
        final double[] v = new double[dim];
        if (attr instanceof Object[]) {
            final Object[] a = (Object[]) attr;
            if (a.length != dim) throw new XMLParseException("attribute '" + traitName + "' needs " + dim + " values for " + who);
            for (int i = 0; i < dim; i++) v[i] = ((Number) a[i]).doubleValue();
            return v;
        }
        if (attr instanceof double[]) {
            final double[] a = (double[]) attr;
            if (a.length != dim) throw new XMLParseException("attribute '" + traitName + "' needs " + dim + " values for " + who);
            System.arraycopy(a, 0, v, 0, dim);
            return v;
        }
        final String[] parts = attr.toString().trim().split("\\s+");
        if (parts.length != dim) throw new XMLParseException("attribute '" + traitName + "' needs " + dim + " values for " + who);
        for (int i = 0; i < dim; i++) v[i] = Double.parseDouble(parts[i]);
        return v;
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
        return "Spectral (FFT) Felsenstein pruning likelihood for independent d-dimensional wrapped-Brownian "
                + "diffusions of protein backbone dihedrals on a flat torus.";
    }

    public Class getReturnType() {
        return TorusFourierTreeLikelihood.class;
    }
}
