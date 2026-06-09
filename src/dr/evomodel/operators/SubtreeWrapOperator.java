/*
 * SubtreeWrapOperator.java
 *
 * Block-flip MCMC operator for the integer wraps of a WrappedContinuousTraitDataModel.
 * Picks a random internal node (a clade), a random (trait, dim) pair, a step in
 * {+1, -1}, and adds 'step' to every wrap entry belonging to a leaf descended
 * from that clade for that (trait, dim). Pinned wraps (taxon 0 gauge fix and
 * missing-data coords) are skipped. If any non-pinned wrap in the clade would
 * leave the parameter's bounds, the proposal is rejected (logq = -infinity).
 *
 * Proposal is symmetric (clade uniform over internal nodes; (trait, dim)
 * uniform; +1/-1 equiprobable), so the log Hastings ratio is 0 for any
 * accepted proposal. The reverse proposal flips the same clade by -step
 * over the same (trait, dim) pair and reaches the original state.
 *
 * Use this alongside the existing per-coord randomWalkIntegerOperator to
 * couple wrap mixing to topology mixing.
 *
 * @author Samir Bhatt
 */

package dr.evomodel.operators;

import dr.evolution.tree.NodeRef;
import dr.evolution.util.Taxon;
import dr.evomodel.tree.TreeModel;
import dr.evomodel.treedatalikelihood.continuous.WrappedContinuousTraitDataModel;
import dr.inference.model.Parameter;
import dr.inference.operators.SimpleMCMCOperator;
import dr.math.MathUtils;

import java.util.ArrayList;
import java.util.List;

public class SubtreeWrapOperator extends SimpleMCMCOperator {

    private final TreeModel treeModel;
    private final WrappedContinuousTraitDataModel dataModel;
    private final Parameter wraps;
    private final int numTraits;
    private final int dimTrait;
    private final int numTaxa;
    private final int[] nodeNumberToTaxonIndex;  // tree external-node number -> wraps taxon index

    public SubtreeWrapOperator(TreeModel treeModel,
                               WrappedContinuousTraitDataModel dataModel,
                               double weight) {
        this.treeModel = treeModel;
        this.dataModel = dataModel;
        this.wraps = dataModel.getWrapsParameter();
        this.numTraits = dataModel.getTraitCount();
        this.dimTrait = dataModel.getTraitDimension();
        this.numTaxa = treeModel.getExternalNodeCount();

        // Build external-node-number -> taxon-index map.
        // The wraps parameter is laid out as (taxonIndex, trait, dim) where
        // taxonIndex matches the data model's underlying CompoundParameter
        // ordering, which in turn matches treeModel.getTaxonIndex(...).
        this.nodeNumberToTaxonIndex = new int[treeModel.getNodeCount()];
        for (int i = 0; i < this.nodeNumberToTaxonIndex.length; i++) {
            this.nodeNumberToTaxonIndex[i] = -1;
        }
        for (int i = 0; i < numTaxa; i++) {
            NodeRef leaf = treeModel.getExternalNode(i);
            Taxon t = treeModel.getNodeTaxon(leaf);
            int taxonIdx = treeModel.getTaxonIndex(t);
            this.nodeNumberToTaxonIndex[leaf.getNumber()] = taxonIdx;
        }

        setWeight(weight);
    }

    @Override
    public double doOperation() {
        final int nInternal = treeModel.getInternalNodeCount();
        if (nInternal < 1) return Double.NEGATIVE_INFINITY;

        // pick a clade (any internal node; root is fine -- means "all leaves")
        NodeRef clade = treeModel.getInternalNode(MathUtils.nextInt(nInternal));
        // pick (trait, dim)
        final int trait = MathUtils.nextInt(numTraits);
        final int dim   = MathUtils.nextInt(dimTrait);
        // pick step
        final int step = MathUtils.nextBoolean() ? 1 : -1;

        // collect taxon indices for leaves under this clade
        List<Integer> leafTaxa = new ArrayList<>();
        collectLeafTaxa(clade, leafTaxa);
        if (leafTaxa.isEmpty()) return Double.NEGATIVE_INFINITY;

        // figure out which wrap entries we'll flip; check bounds
        final double lower = wraps.getBounds().getLowerLimit(0);
        final double upper = wraps.getBounds().getUpperLimit(0);
        int[] toFlip = new int[leafTaxa.size()];
        int nFlip = 0;
        for (int taxonIdx : leafTaxa) {
            int wrapIdx = taxonIdx * numTraits * dimTrait + trait * dimTrait + dim;
            if (dataModel.isWrapPinned(wrapIdx)) continue;
            double proposed = wraps.getParameterValue(wrapIdx) + step;
            if (proposed < lower || proposed > upper) {
                return Double.NEGATIVE_INFINITY;
            }
            toFlip[nFlip++] = wrapIdx;
        }
        if (nFlip == 0) return Double.NEGATIVE_INFINITY;  // entire clade was pinned

        // apply the flip
        for (int i = 0; i < nFlip; i++) {
            int wrapIdx = toFlip[i];
            wraps.setParameterValue(wrapIdx, wraps.getParameterValue(wrapIdx) + step);
        }

        // symmetric proposal => logHR = 0
        return 0.0;
    }

    private void collectLeafTaxa(NodeRef node, List<Integer> out) {
        if (treeModel.isExternal(node)) {
            int taxonIdx = nodeNumberToTaxonIndex[node.getNumber()];
            if (taxonIdx >= 0) out.add(taxonIdx);
        } else {
            final int n = treeModel.getChildCount(node);
            for (int i = 0; i < n; i++) {
                collectLeafTaxa(treeModel.getChild(node, i), out);
            }
        }
    }

    @Override
    public String getOperatorName() {
        return "subtreeWrap(" + wraps.getId() + ")";
    }

    public String getPerformanceSuggestion() { return ""; }
}
