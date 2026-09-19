"""Matched negative controls for the expert feature-to-concept graph.

The semantic test for this adapter is the real expert graph against a graph with the
same statistics and the wrong assignments. Both controls below keep the number of
concepts, the total edge count, the per-concept degree and the global relation-type
counts, so an arm using one is exactly parameter-matched to the real arm and differs
only in which features feed which concept.

Deterministic given a seed, through a private torch.Generator, so they never disturb
the caller's global RNG state. Diagnostic fields are dropped: they are keyed by
feature id, which a control has deliberately reassigned.
"""

from __future__ import annotations

import torch

from .expert_bundle import NO_EDGE, ExpertBundleError, ExpertPriorBundle

EXPERT_GRAPH_CONTROLS = ("real", "permuted", "random_matched")


def _rebuild(bundle: ExpertPriorBundle, mask: torch.Tensor, index: torch.Tensor) -> ExpertPriorBundle:
    return ExpertPriorBundle(
        feature_ids=bundle.feature_ids,
        concept_ids=bundle.concept_ids,
        relation_mask=mask,
        relation_type_index=index,
        relation_type_vocabulary=bundle.relation_type_vocabulary,
    )


def permuted_graph(bundle: ExpertPriorBundle, seed: int) -> ExpertPriorBundle:
    """The same graph with the feature axis permuted: edge (j, c, r) becomes (pi(j), c, r).

    A bijection cannot collide two of a concept's features, so every statistic survives
    exactly, including each concept's relation-type composition. What is destroyed is
    which features a concept draws on; which features are grouped together is not, so
    this arm keeps the co-occurrence structure and gets the groups wrong.

    Redraws if the permutation is the identity: a control that changed nothing would
    silently duplicate the real arm.
    """
    m = bundle.m
    gen = torch.Generator().manual_seed(seed)
    identity = torch.arange(m)
    perm = torch.randperm(m, generator=gen)
    while m > 1 and torch.equal(perm, identity):
        perm = torch.randperm(m, generator=gen)

    mask = torch.zeros_like(bundle.relation_mask)
    index = torch.full_like(bundle.relation_type_index, NO_EDGE)
    mask[perm] = bundle.relation_mask
    index[perm] = bundle.relation_type_index
    return _rebuild(bundle, mask, index)


def random_matched_graph(bundle: ExpertPriorBundle, seed: int) -> ExpertPriorBundle:
    """A fresh random graph with the same concept count, degrees and relation-type counts.

    Each concept draws its own degree worth of distinct features uniformly from all m.
    Relation types are the original multiset of all edge labels, permuted and dealt out,
    so the global counts match while the per-concept composition does not.

    Stronger than `permuted_graph`: the grouping structure is destroyed as well as the
    assignment.
    """
    m, K = bundle.m, bundle.n_concepts
    degrees = bundle.edges_per_concept
    if int(degrees.max()) > m:
        raise ExpertBundleError(
            f"a concept has degree {int(degrees.max())}, more than the {m} available features"
        )
    gen = torch.Generator().manual_seed(seed)

    labels = bundle.relation_type_index[bundle.relation_mask]
    labels = labels[torch.randperm(labels.numel(), generator=gen)]

    mask = torch.zeros(m, K, dtype=torch.bool)
    index = torch.full((m, K), NO_EDGE, dtype=torch.long)
    taken = 0
    for c in range(K):
        degree = int(degrees[c])
        chosen = torch.randperm(m, generator=gen)[:degree]
        mask[chosen, c] = True
        index[chosen, c] = labels[taken:taken + degree]
        taken += degree
    return _rebuild(bundle, mask, index)


def apply_graph_control(bundle: ExpertPriorBundle, control: str, seed: int) -> ExpertPriorBundle:
    """Dispatch one named control. "real" returns the bundle unchanged."""
    if control not in EXPERT_GRAPH_CONTROLS:
        raise ValueError(f"unknown control {control!r}; choose from {EXPERT_GRAPH_CONTROLS}")
    if control == "real":
        return bundle
    if control == "permuted":
        return permuted_graph(bundle, seed)
    return random_matched_graph(bundle, seed)
