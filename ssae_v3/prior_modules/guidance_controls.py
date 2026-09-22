"""Matched negative controls for the prior-guidance bundle.

Each control replaces one source (the embedding subspace or the graph subspace) with
a same-capacity object that destroys a specific, named piece of structure, so a
measured gain can be checked against "extra capacity" rather than "the LLM's/expert's
actual content". Deterministic given a seed, always through a private
`torch.Generator`, never the caller's global RNG state.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .controls import random_basis
from .guidance_bundle import (
    GuidanceBundleError,
    PriorGuidanceBundle,
    build_graph_subspace,
)

# A permuted graph that reproduces the original is a no-op, so draws are retried.
_MAX_PERMUTATION_DRAWS = 100

EMBEDDING_VARIANTS = ("none", "real", "random")
GRAPH_VARIANTS = ("none", "real", "permuted", "degree_matched")


def make_random_embedding_control(m: int, k: int, seed: int) -> Tensor:
    """A random rank-k orthonormal basis of R^m, standing in for U_k.

    Delegates to `controls.random_basis`, which is exactly this construction (the Q
    factor of a QR decomposition of a random Gaussian matrix), rather than a second
    copy of the same algorithm.
    """
    return random_basis(m, k, seed)


def make_permuted_graph_control(C: Tensor, seed: int) -> Tensor:
    """Permute the feature rows of C, preserving concept count, edge count, per-concept
    degree and rank exactly, while destroying which feature sits at each position.

    Redraws while the result equals C, so a no-op control cannot silently pass as the
    real graph. Testing the matrix rather than the permutation matters because rows of
    equal degree attached to the same concepts are identical, so a non-identity
    permutation can still leave the incidence unchanged.
    """
    m = C.shape[0]
    gen = torch.Generator().manual_seed(seed)
    for _ in range(_MAX_PERMUTATION_DRAWS):
        perm = torch.randperm(m, generator=gen)
        permuted = C[perm].clone()
        if not torch.equal(permuted, C):
            return permuted
    raise GuidanceBundleError(
        "every row permutation of this graph reproduces it; there is nothing for a "
        "permuted control to destroy"
    )


def _degree_sequences(C: Tensor) -> tuple:
    return C.sum(dim=1), C.sum(dim=0)


def make_degree_matched_graph_control(
    C: Tensor,
    seed: int,
    target_rank: int,
    max_attempts: int = 20,
) -> Tensor:
    """Degree-preserving rewiring of the bipartite (feature, concept) incidence.

    Repeated double-edge swaps: pick edges (f1, c1) and (f2, c2) with f1 != f2,
    c1 != c2, and neither (f1, c2) nor (f2, c1) already present, then swap them to
    (f1, c2) and (f2, c1). Each swap preserves both degree sequences and the edge
    count exactly. Runs `10 * n_edges` successful swaps, then requires the result to
    have rank `target_rank` and to differ from C; otherwise the whole rewiring is
    restarted with the next seed offset, up to `max_attempts` times.
    """
    feature_degrees, concept_degrees = _degree_sequences(C)
    n_edges = int(C.sum())
    if n_edges == 0:
        raise GuidanceBundleError("cannot rewire a graph with no edges")

    for attempt in range(max_attempts):
        gen = torch.Generator().manual_seed(seed + attempt)
        working = C.clone()
        target_swaps = 10 * n_edges
        done = 0
        stalls = 0
        max_stalls = 200 * max(n_edges, 1)
        while done < target_swaps and stalls < max_stalls:
            edges = working.nonzero(as_tuple=False)
            idx = torch.randint(0, edges.shape[0], (2,), generator=gen)
            f1, c1 = int(edges[idx[0], 0]), int(edges[idx[0], 1])
            f2, c2 = int(edges[idx[1], 0]), int(edges[idx[1], 1])
            if f1 == f2 or c1 == c2:
                stalls += 1
                continue
            if bool(working[f1, c2]) or bool(working[f2, c1]):
                stalls += 1
                continue
            working[f1, c1] = 0.0
            working[f2, c2] = 0.0
            working[f1, c2] = 1.0
            working[f2, c1] = 1.0
            done += 1
            stalls = 0

        if done < target_swaps:
            continue
        new_feature_degrees, new_concept_degrees = _degree_sequences(working)
        if not torch.equal(new_feature_degrees, feature_degrees):
            continue
        if not torch.equal(new_concept_degrees, concept_degrees):
            continue
        if torch.equal(working, C):
            continue
        _, rank = build_graph_subspace(working)
        if rank != target_rank:
            continue
        return working

    raise GuidanceBundleError(
        f"could not produce a degree-matched graph control at rank {target_rank} "
        f"within {max_attempts} attempts"
    )


def apply_guidance_controls(
    bundle: PriorGuidanceBundle,
    embedding_variant: str,
    graph_variant: str,
    seed: int,
) -> PriorGuidanceBundle:
    """Return a new PriorGuidanceBundle with the requested controls applied.

    Raises if a variant asks for a control of a prior the input bundle does not
    carry. Records the applied variant and the seed in the corresponding metadata
    dict (keys "variant" and "control_seed").
    """
    if embedding_variant not in EMBEDDING_VARIANTS:
        raise GuidanceBundleError(
            f"unknown embedding variant {embedding_variant!r}; choose from {EMBEDDING_VARIANTS}"
        )
    if graph_variant not in GRAPH_VARIANTS:
        raise GuidanceBundleError(
            f"unknown graph variant {graph_variant!r}; choose from {GRAPH_VARIANTS}"
        )

    U_k: Optional[Tensor] = bundle.U_k
    P_U: Optional[Tensor] = bundle.P_U
    embedding_rank = bundle.embedding_rank
    embedding_meta = dict(bundle.embedding_source_metadata)

    if embedding_variant == "none":
        U_k, P_U, embedding_rank = None, None, 0
        embedding_meta = {}
    elif embedding_variant == "random":
        if bundle.U_k is None:
            raise GuidanceBundleError("embedding variant 'random' requested but bundle has no embedding prior")
        U_k = make_random_embedding_control(bundle.m, bundle.embedding_rank, seed)
        P_U = U_k @ U_k.T
        embedding_meta["variant"] = "random"
        embedding_meta["control_seed"] = seed
    elif embedding_variant == "real":
        if bundle.U_k is None:
            raise GuidanceBundleError("embedding variant 'real' requested but bundle has no embedding prior")
        embedding_meta["variant"] = "real"

    C: Optional[Tensor] = bundle.C
    Q_C: Optional[Tensor] = bundle.Q_C
    P_C: Optional[Tensor] = bundle.P_C
    graph_rank = bundle.graph_rank
    graph_meta = dict(bundle.graph_source_metadata)

    if graph_variant == "none":
        C, Q_C, P_C, graph_rank = None, None, None, 0
        graph_meta = {}
    elif graph_variant == "permuted":
        if bundle.C is None:
            raise GuidanceBundleError("graph variant 'permuted' requested but bundle has no graph prior")
        C = make_permuted_graph_control(bundle.C, seed)
        Q_C, graph_rank = build_graph_subspace(C)
        P_C = Q_C @ Q_C.T
        graph_meta["variant"] = "permuted"
        graph_meta["control_seed"] = seed
    elif graph_variant == "degree_matched":
        if bundle.C is None:
            raise GuidanceBundleError("graph variant 'degree_matched' requested but bundle has no graph prior")
        C = make_degree_matched_graph_control(bundle.C, seed, bundle.graph_rank)
        Q_C, graph_rank = build_graph_subspace(C)
        P_C = Q_C @ Q_C.T
        graph_meta["variant"] = "degree_matched"
        graph_meta["control_seed"] = seed
    elif graph_variant == "real":
        if bundle.C is None:
            raise GuidanceBundleError("graph variant 'real' requested but bundle has no graph prior")
        graph_meta["variant"] = "real"

    return PriorGuidanceBundle(
        feature_ids=bundle.feature_ids,
        U_k=U_k,
        P_U=P_U,
        C=C,
        Q_C=Q_C,
        P_C=P_C,
        embedding_rank=embedding_rank,
        graph_rank=graph_rank,
        embedding_source_metadata=embedding_meta,
        graph_source_metadata=graph_meta,
        source_hashes=bundle.source_hashes,
    )
