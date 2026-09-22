"""Tests for the offline prior-guidance bundle, its subspace builders and controls.

Small synthetic inputs throughout (m around 8-10, a handful of concepts) so the unit
tests need no artifact on disk. One test at the bottom loads the real IHDP artifacts
and is skipped when they are not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ssae_v3.prior_modules.guidance_bundle import (
    GuidanceBundleError,
    PriorGuidanceBundle,
    build_embedding_subspace,
    build_graph_subspace,
    load_guidance_bundle,
    orthonormal_basis,
)
from ssae_v3.prior_modules.guidance_controls import (
    apply_guidance_controls,
    make_degree_matched_graph_control,
    make_permuted_graph_control,
    make_random_embedding_control,
)

FEATURE_IDS = tuple(f"x{i}" for i in range(1, 9))  # m = 8


def _low_rank_embedding(m: int = 8, d: int = 5, rank: int = 3, seed: int = 0) -> torch.Tensor:
    """A (m, d) matrix with a dominant rank-`rank` structure plus small noise."""
    gen = torch.Generator().manual_seed(seed)
    basis = torch.randn(m, rank, generator=gen)
    coeffs = torch.randn(rank, d, generator=gen)
    signal = basis @ coeffs
    noise = 1e-3 * torch.randn(m, d, generator=gen)
    return signal + noise


def _synthetic_graph() -> torch.Tensor:
    """A (10, 4) incidence matrix with overlapping concepts, amenable to rewiring."""
    m, K = 10, 4
    C = torch.zeros(m, K)
    edges = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (2, 1), (3, 1), (4, 1), (5, 1),
        (4, 2), (5, 2), (6, 2), (7, 2),
        (6, 3), (7, 3), (8, 3), (9, 3),
    ]
    for f, c in edges:
        C[f, c] = 1.0
    return C


def _identity_bundle(feature_ids, U_k=None, C=None) -> PriorGuidanceBundle:
    """Build a valid bundle from raw U_k / C, filling in the derived tensors."""
    P_U = embedding_rank = None
    embedding_rank = 0
    if U_k is not None:
        P_U = U_k @ U_k.T
        embedding_rank = U_k.shape[1]
    Q_C = P_C = None
    graph_rank = 0
    if C is not None:
        Q_C, graph_rank = build_graph_subspace(C)
        P_C = Q_C @ Q_C.T
    return PriorGuidanceBundle(
        feature_ids=tuple(feature_ids),
        U_k=U_k,
        P_U=P_U,
        C=C,
        Q_C=Q_C,
        P_C=P_C,
        embedding_rank=embedding_rank,
        graph_rank=graph_rank,
        embedding_source_metadata={},
        graph_source_metadata={},
        source_hashes={},
    )


# --------------------------------------------------------------------------- #
# build_embedding_subspace / orthonormal_basis
# --------------------------------------------------------------------------- #


def test_embedding_subspace_shapes_and_orthonormality():
    V = _low_rank_embedding()
    U_k, k, energy = build_embedding_subspace(V, energy_threshold=0.90)
    assert U_k.shape == (8, k)
    gram = U_k.T @ U_k
    assert torch.allclose(gram, torch.eye(k), atol=1e-4)
    assert 0.0 < energy <= 1.0


def test_embedding_subspace_projector_symmetric_idempotent_rank():
    V = _low_rank_embedding()
    U_k, k, _ = build_embedding_subspace(V, energy_threshold=0.90)
    P_U = U_k @ U_k.T
    assert torch.allclose(P_U, P_U.T, atol=1e-5)
    assert torch.allclose(P_U @ P_U, P_U, atol=1e-4)
    assert int(torch.linalg.matrix_rank(P_U.to(torch.float64), atol=1e-4)) == k


def test_k_selection_honours_energy_threshold_and_never_reaches_m():
    V = _low_rank_embedding(m=8, d=5, rank=3)
    _, k_loose, energy_loose = build_embedding_subspace(V, energy_threshold=0.5)
    _, k_tight, energy_tight = build_embedding_subspace(V, energy_threshold=0.999)
    assert k_loose <= k_tight
    assert energy_loose >= 0.5
    assert energy_tight >= 0.999 or k_tight == 7  # clamped to m - 1
    assert k_tight <= 7  # m - 1, projector can never be the identity


def test_graph_subspace_orthonormal_and_rank_recorded():
    C = _synthetic_graph()
    Q_C, rank = build_graph_subspace(C)
    assert Q_C.shape == (10, rank)
    gram = Q_C.T @ Q_C
    assert torch.allclose(gram, torch.eye(rank), atol=1e-4)
    P_C = Q_C @ Q_C.T
    assert torch.allclose(P_C, P_C.T, atol=1e-5)
    assert torch.allclose(P_C @ P_C, P_C, atol=1e-4)


def test_orthonormal_basis_rejects_rank_zero():
    zeros = torch.zeros(6, 3)
    with pytest.raises(GuidanceBundleError):
        orthonormal_basis(zeros)


def test_orthonormal_basis_rejects_non_finite():
    a = torch.randn(6, 3)
    a[0, 0] = float("nan")
    with pytest.raises(GuidanceBundleError):
        orthonormal_basis(a)


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #


def test_random_embedding_control_preserves_rank_orthonormality_and_differs():
    V = _low_rank_embedding()
    U_k, k, _ = build_embedding_subspace(V, energy_threshold=0.90)
    control = make_random_embedding_control(8, k, seed=1)
    assert control.shape == (8, k)
    gram = control.T @ control
    assert torch.allclose(gram, torch.eye(k), atol=1e-4)
    assert not torch.allclose(control, U_k)


def test_permuted_graph_preserves_rank_edges_and_degree_sequence():
    C = _synthetic_graph()
    _, rank = build_graph_subspace(C)
    control = make_permuted_graph_control(C, seed=1)
    assert int(control.sum()) == int(C.sum())
    assert torch.equal(torch.sort(control.sum(dim=0)).values, torch.sort(C.sum(dim=0)).values)
    _, control_rank = build_graph_subspace(control)
    assert control_rank == rank
    assert not torch.equal(control, C)


def test_degree_matched_graph_preserves_both_degrees_edges_rank_and_differs():
    C = _synthetic_graph()
    _, rank = build_graph_subspace(C)
    control = make_degree_matched_graph_control(C, seed=1, target_rank=rank)
    assert torch.equal(control.sum(dim=1), C.sum(dim=1))
    assert torch.equal(control.sum(dim=0), C.sum(dim=0))
    assert int(control.sum()) == int(C.sum())
    _, control_rank = build_graph_subspace(control)
    assert control_rank == rank
    assert not torch.equal(control, C)


def test_degree_matched_graph_raises_when_unreachable():
    # a single edge cannot be rewired at all: no second edge to swap with.
    C = torch.zeros(4, 2)
    C[0, 0] = 1.0
    C[1, 1] = 1.0
    with pytest.raises(GuidanceBundleError):
        make_degree_matched_graph_control(C, seed=1, target_rank=2, max_attempts=3)


# --------------------------------------------------------------------------- #
# bundle validation
# --------------------------------------------------------------------------- #


def test_bundle_rejects_duplicate_feature_ids():
    V = _low_rank_embedding(m=8)
    U_k, _, _ = build_embedding_subspace(V, energy_threshold=0.90)
    ids = ("x1",) * 8
    with pytest.raises(GuidanceBundleError):
        _identity_bundle(ids, U_k=U_k)


def test_bundle_rejects_row_count_mismatch():
    V = _low_rank_embedding(m=8)
    U_k, _, _ = build_embedding_subspace(V, energy_threshold=0.90)
    with pytest.raises(GuidanceBundleError):
        _identity_bundle(FEATURE_IDS[:7], U_k=U_k)


def test_bundle_rejects_non_idempotent_projector():
    V = _low_rank_embedding(m=8)
    U_k, k, _ = build_embedding_subspace(V, energy_threshold=0.90)
    bad_P_U = torch.randn(8, 8)
    with pytest.raises(GuidanceBundleError):
        PriorGuidanceBundle(
            feature_ids=FEATURE_IDS,
            U_k=U_k,
            P_U=bad_P_U,
            C=None,
            Q_C=None,
            P_C=None,
            embedding_rank=k,
            graph_rank=0,
            embedding_source_metadata={},
            graph_source_metadata={},
            source_hashes={},
        )


def test_bundle_rejects_all_zero_concept_column():
    C = _synthetic_graph()[:8, :3].clone()
    C[:, 2] = 0.0
    with pytest.raises(GuidanceBundleError):
        _identity_bundle(FEATURE_IDS, C=C)


def test_bundle_rejects_non_binary_C():
    C = _synthetic_graph()[:8, :3].clone()
    C[0, 0] = 0.5
    with pytest.raises(GuidanceBundleError):
        _identity_bundle(FEATURE_IDS, C=C)


def test_bundle_with_neither_prior_is_legal():
    bundle = _identity_bundle(FEATURE_IDS)
    assert bundle.embedding_rank == 0
    assert bundle.graph_rank == 0
    assert bundle.m == 8
    assert bundle.n_concepts == 0
    assert bundle.n_edges == 0


# --------------------------------------------------------------------------- #
# apply_guidance_controls
# --------------------------------------------------------------------------- #


def _both_priors_bundle() -> PriorGuidanceBundle:
    V = _low_rank_embedding(m=10)
    U_k, _, _ = build_embedding_subspace(V, energy_threshold=0.90)
    C = _synthetic_graph()
    return _identity_bundle([f"x{i}" for i in range(1, 11)], U_k=U_k, C=C)


def test_apply_guidance_controls_raises_for_absent_embedding_prior():
    bundle = _identity_bundle(FEATURE_IDS, C=_synthetic_graph()[:8, :3])
    with pytest.raises(GuidanceBundleError):
        apply_guidance_controls(bundle, embedding_variant="random", graph_variant="real", seed=1)


def test_apply_guidance_controls_raises_for_absent_graph_prior():
    V = _low_rank_embedding(m=8)
    U_k, _, _ = build_embedding_subspace(V, energy_threshold=0.90)
    bundle = _identity_bundle(FEATURE_IDS, U_k=U_k)
    with pytest.raises(GuidanceBundleError):
        apply_guidance_controls(bundle, embedding_variant="real", graph_variant="permuted", seed=1)


def test_apply_guidance_controls_records_variant_and_seed():
    bundle = _both_priors_bundle()
    controlled = apply_guidance_controls(bundle, embedding_variant="random", graph_variant="permuted", seed=7)
    assert controlled.embedding_source_metadata["variant"] == "random"
    assert controlled.embedding_source_metadata["control_seed"] == 7
    assert controlled.graph_source_metadata["variant"] == "permuted"
    assert controlled.graph_source_metadata["control_seed"] == 7
    assert controlled.embedding_rank == bundle.embedding_rank
    assert controlled.graph_rank == bundle.graph_rank


def test_apply_guidance_controls_none_drops_the_prior():
    bundle = _both_priors_bundle()
    controlled = apply_guidance_controls(bundle, embedding_variant="none", graph_variant="none", seed=1)
    assert controlled.U_k is None and controlled.P_U is None and controlled.embedding_rank == 0
    assert controlled.C is None and controlled.Q_C is None and controlled.P_C is None
    assert controlled.graph_rank == 0


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def test_random_embedding_control_deterministic_given_seed():
    a = make_random_embedding_control(8, 3, seed=42)
    b = make_random_embedding_control(8, 3, seed=42)
    c = make_random_embedding_control(8, 3, seed=43)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_permuted_graph_control_deterministic_given_seed():
    C = _synthetic_graph()
    a = make_permuted_graph_control(C, seed=42)
    b = make_permuted_graph_control(C, seed=42)
    c = make_permuted_graph_control(C, seed=43)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_degree_matched_graph_control_deterministic_given_seed():
    C = _synthetic_graph()
    _, rank = build_graph_subspace(C)
    a = make_degree_matched_graph_control(C, seed=42, target_rank=rank)
    b = make_degree_matched_graph_control(C, seed=42, target_rank=rank)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# real artifact
# --------------------------------------------------------------------------- #

_REAL_EMBEDDINGS = Path("artifacts/ihdp/expert_prior_2/feature_embeddings.pt")


@pytest.mark.skipif(not _REAL_EMBEDDINGS.exists(), reason="real IHDP expert-prior artifact not present")
def test_real_ihdp_guidance_bundle():
    from ssae_v3.data.ihdp import FEATURE_NAMES

    bundle = load_guidance_bundle(
        FEATURE_NAMES,
        dataset="ihdp",
        expert_dir=Path("artifacts/ihdp/expert_prior_2"),
    )
    assert bundle.embedding_rank == 13
    assert bundle.graph_rank == 6
    assert bundle.m == 25
    assert bundle.n_concepts == 6
    assert bundle.n_edges == 30
    assert "feature_embeddings.pt" in bundle.source_hashes
    assert "expert_prior.yaml" in bundle.source_hashes
