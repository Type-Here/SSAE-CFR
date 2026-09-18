"""The semantic prior bundle: the SVD-derived P_U/W_r/Q objects, their persistence,
and the loader's covariate-order guard.

Uses synthetic V matrices throughout so the suite runs without a built artifact; a
handful of checks against the real, committed IHDP bundle are guarded with
skipif so they run when the artifact exists but never block portability.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssae_v3.data.roles import repo_root
from ssae_v3.prior_build.projector import (
    build_prior_bundle,
    choose_k_svd,
    load_bundle,
    save_bundle,
)
from ssae_v3.prior_modules.loader import load_prior_tensors

REAL_BUNDLE_PATH = repo_root() / "artifacts" / "ihdp" / "prior_bundle.npz"


def _synthetic_v(m: int = 12, d: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, d))


def test_bundle_projector_is_symmetric_idempotent_and_rank_k():
    """P_U is symmetric, idempotent, and has rank exactly k_U."""
    v = _synthetic_v()
    bundle = build_prior_bundle(v, k_U=5, r_W_rank=4, center=True)
    p_u = bundle.P_U

    assert np.allclose(p_u, p_u.T, atol=1e-8)
    assert np.allclose(p_u @ p_u, p_u, atol=1e-6)
    assert np.isclose(np.trace(p_u), 5, atol=1e-6)
    assert np.linalg.matrix_rank(p_u, tol=1e-6) == 5


def test_q_equals_centered_v_times_w_and_u_times_sigma_exactly_once():
    """Q == V_c @ W_r == U_r * singular_values; Sigma is applied exactly once."""
    v = _synthetic_v()
    bundle = build_prior_bundle(v, k_U=5, r_W_rank=4, center=True)

    v_c = v - v.mean(axis=0, keepdims=True)
    q_from_w = v_c @ bundle.W_r
    assert np.allclose(bundle.Q, q_from_w, atol=1e-8)

    u, s, _ = np.linalg.svd(v_c, full_matrices=False)
    # sign-canonicalize U the same way build_prior_bundle does, so U_r lines up with W_r
    for i in range(u.shape[1]):
        j = int(np.argmax(np.abs(u[:, i])))
        if u[j, i] < 0:
            u[:, i] = -u[:, i]
    q_from_u_sigma = u[:, :4] * s[:4]
    assert np.allclose(bundle.Q, q_from_u_sigma, atol=1e-6)


def test_q_tilde_has_unit_rms():
    """rms(Q_tilde) == 1, within tolerance."""
    v = _synthetic_v()
    bundle = build_prior_bundle(v, k_U=5, r_W_rank=4, center=True)
    rms = np.sqrt(np.mean(bundle.Q_tilde ** 2))
    assert abs(rms - 1.0) < 1e-3


def test_centering_makes_pu_annihilate_the_ones_vector():
    """With center=True, P_U @ ones is ~0: the uniform-shift direction is unmodeled."""
    v = _synthetic_v()
    bundle = build_prior_bundle(v, k_U=5, r_W_rank=4, center=True)
    ones = np.ones(bundle.m)
    assert np.allclose(bundle.P_U @ ones, 0.0, atol=1e-6)


def test_centering_reveals_more_rank_than_a_dominant_offset_hides():
    """An anisotropic V's uncentered rank at 90% energy is smaller than its centered rank."""
    rng = np.random.default_rng(1)
    m, d = 20, 40
    base = rng.standard_normal((m, d)) * 0.5
    offset = rng.standard_normal(d) * 50.0
    v_aniso = base + offset  # every row dominated by the same large shared offset

    k_uncentered = choose_k_svd(v_aniso, energy_threshold=0.90, center=False)
    k_centered = choose_k_svd(v_aniso, energy_threshold=0.90, center=True)
    assert k_centered > k_uncentered


def test_bundle_build_is_deterministic():
    """Building twice from the same V gives bitwise-identical arrays."""
    v = _synthetic_v()
    b1 = build_prior_bundle(v, k_U=5, r_W_rank=4, center=True)
    b2 = build_prior_bundle(v, k_U=5, r_W_rank=4, center=True)

    assert np.array_equal(b1.P_U, b2.P_U)
    assert np.array_equal(b1.Q, b2.Q)
    assert np.array_equal(b1.Q_tilde, b2.Q_tilde)
    assert np.array_equal(b1.U_k, b2.U_k)
    assert np.array_equal(b1.W_r, b2.W_r)
    assert b1.s_Q == b2.s_Q


def test_save_and_load_bundle_round_trips_exactly(tmp_path):
    """save_bundle -> load_bundle reproduces every array and scalar field exactly."""
    v = _synthetic_v()
    names = [f"x{i}" for i in range(v.shape[0])]
    bundle = build_prior_bundle(v, k_U=5, r_W_rank=4, center=True, feature_names=names)
    path = save_bundle(bundle, tmp_path / "prior_bundle.npz")
    loaded = load_bundle(path)

    assert np.array_equal(bundle.P_U, loaded.P_U)
    assert np.array_equal(bundle.Q, loaded.Q)
    assert np.array_equal(bundle.Q_tilde, loaded.Q_tilde)
    assert np.array_equal(bundle.U_k, loaded.U_k)
    assert np.array_equal(bundle.W_r, loaded.W_r)
    assert bundle.k_U == loaded.k_U
    assert bundle.r_W_rank == loaded.r_W_rank
    assert bundle.centered == loaded.centered
    assert list(bundle.feature_names) == list(loaded.feature_names)
    assert bundle.s_Q == loaded.s_Q


def test_loader_rejects_a_reordered_feature_list(tmp_path):
    """load_prior_tensors names the first mismatched covariate on a reordered feature list."""
    v = _synthetic_v(m=6, d=10)
    names = [f"x{i}" for i in range(6)]
    bundle = build_prior_bundle(v, k_U=3, r_W_rank=2, center=True, feature_names=names)
    path = save_bundle(bundle, tmp_path / "prior_bundle.npz")

    reordered = [names[1], names[0]] + names[2:]
    with pytest.raises(ValueError, match="index 0"):
        load_prior_tensors(reordered, path=path)


def test_loader_rejects_a_wrong_length_feature_list(tmp_path):
    """load_prior_tensors raises when the feature list has the wrong length."""
    v = _synthetic_v(m=6, d=10)
    names = [f"x{i}" for i in range(6)]
    bundle = build_prior_bundle(v, k_U=3, r_W_rank=2, center=True, feature_names=names)
    path = save_bundle(bundle, tmp_path / "prior_bundle.npz")

    with pytest.raises(ValueError):
        load_prior_tensors(names[:-1], path=path)


@pytest.mark.skipif(not REAL_BUNDLE_PATH.exists(), reason="real IHDP prior bundle not built")
def test_real_ihdp_bundle_projector_is_a_genuine_projector():
    """The real, committed IHDP prior bundle's P_U is symmetric and idempotent."""
    bundle = load_bundle(REAL_BUNDLE_PATH)
    p_u = bundle.P_U
    assert np.allclose(p_u, p_u.T, atol=1e-6)
    assert np.allclose(p_u @ p_u, p_u, atol=1e-4)
