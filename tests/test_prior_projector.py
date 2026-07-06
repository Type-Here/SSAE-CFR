"""Tests for the prior projector: P_U properties and k_svd selection.

P_U must be a genuine orthogonal projector (symmetric, idempotent, eigenvalues in
{0, 1}, rank exactly k_svd) because the whole feature-space decomposition x = P_U x +
(I - P_U) x relies on those algebraic facts. The k_svd chooser must honour its two
knobs: the energy threshold and the protected-covariate retention floor.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssae_cfr.prior import build_projector, choose_k_svd, retention, svd_energy


def test_projector_is_orthogonal_projector(V, m):
    k = choose_k_svd(V, energy_threshold=0.90)
    P = build_projector(V, k)

    assert P.shape == (m, m)
    assert np.allclose(P, P.T), "P_U must be symmetric"
    assert np.allclose(P @ P, P, atol=1e-8), "P_U must be idempotent"
    assert np.linalg.matrix_rank(P, tol=1e-8) == k
    assert np.isclose(np.trace(P), k), "trace of a projector equals its rank"

    eig = np.linalg.eigvalsh(P)
    assert np.allclose(np.round(eig), eig, atol=1e-6), "eigenvalues must be 0 or 1"
    assert set(np.round(eig).astype(int).tolist()) <= {0, 1}


def test_decomposition_is_orthogonal(V):
    k = choose_k_svd(V, energy_threshold=0.90)
    P = build_projector(V, k)
    m = P.shape[0]
    x = np.random.default_rng(1).standard_normal(m)

    x_prior = P @ x
    x_res = (np.eye(m) - P) @ x
    assert np.allclose(x, x_prior + x_res), "the two parts must sum back to x"
    assert np.isclose(x_prior @ x_res, 0.0, atol=1e-8), "prior and residual are orthogonal"


def test_energy_threshold_is_monotone_in_k(V):
    k_low = choose_k_svd(V, energy_threshold=0.50)
    k_high = choose_k_svd(V, energy_threshold=0.95)
    assert k_low <= k_high, "a higher energy target needs at least as many directions"


def test_energy_curve_is_a_valid_cdf(V):
    e = svd_energy(V)
    assert np.all(np.diff(e) >= -1e-12), "cumulative energy is non-decreasing"
    assert np.isclose(e[-1], 1.0), "cumulative energy reaches 1"
    assert np.all(e >= -1e-12) and np.all(e <= 1.0 + 1e-12)


def test_k_never_reaches_m(V, m):
    # k_max defaults to m-1 so P_U can never be the identity (a vacuous prior)
    k = choose_k_svd(V, energy_threshold=1.0)
    assert k <= m - 1


def test_protected_floor_bumps_k(V):
    protected = [3, 7]
    k_plain = choose_k_svd(V, energy_threshold=0.50)
    k_prot = choose_k_svd(V, energy_threshold=0.50, protected=protected, retention_floor=0.5)

    assert k_prot >= k_plain, "the floor can only raise k, never lower it"
    ret = retention(build_projector(V, k_prot))
    assert np.all(ret[protected] >= 0.5), "protected covariates must clear the floor"


def test_retention_is_diagonal_in_unit_range(V):
    P = build_projector(V, choose_k_svd(V))
    ret = retention(P)
    assert np.allclose(ret, np.diag(P))
    assert np.all(ret >= -1e-9) and np.all(ret <= 1.0 + 1e-9)


def test_build_projector_rejects_bad_k(V, m):
    with pytest.raises(ValueError):
        build_projector(V, 0)
    with pytest.raises(ValueError):
        build_projector(V, m + 1)