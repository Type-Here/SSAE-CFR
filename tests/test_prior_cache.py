"""Tests for P_U persistence and the train-time prior-loading guard.

The projector round-trips through disk unchanged, its sidecar carries the covariate
order, and the training loader refuses a prior built for a different covariate set (a
silent mismatch would corrupt the projection geometry).
"""

from __future__ import annotations

import numpy as np
import pytest

from ssae_cfr.data.base import Dataset
from ssae_cfr.prior import build_projector, choose_k_svd, load_projector, placeholder_embeddings, save_projector
from ssae_cfr.train import load_prior_for

M = 10


def _dataset(m=M, name="toy"):
    rng = np.random.default_rng(0)
    n = 40
    return Dataset(
        name=name,
        x=rng.standard_normal((n, m)),
        t=(rng.random(n) < 0.5).astype(int),
        yf=rng.standard_normal(n),
        feature_names=[f"f{i}" for i in range(m)],
    )


def _save_prior(tmp_path, feature_names, m=M):
    V = placeholder_embeddings(m, d_LLM=16, seed=0)
    P_U = build_projector(V, choose_k_svd(V))
    path = tmp_path / "P_U.npz"
    save_projector(P_U, path, {"feature_names": list(feature_names)})
    return path, P_U


def test_projector_round_trip(tmp_path):
    ds = _dataset()
    path, P_U = _save_prior(tmp_path, ds.feature_names)
    loaded, meta = load_projector(path)
    assert np.allclose(loaded, P_U)
    assert meta["feature_names"] == list(ds.feature_names)


def test_load_prior_for_accepts_matching_dataset(tmp_path):
    ds = _dataset()
    path, P_U = _save_prior(tmp_path, ds.feature_names)
    P = load_prior_for(ds, str(path))
    assert P.shape == (ds.m, ds.m)
    assert np.allclose(P, P_U)


def test_load_prior_for_rejects_mismatched_covariates(tmp_path):
    ds = _dataset()
    # prior built with different names -> must be refused
    path, _ = _save_prior(tmp_path, [f"other{i}" for i in range(M)])
    with pytest.raises(ValueError):
        load_prior_for(ds, str(path))


def test_load_prior_for_rejects_wrong_size(tmp_path):
    ds = _dataset(m=M)
    # prior for a wider covariate set (no feature_names in sidecar -> shape guard fires)
    V = placeholder_embeddings(M + 3, d_LLM=16, seed=0)
    P_U = build_projector(V, choose_k_svd(V))
    path = tmp_path / "P_U.npz"
    save_projector(P_U, path, {})  # empty meta, so only the shape check protects us
    with pytest.raises(ValueError):
        load_prior_for(ds, str(path))
