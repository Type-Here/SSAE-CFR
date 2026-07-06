"""Tests for the embeddings module: the placeholder generator and the V cache.

The heavy `build_embeddings` path (a real LLM forward) is not exercised here - it runs
on the university machine and needs the model weights. What we can and must test is the
offline plumbing: the placeholder V is well-shaped and reproducible, and the disk cache
round-trips V while recording the model and covariate order in its sidecar.
"""

from __future__ import annotations

import json

import numpy as np

from ssae_cfr.prior import cache_embeddings, load_embeddings, placeholder_embeddings


def test_placeholder_shape_and_no_nans():
    V = placeholder_embeddings(25, d_LLM=64, seed=0)
    assert V.shape == (25, 64)
    assert np.isfinite(V).all()


def test_placeholder_is_reproducible_and_seed_sensitive():
    a = placeholder_embeddings(10, d_LLM=32, seed=1)
    b = placeholder_embeddings(10, d_LLM=32, seed=1)
    c = placeholder_embeddings(10, d_LLM=32, seed=2)
    assert np.array_equal(a, b), "same seed must give the same V"
    assert not np.array_equal(a, c), "different seeds must differ"


def test_placeholder_rows_are_distinct():
    V = placeholder_embeddings(8, d_LLM=16, seed=0)
    # no two covariate rows should be identical (degenerate embeddings)
    for i in range(V.shape[0]):
        for j in range(i + 1, V.shape[0]):
            assert not np.array_equal(V[i], V[j])


def test_cache_round_trip(tmp_path):
    V = placeholder_embeddings(6, d_LLM=12, seed=3)
    names = [f"x{i}" for i in range(6)]
    path = tmp_path / "V.npz"

    cache_embeddings(V, path, model_name="placeholder", feature_names=names)
    loaded = load_embeddings(path)

    assert np.allclose(loaded, V)
    meta = json.loads((tmp_path / "V.json").read_text())
    assert meta["model_name"] == "placeholder"
    assert meta["d_LLM"] == 12
    assert meta["m"] == 6
    assert meta["feature_names"] == names


def test_cache_rejects_name_length_mismatch(tmp_path):
    V = placeholder_embeddings(6, d_LLM=12, seed=3)
    try:
        cache_embeddings(V, tmp_path / "V.npz", model_name="x", feature_names=["only", "two"])
    except ValueError:
        return
    raise AssertionError("cache_embeddings should reject a feature_names length mismatch")