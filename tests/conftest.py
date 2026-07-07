"""Shared pytest fixtures for the SSAE-CFR test suite.

The package lives at the repo root (`ssae_cfr/`), so running pytest from the repo
root makes it importable without extra path juggling. Fixtures here provide the small
reproducible inputs the component tests share (a placeholder embedding matrix, a tiny
standardized covariate batch), so individual tests stay focused on behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssae_cfr.prior import placeholder_embeddings

M = 25  # IHDP covariate width, a convenient small size for the whole suite


@pytest.fixture
def m() -> int:
    return M


@pytest.fixture
def V() -> np.ndarray:
    """A reproducible placeholder embedding matrix (m x d_LLM)."""
    return placeholder_embeddings(M, d_LLM=64, seed=0)


@pytest.fixture
def xt() -> tuple[np.ndarray, np.ndarray]:
    """A small standardized-ish covariate batch and a binary treatment vector."""
    rng = np.random.default_rng(7)
    n = 200
    x = rng.standard_normal((n, M))
    t = (rng.random(n) < 0.5).astype(np.int64)
    return x, t