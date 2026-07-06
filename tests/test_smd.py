"""Tests for the SMD noise modulator.

omega must behave like an imbalance gauge: zero when treated and control look alike,
growing (toward 1) as they separate, monotone in both the imbalance and the alpha gain,
and always a plain detached scalar in [0, 1).
"""

from __future__ import annotations

import numpy as np

from ssae_cfr.utils.smd import omega_from_smd, smd_per_covariate


def _split_batch(shift: float, n: int = 4000, m: int = 6, seed: int = 0):
    """Treated and control drawn from the same law but with a mean `shift` on treated."""
    rng = np.random.default_rng(seed)
    t = np.array([0] * (n // 2) + [1] * (n // 2))
    x = rng.standard_normal((n, m))
    x[t == 1] += shift
    return x, t


def test_smd_zero_when_balanced():
    x, t = _split_batch(shift=0.0)
    smd = smd_per_covariate(x, t)
    assert smd.shape == (x.shape[1],)
    assert np.all(smd < 0.15), "no shift -> small SMD on every covariate"
    assert omega_from_smd(x, t) < 0.2


def test_smd_grows_with_shift():
    _, t = _split_batch(shift=0.0)
    o_small = omega_from_smd(*_split_batch(shift=0.5))
    o_large = omega_from_smd(*_split_batch(shift=3.0))
    assert o_small < o_large, "more imbalance -> more noise"


def test_omega_in_unit_interval_and_scalar():
    x, t = _split_batch(shift=2.0)
    o = omega_from_smd(x, t)
    assert isinstance(o, float)
    assert 0.0 <= o < 1.0


def test_omega_monotone_in_alpha():
    x, t = _split_batch(shift=1.0)
    o1 = omega_from_smd(x, t, alpha_smd=0.5)
    o2 = omega_from_smd(x, t, alpha_smd=4.0)
    assert o1 < o2, "a larger alpha gain maps the same imbalance to more noise"


def test_smd_empty_group_returns_zeros():
    x = np.random.default_rng(0).standard_normal((10, 4))
    t = np.zeros(10, dtype=int)  # no treated units
    assert np.all(smd_per_covariate(x, t) == 0.0)
    assert omega_from_smd(x, t) == 0.0