"""Tests for the evaluation metrics.

Each metric is pinned against a case where the answer is known by hand: a perfect
estimator scores zero error, a sharp null reduces PEHE to the RMS of the estimate, an
oracle policy beats both constant policies, the E-value of a null effect is 1, and a
representation that removes a planted imbalance scores near 1 on SMD reduction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ssae_cfr.utils.metrics import (
    approximate_risk_ratio,
    approximate_risk_ratio_ci,
    e_value,
    e_value_ci,
    eps_ate,
    pehe,
    pehe_against_zero,
    policy_risk,
    policy_risk_table,
    policy_value,
    smd_reduction,
)


# -- oracle metrics --------------------------------------------------------


def test_pehe_zero_for_a_perfect_estimator():
    tau = np.array([-1.0, 0.5, 2.0, 3.5])
    assert pehe(tau, tau) == pytest.approx(0.0)
    assert eps_ate(tau, tau) == pytest.approx(0.0)


def test_pehe_is_the_rms_error():
    tau_true = np.array([0.0, 0.0, 0.0, 0.0])
    tau_hat = np.array([1.0, -1.0, 2.0, -2.0])
    assert pehe(tau_hat, tau_true) == pytest.approx(np.sqrt(10.0 / 4.0))


def test_eps_ate_ignores_errors_that_cancel():
    """The two metrics disagree on purpose: eps_ATE is blind to cancelling errors."""
    tau_true = np.zeros(4)
    tau_hat = np.array([1.0, -1.0, 1.0, -1.0])
    assert eps_ate(tau_hat, tau_true) == pytest.approx(0.0)
    assert pehe(tau_hat, tau_true) == pytest.approx(1.0)


def test_pehe_accepts_a_scalar_oracle_and_matches_against_zero():
    tau_hat = np.array([0.3, -0.4, 0.5])
    assert pehe(tau_hat, 0.0) == pytest.approx(pehe_against_zero(tau_hat))
    assert pehe_against_zero(tau_hat) == pytest.approx(np.sqrt(np.mean(tau_hat ** 2)))


def test_pehe_accepts_torch_tensors():
    tau_hat = torch.tensor([1.0, -1.0], requires_grad=True)
    tau_true = torch.zeros(2)
    assert pehe(tau_hat, tau_true) == pytest.approx(1.0)


def test_pehe_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        pehe(np.zeros(3), np.zeros(4))


# -- policy metrics --------------------------------------------------------


def _heterogeneous_arms(n: int = 400, seed: int = 0):
    """Half the units benefit from treatment, half are harmed; treatment is randomized.

    Unit i has y(1) = 1, y(0) = 0 when `good` is true and the reverse otherwise, so the
    oracle policy scores a value of 1 and both constant policies score 0.5.
    """
    rng = np.random.default_rng(seed)
    good = np.arange(n) < n // 2
    t = (rng.random(n) < 0.5).astype(np.int64)
    yf = np.where(good, t, 1 - t).astype(np.float64)
    tau_oracle = np.where(good, 1.0, -1.0)
    return tau_oracle, t, yf


def test_policy_value_of_the_oracle_policy_is_maximal():
    tau, t, yf = _heterogeneous_arms()
    assert policy_value(tau, t, yf) == pytest.approx(1.0)


def test_policy_risk_table_ranks_the_oracle_above_both_constant_policies():
    tau, t, yf = _heterogeneous_arms()
    table = policy_risk_table(tau, t, yf)
    assert table["policy"] == pytest.approx(0.0)
    assert table["treat_all"] == pytest.approx(0.5, abs=0.06)
    assert table["treat_none"] == pytest.approx(0.5, abs=0.06)
    assert table["policy"] < min(table["treat_all"], table["treat_none"])


def test_policy_risk_flips_with_the_outcome_convention():
    """A mortality-style outcome is a loss already, so risk = value, not 1 - value."""
    tau, t, yf = _heterogeneous_arms()
    benefit = policy_risk(tau, t, yf, higher_is_better=True)
    loss = policy_risk(tau, t, yf, higher_is_better=False)
    assert benefit == pytest.approx(1.0 - loss)


def test_treat_all_value_is_the_treated_arm_mean():
    _, t, yf = _heterogeneous_arms()
    treat_all = np.ones(t.shape[0])
    assert policy_value(treat_all, t, yf) == pytest.approx(yf[t == 1].mean())


def test_policy_value_is_nan_when_an_arm_is_missing():
    tau = np.ones(4)
    t = np.zeros(4, dtype=int)  # nobody was actually treated
    yf = np.array([1.0, 0.0, 1.0, 0.0])
    assert np.isnan(policy_value(tau, t, yf))


def test_policy_value_honours_supplied_weights():
    """Up-weighting the units with outcome 1 must pull the estimated value upward."""
    tau = np.ones(4)
    t = np.ones(4, dtype=int)
    yf = np.array([1.0, 1.0, 0.0, 0.0])
    w = np.array([3.0, 3.0, 1.0, 1.0])
    assert policy_value(tau, t, yf) == pytest.approx(0.5)
    assert policy_value(tau, t, yf, weights=w) == pytest.approx(0.75)


def test_policy_metrics_reject_ragged_inputs():
    with pytest.raises(ValueError):
        policy_value(np.zeros(3), np.zeros(3), np.zeros(4))


# -- E-value ---------------------------------------------------------------


def test_e_value_of_a_null_effect_is_one():
    assert e_value(1.0) == pytest.approx(1.0)


def test_e_value_is_symmetric_under_inversion():
    """A protective RR and its reciprocal describe equally strong effects."""
    assert e_value(2.0) == pytest.approx(e_value(0.5))


def test_e_value_grows_with_effect_size():
    assert e_value(1.5) < e_value(3.0) < e_value(10.0)


def test_e_value_matches_the_closed_form():
    rr = 2.0
    assert e_value(rr) == pytest.approx(rr + np.sqrt(rr * (rr - 1.0)))


def test_e_value_rejects_a_nonpositive_risk_ratio():
    with pytest.raises(ValueError):
        e_value(0.0)


def test_e_value_ci_is_one_when_the_interval_covers_the_null():
    assert e_value_ci(0.8, 1.4) == pytest.approx(1.0)


def test_e_value_ci_uses_the_limit_nearest_the_null():
    assert e_value_ci(1.5, 4.0) == pytest.approx(e_value(1.5))
    assert e_value_ci(0.2, 0.5) == pytest.approx(e_value(0.5))


def test_e_value_ci_never_exceeds_the_point_estimate_e_value():
    rr, lo, hi = 3.0, 1.5, 6.0
    assert e_value_ci(lo, hi) <= e_value(rr)


def test_e_value_ci_rejects_reversed_limits():
    with pytest.raises(ValueError):
        e_value_ci(2.0, 1.0)


def test_continuous_outcome_approximation_is_null_preserving():
    assert approximate_risk_ratio(0.0) == pytest.approx(1.0)
    lo, hi = approximate_risk_ratio_ci(0.0, 0.5)
    assert lo < 1.0 < hi
    assert e_value_ci(lo, hi) == pytest.approx(1.0)


def test_continuous_outcome_approximation_feeds_the_e_value():
    d, se = 0.6, 0.1
    lo, hi = approximate_risk_ratio_ci(d, se)
    assert lo < approximate_risk_ratio(d) < hi
    assert e_value_ci(lo, hi) > 1.0


# -- balance ---------------------------------------------------------------


def test_smd_reduction_is_one_when_the_representation_is_balanced():
    rng = np.random.default_rng(0)
    n, m = 600, 5
    t = np.array([0] * (n // 2) + [1] * (n // 2))
    x = rng.standard_normal((n, m))
    x[t == 1] += 2.0  # a large planted imbalance
    z = rng.standard_normal((n, 3))  # a representation that carries none of it
    assert smd_reduction(x, z, t) > 0.9


def test_smd_reduction_is_zero_when_nothing_changes():
    rng = np.random.default_rng(1)
    n = 400
    t = np.array([0] * (n // 2) + [1] * (n // 2))
    x = rng.standard_normal((n, 4))
    x[t == 1] += 1.5
    assert smd_reduction(x, x, t) == pytest.approx(0.0)


def test_smd_reduction_is_negative_when_the_representation_is_worse():
    rng = np.random.default_rng(2)
    n = 400
    t = np.array([0] * (n // 2) + [1] * (n // 2))
    x = rng.standard_normal((n, 4))
    x[t == 1] += 0.2
    z = rng.standard_normal((n, 4))
    z[t == 1] += 3.0
    assert smd_reduction(x, z, t) < 0.0


def test_smd_reduction_accepts_torch_representations():
    rng = np.random.default_rng(3)
    n = 200
    t = np.array([0] * (n // 2) + [1] * (n // 2))
    x = rng.standard_normal((n, 4))
    x[t == 1] += 2.0
    z = torch.zeros(n, 3)
    assert smd_reduction(x, z, t) == pytest.approx(1.0, abs=1e-6)


def test_smd_reduction_rejects_a_1d_representation():
    t = np.array([0, 0, 1, 1])
    x = np.zeros((4, 2))
    with pytest.raises(ValueError):
        smd_reduction(x, np.zeros(4), t)