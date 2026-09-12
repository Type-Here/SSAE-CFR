"""Tests for the loss terms, the warm-up schedule, and the total-loss assembly."""

from __future__ import annotations

import math

import pytest
import torch

from ssae_cfr.config import load_config
from ssae_cfr.losses import factual_loss, mmd_rbf, preference_loss, total_loss
from ssae_cfr.utils.schedules import linear_warmup


# -- factual -----------------------------------------------------------------

def test_factual_continuous_matches_manual_mse():
    y0 = torch.tensor([1.0, 2.0, 3.0])
    y1 = torch.tensor([2.0, 0.0, 1.0])
    t = torch.tensor([0, 1, 0])
    yf = torch.tensor([1.5, 0.5, 2.0])
    yf_hat = t.float() * y1 + (1 - t.float()) * y0
    expected = torch.mean((yf_hat - yf) ** 2)
    assert torch.allclose(factual_loss(y0, y1, t, yf, "continuous"), expected)


def test_factual_binary_uses_logits():
    y0 = torch.zeros(4)
    y1 = torch.zeros(4)  # logit 0 -> prob 0.5
    t = torch.tensor([0, 1, 0, 1])
    yf = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = factual_loss(y0, y1, t, yf, "binary")
    assert torch.allclose(loss, torch.tensor(math.log(2.0)), atol=1e-6)


def test_factual_rejects_unknown_outcome_type():
    with pytest.raises(ValueError):
        factual_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2), "poisson")


# -- preference --------------------------------------------------------------

def test_preference_is_the_mean_admission():
    b = torch.tensor([[1.0, 0.0, 0.5], [0.0, 0.0, 1.0]])  # sum 2.5 over 6 entries
    assert torch.allclose(preference_loss(b), torch.tensor(2.5 / 6.0))


def test_preference_is_an_l1_norm_over_m():
    """b >= 0, so mean_j b_j is ||b||_1 / m - the sparsity claim, relocated to the gate."""
    b = torch.rand(7, 5)
    assert torch.allclose(preference_loss(b), b.abs().sum(dim=-1).mean() / 5.0)


def test_preference_is_bounded_by_the_gate():
    """Unlike the unbounded norm penalty it replaces, this cannot exceed 1."""
    assert preference_loss(torch.zeros(8, 5)) == 0.0
    assert preference_loss(torch.ones(8, 5)) == 1.0


# -- mmd ---------------------------------------------------------------------

def test_mmd_nonnegative_and_near_zero_for_identical_groups():
    z = torch.randn(64, 4)
    z = torch.cat([z, z], dim=0)                       # identical clouds
    t = torch.tensor([1] * 64 + [0] * 64)
    val = mmd_rbf(z, t)
    assert val >= 0.0
    assert val < 1e-4


def test_mmd_larger_for_separated_groups():
    torch.manual_seed(0)
    z_t = torch.randn(64, 4)
    close = torch.cat([z_t, torch.randn(64, 4)], dim=0)
    far = torch.cat([z_t, torch.randn(64, 4) + 8.0], dim=0)
    t = torch.tensor([1] * 64 + [0] * 64)
    assert mmd_rbf(far, t) > mmd_rbf(close, t)


def test_mmd_zero_when_one_arm_missing():
    z = torch.randn(10, 4)
    t = torch.ones(10)  # no controls
    val = mmd_rbf(z, t)
    assert float(val) == 0.0


# -- schedule ----------------------------------------------------------------

def test_linear_warmup_ramp():
    assert linear_warmup(0, 1.0, 10) == 0.0
    assert linear_warmup(5, 1.0, 10) == pytest.approx(0.5)
    assert linear_warmup(10, 1.0, 10) == pytest.approx(1.0)
    assert linear_warmup(50, 1.0, 10) == pytest.approx(1.0)  # holds after warmup


def test_linear_warmup_zero_epochs_is_immediate():
    assert linear_warmup(0, 2.0, 0) == 2.0


# -- total -------------------------------------------------------------------

def test_total_loss_weights_and_breakdown():
    cfg = load_config(None, alpha_mmd=2.0, beta_l1=0.1, lambda_rec=3.0, gamma_pref=1.0,
                      gamma_warmup=10)
    terms = {
        "L_fact": torch.tensor(1.0),
        "L_mmd": torch.tensor(0.5),
        "L_sparse": torch.tensor(4.0),
        "L_rec": torch.tensor(2.0),
        "L_pref": torch.tensor(10.0),
    }
    # epoch 0 -> gamma 0, so L_pref drops out
    total, bd = total_loss(terms, cfg, epoch=0)
    expected = 1.0 + 2.0 * 0.5 + 0.1 * 4.0 + 3.0 * 2.0 + 0.0 * 10.0
    assert total.item() == pytest.approx(expected)
    assert bd["gamma"] == 0.0
    assert bd["L_total"] == pytest.approx(expected)
    # at full warmup gamma=1 -> the preference term contributes
    total2, bd2 = total_loss(terms, cfg, epoch=10)
    assert total2.item() == pytest.approx(expected + 10.0)
    assert bd2["gamma"] == pytest.approx(1.0)


def test_total_loss_requires_all_terms():
    cfg = load_config(None)
    with pytest.raises(KeyError):
        total_loss({"L_fact": torch.tensor(1.0)}, cfg, epoch=0)


# -- loss shares -------------------------------------------------------------

def _shares_terms():
    return {
        "L_fact": torch.tensor(1.0),
        "L_mmd": torch.tensor(0.5),
        "L_sparse": torch.tensor(4.0),
        "L_rec": torch.tensor(2.0),
        "L_pref": torch.tensor(10.0),
    }


def test_shares_are_weighted_and_sum_to_one():
    cfg = load_config(None, alpha_mmd=2.0, beta_l1=0.1, lambda_rec=3.0, gamma_pref=1.0,
                      gamma_warmup=0)
    _, bd = total_loss(_shares_terms(), cfg, epoch=0)
    weighted = {"L_fact": 1.0, "L_mmd": 1.0, "L_sparse": 0.4, "L_rec": 6.0, "L_pref": 10.0}
    budget = sum(weighted.values())
    for name, value in weighted.items():
        assert bd[f"w_{name}"] == pytest.approx(value)
        assert bd[f"share_{name}"] == pytest.approx(value / budget)
    assert sum(bd[f"share_{n}"] for n in weighted) == pytest.approx(1.0)


def test_share_follows_the_weight_not_the_raw_value():
    """The point of the shares: a large term at a tiny weight is a small share."""
    cfg = load_config(None, alpha_mmd=1.0, beta_l1=1e-6, lambda_rec=1.0, gamma_pref=0.0)
    _, bd = total_loss(_shares_terms(), cfg, epoch=0)
    assert bd["L_sparse"] == pytest.approx(4.0)          # the largest raw value but one
    assert bd["share_L_sparse"] < 1e-5                   # and effectively absent


def test_share_of_a_zero_weight_term_is_zero():
    cfg = load_config(None, gamma_pref=0.0)
    _, bd = total_loss(_shares_terms(), cfg, epoch=100)
    assert bd["gamma"] == 0.0
    assert bd["share_L_pref"] == 0.0


def test_shares_stay_a_budget_when_mmd_is_negative():
    """The biased MMD estimator can go slightly negative; shares must stay in [0, 1]."""
    cfg = load_config(None, alpha_mmd=1.0, beta_l1=0.0, lambda_rec=1.0, gamma_pref=0.0)
    terms = _shares_terms()
    terms["L_mmd"] = torch.tensor(-0.02)
    _, bd = total_loss(terms, cfg, epoch=0)
    assert bd["share_L_mmd"] < 0.0                       # sign is preserved
    assert sum(abs(bd[f"share_{n}"]) for n in _shares_terms()) == pytest.approx(1.0)


def test_format_shares_renders_percentages():
    from ssae_cfr.train import format_shares

    cfg = load_config(None, alpha_mmd=0.0, beta_l1=0.0, lambda_rec=1.0, gamma_pref=0.0)
    _, bd = total_loss(_shares_terms(), cfg, epoch=0)
    rendered = format_shares(bd)
    assert "fac 33%" in rendered and "rec 67%" in rendered