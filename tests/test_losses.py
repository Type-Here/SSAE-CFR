"""Tests for the loss terms, the gamma schedule, and the total-loss assembly."""

from __future__ import annotations

import math

import pytest
import torch

from ssae_cfr.config import load_config
from ssae_cfr.losses import align_loss, factual_loss, mmd_rbf, total_loss
from ssae_cfr.utils.schedules import gamma_warmup


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


# -- align -------------------------------------------------------------------

def test_align_is_mean_squared_norm():
    mu_res = torch.tensor([[3.0, 4.0], [0.0, 0.0]])  # norms^2: 25, 0 -> mean 12.5
    assert torch.allclose(align_loss(mu_res), torch.tensor(12.5))


def test_align_zero_at_zero():
    assert align_loss(torch.zeros(8, 5)) == 0.0


# -- mmd ---------------------------------------------------------------------

def test_mmd_nonnegative_and_near_zero_for_identical_groups():
    z = torch.randn(64, 4)
    z_mod = torch.cat([z, z], dim=0)                       # identical clouds
    t = torch.tensor([1] * 64 + [0] * 64)
    val = mmd_rbf(z_mod, t)
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
    z_mod = torch.randn(10, 4)
    t = torch.ones(10)  # no controls
    val = mmd_rbf(z_mod, t)
    assert float(val) == 0.0


# -- schedule ----------------------------------------------------------------

def test_gamma_warmup_ramp():
    assert gamma_warmup(0, 1.0, 10) == 0.0
    assert gamma_warmup(5, 1.0, 10) == pytest.approx(0.5)
    assert gamma_warmup(10, 1.0, 10) == pytest.approx(1.0)
    assert gamma_warmup(50, 1.0, 10) == pytest.approx(1.0)  # holds after warmup


def test_gamma_warmup_zero_epochs_is_immediate():
    assert gamma_warmup(0, 2.0, 0) == 2.0


# -- total -------------------------------------------------------------------

def test_total_loss_weights_and_breakdown():
    cfg = load_config(None, alpha_mmd=2.0, beta_l1=0.1, lambda_rec=3.0, gamma_align=1.0,
                      gamma_warmup=10)
    terms = {
        "L_fact": torch.tensor(1.0),
        "L_mmd": torch.tensor(0.5),
        "L_sparse": torch.tensor(4.0),
        "L_rec": torch.tensor(2.0),
        "L_align": torch.tensor(10.0),
    }
    # epoch 0 -> gamma 0, so L_align drops out
    total, bd = total_loss(terms, cfg, epoch=0)
    expected = 1.0 + 2.0 * 0.5 + 0.1 * 4.0 + 3.0 * 2.0 + 0.0 * 10.0
    assert total.item() == pytest.approx(expected)
    assert bd["gamma"] == 0.0
    assert bd["L_total"] == pytest.approx(expected)
    # at full warmup gamma=1 -> align contributes
    total2, bd2 = total_loss(terms, cfg, epoch=10)
    assert total2.item() == pytest.approx(expected + 10.0)
    assert bd2["gamma"] == pytest.approx(1.0)


def test_total_loss_requires_all_terms():
    cfg = load_config(None)
    with pytest.raises(KeyError):
        total_loss({"L_fact": torch.tensor(1.0)}, cfg, epoch=0)