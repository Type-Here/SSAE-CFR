"""Smoke tests for the training/evaluation harness: the fit loop, early stopping,
the total-loss assembly, aggregation, and the adapter-presence contract.

Kept fast: tiny synthetic datasets and tiny model dimensions throughout.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ssae_v3.data.base import Dataset
from ssae_v3.hparams import load_config
from ssae_v3.losses.total import total_loss
from ssae_v3.model_v3 import SSAECFRv3
from ssae_v3.training.evaluate import aggregate
from ssae_v3.training.train import fit

M = 5


def _synthetic_dataset(n: int = 40, m: int = M, seed: int = 0, name: str = "synthetic") -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, m))
    t = (rng.random(n) < 0.5).astype(np.int64)
    yf = x[:, 0] + 0.5 * t + rng.standard_normal(n) * 0.1
    return Dataset(name=name, x=x, t=t, yf=yf, feature_names=[f"x{i}" for i in range(m)])


def _tiny_cfg(**overrides):
    base = dict(
        in_channels=M,
        d_u=8,
        encoder_hidden=(8,),
        decoder_hidden=(8,),
        head_hidden=(8,),
        epochs=20,
        lr=1e-2,
        is_noise_active=False,  # determinism, not a claim about the noise ablation
    )
    base.update(overrides)
    return load_config(None, **base)


def test_fit_reduces_loss_on_a_small_synthetic_dataset():
    """A few epochs of fit lowers the total training loss on a tiny synthetic dataset."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(model_variant="empirical", epochs=40)
    ds = _synthetic_dataset()
    model = SSAECFRv3(cfg)
    history = fit(model, ds, cfg, verbose=False)
    assert history[0]["L_total"] > history[-1]["L_total"]


def test_patience_zero_never_triggers_early_stopping():
    """patience=0 runs every configured epoch and never rewinds the model."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(model_variant="empirical", epochs=8, patience=0)
    ds = _synthetic_dataset()
    val = _synthetic_dataset(n=10, seed=1)
    model = SSAECFRv3(cfg)
    history = fit(model, ds, cfg, val=val, verbose=False)

    assert len(history) == cfg.epochs
    assert not any("early_stopped_to_epoch" in entry for entry in history)


def test_small_patience_triggers_and_records_the_rewind_epoch():
    """A tiny patience with es_check_every=1 stops early and logs the rewound epoch."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(
        model_variant="empirical", epochs=30, patience=1, es_check_every=1,
        lr=0.0, min_delta=0.0,
    )
    ds = _synthetic_dataset()
    val = _synthetic_dataset(n=10, seed=1)
    model = SSAECFRv3(cfg)
    history = fit(model, ds, cfg, val=val, verbose=False)

    assert len(history) < cfg.epochs
    assert "early_stopped_to_epoch" in history[-1]
    assert history[-1]["early_stopped_to_epoch"] >= 0


def test_total_loss_returns_four_terms_and_normalized_shares():
    """total_loss returns w_*/share_* for all four terms, shares summing to 1 in magnitude."""
    cfg = load_config(None, in_channels=4, alpha_mmd=2.0, beta_l1=0.5, lambda_rec=1.5)
    terms = {
        "L_fact": torch.tensor(1.0),
        "L_mmd": torch.tensor(0.2),
        "L_sparse": torch.tensor(0.3),
        "L_rec": torch.tensor(0.6),
    }
    total, breakdown = total_loss(terms, cfg)
    assert torch.is_tensor(total)

    names = ("L_fact", "L_mmd", "L_sparse", "L_rec")
    for name in names:
        assert f"w_{name}" in breakdown
        assert f"share_{name}" in breakdown

    shares = [breakdown[f"share_{name}"] for name in names]
    assert abs(sum(abs(s) for s in shares) - 1.0) < 1e-8


def test_total_loss_raises_on_a_missing_term():
    """total_loss raises KeyError when a required term is absent."""
    cfg = load_config(None, in_channels=4)
    terms = {"L_fact": torch.tensor(1.0), "L_mmd": torch.tensor(0.0), "L_sparse": torch.tensor(0.0)}
    with pytest.raises(KeyError):
        total_loss(terms, cfg)


def test_aggregate_reports_mean_median_std_and_ignores_nan():
    """aggregate carries mean/median/std per metric and drops nan entries."""
    runs = [
        {"pehe": 1.0, "smd": 0.1},
        {"pehe": 2.0, "smd": float("nan")},
        {"pehe": 3.0, "smd": 0.3},
    ]
    summary = aggregate(runs)

    assert summary["pehe"]["mean"] == pytest.approx(2.0)
    assert summary["pehe"]["median"] == pytest.approx(2.0)
    assert summary["pehe"]["n_runs"] == 3

    assert summary["smd"]["n_runs"] == 2
    assert summary["smd"]["mean"] == pytest.approx(0.2)
    assert summary["smd"]["median"] == pytest.approx(0.2)


def test_variant_requiring_an_adapter_without_its_prior_is_impossible():
    """A variant whose branch needs P_U/q_tilde raises when that prior is not given."""
    for variant in ("u_adapter", "w_adapter", "u_w_adapter"):
        cfg = load_config(None, in_channels=M, model_variant=variant)
        with pytest.raises(ValueError):
            SSAECFRv3(cfg)
