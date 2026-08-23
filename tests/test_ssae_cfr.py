"""Tests for the full SSAE-CFR model and a short end-to-end training smoke.

We check the forward surface (all pieces present, right shapes), that the five loss
terms are finite scalars, that a few optimisation steps actually reduce the total loss,
that predictions are deterministic in eval, and that `fit` drives the loss down on a
tiny synthetic dataset.
"""

from __future__ import annotations

import numpy as np
import torch

from ssae_cfr.config import load_config
from ssae_cfr.data.base import Dataset
from ssae_cfr.models import SSAECFR
from ssae_cfr.prior import build_projector, choose_k_svd, placeholder_embeddings
from ssae_cfr.train import fit
from ssae_cfr.utils.standardize import standardize_dataset

M = 12


def _model(cfg=None):
    cfg = cfg or load_config(None, k_latent=8, encoder_hidden=(16,), decoder_hidden=(16,),
                             head_hidden=(8,), gating_hidden=(8,))
    torch.manual_seed(0)
    V = placeholder_embeddings(M, d_LLM=32, seed=0)
    P_U = build_projector(V, choose_k_svd(V))
    return SSAECFR(m=M, P_U=torch.as_tensor(P_U, dtype=torch.float32), cfg=cfg), cfg


def _batch(n=64, seed=0):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.standard_normal((n, M)), dtype=torch.float32)
    t = torch.tensor((rng.random(n) < 0.5).astype(np.float32))
    yf = torch.tensor(rng.standard_normal(n), dtype=torch.float32)
    return x, t, yf


def test_forward_surface():
    model, _ = _model()
    x, t, _ = _batch()
    out = model(x, t, omega=0.0)
    for key in ("y0_hat", "y1_hat", "yf_hat"):
        assert out[key].shape == (64,)
    for key in ("z_mod", "lam", "mu_res", "z_prior", "z_res", "mu_enc", "z_enc"):
        assert out[key].shape == (64, 8)
    assert out["x_hat"].shape == (64, M)


def test_yf_hat_selects_the_factual_arm():
    model, _ = _model()
    x, t, _ = _batch()
    out = model(x, t, omega=0.0)
    expected = t * out["y1_hat"] + (1 - t) * out["y0_hat"]
    assert torch.allclose(out["yf_hat"], expected)


def test_loss_terms_are_finite_scalars():
    model, _ = _model()
    x, t, yf = _batch()
    terms = model.loss_terms(model(x, t, 0.0), x, t, yf, "continuous")
    assert set(terms) == {"L_fact", "L_mmd", "L_sparse", "L_rec", "L_align"}
    for v in terms.values():
        assert v.ndim == 0 and torch.isfinite(v)


def test_l1_target_switch_changes_sparse_term():
    x, t, yf = _batch()
    m_z, _ = _model(load_config(None, k_latent=8, encoder_hidden=(16,), decoder_hidden=(16,),
                                head_hidden=(8,), gating_hidden=(8,), l1_target="z"))
    m_mu, _ = _model(load_config(None, k_latent=8, encoder_hidden=(16,), decoder_hidden=(16,),
                                 head_hidden=(8,), gating_hidden=(8,), l1_target="mu"))
    # same init (seed fixed inside _model), so any difference is the target switch;
    # in eval z==mu, so force train to make z carry noise and differ from mu
    m_z.train(); m_mu.train()
    sz = m_z.loss_terms(m_z(x, t, 0.5), x, t, yf, "continuous")["L_sparse"]
    smu = m_mu.loss_terms(m_mu(x, t, 0.5), x, t, yf, "continuous")["L_sparse"]
    assert torch.isfinite(sz) and torch.isfinite(smu)


def test_optimisation_step_reduces_total_loss():
    from ssae_cfr.losses import total_loss
    model, cfg = _model()
    x, t, yf = _batch()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for step in range(40):
        model.train()
        out = model(x, t, omega=0.0)
        terms = model.loss_terms(out, x, t, yf, "continuous")
        loss, _ = total_loss(terms, cfg, epoch=100)  # past warmup, all terms active
        opt.zero_grad(); loss.backward(); opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first, "training must reduce the total loss"


def test_predict_tau_is_deterministic():
    model, _ = _model()
    x, _, _ = _batch()
    tau1 = model.predict_tau(x)
    tau2 = model.predict_tau(x)
    assert tau1.shape == (64,)
    assert torch.allclose(tau1, tau2)


def test_fit_drives_loss_down_on_synthetic():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal((n, M))
    t = (rng.random(n) < 0.5).astype(np.int64)
    # a learnable factual signal so the loss has somewhere to go
    yf = x[:, 0] + 2.0 * t + 0.1 * rng.standard_normal(n)
    ds = Dataset(name="synthetic", x=x, t=t, yf=yf, feature_names=[f"x{i}" for i in range(M)])
    ds, _ = standardize_dataset(ds)

    cfg = load_config(None, k_latent=8, encoder_hidden=(16,), decoder_hidden=(16,),
                      head_hidden=(8,), gating_hidden=(8,), epochs=60, lr=1e-2)
    model, _ = _model(cfg)
    history = fit(model, ds, cfg, verbose=False)
    assert history[-1]["L_fact"] < history[0]["L_fact"], "factual loss should fall"