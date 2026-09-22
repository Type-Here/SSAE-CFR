"""Tests for SSAECFRv4: the empirical host guided at its input by a fixed prior
subspace, with no adapter and zero added trainable parameters.

Small synthetic data and tiny model dimensions throughout; nothing here needs a GPU
or a built prior artifact. Projectors are built from random orthonormal bases via QR,
the same way tests/test_prior_guidance.py does.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ssae_v3.data.base import Dataset
from ssae_v3.hparams import load_config
from ssae_v3.model_v3 import SSAECFRv3
from ssae_v3.model_v4 import SSAECFRv4
from ssae_v3.prior_modules import PriorGuidance
from ssae_v3.training.train import fit

M = 6


def _random_basis(m: int, rank: int, seed: int) -> torch.Tensor:
    """An orthonormal (m, rank) basis via the Q factor of a random QR."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(m, rank, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    return q.to(dtype=torch.float32)


def _make_projector(m: int, rank: int, seed: int):
    q = _random_basis(m, rank, seed)
    p = q @ q.t()
    return p, q


def _build_guidance(
    mode: str, m: int, gamma_prior: float = 0.3, r_u: int = 3, r_c: int = 2, seed: int = 0
) -> PriorGuidance:
    kwargs = {}
    if mode in ("embedding", "both"):
        p_u, q_u = _make_projector(m, r_u, seed + 1)
        kwargs.update(p_u=p_u, q_u=q_u)
    if mode in ("graph", "both"):
        p_c, q_c = _make_projector(m, r_c, seed + 2)
        kwargs.update(p_c=p_c, q_c=q_c)
    return PriorGuidance(mode=mode, gamma_prior=gamma_prior, **kwargs)


def _tiny_cfg(**overrides):
    base = dict(
        in_channels=M,
        d_u=8,
        encoder_hidden=(8,),
        decoder_hidden=(8,),
        head_hidden=(8,),
        epochs=20,
        lr=1e-2,
        is_noise_active=False,
    )
    base.update(overrides)
    return load_config(None, **base)


def _synthetic_dataset(n: int = 40, m: int = M, seed: int = 0, name: str = "synthetic") -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, m))
    t = (rng.random(n) < 0.5).astype(np.int64)
    yf = x[:, 0] + 0.5 * t + rng.standard_normal(n) * 0.1
    return Dataset(name=name, x=x, t=t, yf=yf, feature_names=[f"x{i}" for i in range(m)])


# --------------------------------------------------------------- prior-off equivalence


def test_prior_off_equivalence_forward_outputs():
    """SSAECFRv4(prior_mode='none') reproduces SSAECFRv3(model_variant='empirical')
    exactly, once their empirical/heads weights are copied across."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(model_variant="empirical")
    v3 = SSAECFRv3(cfg)
    v4 = SSAECFRv4(cfg)
    v4.empirical.load_state_dict(v3.empirical.state_dict())
    v4.heads.load_state_dict(v3.heads.state_dict())
    v3.eval()
    v4.eval()

    x = torch.randn(10, M)
    t = (torch.rand(10) < 0.5).float()
    out3 = v3(x, t, omega=0.0)
    out4 = v4(x, t, omega=0.0)
    for key in ("y0_hat", "y1_hat", "x_hat", "u"):
        assert torch.equal(out3[key], out4[key]), key


def test_seeded_construction_matches_v3_empirical():
    """Under one seed, SSAECFRv3(empirical) and SSAECFRv4(prior_mode='none') build
    identical encoder/head weights: guidance construction consumes no RNG, and both
    models build Empirical then OutcomeHeads in the same order."""
    torch.manual_seed(123)
    cfg = _tiny_cfg(model_variant="empirical")
    v3 = SSAECFRv3(cfg)
    torch.manual_seed(123)
    v4 = SSAECFRv4(cfg)

    for (n3, p3), (n4, p4) in zip(v3.empirical.state_dict().items(), v4.empirical.state_dict().items()):
        assert torch.equal(p3, p4), f"empirical.{n3}"
    for (n3, p3), (n4, p4) in zip(v3.heads.state_dict().items(), v4.heads.state_dict().items()):
        assert torch.equal(p3, p4), f"heads.{n3}"


# ------------------------------------------------------------- parameter matching


def test_parameter_count_identical_across_prior_modes():
    """Trainable parameter count is unchanged by prior_mode; guidance itself is inert."""
    counts = {}
    for mode in ("none", "embedding", "graph", "both"):
        kwargs = {} if mode == "none" else dict(prior_mode=mode, gamma_prior=0.3, lambda_prior=0.1)
        cfg = _tiny_cfg(model_variant="empirical", **kwargs)
        guidance = None if mode == "none" else _build_guidance(mode, M, seed=1)
        model = SSAECFRv4(cfg, guidance=guidance)
        assert sum(p.numel() for p in model.guidance.parameters()) == 0
        counts[mode] = sum(p.numel() for p in model.parameters())
    assert len(set(counts.values())) == 1, counts


# ----------------------------------------------------------- reconstruction target


def test_reconstruction_loss_uses_original_x_not_guided():
    torch.manual_seed(0)
    guidance = _build_guidance("embedding", M, gamma_prior=0.5, seed=2)
    cfg = _tiny_cfg(model_variant="empirical", prior_mode="embedding", gamma_prior=0.5, lambda_prior=0.2)
    model = SSAECFRv4(cfg, guidance=guidance)
    model.eval()

    x = torch.randn(9, M)
    t = (torch.rand(9) < 0.5).float()
    yf = torch.randn(9)
    out = model(x, t, omega=0.0)
    terms = model.loss_terms(out, x, t, yf)

    expected = F.mse_loss(out["x_hat"], x)
    assert torch.allclose(terms["L_rec"], expected)

    x_guided, _ = guidance.transform(x)
    assert not torch.allclose(x_guided, x)
    wrong = F.mse_loss(out["x_hat"], x_guided)
    assert not torch.allclose(terms["L_rec"], wrong)


def test_balance_representation_is_u():
    cfg = _tiny_cfg(model_variant="empirical")
    model = SSAECFRv4(cfg)
    x = torch.randn(5, M)
    t = (torch.rand(5) < 0.5).float()
    out = model(x, t, omega=0.0)
    assert model.balance_representation(out) is out["u"]


# --------------------------------------------------------------------- L_guidance


def test_l_guidance_present_only_when_active():
    cfg_off = _tiny_cfg(model_variant="empirical")
    model_off = SSAECFRv4(cfg_off)
    x = torch.randn(6, M)
    t = (torch.rand(6) < 0.5).float()
    yf = torch.randn(6)
    out_off = model_off(x, t, omega=0.0)
    terms_off = model_off.loss_terms(out_off, x, t, yf)
    assert "L_guidance" not in terms_off

    guidance = _build_guidance("embedding", M, gamma_prior=0.4, seed=3)
    cfg_on = _tiny_cfg(model_variant="empirical", prior_mode="embedding", gamma_prior=0.4, lambda_prior=0.25)
    model_on = SSAECFRv4(cfg_on, guidance=guidance)
    out_on = model_on(x, t, omega=0.0)
    terms_on = model_on.loss_terms(out_on, x, t, yf)
    assert "L_guidance" in terms_on

    total, breakdown = model_on.total_loss(terms_on, cfg_on)
    expected_total = (
        terms_on["L_fact"]
        + cfg_on.alpha_mmd * terms_on["L_mmd"]
        + cfg_on.beta_l1 * terms_on["L_sparse"]
        + cfg_on.lambda_rec * terms_on["L_rec"]
        + cfg_on.lambda_prior * terms_on["L_guidance"]
    )
    assert torch.allclose(total, expected_total)
    assert "share_L_guidance" in breakdown


def test_lambda_prior_zero_matches_four_term_total():
    guidance = _build_guidance("embedding", M, gamma_prior=0.4, seed=4)
    cfg = _tiny_cfg(model_variant="empirical", prior_mode="embedding", gamma_prior=0.4, lambda_prior=0.0)
    model = SSAECFRv4(cfg, guidance=guidance)

    x = torch.randn(6, M)
    t = (torch.rand(6) < 0.5).float()
    yf = torch.randn(6)
    out = model(x, t, omega=0.0)
    terms = model.loss_terms(out, x, t, yf)
    total, _ = model.total_loss(terms, cfg)
    four_term = (
        terms["L_fact"]
        + cfg.alpha_mmd * terms["L_mmd"]
        + cfg.beta_l1 * terms["L_sparse"]
        + cfg.lambda_rec * terms["L_rec"]
    )
    assert torch.allclose(total, four_term)


def test_optimizer_step_leaves_guidance_buffers_unchanged():
    torch.manual_seed(5)
    guidance = _build_guidance("both", M, gamma_prior=0.3, seed=6)
    buffers_before = {k: v.clone() for k, v in guidance.state_dict().items()}
    cfg = _tiny_cfg(model_variant="empirical", prior_mode="both", gamma_prior=0.3, lambda_prior=0.2, lr=1e-2)
    model = SSAECFRv4(cfg, guidance=guidance)

    x = torch.randn(12, M)
    t = (torch.rand(12) < 0.5).float()
    yf = torch.randn(12)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    out = model(x, t, omega=0.0)
    terms = model.loss_terms(out, x, t, yf)
    loss1, _ = model.total_loss(terms, cfg)
    opt.zero_grad()
    loss1.backward()
    opt.step()

    out2 = model(x, t, omega=0.0)
    terms2 = model.loss_terms(out2, x, t, yf)
    loss2, _ = model.total_loss(terms2, cfg)

    assert float(loss1.detach()) != pytest.approx(float(loss2.detach()))
    for key, value in model.guidance.state_dict().items():
        assert torch.equal(value, buffers_before[key]), key


# ------------------------------------------------------------- constructor validation


def test_rejects_adapter_flag_from_model_variant():
    cfg = _tiny_cfg(model_variant="u_adapter")
    with pytest.raises(ValueError):
        SSAECFRv4(cfg)


def test_rejects_adapter_flag_set_by_plain_assignment():
    """A config mutated by plain attribute assignment (skipping DefaultConfig.validate)
    must still be rejected: SSAECFRv4 cannot trust the flag came through validate()."""
    cfg = _tiny_cfg(model_variant="empirical")
    cfg.use_w_adapter = True
    with pytest.raises(ValueError):
        SSAECFRv4(cfg)


def test_rejects_prior_mode_guidance_mismatch():
    guidance = _build_guidance("embedding", M, seed=7)
    cfg = _tiny_cfg(model_variant="empirical", prior_mode="graph", gamma_prior=0.1, lambda_prior=0.1)
    with pytest.raises(ValueError, match="embedding"):
        SSAECFRv4(cfg, guidance=guidance)


def test_rejects_live_guidance_with_prior_mode_none():
    guidance = _build_guidance("embedding", M, seed=8)
    cfg = _tiny_cfg(model_variant="empirical", prior_mode="none")
    with pytest.raises(ValueError):
        SSAECFRv4(cfg, guidance=guidance)


def test_rejects_guidance_m_mismatch():
    guidance = _build_guidance("embedding", M + 2, seed=9)
    cfg = _tiny_cfg(model_variant="empirical", prior_mode="embedding", gamma_prior=0.2, lambda_prior=0.1)
    with pytest.raises(ValueError):
        SSAECFRv4(cfg, guidance=guidance)


# --------------------------------------------------------------- prediction interface


def test_predict_tau_and_potential_outcomes():
    cfg = _tiny_cfg(model_variant="empirical")
    model = SSAECFRv4(cfg, y_loc=2.0, y_scale=3.0)
    model.train()
    x = torch.randn(4, M)

    y0, y1 = model.potential_outcomes(x)
    assert model.training is True
    assert not y0.requires_grad
    assert not y1.requires_grad

    tau = model.predict_tau(x)
    assert model.training is True
    assert not tau.requires_grad
    assert torch.allclose(tau, y1 - y0, atol=1e-5)

    # The outcome-scale affine was actually applied (loc/scale != 0/1): the raw head
    # output must differ from the scaled potential outcome.
    model.eval()
    with torch.no_grad():
        out = model(x, torch.zeros(4), omega=0.0)
    assert not torch.allclose(y0, out["y0_hat"])


# ------------------------------------------------------------------- end-to-end fit


def test_end_to_end_training_with_guidance():
    torch.manual_seed(11)
    guidance = _build_guidance("embedding", M, gamma_prior=0.3, seed=12)
    cfg = _tiny_cfg(
        model_variant="empirical", prior_mode="embedding", gamma_prior=0.3,
        lambda_prior=0.1, epochs=6,
    )
    model = SSAECFRv4(cfg, guidance=guidance)
    ds = _synthetic_dataset()
    history = fit(model, ds, cfg, verbose=False)

    assert len(history) == cfg.epochs
    assert "share_L_guidance" in history[-1]
