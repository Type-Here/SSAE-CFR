"""Tests for the PGAG decomposition + gating.

We lock down: shapes; the gate lies in [0, 1]; z_mod is exactly the gated convex
combination of z_prior and z_res; the encoder is shared (same object, one parameter
set) across the passes; mu_res is deterministic while z_mod is stochastic in training;
the fixed P_U is a non-trainable buffer; and gradients reach both the encoder and gate.
"""

from __future__ import annotations

import torch

from ssae_cfr.models import PGAG, Encoder
from ssae_cfr.prior import build_projector, choose_k_svd, placeholder_embeddings

M, K = 25, 8


def _pgag():
    torch.manual_seed(0)
    V = placeholder_embeddings(M, d_LLM=64, seed=0)
    P_U = build_projector(V, choose_k_svd(V))
    enc = Encoder(m=M, hidden=(16,), k_latent=K)
    return PGAG(encoder=enc, P_U=P_U, gating_hidden=(16,)), enc


def test_pgag_output_shapes():
    pgag, _ = _pgag()
    x = torch.randn(30, M)
    out = pgag(x, omega=0.0)
    for field in (out.z_mod, out.lam, out.mu_res, out.z_prior, out.z_res):
        assert field.shape == (30, K)


def test_gate_in_unit_interval():
    pgag, _ = _pgag()
    out = pgag(torch.randn(30, M), omega=0.0)
    assert torch.all(out.lam >= 0.0) and torch.all(out.lam <= 1.0)


def test_zmod_is_gated_convex_combination():
    pgag, _ = _pgag()
    out = pgag(torch.randn(30, M), omega=0.0)
    expected = out.lam * out.z_prior + (1.0 - out.lam) * out.z_res
    assert torch.allclose(out.z_mod, expected, atol=1e-6)


def test_encoder_is_shared():
    pgag, enc = _pgag()
    assert pgag.encoder is enc, "all PGAG passes must use the one shared encoder"


def test_mu_res_deterministic_but_zmod_stochastic_in_training():
    pgag, _ = _pgag()
    pgag.train()
    x = torch.randn(16, M)
    out1 = pgag(x, omega=0.7)
    out2 = pgag(x, omega=0.7)
    assert torch.allclose(out1.mu_res, out2.mu_res), "mu_res must be noise-free"
    assert not torch.allclose(out1.z_mod, out2.z_mod), "z_mod must carry the noise"


def test_zmod_deterministic_in_eval():
    pgag, _ = _pgag()
    pgag.eval()
    x = torch.randn(16, M)
    assert torch.allclose(pgag(x, omega=0.7).z_mod, pgag(x, omega=0.7).z_mod)


def test_P_U_is_non_trainable_buffer():
    pgag, _ = _pgag()
    assert "P_U" in dict(pgag.named_buffers())
    assert "P_U" not in dict(pgag.named_parameters())


def test_gradients_reach_encoder_and_gate():
    pgag, enc = _pgag()
    pgag.train()
    out = pgag(torch.randn(16, M), omega=0.3)
    out.z_mod.sum().backward()
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    gate_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in pgag.gate.parameters())
    assert enc_grad and gate_grad


def test_rejects_mismatched_P_U():
    enc = Encoder(m=M, hidden=(16,), k_latent=K)
    bad = torch.eye(M + 1)
    try:
        PGAG(encoder=enc, P_U=bad, gating_hidden=(16,))
    except ValueError:
        return
    raise AssertionError("PGAG should reject a P_U whose size does not match m")