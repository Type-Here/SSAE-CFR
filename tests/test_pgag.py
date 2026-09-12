"""Tests for the PGAG decomposition + admission gate.

We lock down the things the rewrite exists for. The decomposition is exact; the gate
lies in [0, 1]; `x_mod` is the additive `x_prior + b * x_res`, so `b = 1` reaches the
whole covariate vector - the reachability the previous convex gate could not express.
The gate reads the deterministic `x`, so noise cannot change its decision. The encoder
runs exactly once, so there is a single code rather than two parallel representations.
Plus the invariants the previous version already had: shared encoder, P_U a non-trainable
buffer, gradients reaching both the encoder and the gate.
"""

from __future__ import annotations

import pytest
import torch

from ssae_cfr.models import PGAG, Encoder
from ssae_cfr.prior import build_projector, choose_k_svd, placeholder_embeddings

M, K = 25, 8


def _pgag(b_mode: str = "learned", noise_scale: str = "absolute"):
    torch.manual_seed(0)
    V = placeholder_embeddings(M, d_LLM=64, seed=0)
    P_U = build_projector(V, choose_k_svd(V))
    enc = Encoder(m=M, hidden=(16,), k_latent=K, noise_scale=noise_scale)
    return PGAG(encoder=enc, P_U=P_U, gating_hidden=(16,), b_mode=b_mode), enc


def test_pgag_output_shapes():
    pgag, _ = _pgag()
    x = torch.randn(30, M)
    out = pgag(x, omega=0.0)
    for field in (out.z, out.mu):
        assert field.shape == (30, K)
    for field in (out.b, out.x_mod, out.x_prior, out.x_res):
        assert field.shape == (30, M), "the gate and the split live in covariate space"


def test_decomposition_is_exact():
    pgag, _ = _pgag()
    x = torch.randn(30, M)
    out = pgag(x, omega=0.0)
    assert torch.allclose(out.x_prior + out.x_res, x, atol=1e-5)


def test_gate_in_unit_interval():
    pgag, _ = _pgag()
    out = pgag(torch.randn(30, M), omega=0.0)
    assert torch.all(out.b >= 0.0) and torch.all(out.b <= 1.0)


def test_x_mod_is_the_additive_combination():
    pgag, _ = _pgag()
    out = pgag(torch.randn(30, M), omega=0.0)
    assert torch.allclose(out.x_mod, out.x_prior + out.b * out.x_res, atol=1e-6)


def test_b_one_reaches_the_whole_covariate_vector():
    """The point of the additive form: the model WITHOUT a prior is a value of b.

    A convex gate could not express this - it could only interpolate between the two
    halves, never recover their sum.
    """
    pgag, _ = _pgag(b_mode="one")
    x = torch.randn(30, M)
    out = pgag(x, omega=0.0)
    assert torch.allclose(out.x_mod, x, atol=1e-5)


def test_b_zero_is_the_prior_only_model():
    pgag, _ = _pgag(b_mode="zero")
    x = torch.randn(30, M)
    out = pgag(x, omega=0.0)
    assert torch.allclose(out.x_mod, out.x_prior, atol=1e-6)
    assert torch.all(out.b == 0.0)


def test_gate_reads_a_deterministic_input():
    """Same patient, same decision - the noise is downstream of the gate."""
    pgag, _ = _pgag()
    pgag.train()
    x = torch.randn(16, M)
    out1, out2 = pgag(x, omega=0.7), pgag(x, omega=0.7)
    assert torch.allclose(out1.b, out2.b), "b must not depend on the injected noise"
    assert torch.allclose(out1.mu, out2.mu), "mu must be noise-free"
    assert not torch.allclose(out1.z, out2.z), "z must carry the noise"


def test_encoder_runs_exactly_once():
    """One code, not two: the decoder, the heads and the MMD all read the same z."""
    pgag, enc = _pgag()
    calls = []
    original = enc.encode
    enc.encode = lambda x, omega=0.0: (calls.append(1), original(x, omega))[1]
    pgag(torch.randn(8, M), omega=0.0)
    assert len(calls) == 1


def test_z_deterministic_in_eval():
    pgag, _ = _pgag()
    pgag.eval()
    x = torch.randn(16, M)
    assert torch.allclose(pgag(x, omega=0.7).z, pgag(x, omega=0.7).z)


def test_encoder_is_shared():
    pgag, enc = _pgag()
    assert pgag.encoder is enc, "the PGAG pass must use the one shared encoder"


def test_P_U_is_non_trainable_buffer():
    pgag, _ = _pgag()
    assert "P_U" in dict(pgag.named_buffers())
    assert "P_U" not in dict(pgag.named_parameters())


def test_gradients_reach_encoder_and_gate():
    pgag, enc = _pgag()
    pgag.train()
    out = pgag(torch.randn(16, M), omega=0.3)
    out.z.sum().backward()
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    gate_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in pgag.gate.parameters())
    assert enc_grad and gate_grad


def test_a_fixed_gate_takes_no_gradient():
    """b_mode='one'/'zero' are ablations, so the gate MLP must stay out of the graph."""
    pgag, _ = _pgag(b_mode="one")
    pgag.train()
    pgag(torch.randn(16, M), omega=0.0).z.sum().backward()
    assert all(p.grad is None for p in pgag.gate.parameters())


def test_rejects_mismatched_P_U():
    enc = Encoder(m=M, hidden=(16,), k_latent=K)
    with pytest.raises(ValueError):
        PGAG(encoder=enc, P_U=torch.eye(M + 1), gating_hidden=(16,))


def test_rejects_unknown_b_mode():
    enc = Encoder(m=M, hidden=(16,), k_latent=K)
    with pytest.raises(ValueError):
        PGAG(encoder=enc, P_U=torch.eye(M), gating_hidden=(16,), b_mode="sometimes")
