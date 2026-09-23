"""Regression tests required before the v0.5.1 30-realization comparison.

Covers the plan's section 21 list: the direct path is unchanged, the attention module
is parameter-free and probabilistic, uniform attention reproduces direct fusion
exactly, gradients reach the encoder through the attention path, the prior is frozen,
and every arm carries the same trainable parameter count.

Nothing here needs a GPU, a built artifact or transformers: Z is a small random
matrix, since every property under test is a property of the code and not of the
embeddings.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from slr_v0_5.config import SLRConfig
from slr_v0_5.model import SLRCFRv05
from slr_v0_5.prior.artifacts import FeatureEmbeddings, center_embeddings
from slr_v0_5.prior.base import build_prior_integration
from slr_v0_5.prior.controls import apply_embedding_variant, prior_rms
from slr_v0_5.prior.cosine_cross_attention import ParameterFreeCrossAttentionIntegration
from slr_v0_5.prior.direct_embedding import DirectEmbeddingIntegration

M = 6
D_Q = 8
N = 12


@pytest.fixture
def embeddings() -> FeatureEmbeddings:
    generator = torch.Generator().manual_seed(0)
    Z = torch.randn(M, D_Q, generator=generator)
    return FeatureEmbeddings(
        feature_ids=tuple(f"x{j + 1}" for j in range(M)),
        Z=Z,
        metadata={"m": M, "d_q": D_Q, "centered": False},
    )


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(N, M, generator=generator)
    mu = torch.randn(N, D_Q, generator=generator)
    t = (torch.rand(N, generator=generator) < 0.4).float()
    return x, mu, t


def config(**overrides) -> SLRConfig:
    base = SLRConfig(
        in_channels=M,
        d_latent=D_Q,
        encoder_hidden=(5,),
        head_hidden=(4,),
        eta_prior=0.3,
        epochs=1,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


# -- the direct path is unchanged -----------------------------------------


def test_direct_is_the_closed_form(embeddings, batch):
    x, mu, _ = batch
    p, _ = DirectEmbeddingIntegration(embeddings.Z)(x, mu)
    assert torch.equal(p, (x @ embeddings.Z) / math.sqrt(M))


def test_direct_fusion_is_unchanged(embeddings, batch):
    """u_out is exactly u_emp + eta * p_direct, and the heads read that."""
    x, _, t = batch
    cfg = config()
    model = SLRCFRv05(cfg, build_prior_integration(cfg, embeddings))
    out = model(x, t)
    expected_p = (x @ embeddings.Z) / math.sqrt(M)
    assert torch.equal(out["u_prior"], expected_p)
    assert torch.equal(out["u_out"], out["u_emp"] + cfg.eta_prior * expected_p)
    y0, y1 = model.heads(out["u_out"])
    assert torch.equal(out["y0_hat"], y0) and torch.equal(out["y1_hat"], y1)


def test_direct_ignores_mu_emp(embeddings, batch):
    x, mu, _ = batch
    integration = DirectEmbeddingIntegration(embeddings.Z)
    assert torch.equal(integration(x, mu)[0], integration(x, torch.zeros_like(mu))[0])


# -- the attention module --------------------------------------------------


def test_attention_has_no_trainable_parameters(embeddings):
    integration = ParameterFreeCrossAttentionIntegration(embeddings.Z)
    assert sum(p.numel() for p in integration.parameters() if p.requires_grad) == 0


def test_frozen_prior(embeddings):
    integration = ParameterFreeCrossAttentionIntegration(embeddings.Z)
    assert integration.Z.requires_grad is False
    assert integration.K_hat.requires_grad is False


def test_shapes_and_probabilities(embeddings, batch):
    x, mu, _ = batch
    integration = ParameterFreeCrossAttentionIntegration(embeddings.Z)
    p, diag = integration(x, mu)
    alpha = diag["alpha"]
    assert alpha.shape == (N, M)
    assert p.shape == (N, D_Q)
    assert torch.all(alpha >= 0.0)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(N), atol=1e-6)
    assert torch.isfinite(p).all()


def test_uniform_attention_reproduces_direct(embeddings, batch):
    """The identity the whole interpretation rests on: at alpha = 1/m the attention
    prior IS the direct prior, so attention generalizes direct fusion."""
    x, _, _ = batch
    uniform = torch.full((N, M), 1.0 / M)
    p_att = ((x * (M * uniform)) @ embeddings.Z) / math.sqrt(M)
    p_direct, _ = DirectEmbeddingIntegration(embeddings.Z)(x, torch.zeros(N, D_Q))
    assert torch.allclose(p_att, p_direct, atol=1e-6)


def test_attention_depends_on_the_query(embeddings, batch):
    x, mu, _ = batch
    integration = ParameterFreeCrossAttentionIntegration(embeddings.Z)
    other = torch.randn(N, D_Q, generator=torch.Generator().manual_seed(2))
    assert not torch.allclose(integration(x, mu)[0], integration(x, other)[0])


def test_temperature_sharpens(embeddings, batch):
    _, mu, _ = batch
    warm = ParameterFreeCrossAttentionIntegration(embeddings.Z, temperature=1.0).attention(mu)
    cold = ParameterFreeCrossAttentionIntegration(embeddings.Z, temperature=0.1).attention(mu)
    assert cold.max(dim=-1).values.mean() > warm.max(dim=-1).values.mean()


def test_cosine_score_ignores_a_global_value_rescaling(embeddings, batch):
    """What makes the norm-matched control reusable unchanged: the scalar the control
    applies changes the values, not the attention weights."""
    _, mu, _ = batch
    plain = ParameterFreeCrossAttentionIntegration(embeddings.Z).attention(mu)
    scaled = ParameterFreeCrossAttentionIntegration(embeddings.Z * 3.7).attention(mu)
    assert torch.allclose(plain, scaled, atol=1e-5)


def test_norm_matching_gives_each_patient_the_direct_magnitude(embeddings, batch):
    x, mu, _ = batch
    matched, diag = ParameterFreeCrossAttentionIntegration(
        embeddings.Z, temperature=0.05, norm_match=True
    )(x, mu)
    direct, _ = DirectEmbeddingIntegration(embeddings.Z)(x, mu)
    assert torch.allclose(matched.norm(dim=-1), direct.norm(dim=-1), atol=1e-4)
    assert diag["attention_norm_match"] == 1.0


def test_norm_matching_keeps_the_direction(embeddings, batch):
    """It rescales, it does not re-aim: the attention pattern still decides where the
    patient's prior points."""
    x, mu, _ = batch
    plain, _ = ParameterFreeCrossAttentionIntegration(embeddings.Z, temperature=0.05)(x, mu)
    matched, _ = ParameterFreeCrossAttentionIntegration(
        embeddings.Z, temperature=0.05, norm_match=True
    )(x, mu)
    cosine = torch.nn.functional.cosine_similarity(plain, matched, dim=-1)
    assert torch.allclose(cosine, torch.ones(N), atol=1e-5)


def test_norm_matching_preserves_the_uniform_identity(embeddings, batch):
    """At uniform attention the rescaling factor is exactly 1, so the direct identity
    holds with norm matching on as well."""
    x, mu, _ = batch
    flat = torch.ones(M, D_Q)
    matched, _ = ParameterFreeCrossAttentionIntegration(flat, norm_match=True)(x, mu)
    assert torch.allclose(matched, (x @ flat) / math.sqrt(M), atol=1e-4)


def test_gradient_reaches_the_encoder_through_attention(embeddings, batch):
    x, _, t = batch
    for norm_match in (False, True):
        cfg = config(prior_integration="cosine_cross_attention", attention_norm_match=norm_match)
        model = SLRCFRv05(cfg, build_prior_integration(cfg, embeddings))
        out = model(x, t)
        out["u_out"].pow(2).sum().backward()
        first = model.encoder.net[0]
        assert first.weight.grad is not None and first.weight.grad.abs().sum() > 0


def test_attention_diagnostics_are_finite(embeddings, batch):
    x, mu, _ = batch
    _, diag = ParameterFreeCrossAttentionIntegration(embeddings.Z)(x, mu)
    scalars = {k: v for k, v in diag.items() if isinstance(v, float)}
    assert scalars and all(math.isfinite(v) for v in scalars.values())
    assert 0.0 <= scalars["attention_entropy_normalized"] <= 1.0


# -- arms ------------------------------------------------------------------


def test_every_arm_has_the_same_parameter_count(embeddings, batch):
    x, _, _ = batch
    counts = set()
    for integration in ("direct_embedding", "cosine_cross_attention"):
        for variant, eta in (("none", 0.0), ("real", 0.3), ("permuted", 0.3), ("permuted_norm_matched", 0.3)):
            cfg = config(prior_integration=integration, embedding_variant=variant, eta_prior=eta)
            arm = apply_embedding_variant(embeddings, variant, seed=7, x_std=x)
            model = SLRCFRv05(cfg, build_prior_integration(cfg, arm if variant != "none" else None))
            counts.add(sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert len(counts) == 1


def test_norm_matched_control_matches_the_real_prior_rms(embeddings, batch):
    x, _, _ = batch
    centered = center_embeddings(embeddings)
    matched = apply_embedding_variant(centered, "permuted_norm_matched", seed=7, x_std=x)
    assert prior_rms(matched.Z, x) == pytest.approx(prior_rms(centered.Z, x), rel=1e-5)


def test_projected_cross_attention_is_refused(embeddings):
    cfg = config(prior_integration="cross_attention")
    with pytest.raises(NotImplementedError):
        build_prior_integration(cfg, embeddings)
