"""Tests for PriorGuidance: the fixed covariate-space transform and its
subspace-alignment loss.

Nothing here needs a GPU or a built prior artifact; projectors and bases are built
from small random orthonormal matrices via QR.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from ssae_v3.core.modules import Encoder
from ssae_v3.prior_modules.prior_guidance import (
    PRIOR_MODES,
    PriorGuidance,
    first_linear_weight,
)

M = 7


def random_basis(m: int, rank: int, seed: int) -> torch.Tensor:
    """An orthonormal (m, rank) basis via the Q factor of a random QR."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(m, rank, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    return q.to(dtype=torch.float32)


def make_prior(m: int, rank: int, seed: int):
    q = random_basis(m, rank, seed)
    p = q @ q.t()
    return p, q


def disjoint_priors(m: int, r_u: int, r_c: int, seed: int):
    """p_u/q_u and p_c/q_c on complementary, non-overlapping subspaces."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(m, r_u + r_c, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    q = q.to(dtype=torch.float32)
    q_u, q_c = q[:, :r_u], q[:, r_u:]
    return (q_u @ q_u.t()), q_u, (q_c @ q_c.t()), q_c


# --------------------------------------------------------------------------- transform


def test_none_mode_returns_input_object_and_zero_loss():
    guidance = PriorGuidance(mode="none")
    x = torch.randn(4, M)
    x_guided, diag = guidance.transform(x)

    assert x_guided is x
    assert diag["x_prior_norm"] == 0.0
    assert diag["x_guided_delta_ratio"] == 0.0

    weight = nn.Linear(M, 5).weight
    loss, loss_diag = guidance.guidance_loss(weight)
    assert float(loss) == 0.0
    assert loss.grad_fn is None
    assert math.isnan(loss_diag["alignment_active"])
    assert math.isnan(loss_diag["alignment_U"])
    assert math.isnan(loss_diag["alignment_C"])


def test_embedding_only_transform_matches_formula():
    p_u, q_u = make_prior(M, 3, seed=1)
    gamma = 0.4
    guidance = PriorGuidance(p_u=p_u, q_u=q_u, mode="embedding", gamma_prior=gamma)
    x = torch.randn(5, M)
    x_guided, _ = guidance.transform(x)
    expected = x + gamma * (x @ p_u)
    assert torch.allclose(x_guided, expected, atol=1e-6)


def test_graph_only_transform_matches_formula():
    p_c, q_c = make_prior(M, 2, seed=2)
    gamma = 0.9
    guidance = PriorGuidance(p_c=p_c, q_c=q_c, mode="graph", gamma_prior=gamma)
    x = torch.randn(5, M)
    x_guided, _ = guidance.transform(x)
    expected = x + gamma * (x @ p_c)
    assert torch.allclose(x_guided, expected, atol=1e-6)


def test_combined_transform_averages_not_sums():
    p_u, q_u = make_prior(M, 3, seed=1)
    p_c, q_c = make_prior(M, 2, seed=2)
    gamma = 0.5
    guidance = PriorGuidance(
        p_u=p_u, q_u=q_u, p_c=p_c, q_c=q_c, mode="both", gamma_prior=gamma
    )
    x = torch.randn(4, M)
    x_guided, _ = guidance.transform(x)

    averaged = x + gamma * 0.5 * (x @ p_u + x @ p_c)
    summed = x + gamma * (x @ p_u + x @ p_c)
    assert torch.allclose(x_guided, averaged, atol=1e-6)
    assert not torch.allclose(x_guided, summed, atol=1e-4)


def test_x_guided_equals_x_at_T_buffer():
    p_u, q_u = make_prior(M, 3, seed=3)
    guidance = PriorGuidance(p_u=p_u, q_u=q_u, mode="embedding", gamma_prior=0.7)
    x = torch.randn(6, M)
    x_guided, _ = guidance.transform(x)
    assert torch.allclose(x_guided, x @ guidance.T, atol=1e-6)


# ------------------------------------------------------------------------ parameters


@pytest.mark.parametrize("mode", PRIOR_MODES)
def test_zero_trainable_parameters(mode):
    kwargs = {}
    if mode in ("embedding", "both"):
        p_u, q_u = make_prior(M, 3, seed=10)
        kwargs.update(p_u=p_u, q_u=q_u)
    if mode in ("graph", "both"):
        p_c, q_c = make_prior(M, 2, seed=11)
        kwargs.update(p_c=p_c, q_c=q_c)
    guidance = PriorGuidance(mode=mode, gamma_prior=0.3, **kwargs)
    assert list(guidance.parameters()) == []


# -------------------------------------------------------------------- constructor validation


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        PriorGuidance(mode="bogus")


def test_rejects_mode_missing_its_prior():
    with pytest.raises(ValueError):
        PriorGuidance(mode="embedding")


def test_rejects_unused_prior_supplied():
    p_u, q_u = make_prior(M, 3, seed=1)
    with pytest.raises(ValueError):
        PriorGuidance(p_u=p_u, q_u=q_u, mode="none")


def test_rejects_nonsquare_p():
    q_u = random_basis(M, 3, seed=1)
    p_bad = torch.randn(M, M + 1)
    with pytest.raises(ValueError, match="square"):
        PriorGuidance(p_u=p_bad, q_u=q_u, mode="embedding")


def test_rejects_m_mismatch_between_p_and_q():
    p_u, _ = make_prior(M, 3, seed=1)
    q_bad = random_basis(M + 1, 3, seed=2)
    with pytest.raises(ValueError):
        PriorGuidance(p_u=p_u, q_u=q_bad, mode="embedding")


def test_rejects_negative_gamma():
    p_u, q_u = make_prior(M, 3, seed=1)
    with pytest.raises(ValueError, match="gamma_prior"):
        PriorGuidance(p_u=p_u, q_u=q_u, mode="embedding", gamma_prior=-0.1)


# ---------------------------------------------------------------------------- loss


def test_loss_zero_inside_active_span_positive_outside():
    p_u, q_u = make_prior(M, 3, seed=5)
    guidance = PriorGuidance(p_u=p_u, q_u=q_u, mode="embedding", gamma_prior=0.0)
    h = 4

    coeff = torch.randn(3, h)
    weight_inside = (q_u @ coeff).t()  # W_eff = W1 lies exactly in span(q_u)
    loss_inside, _ = guidance.guidance_loss(weight_inside)
    assert float(loss_inside) < 1e-6

    weight_outside = torch.randn(h, M)
    loss_outside, _ = guidance.guidance_loss(weight_outside)
    assert float(loss_outside) > 1e-3


def test_loss_invariant_to_global_scale():
    p_u, q_u = make_prior(M, 3, seed=6)
    guidance = PriorGuidance(p_u=p_u, q_u=q_u, mode="embedding", gamma_prior=0.6)
    weight = torch.randn(5, M)
    loss1, _ = guidance.guidance_loss(weight)
    loss2, _ = guidance.guidance_loss(weight * 3.7)
    assert torch.allclose(loss1, loss2, atol=1e-6)


def test_both_mode_uses_union_not_intersection():
    p_u, q_u, p_c, q_c = disjoint_priors(M, 3, 2, seed=7)
    guidance = PriorGuidance(
        p_u=p_u, q_u=q_u, p_c=p_c, q_c=q_c, mode="both", gamma_prior=0.0
    )
    h = 4
    coeff = torch.randn(3, h)
    weight = (q_u @ coeff).t()  # W_eff lies entirely inside span(q_u), not span(q_c)
    loss, _ = guidance.guidance_loss(weight)
    assert float(loss) < 1e-6


def test_active_rank_is_union_rank_not_sum_when_overlapping():
    q_u = random_basis(M, 3, seed=8)
    gen = torch.Generator().manual_seed(9)
    raw = torch.randn(M, 1, generator=gen, dtype=torch.float64)
    q_u64 = q_u.to(torch.float64)
    w = raw - q_u64 @ (q_u64.t() @ raw)
    w = w / w.norm()
    # q_c's first column IS q_u's first column: rank(union) = 3 + 1, not 3 + 2
    q_c = torch.cat([q_u64[:, :1], w], dim=1).to(dtype=torch.float32)
    p_u = q_u @ q_u.t()
    p_c = q_c @ q_c.t()

    guidance = PriorGuidance(
        p_u=p_u, q_u=q_u, p_c=p_c, q_c=q_c, mode="both", gamma_prior=0.0
    )
    assert guidance.active_rank == 4
    assert guidance.active_rank != q_u.shape[1] + q_c.shape[1]


# --------------------------------------------------------------------- first_linear_weight


def test_first_linear_weight_sequential():
    net = nn.Sequential(nn.Linear(M, 5), nn.ReLU(), nn.Linear(5, 2))
    weight = first_linear_weight(net)
    assert weight is net[0].weight


def test_first_linear_weight_encoder():
    enc = Encoder(M, (8,), 4)
    weight = first_linear_weight(enc)
    assert weight is enc.net[0].weight


def test_first_linear_weight_raises_without_linear():
    net = nn.Sequential(nn.ReLU(), nn.Tanh())
    with pytest.raises(ValueError, match="no nn.Linear"):
        first_linear_weight(net)


def test_first_linear_weight_raises_when_earlier_module_owns_params():
    net = nn.Sequential(nn.BatchNorm1d(M), nn.Linear(M, 5))
    with pytest.raises(ValueError):
        first_linear_weight(net)


# ------------------------------------------------------------------------------ buffers


def test_registered_buffers_do_not_alias_caller_tensors():
    p_u, q_u = make_prior(M, 3, seed=12)
    p_u_copy, q_u_copy = p_u.clone(), q_u.clone()
    guidance = PriorGuidance(p_u=p_u, q_u=q_u, mode="embedding", gamma_prior=0.2)

    p_u.mul_(0.0)
    q_u.mul_(0.0)

    assert torch.allclose(guidance.p_u, p_u_copy)
    assert torch.allclose(guidance.q_u, q_u_copy)
