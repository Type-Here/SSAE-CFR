"""Negative-control transformations of the semantic prior.

A capacity-matched control must have the same algebraic shape as the real prior
object it stands in for (so a measured gain cannot be attributed to extra capacity),
must genuinely destroy the semantic content it targets, and must actually change the
adapter it feeds - a control that changed nothing would make its arm a silent
duplicate of the real one.
"""

from __future__ import annotations

import torch

import pytest

from ssae_v3.prior_modules.controls import (
    CONTROLS,
    apply_control,
    random_projector,
    random_semantics,
    shuffled_semantics,
)
from ssae_v3.prior_modules.u_adapter import UStructuralAdapter
from ssae_v3.prior_modules.w_adapter import WSemanticAdapter


def test_random_projector_is_a_genuine_projector():
    """random_projector is symmetric, idempotent, and has trace == rank."""
    m, rank = 8, 3
    p = random_projector(m, rank, seed=0)
    assert torch.allclose(p, p.T, atol=1e-5)
    assert torch.allclose(p @ p, p, atol=1e-4)
    assert abs(float(torch.trace(p)) - rank) < 1e-4


def test_random_projector_is_seed_reproducible_and_seed_sensitive():
    """The same seed reproduces the same projector; a different seed gives a different one."""
    p1 = random_projector(8, 3, seed=0)
    p2 = random_projector(8, 3, seed=0)
    p3 = random_projector(8, 3, seed=1)
    assert torch.equal(p1, p2)
    assert not torch.equal(p1, p3)


def test_shuffled_semantics_is_a_true_permutation_and_not_identity():
    """shuffled_semantics permutes rows bijectively and never returns the identity."""
    m, r = 6, 4
    q = torch.randn(m, r)
    shuffled = shuffled_semantics(q, seed=0)

    assert shuffled.shape == q.shape
    assert not torch.equal(shuffled, q)

    used = set()
    for i in range(m):
        matches = [j for j in range(m) if torch.equal(shuffled[i], q[j])]
        assert len(matches) == 1
        used.add(matches[0])
    assert used == set(range(m))


def test_random_semantics_matches_shape_and_rms():
    """random_semantics matches q_tilde's shape and root-mean-square magnitude."""
    q = torch.randn(6, 4) * 3.0
    r = random_semantics(q, seed=0)

    assert r.shape == q.shape
    real_rms = torch.sqrt(torch.mean(q.double() ** 2))
    got_rms = torch.sqrt(torch.mean(r.double() ** 2))
    assert abs(float(real_rms) - float(got_rms)) < 1e-4


def test_apply_control_dispatches_every_named_control():
    """apply_control handles every name in CONTROLS and rejects an unknown one."""
    m, r = 6, 3
    p_u = random_projector(m, r, seed=0)
    q_tilde = torch.randn(m, r)

    for control in CONTROLS:
        p_out, q_out = apply_control(p_u, q_tilde, control, seed=0)
        assert p_out is not None
        assert q_out is not None

    with pytest.raises(ValueError):
        apply_control(p_u, q_tilde, "not-a-real-control", seed=0)


def test_apply_control_leaves_an_absent_branch_as_none():
    """A branch not supplied (None) is never fabricated by a control."""
    m, r = 6, 3
    q_tilde = torch.randn(m, r)
    p_out, q_out = apply_control(None, q_tilde, "nonsemantic", seed=0)
    assert p_out is None
    assert q_out is not None

    p_u = random_projector(m, r, seed=0)
    p_out2, q_out2 = apply_control(p_u, None, "nonsemantic", seed=0)
    assert p_out2 is not None
    assert q_out2 is None


def test_shuffled_semantics_control_actually_changes_the_w_branch():
    """Swapping in shuffled semantics changes a_W, so the control is not a no-op."""
    m, r, d_u = 7, 3, 6
    q_tilde = torch.randn(m, r)
    shuffled = shuffled_semantics(q_tilde, seed=0)

    torch.manual_seed(42)
    real_adapter = WSemanticAdapter(q_tilde, d_u, d_token=6, phi_hidden=(6,), d_s=6, rho_hidden=(6,))
    torch.nn.init.normal_(real_adapter.a_w_head[-1].weight, std=1.0)
    torch.nn.init.normal_(real_adapter.a_w_head[-1].bias, std=1.0)

    torch.manual_seed(42)
    control_adapter = WSemanticAdapter(
        shuffled, d_u, d_token=6, phi_hidden=(6,), d_s=6, rho_hidden=(6,)
    )
    torch.nn.init.normal_(control_adapter.a_w_head[-1].weight, std=1.0)
    torch.nn.init.normal_(control_adapter.a_w_head[-1].bias, std=1.0)

    x = torch.randn(5, m)
    assert not torch.allclose(real_adapter(x), control_adapter(x), atol=1e-6)


def test_random_projector_control_actually_changes_the_u_branch():
    """Swapping in a random rank-matched projector changes c, so the control is not a no-op."""
    m, rank, d_u = 7, 3, 6
    p_u = random_projector(m, rank, seed=0)
    p_control = random_projector(m, rank, seed=99)

    torch.manual_seed(7)
    real_adapter = UStructuralAdapter(p_u, d_u, hidden=(6,))
    torch.nn.init.normal_(real_adapter.net[-1].weight, std=1.0)
    torch.nn.init.normal_(real_adapter.net[-1].bias, std=1.0)

    torch.manual_seed(7)
    control_adapter = UStructuralAdapter(p_control, d_u, hidden=(6,))
    torch.nn.init.normal_(control_adapter.net[-1].weight, std=1.0)
    torch.nn.init.normal_(control_adapter.net[-1].bias, std=1.0)

    x = torch.randn(5, m)
    assert not torch.allclose(real_adapter(x), control_adapter(x), atol=1e-6)
