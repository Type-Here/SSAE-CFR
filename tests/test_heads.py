"""Tests for the TARNet outcome heads.

The two heads must be independent (separate parameters, generally different outputs),
emit one scalar per unit, and pass gradient back.
"""

from __future__ import annotations

import torch

from ssae_cfr.models import OutcomeHeads

K = 8


def test_head_output_shapes():
    heads = OutcomeHeads(k_latent=K, hidden=(16,))
    z = torch.randn(20, K)
    y0, y1 = heads(z)
    assert y0.shape == (20,)
    assert y1.shape == (20,)


def test_heads_are_independent():
    torch.manual_seed(0)
    heads = OutcomeHeads(k_latent=K, hidden=(16,))
    z = torch.randn(20, K)
    y0, y1 = heads(z)
    # two independently initialised heads should not produce identical predictions
    assert not torch.allclose(y0, y1)


def test_gradients_flow_to_both_heads():
    heads = OutcomeHeads(k_latent=K, hidden=(16,))
    z = torch.randn(20, K)
    y0, y1 = heads(z)
    (y0.sum() + y1.sum()).backward()
    assert all(p.grad is not None for p in heads.h0.parameters())
    assert all(p.grad is not None for p in heads.h1.parameters())