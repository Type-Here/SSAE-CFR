"""Tests for the SSAE encoder and decoder.

The contract we lock down: the encoder returns a (mu, z) pair of the right width; mu is
deterministic; z equals mu when omega is 0 or in eval, and becomes noisy (but centred
on mu) only while training with omega != 0; the decoder maps a code back to R^m; and
gradients flow through both.
"""

from __future__ import annotations

import torch

from ssae_cfr.models import Decoder, Encoder

M, K = 25, 8


def _encoder():
    torch.manual_seed(0)
    return Encoder(m=M, hidden=(16,), k_latent=K)


def test_encoder_output_shapes():
    enc = _encoder()
    x = torch.randn(32, M)
    mu, z = enc.encode(x, omega=0.0)
    assert mu.shape == (32, K)
    assert z.shape == (32, K)


def test_mu_is_deterministic():
    enc = _encoder().train()
    x = torch.randn(16, M)
    mu1, _ = enc.encode(x, omega=0.5)
    mu2, _ = enc.encode(x, omega=0.5)
    assert torch.allclose(mu1, mu2), "mu must not depend on the noise"


def test_z_equals_mu_without_noise():
    enc = _encoder().train()
    x = torch.randn(16, M)
    # omega == 0 -> no noise even in training
    mu, z = enc.encode(x, omega=0.0)
    assert torch.equal(mu, z)
    # eval mode -> no noise even with omega > 0
    enc.eval()
    mu, z = enc.encode(x, omega=0.9)
    assert torch.equal(mu, z)


def test_z_is_noisy_but_centred_on_mu_in_training():
    enc = _encoder().train()
    x = torch.randn(8, M)
    mu, z = enc.encode(x, omega=0.7)
    assert not torch.equal(mu, z), "training + omega>0 must inject noise"
    # averaging many draws should concentrate around mu (unbiased noise)
    draws = torch.stack([enc.encode(x, omega=0.7)[1] for _ in range(2000)])
    assert torch.allclose(draws.mean(dim=0), mu, atol=0.1)


def test_decoder_reconstructs_covariate_shape():
    dec = Decoder(k_latent=K, hidden=(16,), m=M)
    z = torch.randn(10, K)
    x_hat = dec(z)
    assert x_hat.shape == (10, M)


def test_gradients_flow_through_encoder_and_decoder():
    enc, dec = _encoder().train(), Decoder(k_latent=K, hidden=(16,), m=M)
    x = torch.randn(16, M)
    _, z = enc.encode(x, omega=0.3)
    loss = torch.nn.functional.mse_loss(dec(z), x)
    loss.backward()
    enc_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    dec_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.parameters())
    assert enc_grad and dec_grad


def test_batchnorm_variant_builds_and_runs():
    enc = Encoder(m=M, hidden=(16,), k_latent=K, batchnorm=True).train()
    mu, z = enc.encode(torch.randn(8, M), omega=0.0)
    assert mu.shape == (8, K)