"""Tests for the SSAE encoder and decoder.

The contract we lock down: the encoder returns a (mu, z) pair of the right width; mu is
deterministic; z equals mu when omega is 0 or in eval, and becomes noisy (but centred
on mu) only while training with omega != 0; the decoder maps a code back to R^m; and
gradients flow through both.

Plus the noise-scale switch. Under "absolute" the noise amplitude ignores the size of the
code, which is the escape hatch the encoder was measured to use: it cannot lower omega,
so it inflates mu instead and the relative noise decays as training proceeds. Under
"relative" the amplitude tracks ||mu||, so the noise-to-signal ratio is omega whatever
the scale - and the scale factor is detached, so the ratio is not gameable from the other
side either.
"""

from __future__ import annotations

import pytest
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


# -- noise scale -------------------------------------------------------------

def _noise_sd(enc: Encoder, x: torch.Tensor, omega: float, draws: int = 400) -> float:
    """Empirical sd of (z - mu), pooled over draws and coordinates."""
    mu, _ = enc.encode(x, omega=0.0)
    noise = torch.stack([enc.encode(x, omega=omega)[1] - mu for _ in range(draws)])
    return float(noise.detach().std())


def test_absolute_noise_ignores_the_size_of_the_code():
    """omega is an absolute amplitude, so inflating mu dilutes the noise. That is P3."""
    torch.manual_seed(0)
    enc = Encoder(m=M, hidden=(16,), k_latent=K, noise_scale="absolute").train()
    x = torch.randn(8, M)
    small = _noise_sd(enc, x, omega=0.5)
    with torch.no_grad():  # scale the code up tenfold, leaving everything else alone
        enc.net[-1].weight *= 10.0
        enc.net[-1].bias *= 10.0
    assert _noise_sd(enc, x, omega=0.5) == pytest.approx(small, rel=0.1)


def test_relative_noise_tracks_the_size_of_the_code():
    torch.manual_seed(0)
    enc = Encoder(m=M, hidden=(16,), k_latent=K, noise_scale="relative").train()
    x = torch.randn(8, M)
    small = _noise_sd(enc, x, omega=0.5)
    with torch.no_grad():
        enc.net[-1].weight *= 10.0
        enc.net[-1].bias *= 10.0
    assert _noise_sd(enc, x, omega=0.5) == pytest.approx(10.0 * small, rel=0.1)


def test_relative_noise_to_signal_ratio_is_omega():
    torch.manual_seed(0)
    enc = Encoder(m=M, hidden=(16,), k_latent=K, noise_scale="relative").train()
    x = torch.randn(64, M)
    mu, _ = enc.encode(x, omega=0.0)
    mu = mu.detach()
    omega = 0.4
    draws = torch.stack([enc.encode(x, omega=omega)[1].detach() - mu for _ in range(400)])
    # per unit: E||noise||^2 = omega^2 ||mu||^2, so the ratio of norms is omega
    ratio = draws.norm(dim=-1).mean(dim=0) / mu.norm(dim=-1)
    # the mean chi-distributed norm sits slightly below sqrt(k) * sd, hence the tolerance
    assert float(ratio.mean()) == pytest.approx(omega, rel=0.15)


def test_relative_noise_scale_is_detached():
    """The amplitude measures the code; no gradient may flow back through it."""
    torch.manual_seed(0)
    enc = Encoder(m=M, hidden=(16,), k_latent=K, noise_scale="relative").train()
    x = torch.randn(16, M)
    mu, z = enc.encode(x, omega=0.5)
    # d z / d mu is the identity if the scale is detached, so the noise contributes
    # nothing to the gradient of (z - mu).sum() with respect to the encoder weights
    (z - mu.detach()).sum().backward(retain_graph=True)
    grad_z = [p.grad.clone() for p in enc.parameters()]
    enc.zero_grad()
    mu.sum().backward()
    assert all(torch.allclose(a, p.grad, atol=1e-5) for a, p in zip(grad_z, enc.parameters()))


def test_rejects_unknown_noise_scale():
    with pytest.raises(ValueError):
        Encoder(m=M, hidden=(16,), k_latent=K, noise_scale="proportional")