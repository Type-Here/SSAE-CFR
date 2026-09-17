"""The empirical branch of the SSAE pipeline: Encoder -> mu -> NOISE -> u -> Decoder.

Only the empirical part - no heads, no outcome scaling, no losses. Composition with the
U/W adapters and the heads happens in the model that owns this branch.
"""

from __future__ import annotations

from typing import Tuple

from torch import Tensor, nn

from .modules import Decoder, Encoder
from .noise import NoiseInjector
from ..hparams import DefaultConfig


class Empirical(nn.Module):
    """Unfiltered empirical host: x -> mu -> u -> x_hat."""

    def __init__(self, cfg: DefaultConfig) -> None:
        super().__init__()
        self.in_channels = cfg.in_channels
        self.is_noise_active = cfg.is_noise_active
        self.encoder = Encoder(cfg.in_channels, cfg.encoder_hidden, cfg.d_u, cfg.activation, cfg.batchnorm)
        self.decoder = Decoder(cfg.d_u, cfg.decoder_hidden, cfg.in_channels, cfg.activation, cfg.batchnorm)
        self.noise_injector = NoiseInjector(cfg.d_u, cfg.noise_dist, cfg.noise_scale)

    def forward(self, x: Tensor, omega: float = 0.0) -> Tuple[Tensor, Tensor, Tensor]:
        """Return (mu, u, x_hat)."""
        mu = self.encoder(x)
        u = self.noise_injector(mu, omega) if self.is_noise_active else mu
        x_hat = self.decoder(u)
        return mu, u, x_hat
