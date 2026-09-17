"""Noise injector: u = mu + omega * s * eps, applied between the encoder and the decoder.

The noise exists to create overlap: with little common support between the arms the
deterministic codes are cleanly separable, so the MMD kernel sees two far-apart point
clouds with a vanishing cross term and an uninformative gradient. Smearing each cloud
restores a gradient that points somewhere.

omega is not constructor state - it is recomputed per batch from the SMD modulator (see
utils/smd.py) and passed to forward. `s` is 1 for "absolute" and
`||mu||_detached / sqrt(d_u)` for "relative" (detached: the amplitude is a measurement of
the code, not a parameter of it). "absolute" is the default because it is what every
existing number was produced under. Noise is added only while training and omega != 0;
otherwise forward returns mu itself.
"""

from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor, nn

from ..hparams import NOISE_SCALES, NOISE_TYPES

_EPS_SAMPLERS = {
    "gaussian": torch.randn_like,
    "laplace": lambda mu: torch.distributions.laplace.Laplace(
        torch.zeros((), device=mu.device, dtype=mu.dtype),
        torch.ones((), device=mu.device, dtype=mu.dtype),
    ).sample(mu.shape),
}


class NoiseInjector(nn.Module):
    """Injects omega-scaled noise into a code: u = mu + omega * s * eps."""

    def __init__(
        self,
        d_u: int,
        noise_type: str = "gaussian",
        noise_scale: str = "absolute",
    ) -> None:
        super().__init__()
        if noise_type not in NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {NOISE_TYPES}; got {noise_type!r}")
        if noise_scale not in NOISE_SCALES:
            raise ValueError(f"noise_scale must be one of {NOISE_SCALES}; got {noise_scale!r}")
        self.d_u = d_u
        self.noise_type = noise_type
        self.noise_scale = noise_scale

    def forward(self, mu: Tensor, omega: float) -> Tensor:
        if not self.training or omega == 0.0:
            return mu
        eps = _EPS_SAMPLERS[self.noise_type](mu)
        scale: Union[Tensor, float] = 1.0
        if self.noise_scale == "relative":
            # detached: the amplitude is a measurement of the code, not a parameter of
            # it, so no gradient may flow back through the noise scale
            scale = mu.detach().norm(dim=-1, keepdim=True) / math.sqrt(self.d_u)
        return mu + omega * scale * eps
