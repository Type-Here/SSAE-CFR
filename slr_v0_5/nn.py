"""Trainable building blocks: MLP factory, encoder, noise injector, outcome heads.

Adapted from the v0.3/v0.4 core. The decoder is gone - v0.5 has no reconstruction
path - and the encoder's output width is the latent width `d_latent`, which the
runner sets to the embedding dimension of the prior artifact.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import torch
from torch import Tensor, nn

_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}

_EPS_SAMPLERS = {
    "gaussian": torch.randn_like,
    "laplace": lambda mu: torch.distributions.laplace.Laplace(
        torch.zeros((), device=mu.device, dtype=mu.dtype),
        torch.ones((), device=mu.device, dtype=mu.dtype),
    ).sample(mu.shape),
}


def _activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}; expected one of {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]()


def build_mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    activation: str = "relu",
    batchnorm: bool = False,
) -> nn.Sequential:
    """(Linear -> [BatchNorm] -> activation) per hidden width, then a linear output layer."""
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers.append(nn.Linear(prev, width))
        if batchnorm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(_activation(activation))
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Deterministic MLP encoder R^m -> R^{d_latent}. Noise is applied outside."""

    def __init__(
        self,
        m: int,
        hidden: Sequence[int],
        d_latent: int,
        activation: str = "relu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.m = m
        self.d_latent = d_latent
        self.net = build_mlp(m, hidden, d_latent, activation, batchnorm)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class NoiseInjector(nn.Module):
    """u_emp = mu_emp + omega * s * eps, applied to the empirical branch only.

    omega is not constructor state: it is recomputed per batch from the SMD modulator
    and passed to forward. `s` is 1 for "absolute" and ||mu||_detached / sqrt(d) for
    "relative". Returns mu unchanged outside training or at omega = 0.
    """

    def __init__(self, d_latent: int, noise_type: str = "gaussian", noise_scale: str = "absolute") -> None:
        super().__init__()
        if noise_type not in _EPS_SAMPLERS:
            raise ValueError(f"unknown noise_type {noise_type!r}")
        if noise_scale not in ("absolute", "relative"):
            raise ValueError(f"unknown noise_scale {noise_scale!r}")
        self.d_latent = d_latent
        self.noise_type = noise_type
        self.noise_scale = noise_scale

    def forward(self, mu: Tensor, omega: float) -> Tensor:
        if not self.training or omega == 0.0:
            return mu
        eps = _EPS_SAMPLERS[self.noise_type](mu)
        scale: Union[Tensor, float] = 1.0
        if self.noise_scale == "relative":
            # detached: the amplitude is a measurement of the code, not a parameter of it
            scale = mu.detach().norm(dim=-1, keepdim=True) / math.sqrt(self.d_latent)
        return mu + omega * scale * eps


class OutcomeHeads(nn.Module):
    """Two independent TARNet-style heads mapping u_out to the potential outcomes.

    No output activation: a head emits a raw scalar (a logit for binary outcomes),
    which the factual loss interprets.
    """

    def __init__(
        self,
        d_latent: int,
        hidden: Sequence[int],
        activation: str = "relu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.h0 = build_mlp(d_latent, hidden, 1, activation, batchnorm)
        self.h1 = build_mlp(d_latent, hidden, 1, activation, batchnorm)

    def forward(self, u_out: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (y0_hat, y1_hat), each shape (n,)."""
        return self.h0(u_out).squeeze(-1), self.h1(u_out).squeeze(-1)
