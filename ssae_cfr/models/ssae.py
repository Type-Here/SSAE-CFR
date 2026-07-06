"""SSAE: the stochastic denoising sparse autoencoder (encoder + decoder).

    mu = encoder_mlp(x)           # deterministic bottleneck, R^{k_latent}
    z  = mu + omega * eps         # stochastic; eps ~ N(0, I); omega a detached scalar
    x_hat = decoder_mlp(z)        # reconstruction, feeds L_rec = MSE(x_hat, x)

The single defining choice: the encoder exposes BOTH outputs from one pass. `mu` is the
deterministic code (used by the alignment loss, which must not see the exploration
noise); `z` is the stochastic code (used by the gating, heads and the MMD, so the
injected overlap actually reaches the balancing term). This is not a VAE - there is no
learned variance and no KL. The noise scale `omega` comes from outside the graph (the
SMD modulator), so the encoder cannot tune it away.

Noise is only added while training and when omega != 0; in eval, or with omega == 0,
z == mu, giving deterministic predictions. The same encoder instance is used for all
three PGAG passes (x_prior, x_res, and the plain reconstruction pass) - it is shared
for geometric consistency, so all inputs must share the covariate dimension m.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import Tensor, nn

_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
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
    activation: str = "elu",
    batchnorm: bool = False,
) -> nn.Sequential:
    """A plain MLP: (Linear -> [BatchNorm] -> activation) per hidden width, then a final
    linear map to `out_dim` with no activation (the code/output layer stays linear).

    Shared by the encoder, decoder, heads and gating so their construction is uniform.
    """
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
    """MLP encoder R^m -> R^{k_latent} producing the (mu, z) pair.

    `encode(x, omega)` returns `(mu, z)`; `forward` is an alias. `mu` is always the
    deterministic bottleneck. `z = mu + omega * eps` with fresh eps ~ N(0, I) only while
    training and when `omega != 0`; otherwise `z is mu`.
    """

    def __init__(
        self,
        m: int,
        hidden: Sequence[int],
        k_latent: int,
        activation: str = "elu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.m = m
        self.k_latent = k_latent
        self.net = build_mlp(m, hidden, k_latent, activation, batchnorm)

    def encode(self, x: Tensor, omega: float = 0.0) -> Tuple[Tensor, Tensor]:
        mu = self.net(x)
        if self.training and omega != 0.0:
            eps = torch.randn_like(mu)
            z = mu + omega * eps
        else:
            z = mu
        return mu, z

    def forward(self, x: Tensor, omega: float = 0.0) -> Tuple[Tensor, Tensor]:
        return self.encode(x, omega)


class Decoder(nn.Module):
    """MLP decoder R^{k_latent} -> R^m mapping a code back to covariate space.

    Feeds the reconstruction loss L_rec = MSE(x_hat, x). The reconstruction is what
    keeps the code informative in the absence of a KL term, and gives the L1 sparsity
    something meaningful to act on.
    """

    def __init__(
        self,
        k_latent: int,
        hidden: Sequence[int],
        m: int,
        activation: str = "elu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.net = build_mlp(k_latent, hidden, m, activation, batchnorm)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)