"""SSAE: the stochastic sparse autoencoder (encoder + decoder).

    mu = encoder_mlp(x)           # deterministic bottleneck, R^{k_latent}
    z  = mu + omega * s * eps     # stochastic; eps ~ N(0, I); omega a detached scalar
    x_hat = decoder_mlp(z)        # reconstruction, feeds L_rec = MSE(x_hat, x)

The single defining choice: the encoder exposes BOTH outputs from one pass. `mu` is the
deterministic code; `z` is the stochastic one, which is what feeds the heads and the MMD,
so the injected overlap actually reaches the balancing term. This is not a VAE - there is
no learned variance and no KL. The noise scale `omega` comes from outside the graph (the
SMD modulator), so the encoder cannot tune it away.

Absolute vs relative noise
--------------------------
`omega` outside the graph does stop the encoder from lowering the noise. But with
`s = 1` (`noise_scale="absolute"`) omega sets an *absolute* amplitude, and the encoder
reaches the same end from the other side: it inflates the signal. Measured, with the
noise on, ||z_prior|| grows 2.60 -> 5.93 between epoch 50 and 299, against 1.72 -> 3.06
with the noise off and everything else equal, and the relative noise inside the code
decays 0.72 -> 0.31 over training. The overlap the MMD needs fades exactly as the model
learns.

`noise_scale="relative"` sets `s = ||mu||_detached / sqrt(k_latent)` per unit, so the
expected noise energy is `omega^2 * ||mu||^2` and the noise-to-signal ratio is exactly
omega whatever the scale of mu. Inflating the signal no longer helps - the noise inflates
with it - and because `||mu||` enters detached it is not gameable from the other side
either. Only the unit of measurement changes; omega is still driven by the measured
imbalance on the raw covariates and is still not learnable.

The default is "absolute", which is the behaviour every existing number was produced
under. This is deliberate: the change is a design argument, not yet a measurement, and
flipping it in the same run as another change would confound both.

Noise is only added while training and when omega != 0; in eval, or with omega == 0,
z == mu, giving deterministic predictions.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from ..config.hparams import NOISE_SCALES

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
    deterministic bottleneck. `z = mu + omega * s * eps` with fresh eps ~ N(0, I) only
    while training and when `omega != 0`; otherwise `z is mu`. `s` is 1 for
    `noise_scale="absolute"` and `||mu||_detached / sqrt(k_latent)` for `"relative"`.
    """

    def __init__(
        self,
        m: int,
        hidden: Sequence[int],
        k_latent: int,
        activation: str = "elu",
        batchnorm: bool = False,
        noise_scale: str = "absolute",
    ) -> None:
        super().__init__()
        if noise_scale not in NOISE_SCALES:
            raise ValueError(f"noise_scale must be one of {NOISE_SCALES}; got {noise_scale!r}")
        self.m = m
        self.k_latent = k_latent
        self.noise_scale = noise_scale
        self.net = build_mlp(m, hidden, k_latent, activation, batchnorm)

    def encode(self, x: Tensor, omega: float = 0.0) -> Tuple[Tensor, Tensor]:
        mu = self.net(x)
        if self.training and omega != 0.0:
            eps = torch.randn_like(mu)
            scale: Union[float, Tensor] = 1.0
            if self.noise_scale == "relative":
                # detached: the amplitude is a measurement of the code, not a parameter
                # of it, so no gradient may flow back through the noise scale
                scale = mu.detach().norm(dim=-1, keepdim=True) / math.sqrt(self.k_latent)
            z = mu + omega * scale * eps
        else:
            z = mu
        return mu, z

    def forward(self, x: Tensor, omega: float = 0.0) -> Tuple[Tensor, Tensor]:
        return self.encode(x, omega)


class Decoder(nn.Module):
    """MLP decoder R^{k_latent} -> R^m mapping a code back to covariate space.

    Feeds the reconstruction loss L_rec = MSE(x_hat, x). Its target is the whole x while
    the code is built from the admitted part of x, so besides keeping the code
    informative in the absence of a KL term it is the force that pushes the admission
    gate open - the counterweight to L_pref.
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