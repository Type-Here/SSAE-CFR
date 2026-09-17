"""Shared building blocks: MLP factory, activations, and the encoder/decoder pair.

    mu = encoder(x)      # deterministic bottleneck, R^{d_u}
    ... noise applied externally (see core/noise.py) ...
    x_hat = decoder(u)   # reconstruction, feeds L_rec = MSE(x_hat, x)

The encoder is deterministic; noise injection is a separate module applied between the
encoder and the decoder, so this file has no notion of omega or stochasticity.
"""

from __future__ import annotations

from typing import Sequence

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
    activation: str = "relu",
    batchnorm: bool = False,
) -> nn.Sequential:
    """(Linear -> [BatchNorm] -> activation) per hidden width, then a linear output layer.

    Shared by the encoder, decoder, heads and adapters so their construction is uniform.
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
    """Deterministic MLP encoder R^m -> R^{d_u}. Noise is applied outside this module."""

    def __init__(
        self,
        m: int,
        hidden: Sequence[int],
        d_u: int,
        activation: str = "relu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.m = m
        self.d_u = d_u
        self.net = build_mlp(m, hidden, d_u, activation, batchnorm)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """MLP decoder R^{d_u} -> R^m mapping a code back to covariate space.

    Feeds L_rec = MSE(x_hat, x). Reads the empirical code only, so the semantic adapters
    are never asked to carry information they only exist to correct.
    """

    def __init__(
        self,
        d_u: int,
        hidden: Sequence[int],
        m: int,
        activation: str = "relu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.net = build_mlp(d_u, hidden, m, activation, batchnorm)

    def forward(self, u: Tensor) -> Tensor:
        return self.net(u)