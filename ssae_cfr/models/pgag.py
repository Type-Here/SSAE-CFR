"""PGAG: Prior-Guided Adaptive Gating in feature space (Variant A).

The prior enters here. We split each covariate vector, in feature space R^m, into the
part the semantic prior explains and the part it does not:

    x_prior = P_U x            # the clinically explainable component
    x_res   = x - x_prior      # = (I - P_U) x, the residual

Both halves go through the SAME encoder (shared for geometric consistency), then a
gating network decides, per latent dimension, how much to trust each:

    z_prior = Enc(x_prior)          # stochastic branch
    z_res   = Enc(x_res)            # stochastic branch
    mu_res  = Enc(x_res)            # deterministic branch, for L_align only
    lam     = sigmoid(gate(concat(z_prior, z_res)))   # in [0, 1]^{k_latent}
    z_mod   = lam * z_prior + (1 - lam) * z_res

`lam` is a per-dimension gate, not a single scalar, so the model can lean on the prior
along some latent coordinates and on the residual along others. Its mean and spread are
a headline diagnostic: `lam -> 0` everywhere means the model is ignoring the prior.

`P_U` is fixed and analytic (Variant A), stored as a non-trainable buffer. `mu_res` is
surfaced separately because the alignment loss shrinks the deterministic residual
encoding - never the noisy one.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn

from .ssae import Encoder, build_mlp


class PGAGOutput(NamedTuple):
    """Everything the full model and its diagnostics need from one PGAG pass."""

    z_mod: Tensor      # balanced code fed to the heads and the MMD
    lam: Tensor        # per-dimension gate in [0, 1]^{k_latent}
    mu_res: Tensor     # deterministic residual encoding, for L_align
    z_prior: Tensor    # stochastic prior-subspace code (kept for diagnostics)
    z_res: Tensor      # stochastic residual code (kept for diagnostics)


class PGAG(nn.Module):
    """Feature-space P_U decomposition + shared-encoder passes + adaptive gating."""

    def __init__(
        self,
        encoder: Encoder,
        P_U: Tensor,
        gating_hidden: Sequence[int],
        activation: str = "elu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        k = encoder.k_latent

        P = torch.as_tensor(P_U, dtype=torch.float32)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(f"P_U must be a square matrix; got shape {tuple(P.shape)}")
        if P.shape[0] != encoder.m:
            raise ValueError(f"P_U is {P.shape[0]}x{P.shape[0]} but encoder expects m={encoder.m}")
        # non-trainable: the prior is fixed, it moves with .to(device) but has no grad
        self.register_buffer("P_U", P)

        # gate reads both codes concatenated (2*k) and emits a per-dimension weight (k)
        self.gate = build_mlp(2 * k, gating_hidden, k, activation, batchnorm)

    def forward(self, x: Tensor, omega: float = 0.0) -> PGAGOutput:
        P = self.P_U.to(x.dtype)
        x_prior = x @ P.T            # P_U symmetric, but .T keeps it correct either way
        x_res = x - x_prior          # exact (I - P_U) x, so x_prior + x_res == x

        _, z_prior = self.encoder.encode(x_prior, omega)
        mu_res, z_res = self.encoder.encode(x_res, omega)

        lam = torch.sigmoid(self.gate(torch.cat([z_prior, z_res], dim=-1)))
        z_mod = lam * z_prior + (1.0 - lam) * z_res
        return PGAGOutput(z_mod=z_mod, lam=lam, mu_res=mu_res, z_prior=z_prior, z_res=z_res)