"""PGAG: Prior-Guided Admission Gating in feature space (Variant A).

The prior enters here. We split each covariate vector, in feature space R^m, into the
part the semantic prior explains and the part it does not, then decide how much of the
second to admit:

    x_prior = P_U x                  # the clinically nameable component
    x_res   = x - x_prior            # = (I - P_U) x, exactly, so x_prior + x_res == x
    b       = sigmoid(gate(x))       # the admission gate, in [0, 1]^m
    x_mod   = x_prior + b * x_res    # ADDITIVE
    mu, z   = Enc(x_mod)             # one encoder pass

`b_j` answers, per covariate and per patient: how much of covariate j's undocumented part
do I need for this patient? Its mean and spread are the headline diagnostic, and unlike
the gate this replaces, they are interpretable on their own.

Why additive, in covariate space
--------------------------------
The previous gate mixed two latent codes convexly,
`z_mod = lam * Enc(x_prior) + (1 - lam) * Enc(x_res)`, and that has two measured faults.

A convex combination of two scalars lies between them, but the full-information code is
their *sum* - and the encoder is near-linear in practice (||Enc(x_p) + Enc(x_r) - Enc(x)||
/ ||Enc(x)|| = 0.053), so the sum is not a theoretical object. Measured: 47.8 percent of
coordinates had the full-information code outside the reachable interval, and ||z_mod||
ran at 58-65 percent of ||Enc(x)||. "Use all of x" - the model without a prior -
corresponded to no value of lam at all; lam = 0 gave only the residual, which is the
prior inverted, not the prior ignored. The additive form contains the sum: b = 1 returns
x exactly, b = 0 returns P_U x.

And it acts where the decomposition means something. In R^m the split is exact and every
coordinate is a covariate with a name; after the encoder that identity is gone and the
coordinates are anonymous. Mixing in latent space mixes where the result can be neither
controlled nor read.

Three consequences fall out. There is no separate residual branch to encode, so nothing
can quietly decay into pure noise (previously the residual code was 95 percent injected
noise, and the gate was arbitrating between a signal and that noise). The gate reads `x`,
which is deterministic, so the same patient gets the same decision at every step. And a
single code `z` feeds the decoder, the heads and the MMD, instead of two representations
that met only through shared encoder weights.

`P_U` is fixed and analytic (Variant A), stored as a non-trainable buffer.

The model nests its own ablation
--------------------------------
`b_mode` fixes the gate instead of learning it: "one" is the model without a prior (it
sees all of x), "zero" is the prior-only model. So "with prior vs without" is a flag on
one architecture, compared on the same weights and the same realizations, rather than two
networks whose means get compared - which matters on a benchmark as heavy-tailed as IHDP.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn

from ..config.hparams import B_MODES
from .ssae import Encoder, build_mlp


class PGAGOutput(NamedTuple):
    """Everything the full model and its diagnostics need from one PGAG pass."""

    z: Tensor          # the single stochastic code: decoder, heads and MMD all read it
    mu: Tensor         # its deterministic counterpart
    b: Tensor          # admission gate in [0, 1]^m, per covariate and per unit
    x_mod: Tensor      # x_prior + b * x_res, the encoder's actual input
    x_prior: Tensor    # P_U x
    x_res: Tensor      # (I - P_U) x


class PGAG(nn.Module):
    """Feature-space P_U decomposition + admission gate + one shared-encoder pass."""

    def __init__(
        self,
        encoder: Encoder,
        P_U: Tensor,
        gating_hidden: Sequence[int],
        activation: str = "elu",
        batchnorm: bool = False,
        b_mode: str = "learned",
    ) -> None:
        super().__init__()
        if b_mode not in B_MODES:
            raise ValueError(f"b_mode must be one of {B_MODES}; got {b_mode!r}")
        self.encoder = encoder
        self.b_mode = b_mode

        P = torch.as_tensor(P_U, dtype=torch.float32)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(f"P_U must be a square matrix; got shape {tuple(P.shape)}")
        if P.shape[0] != encoder.m:
            raise ValueError(f"P_U is {P.shape[0]}x{P.shape[0]} but encoder expects m={encoder.m}")
        # non-trainable: the prior is fixed, it moves with .to(device) but has no grad
        self.register_buffer("P_U", P)

        # the gate reads the covariates and emits one admission weight per covariate
        self.gate = build_mlp(encoder.m, gating_hidden, encoder.m, activation, batchnorm)

    def admission(self, x: Tensor) -> Tensor:
        """The gate `b` in [0, 1]^m for each unit; constant under a fixed `b_mode`."""
        if self.b_mode == "one":
            return torch.ones_like(x)
        if self.b_mode == "zero":
            return torch.zeros_like(x)
        return torch.sigmoid(self.gate(x))

    def forward(self, x: Tensor, omega: float = 0.0) -> PGAGOutput:
        P = self.P_U.to(x.dtype)
        x_prior = x @ P.T            # P_U symmetric, but .T keeps it correct either way
        x_res = x - x_prior          # exact (I - P_U) x, so x_prior + x_res == x

        b = self.admission(x)
        x_mod = x_prior + b * x_res
        mu, z = self.encoder.encode(x_mod, omega)
        return PGAGOutput(z=z, mu=mu, b=b, x_mod=x_mod, x_prior=x_prior, x_res=x_res)