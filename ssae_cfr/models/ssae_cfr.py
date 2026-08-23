"""The full SSAE-CFR model (v1, Variant A).

One forward pass runs three encodings through the single shared encoder:

    x        --> encode --> mu_enc, z_enc --> decode --> x_hat      (L_rec, L_sparse)
    P_U x    --> encode --> z_prior                                 (gating)
    (I-P_U)x --> encode --> mu_res, z_res                           (L_align, gating)

then gates the prior/residual codes into z_mod, and reads the two outcome heads off it:

    z_mod --> h0, h1 --> y0_hat, y1_hat --> yf_hat                  (L_fact)
    z_mod | t         --> MMD                                       (L_mmd)

`forward` returns every raw piece (predictions, reconstruction, codes, gate, diagnostics)
without collapsing anything into a loss, so the caller stays in control. `loss_terms`
turns those pieces into the five scalar terms `total_loss` expects. The plain
reconstruction pass encodes the *whole* x (not a decomposed half): it is what keeps the
code informative and gives the L1 sparsity something to compress, given there is no KL.

Outcome scale: with a binary outcome the factual loss is BCE-with-logits, so the heads
emit logits and the sigmoid lives in the loss, not in the model. Anything that reads a
treatment effect off the heads therefore has to squash first - a difference of logits is
a log odds ratio, not the risk difference that a treatment effect on a binary outcome
means. `to_outcome_scale` is the single place that conversion is defined; the model is
told its `outcome_type` at construction so `predict_tau` cannot silently return the
wrong scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..losses import align_loss, factual_loss, mmd_rbf
from .heads import OutcomeHeads
from .pgag import PGAG
from .ssae import Decoder, Encoder

if TYPE_CHECKING:
    from ..config import TrainConfig


def to_outcome_scale(y: Union[Tensor, np.ndarray], outcome_type: str) -> Union[Tensor, np.ndarray]:
    """Map a head output to the scale the outcome actually lives on.

    Continuous outcomes pass through. Binary heads emit logits (BCE-with-logits), so they
    are squashed to probabilities - only then is `y1 - y0` a risk difference, i.e. the
    treatment effect. Accepts torch or numpy so the model and the evaluation harness
    share one definition instead of drifting apart.
    """
    if outcome_type != "binary":
        return y
    if isinstance(y, Tensor):
        return torch.sigmoid(y)
    return 1.0 / (1.0 + np.exp(-np.asarray(y, dtype=np.float64)))


class SSAECFR(nn.Module):
    """Encoder + decoder + PGAG (shared encoder) + two heads, over a fixed P_U."""

    def __init__(
        self,
        m: int,
        P_U: Tensor,
        cfg: "TrainConfig",
        outcome_type: str = "continuous",
    ) -> None:
        super().__init__()
        if outcome_type not in ("continuous", "binary"):
            raise ValueError(
                f"outcome_type must be 'continuous' or 'binary'; got {outcome_type!r}"
            )
        self.m = m
        self.k_latent = cfg.k_latent
        self.l1_target = cfg.l1_target
        self.outcome_type = outcome_type

        self.encoder = Encoder(m, cfg.encoder_hidden, cfg.k_latent, cfg.activation, cfg.batchnorm)
        self.decoder = Decoder(cfg.k_latent, cfg.decoder_hidden, m, cfg.activation, cfg.batchnorm)
        self.pgag = PGAG(self.encoder, P_U, cfg.gating_hidden, cfg.activation, cfg.batchnorm)
        self.heads = OutcomeHeads(cfg.k_latent, cfg.head_hidden, cfg.activation, cfg.batchnorm)

    def forward(self, x: Tensor, t: Tensor, omega: float = 0.0) -> Dict[str, Tensor]:
        # reconstruction pass over the whole x (shared encoder)
        mu_enc, z_enc = self.encoder.encode(x, omega)
        x_hat = self.decoder(z_enc)

        # prior/residual decomposition + gating
        pg = self.pgag(x, omega)
        y0_hat, y1_hat = self.heads(pg.z_mod)
        tf = t.to(y1_hat.dtype)
        yf_hat = tf * y1_hat + (1.0 - tf) * y0_hat

        return {
            "y0_hat": y0_hat,
            "y1_hat": y1_hat,
            "yf_hat": yf_hat,
            "x_hat": x_hat,
            "mu_enc": mu_enc,
            "z_enc": z_enc,
            "z_mod": pg.z_mod,
            "lam": pg.lam,
            "mu_res": pg.mu_res,
            "z_prior": pg.z_prior,
            "z_res": pg.z_res,
        }

    def loss_terms(
        self,
        out: Dict[str, Tensor],
        x: Tensor,
        t: Tensor,
        yf: Tensor,
        outcome_type: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        """Assemble the five scalar loss terms from a forward output.

        `outcome_type` defaults to the model's own; pass it only to override.
        """
        outcome_type = outcome_type or self.outcome_type
        code = out["z_enc"] if self.l1_target == "z" else out["mu_enc"]
        return {
            "L_fact": factual_loss(out["y0_hat"], out["y1_hat"], t, yf, outcome_type),
            "L_mmd": mmd_rbf(out["z_mod"], t),
            "L_sparse": code.abs().sum(dim=-1).mean(),
            "L_rec": F.mse_loss(out["x_hat"], x),
            "L_align": align_loss(out["mu_res"]),
        }

    @torch.no_grad()
    def potential_outcomes(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Deterministic (y0_hat, y1_hat) on the outcome's own scale (eval, no noise).

        For a binary outcome these are probabilities, not the raw head logits.
        """
        was_training = self.training
        self.eval()
        out = self.forward(x, torch.zeros(x.shape[0], device=x.device), omega=0.0)
        if was_training:
            self.train()
        return (
            to_outcome_scale(out["y0_hat"], self.outcome_type),
            to_outcome_scale(out["y1_hat"], self.outcome_type),
        )

    @torch.no_grad()
    def predict_tau(self, x: Tensor) -> Tensor:
        """Deterministic CATE estimate tau_hat(x) = y1_hat - y0_hat (eval, no noise).

        On the outcome's own scale: a risk difference for a binary outcome, since a
        difference of the raw head logits would be a log odds ratio instead.
        """
        y0, y1 = self.potential_outcomes(x)
        return y1 - y0