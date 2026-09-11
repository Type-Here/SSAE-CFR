"""The full SSAE-CFR model (v1, Variant A).

One forward pass, one encoding, one code:

    x --> P_U split --> x_mod = P_U x + b(x) * (I - P_U) x          (L_pref on b)
    x_mod --> encode --> mu, z                                      (L_sparse)
    z     --> decode --> x_hat, reconstructing the WHOLE x          (L_rec)
    z     --> h0, h1 --> y0_hat, y1_hat --> yf_hat                  (L_fact)
    z | t             --> MMD                                       (L_mmd)

`forward` returns every raw piece (predictions, reconstruction, code, gate, diagnostics)
without collapsing anything into a loss, so the caller stays in control. `loss_terms`
turns those pieces into the five scalar terms `total_loss` expects.

The reconstruction target is `x`, not `x_mod`. That is what makes the design work: the
decoder has to rebuild the whole covariate vector from a code built out of an admitted
fraction of it, so `L_rec` pushes `b` toward 1 while `L_pref` pushes it toward 0, and the
balance between `lambda_rec` and `gamma_pref` is what decides `b`. The reconstruction is
no longer a regularizer that keeps the code informative in the absence of a KL - it is
the force that opens the gate, which is why its weight is a first-class tuning target.

A single code also closes a split the previous version carried: the reconstruction used
to run over its own encoding of the whole x while the heads and the MMD read a different,
gated code, so the autoencoder half of the model and the counterfactual-regression half
shared nothing but encoder weights.

Outcome scale: with a binary outcome the factual loss is BCE-with-logits, so the heads
emit logits and the sigmoid lives in the loss, not in the model. Anything that reads a
treatment effect off the heads therefore has to squash first - a difference of logits is
a log odds ratio, not the risk difference that a treatment effect on a binary outcome
means. `to_outcome_scale` is the single place that conversion is defined; the model is
told its `outcome_type` at construction so `predict_tau` cannot silently return the
wrong scale.

A continuous outcome is additionally *standardized inside the model*: the heads are
trained against `(y - y_loc) / y_scale` and `to_outcome_scale` undoes it. This is not
cosmetic. `L_fact` is the only term measured on the outcome's units, while `L_mmd`,
`L_rec` and `L_align` all live on the standardized covariate scale, so without this the
effective balancing weight is `alpha_mmd / var(y)` and a hyperparameter tuned on one
dataset means something different on the next. On IHDP the outcome scale varies about
fifteenfold *across realizations* of the same benchmark, which made a single alpha_mmd
mean fifteen different things within one experiment. The model carries `y_loc`/`y_scale`
as buffers so predictions always come back on the scale the caller supplied data in, and
nothing downstream of the model has to know this happened.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..losses import factual_loss, mmd_rbf, preference_loss
from .heads import OutcomeHeads
from .pgag import PGAG
from .ssae import Decoder, Encoder

if TYPE_CHECKING:
    from ..config import TrainConfig


def to_outcome_scale(
    y: Union[Tensor, np.ndarray],
    outcome_type: str,
    loc: float = 0.0,
    scale: float = 1.0,
) -> Union[Tensor, np.ndarray]:
    """Map a head output to the scale the outcome actually lives on.

    Binary heads emit logits (BCE-with-logits), so they are squashed to probabilities -
    only then is `y1 - y0` a risk difference, i.e. the treatment effect. A probability is
    already on its own scale, so `loc`/`scale` do not apply and are rejected rather than
    silently ignored. Continuous heads are trained against a standardized outcome, so the
    affine map is undone here. Accepts torch or numpy so the model and the evaluation
    harness share one definition instead of drifting apart.
    """
    if outcome_type == "binary":
        if loc != 0.0 or scale != 1.0:
            raise ValueError("a binary outcome is never standardized; got loc/scale != 0/1")
        if isinstance(y, Tensor):
            return torch.sigmoid(y)
        return 1.0 / (1.0 + np.exp(-np.asarray(y, dtype=np.float64)))
    if loc == 0.0 and scale == 1.0:
        return y
    return y * scale + loc


class SSAECFR(nn.Module):
    """Encoder + decoder + PGAG (shared encoder) + two heads, over a fixed P_U."""

    def __init__(
        self,
        m: int,
        P_U: Tensor,
        cfg: "TrainConfig",
        outcome_type: str = "continuous",
        y_loc: float = 0.0,
        y_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if outcome_type not in ("continuous", "binary"):
            raise ValueError(
                f"outcome_type must be 'continuous' or 'binary'; got {outcome_type!r}"
            )
        if y_scale <= 0.0:
            raise ValueError(f"y_scale must be positive; got {y_scale}")
        if outcome_type == "binary" and (y_loc != 0.0 or y_scale != 1.0):
            raise ValueError("a binary outcome is never standardized; leave y_loc/y_scale at 0/1")
        self.m = m
        self.k_latent = cfg.k_latent
        self.l1_target = cfg.l1_target
        self.outcome_type = outcome_type
        # Buffers, not plain floats: they belong to the model's state, move with it
        # across devices, and survive a state_dict round trip. Predictions read on the
        # wrong outcome scale are the kind of bug that produces plausible numbers.
        self.register_buffer("y_loc", torch.tensor(float(y_loc)))
        self.register_buffer("y_scale", torch.tensor(float(y_scale)))

        self.encoder = Encoder(
            m, cfg.encoder_hidden, cfg.k_latent, cfg.activation, cfg.batchnorm, cfg.noise_scale
        )
        self.decoder = Decoder(cfg.k_latent, cfg.decoder_hidden, m, cfg.activation, cfg.batchnorm)
        self.pgag = PGAG(
            self.encoder, P_U, cfg.gating_hidden, cfg.activation, cfg.batchnorm, cfg.b_mode
        )
        self.heads = OutcomeHeads(cfg.k_latent, cfg.head_hidden, cfg.activation, cfg.batchnorm)

    def forward(self, x: Tensor, t: Tensor, omega: float = 0.0) -> Dict[str, Tensor]:
        # prior/residual decomposition + admission gate + the single encoding
        pg = self.pgag(x, omega)
        # the target is the whole x, not the gated x_mod: that is what makes L_rec push
        # the gate open, and what makes the code carry more than the admitted part
        x_hat = self.decoder(pg.z)

        y0_hat, y1_hat = self.heads(pg.z)
        tf = t.to(y1_hat.dtype)
        yf_hat = tf * y1_hat + (1.0 - tf) * y0_hat

        return {
            "y0_hat": y0_hat,
            "y1_hat": y1_hat,
            "yf_hat": yf_hat,
            "x_hat": x_hat,
            "mu": pg.mu,
            "z": pg.z,
            "b": pg.b,
            "x_mod": pg.x_mod,
            "x_res": pg.x_res,
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
        code = out["z"] if self.l1_target == "z" else out["mu"]
        # The heads live on the standardized outcome, so the target is standardized to
        # meet them; `to_outcome_scale` is the inverse, applied on the way out.
        if outcome_type != "binary":
            yf = (yf - self.y_loc) / self.y_scale
        return {
            "L_fact": factual_loss(out["y0_hat"], out["y1_hat"], t, yf, outcome_type),
            "L_mmd": mmd_rbf(out["z"], t),
            "L_sparse": code.abs().sum(dim=-1).mean(),
            "L_rec": F.mse_loss(out["x_hat"], x),
            "L_pref": preference_loss(out["b"]),
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
        loc, scale = self.outcome_affine
        return (
            to_outcome_scale(out["y0_hat"], self.outcome_type, loc, scale),
            to_outcome_scale(out["y1_hat"], self.outcome_type, loc, scale),
        )

    @property
    def outcome_affine(self) -> Tuple[float, float]:
        """(loc, scale) mapping a head output back to the caller's outcome units."""
        return float(self.y_loc), float(self.y_scale)

    @torch.no_grad()
    def predict_tau(self, x: Tensor) -> Tensor:
        """Deterministic CATE estimate tau_hat(x) = y1_hat - y0_hat (eval, no noise).

        On the outcome's own scale: a risk difference for a binary outcome, since a
        difference of the raw head logits would be a log odds ratio instead.
        """
        y0, y1 = self.potential_outcomes(x)
        return y1 - y0