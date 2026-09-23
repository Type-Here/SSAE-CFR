"""SLRCFRv05: a wide empirical encoder fused with a frozen semantic vector.

    x_std --> encoder --> mu_emp --> noise --> u_emp ---+
                    |                                   +--> u_out = u_emp + eta * p
    x_std, mu_emp --> prior_integration ----> p --------+        |
                                                                 +--> heads -> y0, y1
                                                                 +--> MMD, sparse

There is no decoder and no reconstruction. The latent width is the embedding width
d_q, so the empirical code and the semantic vector live in the same space and fuse by
addition - no gate, no concatenation, no adapter. `eta_prior` is a fixed run
hyperparameter, never trainable; at 0 the prior is exactly disabled.

Noise belongs to the empirical branch only: the prior path reads `mu_emp`, the
deterministic pre-noise representation, so a patient's semantic vector is fixed given
the weights. Which prior mechanism runs - direct fusion or cosine cross-attention - is
a configuration choice behind the `PriorIntegration` interface; nothing else in this
model changes with it.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn

from .config import SLRConfig
from .losses import factual_loss, mmd_rbf, sparse_loss, total_loss
from .nn import Encoder, NoiseInjector, OutcomeHeads
from .outcome_scale import to_outcome_scale
from .prior.base import PriorIntegration

ArrayLike = Union[Tensor, np.ndarray]
_EPS = 1e-8


def _mean_norm(a: ArrayLike) -> float:
    """Mean L2 norm across the last axis, for a torch Tensor or a numpy array."""
    if isinstance(a, Tensor):
        a = a.detach().cpu().numpy()
    return float(np.linalg.norm(np.asarray(a), axis=-1).mean())


class SLRCFRv05(nn.Module):
    """Wide-latent counterfactual regressor with direct frozen-embedding fusion."""

    def __init__(
        self,
        cfg: SLRConfig,
        prior_integration: Optional[PriorIntegration] = None,
        outcome_type: str = "continuous",
        y_loc: float = 0.0,
        y_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if outcome_type not in ("continuous", "binary"):
            raise ValueError(f"outcome_type must be 'continuous' or 'binary'; got {outcome_type!r}")
        if y_scale <= 0.0:
            raise ValueError(f"y_scale must be positive; got {y_scale}")
        if outcome_type == "binary" and (y_loc != 0.0 or y_scale != 1.0):
            raise ValueError("a binary outcome is never standardized; leave y_loc/y_scale at 0/1")
        if cfg.d_latent is None:
            raise ValueError(
                "cfg.d_latent is None: the latent width is the embedding dimension of the "
                "prior artifact and must be resolved before the model is built"
            )
        if cfg.embedding_variant == "none" and prior_integration is not None:
            raise ValueError("embedding_variant='none' must be built without a prior integration")
        if cfg.embedding_variant != "none" and prior_integration is None:
            raise ValueError(
                f"embedding_variant={cfg.embedding_variant!r} needs a prior integration; none was given"
            )
        if prior_integration is not None and prior_integration.d_q != cfg.d_latent:
            raise ValueError(
                f"prior emits d_q={prior_integration.d_q} but cfg.d_latent={cfg.d_latent}"
            )

        self.m = cfg.in_channels
        self.d_latent = int(cfg.d_latent)
        self.eta_prior = float(cfg.eta_prior)
        self.noise_enabled = cfg.noise_enabled
        self.outcome_type = outcome_type
        # Buffers, not plain floats: they move with the model across devices and
        # survive a state_dict round trip.
        self.register_buffer("y_loc", torch.tensor(float(y_loc)))
        self.register_buffer("y_scale", torch.tensor(float(y_scale)))

        self.encoder = Encoder(
            cfg.in_channels, cfg.encoder_hidden, self.d_latent, cfg.activation, cfg.batchnorm
        )
        self.noise = NoiseInjector(self.d_latent, cfg.noise_dist, cfg.noise_scale)
        self.prior_integration = prior_integration
        self.heads = OutcomeHeads(self.d_latent, cfg.head_hidden, cfg.activation, cfg.batchnorm)

    def forward(self, x: Tensor, t: Tensor, omega: float = 0.0) -> Dict[str, Union[Tensor, float]]:
        mu_emp = self.encoder(x)
        u_emp = self.noise(mu_emp, omega) if self.noise_enabled else mu_emp

        if self.prior_integration is None:
            # No prior object at all: u_out IS u_emp, not u_emp plus a zero tensor.
            p, prior_diag = None, {}
            u_out = u_emp
        else:
            p, prior_diag = self.prior_integration(x, mu_emp)
            u_out = u_emp + self.eta_prior * p

        y0_hat, y1_hat = self.heads(u_out)
        out: Dict[str, Union[Tensor, float]] = {
            "mu_emp": mu_emp,
            "u_emp": u_emp,
            "u_prior": p,
            "u_out": u_out,
            "y0_hat": y0_hat,
            "y1_hat": y1_hat,
        }
        out.update(prior_diag)
        return out

    def loss_terms(
        self,
        out: Dict[str, Tensor],
        t: Tensor,
        yf: Tensor,
        outcome_type: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        """Factual fit, balance and sparsity - all three read u_out."""
        outcome_type = outcome_type or self.outcome_type
        # The heads live on the standardized outcome, so the target is standardized to
        # meet them; to_outcome_scale undoes this on the way out.
        if outcome_type != "binary":
            yf = (yf - self.y_loc) / self.y_scale

        return {
            "L_fact": factual_loss(out["y0_hat"], out["y1_hat"], t, yf, outcome_type),
            "L_mmd": mmd_rbf(out["u_out"], t),
            "L_sparse": sparse_loss(out["u_out"]),
        }

    def total_loss(self, terms: Mapping[str, Tensor], cfg: SLRConfig) -> Tuple[Tensor, dict]:
        return total_loss(terms, cfg)

    def diagnostics(self, out: Dict[str, object], omega: Optional[float] = None) -> dict:
        """Representation norms, the prior-to-empirical ratio and a sparsity read.

        `prior_to_empirical_norm_ratio` is what says whether the fused semantic
        vector is a nudge or the whole representation.
        """
        u_emp_norm = _mean_norm(out["u_emp"])
        # Scalar entries the prior module reported (p_norm, and the attention
        # statistics when that mechanism is in use) are carried through as they are,
        # so a new integration module needs no change here.
        scores = {
            key: value for key, value in out.items() if isinstance(value, float)
        }
        scores.update({
            "mu_emp_norm": _mean_norm(out["mu_emp"]),
            "u_emp_norm": u_emp_norm,
            "u_out_norm": _mean_norm(out["u_out"]),
            "eta_prior": self.eta_prior,
        })

        u_out = out["u_out"]
        if isinstance(u_out, Tensor):
            u_out = u_out.detach().cpu().numpy()
        u_out = np.asarray(u_out)
        scores["u_out_l1"] = float(np.abs(u_out).sum(axis=-1).mean())
        # Fraction of latent coordinates carrying at least 1 percent of the unit's
        # mean coordinate magnitude - a scale-free read of how much of the wide
        # latent is in use.
        threshold = 0.01 * np.abs(u_out).mean(axis=-1, keepdims=True)
        scores["u_out_active_fraction"] = float((np.abs(u_out) > threshold).mean())

        if out.get("u_prior") is None:
            scores["p_norm"] = 0.0
            scores["prior_contribution_norm"] = 0.0
            scores["prior_to_empirical_norm_ratio"] = 0.0
        else:
            p_norm = _mean_norm(out["u_prior"])
            scores["p_norm"] = p_norm
            scores["prior_contribution_norm"] = self.eta_prior * p_norm
            scores["prior_to_empirical_norm_ratio"] = self.eta_prior * p_norm / (u_emp_norm + _EPS)
        if omega is not None:
            scores["omega"] = omega
        return scores

    def format_epoch(self, last: dict) -> str:
        """A compact one-line summary of the code norms and the prior's share of them."""
        return (
            f"u_emp {last.get('u_emp_norm', float('nan')):.3f} "
            f"u_out {last.get('u_out_norm', float('nan')):.3f} "
            f"prior/emp {last.get('prior_to_empirical_norm_ratio', float('nan')):.3f}"
        )

    @property
    def outcome_affine(self) -> Tuple[float, float]:
        """(loc, scale) mapping a head output back to the caller's outcome units."""
        return float(self.y_loc), float(self.y_scale)

    @torch.no_grad()
    def potential_outcomes(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Deterministic (y0_hat, y1_hat) on the outcome's own scale (eval, no noise)."""
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

    @torch.no_grad()
    def predict_tau(self, x: Tensor) -> Tensor:
        """Deterministic CATE estimate tau_hat(x) = y1_hat - y0_hat, on the outcome scale."""
        y0, y1 = self.potential_outcomes(x)
        return y1 - y0
