"""SSAECFRv4: the empirical host guided, at its input, by a fixed prior subspace.

    x_std --> guidance.transform --> x_guided --> empirical(encoder -> mu -> noise -> u -> decoder) --> x_hat
                                                              |
                                                              u --> heads --> y0_hat, y1_hat

`guidance` (a `PriorGuidance`) additively nudges the encoder's input toward one or two
fixed prior subspaces and pulls the encoder's first layer toward the same subspace
through `L_guidance`; it owns no trainable parameter. There is no adapter branch, no
attention, no gate and nothing learned about the prior itself - every arm of this model
(prior off, embedding, graph, both) has an identical parameter count, since the encoder
and heads are built with exactly the arguments the empirical host uses.

`x_std` always reaches the encoder with coefficient 1: `guidance.transform` only adds
to it, never masks or filters it. The reconstruction target is the original `x_std`,
supplied by the caller to `loss_terms`, never `x_guided`. The MMD acts on `u`, the
encoder's code, exactly where it acts in the empirical host. With `prior_mode="none"`
`guidance.transform` returns the input object itself and `guidance_loss` is exactly
zero with no gradient, so this model IS the empirical backbone.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .core import Empirical, OutcomeHeads, to_outcome_scale
from .hparams import DefaultConfig
from .losses import factual_loss, mmd_rbf, sparse_loss
from .losses.total import total_loss as _total_loss
from .prior_modules import PriorGuidance, first_linear_weight

ArrayLike = Union[Tensor, np.ndarray]


def _mean_norm(a: ArrayLike) -> float:
    """Mean L2 norm across the last axis, for a torch Tensor or a numpy array.

    Accepting either lets the diagnostics run unchanged during training (tensors
    still attached to the graph) and from the evaluation harness (numpy, after the
    forward output has been converted wholesale).
    """
    if isinstance(a, Tensor):
        a = a.detach().cpu().numpy()
    return float(np.linalg.norm(np.asarray(a), axis=-1).mean())


class SSAECFRv4(nn.Module):
    """Empirical host with a fixed, additive input guidance and no adapters."""

    def __init__(
        self,
        cfg: DefaultConfig,
        outcome_type: str = "continuous",
        y_loc: float = 0.0,
        y_scale: float = 1.0,
        guidance: Optional[PriorGuidance] = None,
    ) -> None:
        super().__init__()
        if outcome_type not in ("continuous", "binary"):
            raise ValueError(f"outcome_type must be 'continuous' or 'binary'; got {outcome_type!r}")
        if y_scale <= 0.0:
            raise ValueError(f"y_scale must be positive; got {y_scale}")
        if outcome_type == "binary" and (y_loc != 0.0 or y_scale != 1.0):
            raise ValueError("a binary outcome is never standardized; leave y_loc/y_scale at 0/1")

        if cfg.use_u_adapter or cfg.use_w_adapter or cfg.use_expert_adapter:
            raise ValueError(
                "SSAECFRv4 runs on the unmodified empirical backbone: "
                f"use_u_adapter={cfg.use_u_adapter!r} use_w_adapter={cfg.use_w_adapter!r} "
                f"use_expert_adapter={cfg.use_expert_adapter!r} would leave this arm not "
                "parameter-matched to the others"
            )

        # The guidance must say the same thing the config says. A caller that passed a
        # prior and had it silently ignored would report a run that never happened, and
        # one whose prior was silently dropped would report the wrong arm.
        if guidance is None:
            if cfg.prior_mode != "none":
                raise ValueError(
                    f"cfg.prior_mode={cfg.prior_mode!r} requires a guidance object; none was given"
                )
            guidance = PriorGuidance(mode="none")
        elif guidance.mode != cfg.prior_mode:
            raise ValueError(
                f"guidance.mode={guidance.mode!r} does not match "
                f"cfg.prior_mode={cfg.prior_mode!r}"
            )

        if guidance.mode != "none":
            m_guidance = guidance.T.shape[0]
            if m_guidance != cfg.in_channels:
                raise ValueError(
                    f"guidance covers {m_guidance} features but cfg.in_channels={cfg.in_channels}"
                )

        m = cfg.in_channels
        self.m = m
        self.d_u = cfg.d_u
        self.l1_target = cfg.l1_target
        self.outcome_type = outcome_type
        # Buffers, not plain floats: they move with the model across devices and
        # survive a state_dict round trip. See SSAECFRv3 for the full rationale.
        self.register_buffer("y_loc", torch.tensor(float(y_loc)))
        self.register_buffer("y_scale", torch.tensor(float(y_scale)))

        self.guidance = guidance
        self.empirical = Empirical(cfg)
        self.heads = OutcomeHeads(cfg.d_u, cfg.head_hidden, cfg.activation, cfg.batchnorm)

    def forward(self, x: Tensor, t: Tensor, omega: float = 0.0) -> Dict[str, Union[Tensor, float]]:
        x_guided, transform_diag = self.guidance.transform(x)
        mu, u, x_hat = self.empirical(x_guided, omega)
        y0_hat, y1_hat = self.heads(u)

        return {
            "y0_hat": y0_hat,
            "y1_hat": y1_hat,
            "x_hat": x_hat,
            "mu": mu,
            "u": u,
            # No adapter to fuse: the heads read the encoder code directly, so u_out
            # is the same tensor object as u, not u plus a zero tensor.
            "u_out": u,
            "x_std_norm": transform_diag["x_std_norm"],
            "x_prior_norm": transform_diag["x_prior_norm"],
            "x_guided_delta_ratio": transform_diag["x_guided_delta_ratio"],
        }

    def loss_terms(
        self,
        out: Dict[str, Tensor],
        x: Tensor,
        t: Tensor,
        yf: Tensor,
        outcome_type: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        """Factual fit, balance, sparsity and reconstruction, plus guidance when active.

        `x` is the reconstruction target as passed by the caller - the original
        standardized covariates, never `x_guided` - so the guided input never leaks
        into what `L_rec` is measured against.
        """
        outcome_type = outcome_type or self.outcome_type
        if self.l1_target == "u":
            code = out["u"]
        elif self.l1_target == "mu":
            code = out["mu"]
        else:
            raise ValueError(f"l1_target must be 'u' or 'mu'; got {self.l1_target!r}")

        # The heads live on the standardized outcome, so the target is standardized
        # to meet them; to_outcome_scale undoes this on the way out.
        if outcome_type != "binary":
            yf = (yf - self.y_loc) / self.y_scale

        terms = {
            "L_fact": factual_loss(out["y0_hat"], out["y1_hat"], t, yf, outcome_type),
            "L_mmd": mmd_rbf(self.balance_representation(out), t),
            "L_sparse": sparse_loss(code),
            "L_rec": F.mse_loss(out["x_hat"], x),
        }
        if self.guidance.mode != "none":
            weight = first_linear_weight(self.empirical.encoder)
            guidance_loss, _ = self.guidance.guidance_loss(weight)
            terms["L_guidance"] = guidance_loss
        return terms

    def total_loss(self, terms: Mapping[str, Tensor], cfg: DefaultConfig) -> Tuple[Tensor, dict]:
        """Delegate to `losses.total.total_loss`, which weights L_guidance by lambda_prior
        when the key is present."""
        return _total_loss(terms, cfg)

    def balance_representation(self, out: Dict[str, Tensor]) -> Tensor:
        """`u`: the MMD acts on the encoder's code, there is no fused correction."""
        return out["u"]

    def diagnostics(self, out: Dict[str, object], omega: Optional[float] = None) -> dict:
        """Representation norms and guidance alignment for a forward pass.

        The guidance diagnostics are recomputed here under `torch.no_grad()` from the
        current first-layer weight, so this method stays stateless rather than
        trusting whatever the forward pass happened to log. `omega` is the training
        loop's noise amplitude, which a scored split has no equivalent of; left None
        it is simply absent from the result.
        """
        scores = {
            "u_norm": _mean_norm(out["u"]),
            "u_out_norm": _mean_norm(out["u_out"]),
            "x_std_norm": float(out["x_std_norm"]),
            "x_prior_norm": float(out["x_prior_norm"]),
            "x_guided_delta_ratio": float(out["x_guided_delta_ratio"]),
        }
        with torch.no_grad():
            weight = first_linear_weight(self.empirical.encoder)
            _, guidance_diag = self.guidance.guidance_loss(weight)
        scores.update(guidance_diag)
        if omega is not None:
            scores["omega"] = omega
        return scores

    def format_epoch(self, last: dict) -> str:
        """A compact one-line summary of the code norm and the guidance alignment."""
        return (
            f"u {last.get('u_norm', float('nan')):.3f} "
            f"guidance {last.get('guidance_loss', float('nan')):.4f} "
            f"align {last.get('alignment_active', float('nan')):.3f} "
            f"delta {last.get('x_guided_delta_ratio', float('nan')):.3f}"
        )

    @property
    def outcome_affine(self) -> Tuple[float, float]:
        """(loc, scale) mapping a head output back to the caller's outcome units."""
        return float(self.y_loc), float(self.y_scale)

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

    @torch.no_grad()
    def predict_tau(self, x: Tensor) -> Tensor:
        """Deterministic CATE estimate tau_hat(x) = y1_hat - y0_hat (eval, no noise).

        On the outcome's own scale: a risk difference for a binary outcome, since a
        difference of the raw head logits would be a log-odds ratio instead.
        """
        y0, y1 = self.potential_outcomes(x)
        return y1 - y0
