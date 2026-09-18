"""SSAECFRv3: the empirical host plus two optional, additive, non-destructive corrections.

    x_std  --> empirical --> mu, u --> x_hat        (L_rec on u ONLY)
                              |
    P_U x_std --> u_adapter --> c ---+
                              |      +--> u_shared = u + r_U * c   (MMD acts HERE)
    (x_ij, q~_j) --> w_adapter --> a_W               |
                                                      +--> u_out = u_shared + r_W * a_W
                                                             |
                                                             +--> h0 / h1

The encoder in `core.Empirical` always sees the full standardized `x`; there is no
gate deciding what reaches it and no penalty preferring one information source over
another. When an adapter is absent its correction is not added at all - `u_shared`
(or `u_out`) is the same tensor object as the code before it, not that code plus a
zero tensor - so prior-off is the empirical computation itself, not a
numerically-close approximation of it.
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
from .prior_modules import FixedReliability, UStructuralAdapter, WSemanticAdapter

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


def _mean_scalar(a: ArrayLike) -> float:
    """Mean of every element, for a torch Tensor or a numpy array. See `_mean_norm`."""
    if isinstance(a, Tensor):
        a = a.detach().cpu().numpy()
    return float(np.asarray(a).mean())


class SSAECFRv3(nn.Module):
    """Empirical host + optional U (structural) and W (semantic) corrections."""

    def __init__(
        self,
        cfg: DefaultConfig,
        outcome_type: str = "continuous",
        y_loc: float = 0.0,
        y_scale: float = 1.0,
        P_U: Optional[Tensor] = None,
        q_tilde: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        if outcome_type not in ("continuous", "binary"):
            raise ValueError(f"outcome_type must be 'continuous' or 'binary'; got {outcome_type!r}")
        if y_scale <= 0.0:
            raise ValueError(f"y_scale must be positive; got {y_scale}")
        if outcome_type == "binary" and (y_loc != 0.0 or y_scale != 1.0):
            raise ValueError("a binary outcome is never standardized; leave y_loc/y_scale at 0/1")

        m = cfg.in_channels
        self.m = m
        self.d_u = cfg.d_u
        self.l1_target = cfg.l1_target
        self.outcome_type = outcome_type
        # Buffers, not plain floats: they move with the model across devices and
        # survive a state_dict round trip. L_fact is the only term on the outcome's
        # own units, so a scale silently lost here would make alpha_mmd mean a
        # different thing on every realization (IHDP's outcome scale spans ~15x).
        self.register_buffer("y_loc", torch.tensor(float(y_loc)))
        self.register_buffer("y_scale", torch.tensor(float(y_scale)))

        self.empirical = Empirical(cfg)
        self.heads = OutcomeHeads(cfg.d_u, cfg.head_hidden, cfg.activation, cfg.batchnorm)
        self.reliability = FixedReliability(cfg.r_U, cfg.r_W)

        self.u_adapter = self._build_u_adapter(cfg, m, P_U)
        self.w_adapter = self._build_w_adapter(cfg, m, q_tilde)

    @staticmethod
    def _build_u_adapter(
        cfg: DefaultConfig, m: int, P_U: Optional[Tensor]
    ) -> Optional[UStructuralAdapter]:
        if not cfg.use_u_adapter:
            return None
        if P_U is None:
            raise ValueError(
                f"model_variant={cfg.model_variant!r} requires the U branch (P_U) "
                "but P_U was not given; a silently-ignored flag would make this run "
                "indistinguishable from the empirical variant"
            )
        P_U_t = torch.as_tensor(P_U, dtype=torch.float32)
        if P_U_t.dim() != 2 or P_U_t.shape[0] != P_U_t.shape[1]:
            raise ValueError(f"P_U must be square (m, m); got shape {tuple(P_U_t.shape)}")
        if P_U_t.shape[0] != m:
            raise ValueError(f"P_U is {P_U_t.shape[0]}x{P_U_t.shape[0]} but cfg.in_channels={m}")
        return UStructuralAdapter(P_U_t, cfg.d_u, cfg.u_adapter_hidden, cfg.activation, cfg.batchnorm)

    @staticmethod
    def _build_w_adapter(
        cfg: DefaultConfig, m: int, q_tilde: Optional[Tensor]
    ) -> Optional[WSemanticAdapter]:
        if not cfg.use_w_adapter:
            return None
        if q_tilde is None:
            raise ValueError(
                f"model_variant={cfg.model_variant!r} requires the W branch (q_tilde) "
                "but q_tilde was not given; a silently-ignored flag would make this "
                "run indistinguishable from the empirical variant"
            )
        q_tilde_t = torch.as_tensor(q_tilde, dtype=torch.float32)
        if q_tilde_t.dim() != 2:
            raise ValueError(f"q_tilde must be 2-D (m, r_W); got shape {tuple(q_tilde_t.shape)}")
        if q_tilde_t.shape[0] != m:
            raise ValueError(f"q_tilde has {q_tilde_t.shape[0]} rows but cfg.in_channels={m}")
        return WSemanticAdapter(
            q_tilde_t, cfg.d_u, cfg.d_token, cfg.phi_hidden, cfg.d_s, cfg.rho_hidden,
            cfg.activation, cfg.batchnorm,
        )

    def forward(self, x: Tensor, t: Tensor, omega: float = 0.0) -> Dict[str, Tensor]:
        mu, u, x_hat = self.empirical(x, omega)
        r_U, r_W = self.reliability(x)

        if self.u_adapter is not None:
            c = self.u_adapter(x)
            u_shared = u + r_U * c
        else:
            # No adapter, nothing to fuse: u_shared IS u, the same tensor object,
            # not u plus a zero tensor.
            c = torch.zeros_like(u)
            u_shared = u

        if self.w_adapter is not None:
            a_W = self.w_adapter(x)
            u_out = u_shared + r_W * a_W
        else:
            a_W = torch.zeros_like(u)
            u_out = u_shared

        y0_hat, y1_hat = self.heads(u_out)
        tf = t.to(y1_hat.dtype)
        yf_hat = tf * y1_hat + (1.0 - tf) * y0_hat

        return {
            "y0_hat": y0_hat,
            "y1_hat": y1_hat,
            "yf_hat": yf_hat,
            "x_hat": x_hat,
            "mu": mu,
            "u": u,
            "u_shared": u_shared,
            "u_out": u_out,
            "c": c,
            "a_W": a_W,
            "r_U": r_U,
            "r_W": r_W,
        }

    def loss_terms(
        self,
        out: Dict[str, Tensor],
        x: Tensor,
        t: Tensor,
        yf: Tensor,
        outcome_type: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        """The four loss terms: factual fit, balance, sparsity, reconstruction only."""
        outcome_type = outcome_type or self.outcome_type
        code = out["u"] if self.l1_target == "u" else out["mu"]
        # The heads live on the standardized outcome, so the target is standardized
        # to meet them; to_outcome_scale undoes this on the way out.
        if outcome_type != "binary":
            yf = (yf - self.y_loc) / self.y_scale
        return {
            "L_fact": factual_loss(out["y0_hat"], out["y1_hat"], t, yf, outcome_type),
            "L_mmd": mmd_rbf(self.balance_representation(out), t),
            "L_sparse": sparse_loss(code),
            "L_rec": F.mse_loss(out["x_hat"], x),
        }

    def total_loss(self, terms: Mapping[str, Tensor], cfg: DefaultConfig) -> Tuple[Tensor, dict]:
        """Delegate to `losses.total.total_loss`. No warm-up: every term is on from epoch zero."""
        return _total_loss(terms, cfg)

    def balance_representation(self, out: Dict[str, Tensor]) -> Tensor:
        """`u_shared`: the MMD acts after the U correction, before the W correction."""
        return out["u_shared"]

    def train_diagnostics(self, out: Dict[str, Tensor], omega: float) -> dict:
        """Per-batch watch numbers, `omega` included, for the training loop's log line."""
        return {
            "omega": omega,
            "u_norm": _mean_norm(out["u"]),
            "u_shared_norm": _mean_norm(out["u_shared"]),
            "u_out_norm": _mean_norm(out["u_out"]),
            "c_norm": _mean_norm(out["c"]),
            "a_W_norm": _mean_norm(out["a_W"]),
            "r_U": _mean_scalar(out["r_U"]),
            "r_W": _mean_scalar(out["r_W"]),
        }

    def split_diagnostics(self, out: Dict[str, Tensor]) -> dict:
        """The same norms as `train_diagnostics`, without `omega` - used on a scored split."""
        return {
            "u_norm": _mean_norm(out["u"]),
            "u_shared_norm": _mean_norm(out["u_shared"]),
            "u_out_norm": _mean_norm(out["u_out"]),
            "c_norm": _mean_norm(out["c"]),
            "a_W_norm": _mean_norm(out["a_W"]),
            "r_U": _mean_scalar(out["r_U"]),
            "r_W": _mean_scalar(out["r_W"]),
        }

    def format_epoch(self, last: dict) -> str:
        """A compact one-line summary of the norms, for the training loop's log line."""
        return (
            f"u {last.get('u_norm', float('nan')):.3f} "
            f"u_shared {last.get('u_shared_norm', float('nan')):.3f} "
            f"u_out {last.get('u_out_norm', float('nan')):.3f} "
            f"c {last.get('c_norm', float('nan')):.3f} "
            f"a_W {last.get('a_W_norm', float('nan')):.3f} "
            f"r_U {last.get('r_U', float('nan')):.2f} r_W {last.get('r_W', float('nan')):.2f}"
        )

    @property
    def outcome_affine(self) -> Tuple[float, float]:
        """(loc, scale) mapping a head output back to the caller's outcome units."""
        return float(self.y_loc), float(self.y_scale)

    @torch.no_grad()
    def potential_outcomes(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Deterministic (y0_hat, y1_hat) on the outcome's own scale (eval, no noise).

        All causal predictions derive from u_out; there is no alternate path. For a
        binary outcome these are probabilities, not the raw head logits.
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
