"""The three v0.5 loss terms and their weighted sum.

    L = L_fact + alpha_mmd * L_mmd(u_out) + beta_sparse * L_sparse(u_out)

No reconstruction term, no guidance term, no loss on the embeddings. Factual, MMD
and sparse are adapted unchanged from the v0.3/v0.4 losses; only the assembly is new.

Shares (weighted value over the sum of weighted magnitudes) are reported next to the
raw values because a nominal weight says nothing when terms have natural magnitudes
differing by orders of magnitude. The denominator uses magnitudes: the biased MMD
estimator can come out slightly negative near perfect balance, and a signed
denominator would then report shares that do not describe a budget.
"""

from __future__ import annotations

from typing import Mapping, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from .config import SLRConfig

_EPS = 1e-8


def factual_loss(
    y0_hat: Tensor,
    y1_hat: Tensor,
    t: Tensor,
    yf: Tensor,
    outcome_type: str = "continuous",
) -> Tensor:
    """Mean loss on the observed-arm prediction yf_hat = t*y1_hat + (1-t)*y0_hat."""
    t = t.to(y1_hat.dtype)
    yf_hat = t * y1_hat + (1.0 - t) * y0_hat
    yf = yf.to(yf_hat.dtype)

    if outcome_type == "continuous":
        return F.mse_loss(yf_hat, yf)
    if outcome_type == "binary":
        return F.binary_cross_entropy_with_logits(yf_hat, yf)
    raise ValueError(f"outcome_type must be 'continuous' or 'binary'; got {outcome_type!r}")


def mmd_rbf(u: Tensor, t: Tensor, bandwidth: Optional[float] = None) -> Tensor:
    """Squared RBF-MMD between u[t==1] and u[t==0]. Scalar, >= 0.

    Bandwidth defaults to the median heuristic on the pooled batch. Returns a
    differentiable zero if either arm is absent, so a degenerate batch cannot crash
    training.
    """
    t = t.reshape(-1)
    u_t = u[t == 1]
    u_c = u[t == 0]
    if u_t.shape[0] == 0 or u_c.shape[0] == 0:
        return u.sum() * 0.0

    if bandwidth is None:
        pooled = torch.pdist(u).pow(2)
        med = torch.median(pooled) if pooled.numel() > 0 else u.new_tensor(1.0)
        denom = med + _EPS
    else:
        denom = 2.0 * bandwidth * bandwidth

    def kernel(a: Tensor, b: Tensor) -> Tensor:
        return torch.exp(-torch.cdist(a, b).pow(2) / denom)

    k_tt = kernel(u_t, u_t).mean()
    k_cc = kernel(u_c, u_c).mean()
    k_tc = kernel(u_t, u_c).mean()
    return (k_tt + k_cc - 2.0 * k_tc).clamp_min(0.0)


def sparse_loss(code: Tensor) -> Tensor:
    """Mean L1 norm of `code` over the batch, shape (n, d) -> scalar."""
    return code.abs().sum(dim=-1).mean()


_REQUIRED = ("L_fact", "L_mmd", "L_sparse")


def total_loss(terms: Mapping[str, Tensor], cfg: "SLRConfig") -> Tuple[Tensor, dict]:
    """Weighted total plus a logged breakdown (raw value, weighted value, share)."""
    missing = [name for name in _REQUIRED if name not in terms]
    if missing:
        raise KeyError(f"total_loss missing terms: {missing}")

    weights = {"L_fact": 1.0, "L_mmd": cfg.alpha_mmd, "L_sparse": cfg.beta_sparse}
    total = (
        terms["L_fact"]
        + weights["L_mmd"] * terms["L_mmd"]
        + weights["L_sparse"] * terms["L_sparse"]
    )

    breakdown: dict = {"L_total": float(total.detach())}
    weighted = {}
    for name, weight in weights.items():
        value = float(terms[name].detach())
        breakdown[name] = value
        weighted[name] = weight * value

    budget = sum(abs(v) for v in weighted.values())
    for name, value in weighted.items():
        breakdown[f"w_{name}"] = value
        breakdown[f"share_{name}"] = value / budget if budget > 0.0 else float("nan")
    return total, breakdown
