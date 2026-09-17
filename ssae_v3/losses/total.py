"""Total loss assembly: four terms, no warm-up.

    L = L_fact + alpha_mmd * L_mmd + beta_l1 * L_sparse + lambda_rec * L_rec

Every term is on from epoch zero.

Shares (weight x value, over the total) are reported alongside the raw weights
because a nominal weight says nothing when terms have natural magnitudes differing by
orders of magnitude - e.g. `alpha_mmd = lambda_rec = 1.0` does not mean balancing and
reconstruction matter equally. The share denominator is the sum of the *magnitudes*
of the weighted terms, not their signed total: the biased MMD estimator can come out
slightly negative near perfect balance, and a signed denominator would then report
shares that do not describe a budget.
"""

from __future__ import annotations

from typing import Mapping, Tuple, TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from ..hparams import DefaultConfig

_REQUIRED = ("L_fact", "L_mmd", "L_sparse", "L_rec")


def weighted_breakdown(terms: Mapping[str, Tensor], weights: Mapping[str, float]) -> dict:
    """Per-term raw value, weighted value (`w_<name>`) and share of the objective
    (`share_<name>`, denominator = sum of weighted-term magnitudes)."""
    breakdown: dict = {}
    weighted: dict = {}
    for name, weight in weights.items():
        value = float(terms[name].detach())
        breakdown[name] = value
        weighted[name] = weight * value

    budget = sum(abs(v) for v in weighted.values())
    for name, value in weighted.items():
        breakdown[f"w_{name}"] = value
        breakdown[f"share_{name}"] = value / budget if budget > 0.0 else float("nan")
    return breakdown


def total_loss(terms: Mapping[str, Tensor], cfg: "DefaultConfig") -> Tuple[Tensor, dict]:
    """Combine the four loss terms into the weighted total and a logged breakdown."""
    missing = [name for name in _REQUIRED if name not in terms]
    if missing:
        raise KeyError(f"total_loss missing terms: {missing}")

    weights = {
        "L_fact": 1.0,
        "L_mmd": cfg.alpha_mmd,
        "L_sparse": cfg.beta_l1,
        "L_rec": cfg.lambda_rec,
    }

    total = (
        terms["L_fact"]
        + weights["L_mmd"] * terms["L_mmd"]
        + weights["L_sparse"] * terms["L_sparse"]
        + weights["L_rec"] * terms["L_rec"]
    )

    breakdown = {"L_total": float(total.detach())}
    breakdown.update(weighted_breakdown(terms, weights))
    return total, breakdown
