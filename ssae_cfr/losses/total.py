"""Total loss assembly with the alignment warm-up.

    gamma = warmup(epoch)
    L = L_fact
      + alpha_mmd  * L_mmd
      + beta_l1    * L_sparse
      + lambda_rec * L_rec
      + gamma      * L_pref

Weights come from the TrainConfig; the gamma schedule from `utils.schedules`. Returns the
scalar total plus a breakdown dict (each term's raw scalar value, its weighted value, its
share of the objective, and the current gamma) for the per-run diagnostics.

Why the shares are reported and not just the weights
----------------------------------------------------
A nominal weight says nothing on its own, because the five terms have natural magnitudes
that differ by orders of magnitude. `L_rec` is an unbounded MSE; MMD^2 with an RBF kernel
is bounded (roughly in [0, 2]) and sits near 0.01 once the arms overlap. So `alpha_mmd`
and `lambda_rec` both being 1.0 does not mean balancing and reconstruction matter equally
- measured on a real run, reconstruction commanded 58 percent of the objective and the
balancing term 1.4 percent. The share (weight x value, over the total) is the number that
actually says what the optimizer is working on, so it is computed here rather than left
to be reconstructed after the fact.

The denominator is the sum of the *magnitudes* of the weighted terms, not their signed
total. The biased MMD estimator can come out slightly negative near perfect balance, and
a signed denominator would then report shares that do not describe a budget. With
magnitudes the shares are always in [0, 1] and sum to 1.
"""

from __future__ import annotations

from typing import Mapping, Tuple, TYPE_CHECKING

from torch import Tensor

from ..utils.schedules import linear_warmup

if TYPE_CHECKING:
    from ..config import TrainConfig

_REQUIRED = ("L_fact", "L_mmd", "L_sparse", "L_rec", "L_pref")


def total_loss(
    terms: Mapping[str, Tensor],
    cfg: "TrainConfig",
    epoch: int,
) -> Tuple[Tensor, dict]:
    """Combine the five loss terms into the weighted total and a logged breakdown."""
    missing = [name for name in _REQUIRED if name not in terms]
    if missing:
        raise KeyError(f"total_loss missing terms: {missing}")

    gamma = linear_warmup(epoch, cfg.gamma_pref, cfg.gamma_warmup)
    weights = {
        "L_fact": 1.0,
        "L_mmd": cfg.alpha_mmd,
        "L_sparse": cfg.beta_l1,
        "L_rec": cfg.lambda_rec,
        "L_pref": gamma,
    }

    total = (
        terms["L_fact"]
        + weights["L_mmd"] * terms["L_mmd"]
        + weights["L_sparse"] * terms["L_sparse"]
        + weights["L_rec"] * terms["L_rec"]
        + weights["L_pref"] * terms["L_pref"]
    )

    breakdown = {"L_total": float(total.detach()), "gamma": gamma}
    weighted = {}
    for name in _REQUIRED:
        value = float(terms[name].detach())
        breakdown[name] = value
        weighted[name] = weights[name] * value

    budget = sum(abs(v) for v in weighted.values())
    for name, value in weighted.items():
        breakdown[f"w_{name}"] = value
        breakdown[f"share_{name}"] = value / budget if budget > 0.0 else float("nan")
    return total, breakdown