"""Total loss assembly with the alignment warm-up.

    gamma = warmup(epoch)
    L = L_fact
      + alpha_mmd  * L_mmd
      + beta_l1    * L_sparse
      + lambda_rec * L_rec
      + gamma      * L_align

Weights come from the TrainConfig; the gamma schedule from `utils.schedules`. Returns the
scalar total plus a breakdown dict (each term's raw scalar value, its weight, and the
current gamma) for the per-run diagnostics.
"""

from __future__ import annotations

from typing import Mapping, Tuple, TYPE_CHECKING

from torch import Tensor

from ..utils.schedules import gamma_warmup

if TYPE_CHECKING:
    from ..config import TrainConfig

_REQUIRED = ("L_fact", "L_mmd", "L_sparse", "L_rec", "L_align")


def total_loss(
    terms: Mapping[str, Tensor],
    cfg: "TrainConfig",
    epoch: int,
) -> Tuple[Tensor, dict]:
    """Combine the five loss terms into the weighted total and a logged breakdown."""
    missing = [name for name in _REQUIRED if name not in terms]
    if missing:
        raise KeyError(f"total_loss missing terms: {missing}")

    gamma = gamma_warmup(epoch, cfg.gamma_align, cfg.gamma_warmup)

    total = (
        terms["L_fact"]
        + cfg.alpha_mmd * terms["L_mmd"]
        + cfg.beta_l1 * terms["L_sparse"]
        + cfg.lambda_rec * terms["L_rec"]
        + gamma * terms["L_align"]
    )

    breakdown = {
        "L_total": float(total.detach()),
        "L_fact": float(terms["L_fact"].detach()),
        "L_mmd": float(terms["L_mmd"].detach()),
        "L_sparse": float(terms["L_sparse"].detach()),
        "L_rec": float(terms["L_rec"].detach()),
        "L_align": float(terms["L_align"].detach()),
        "gamma": gamma,
    }
    return total, breakdown