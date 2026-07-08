"""Factual prediction loss L_fact.

Only the factual head is supervised - the outcome we actually observed for each unit:

    yf_hat = t * y1_hat + (1 - t) * y0_hat

The counterfactual head is never given a target; it is shaped indirectly, through the
shared representation and the balancing term. The per-unit loss is MSE for continuous
outcomes and BCE for binary ones (the heads emit logits, so we use the numerically
stable with-logits form), selected by the dataset's outcome type.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def factual_loss(
    y0_hat: Tensor,
    y1_hat: Tensor,
    t: Tensor,
    yf: Tensor,
    outcome_type: str = "continuous",
) -> Tensor:
    """Mean factual loss on the observed-arm prediction.

    `t` may be int or float; it is cast to match the predictions. Returns a scalar.
    """
    t = t.to(y1_hat.dtype)
    yf_hat = t * y1_hat + (1.0 - t) * y0_hat
    yf = yf.to(yf_hat.dtype)

    if outcome_type == "continuous":
        return F.mse_loss(yf_hat, yf)
    if outcome_type == "binary":
        return F.binary_cross_entropy_with_logits(yf_hat, yf)
    raise ValueError(f"outcome_type must be 'continuous' or 'binary'; got {outcome_type!r}")