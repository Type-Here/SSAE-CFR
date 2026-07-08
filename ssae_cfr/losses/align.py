"""Semantic alignment loss L_align.

    L_align = mean_i ||mu_res_i||^2

`mu_res` is the DETERMINISTIC encoding of the residual (I - P_U) x. Shrinking its norm
pushes the model to explain the outcome from the prior-spanned subspace first, and to
lean on the undocumented residual only when it must - the guardrail against resting the
estimate on spurious, unexplained correlations.

Two deliberate choices: we penalise the deterministic branch (penalising the stochastic
one would punish the exploration noise, which is backwards), and the weight gamma is
warmed up over training so a wrong prior cannot railroad the model early (a
self-fulfilling prophecy). The warm-up lives in `utils.schedules`; this term is just the
squared norm.
"""

from __future__ import annotations

from torch import Tensor


def align_loss(mu_res: Tensor) -> Tensor:
    """Mean squared L2 norm of the deterministic residual encoding. Scalar."""
    return mu_res.pow(2).sum(dim=-1).mean()