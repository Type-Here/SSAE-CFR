"""Preference loss L_pref: the cost of admitting undocumented information.

    L_pref = mean_j b_j

`b` is the admission gate in [0, 1]^m: `x_mod = P_U x + b * (I - P_U) x`, so `b_j` is how
much of covariate j's residual - the part the semantic prior does not name - the model
asked for on this unit. Penalizing its mean expresses the project's actual thesis: prefer
the documented explanation, and pay a price for reaching past it.

This replaces L_align = mean ||mu_res||^2, which expressed the same intent as an
unbounded penalty on the norm of a representation. Measured, that was not a preference
but a prohibition: at gamma_align = 1.0 the residual branch was reduced to 1.5 percent of
the prior branch, and the residual subspace carries about as much information about the
true tau as the prior subspace does. A bounded penalty on an admission decision cannot
delete information silently - the quantity it shrinks is the quantity you read.

L_pref is an L1 norm
--------------------
`b` is non-negative, so `mean_j b_j = ||b||_1 / m`. The sparsity the model's name once
claimed does not disappear here, it relocates: from the latent code, where a sparse
coordinate names nothing and the penalty acts on an unbounded norm (which is how a large
beta_l1 collapsed the representation), to covariate space, where every sparse coordinate
is a covariate with a clinical name and the penalty acts on something bounded in [0, 1].

One honest limit: with `b = sigmoid(gate(x))` no coordinate ever reaches exactly zero, so
this drives coordinates into the flat tail rather than switching them off. That is enough
to read which covariates needed their residual; exact zeros would need an L0-style gate.
"""

from __future__ import annotations

from torch import Tensor


def preference_loss(b: Tensor) -> Tensor:
    """Mean admission over units and covariates. Scalar in [0, 1]."""
    return b.mean()