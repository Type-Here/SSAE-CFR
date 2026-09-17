"""Sparsity loss L_sparse: mean over units of the per-unit L1 norm of a code.

Which code it acts on (`u`, the empirical bottleneck, or `mu`, its deterministic
mean) is the caller's choice via the config's `l1_target`.
"""

from __future__ import annotations

from torch import Tensor


def sparse_loss(code: Tensor) -> Tensor:
    """Mean L1 norm of `code` over the batch, shape (n, d) -> scalar."""
    return code.abs().sum(dim=-1).mean()
