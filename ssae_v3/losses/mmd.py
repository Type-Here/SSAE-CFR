"""MMD balancing loss on u_shared.

Empirical (biased) squared MMD between the treated and control codes, RBF kernel:

    MMD^2 = mean k(u_t, u_t') + mean k(u_c, u_c') - 2 mean k(u_t, u_c)

Driving this to zero pulls the treated and control representations into overlap
(the CFR balancing idea). Bandwidth defaults to the median heuristic: with
`bandwidth=None`, the RBF scale is set from the median pairwise squared distance of
the pooled batch, so the kernel is neither saturated nor degenerate for the data at
hand. An explicit `bandwidth` (a sigma) overrides it with denominator 2*sigma^2.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

_EPS = 1e-8


def _sq_dists(a: Tensor, b: Tensor) -> Tensor:
    """Pairwise squared Euclidean distances, shape (len(a), len(b))."""
    return torch.cdist(a, b).pow(2)


def mmd_rbf(u_shared: Tensor, t: Tensor, bandwidth: Optional[float] = None) -> Tensor:
    """Squared RBF-MMD between u_shared[t==1] and u_shared[t==0]. Scalar, >= 0.

    Returns a differentiable zero if either arm is absent from the batch (MMD is
    undefined with only one group), so a degenerate mini-batch cannot crash training.
    """
    t = t.reshape(-1)
    u_t = u_shared[t == 1]
    u_c = u_shared[t == 0]
    if u_t.shape[0] == 0 or u_c.shape[0] == 0:
        return u_shared.sum() * 0.0

    if bandwidth is None:
        pooled = torch.pdist(u_shared).pow(2)  # excludes self-pairs
        med = torch.median(pooled) if pooled.numel() > 0 else u_shared.new_tensor(1.0)
        denom = med + _EPS
    else:
        denom = 2.0 * bandwidth * bandwidth

    def kernel(a: Tensor, b: Tensor) -> Tensor:
        return torch.exp(-_sq_dists(a, b) / denom)

    k_tt = kernel(u_t, u_t).mean()
    k_cc = kernel(u_c, u_c).mean()
    k_tc = kernel(u_t, u_c).mean()
    return (k_tt + k_cc - 2.0 * k_tc).clamp_min(0.0)
