"""MMD balancing loss on z_mod.

Empirical (biased) squared Maximum Mean Discrepancy between the treated and control
codes, with an RBF kernel:

    MMD^2 = mean k(z_t, z_t') + mean k(z_c, z_c') - 2 mean k(z_t, z_c)

Driving this toward zero forces the treated and control code distributions to overlap -
representation-level balancing, the CFR idea. What makes it work here is that z_mod is
the STOCHASTIC code: when the two groups have little common support, the deterministic
codes would be cleanly separable and the kernel would see two far-apart point clouds
with a vanishing cross term and an uninformative gradient. The SMD-scaled noise smears
each cloud out so the kernels overlap and the gradient actually points somewhere.

Bandwidth: the median heuristic. With `bandwidth=None` we set the RBF scale from the
median of the pairwise squared distances of the pooled batch, so the kernel is neither
saturated (all ones) nor degenerate (all zeros) for the data at hand. Passing an
explicit `bandwidth` (a sigma) overrides it with denominator 2*sigma^2.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

_EPS = 1e-8


def _sq_dists(a: Tensor, b: Tensor) -> Tensor:
    """Pairwise squared Euclidean distances, shape (len(a), len(b))."""
    return torch.cdist(a, b).pow(2)


def mmd_rbf(z_mod: Tensor, t: Tensor, bandwidth: Optional[float] = None) -> Tensor:
    """Squared RBF-MMD between z_mod[t==1] and z_mod[t==0]. Scalar, >= 0.

    Returns a differentiable zero if either arm is absent from the batch (MMD is
    undefined with only one group), so a degenerate mini-batch cannot crash training.
    """
    t = t.reshape(-1)
    z_t = z_mod[t == 1]
    z_c = z_mod[t == 0]
    if z_t.shape[0] == 0 or z_c.shape[0] == 0:
        return z_mod.sum() * 0.0

    if bandwidth is None:
        # median of pooled pairwise squared distances (pdist excludes self-pairs)
        pooled = torch.pdist(z_mod).pow(2)
        med = torch.median(pooled) if pooled.numel() > 0 else z_mod.new_tensor(1.0)
        denom = med + _EPS
    else:
        denom = 2.0 * bandwidth * bandwidth

    def kernel(a: Tensor, b: Tensor) -> Tensor:
        return torch.exp(-_sq_dists(a, b) / denom)

    k_tt = kernel(z_t, z_t).mean()
    k_cc = kernel(z_c, z_c).mean()
    k_tc = kernel(z_t, z_c).mean()
    return (k_tt + k_cc - 2.0 * k_tc).clamp_min(0.0)