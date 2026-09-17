"""Reliability: how much semantic augmentation is used, never how much data is seen.

    r_U, r_W = reliability(x)
    u_shared = u + r_U * c
    u_out    = u_shared + r_W * a_W

Reliability scales the U/W corrections added on top of the empirical code. It must
never be read as gating whether the empirical encoder sees x - the empirical path
(x_std -> encoder -> u -> decoder -> x_hat) is unconditional; r_U/r_W only scale what
is added after the fact. Confusing the two would resurrect the admission gate this
design replaces.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


class FixedReliability(nn.Module):
    """Two fixed reliability constants, broadcast to every patient and code dim.

    `forward(x)` returns `(r_U, r_W)`, each shape `(n, 1)`. Both come from
    non-trainable buffers: they move with `.to(device)` and take x's dtype, but carry
    no gradient and are never touched by an optimizer.

    The signature is written to be swapped later for a learned, per-patient
    estimator without changing the model that calls it.
    """

    def __init__(self, r_U: float, r_W: float) -> None:
        super().__init__()
        self.register_buffer("r_U_const", torch.tensor(float(r_U)))
        self.register_buffer("r_W_const", torch.tensor(float(r_W)))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        n = x.shape[0]
        r_U = self.r_U_const.to(device=x.device, dtype=x.dtype).expand(n, 1)
        r_W = self.r_W_const.to(device=x.device, dtype=x.dtype).expand(n, 1)
        return r_U, r_W
