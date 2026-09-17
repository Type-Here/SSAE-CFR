"""U-Adapter: the structural correction, from covariate space only.

    P_U x_std -> A_U -> c

`c` is added to the empirical code (`u_shared = u + r_U * c`), never to the raw
covariates. `A_U` is an MLP over R^m, since P_U x_std is a rank-k_U vector that still
lives in the full m-dimensional covariate space.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from ..core.modules import build_mlp


class UStructuralAdapter(nn.Module):
    """Maps P_U x_std to a correction c in R^{d_u}.

    `P_U` is stored as a non-trainable buffer: it moves with `.to(device)` and is
    saved in `state_dict`, but is never optimized. The projection is computed inside
    `forward` so the adapter structurally never receives anything but P_U x_std - not
    x, not u, not the residual (I - P_U) x.

    The final linear layer is zero-initialized (weight and bias), so c is exactly
    zero at init, whatever the input.
    """

    def __init__(
        self,
        P_U: Tensor,
        d_u: int,
        hidden: Sequence[int] = (32,),
        activation: str = "relu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if P_U.dim() != 2 or P_U.shape[0] != P_U.shape[1]:
            raise ValueError(f"P_U must be square (m, m); got shape {tuple(P_U.shape)}")
        # cloned so this buffer never aliases the caller's tensor: `.to()` alone is a
        # no-op (same storage) when the dtype already matches
        self.register_buffer("P_U", P_U.detach().to(dtype=torch.float32).clone())
        m = P_U.shape[0]
        self.net = build_mlp(m, hidden, d_u, activation, batchnorm)
        self._zero_final_layer()

    def _zero_final_layer(self) -> None:
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x_std: Tensor) -> Tensor:
        """Project x_std onto the covariate-space prior, then map to a correction c."""
        p_u = self.P_U.to(device=x_std.device, dtype=x_std.dtype)
        projected = x_std @ p_u.T
        return self.net(projected)
