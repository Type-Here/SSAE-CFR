"""U-Adapter: the structural correction, from covariate space only.

    U_k^T x_std -> A_U -> c

`c` is added to the empirical code (`u_shared = u + r_U * c`), never to the raw
covariates. `A_U` is an MLP over R^{k_U}: the adapter reads the coordinates of x in
the prior's basis rather than their embedding P_U x = U_k (U_k^T x) back into R^m.
The two carry exactly the same information and span the same function class - any
first layer W on P_U x equals the layer W U_k on U_k^T x, and conversely, since
U_k^T U_k = I - so this is a reparameterization with k_U inputs instead of m, not a
change of what the branch may see.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from ..core.modules import build_mlp


class UStructuralAdapter(nn.Module):
    """Maps U_k^T x_std to a correction c in R^{d_u}.

    `U_k` is stored as a non-trainable buffer: it moves with `.to(device)` and is
    saved in `state_dict`, but is never optimized. The projection is computed inside
    `forward` so the adapter structurally never receives anything but the prior's own
    coordinates of x - not x, not u, not the residual (I - P_U) x.

    The final linear layer is zero-initialized (weight and bias), so c is exactly
    zero at init, whatever the input.
    """

    def __init__(
        self,
        U_k: Tensor,
        d_u: int,
        hidden: Sequence[int] = (32,),
        activation: str = "relu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        # cloned so this buffer never aliases the caller's tensor: `.to()` alone is a
        # no-op (same storage) when the dtype already matches
        self.register_buffer("U_k", U_k.detach().to(dtype=torch.float32).clone())
        k_u = U_k.shape[1]
        self.net = build_mlp(k_u, hidden, d_u, activation, batchnorm)
        self._zero_final_layer()

    def _zero_final_layer(self) -> None:
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x_std: Tensor) -> Tensor:
        """Project x_std onto the covariate-space prior, then map to a correction c."""
        U_k = self.U_k.to(device=x_std.device, dtype=x_std.dtype)
        z_u = x_std @ U_k  # [batch, k_U]
        c = self.net(z_u)  # [batch, d_u]
        return c