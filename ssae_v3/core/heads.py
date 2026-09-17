"""TARNet-style outcome heads h0, h1 over u_out.

    y0_hat, y1_hat = h0(u_out), h1(u_out)
    yf_hat = t * y1_hat + (1 - t) * y0_hat

Two independent heads (rather than concatenating t into one regressor) so a weak
treatment effect is not washed out. No output activation: the heads emit a raw scalar
(a logit for binary outcomes), which the factual loss interprets.
"""

from __future__ import annotations

from typing import Sequence, Tuple
from torch import Tensor, nn
from .modules import build_mlp


class OutcomeHeads(nn.Module):
    """Two independent heads mapping u_out (R^{d_u}) to the two potential outcomes."""

    def __init__(
        self,
        d_u: int,
        hidden: Sequence[int],
        activation: str = "relu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.h0 = build_mlp(d_u, hidden, 1, activation, batchnorm)
        self.h1 = build_mlp(d_u, hidden, 1, activation, batchnorm)

    def forward(self, u_out: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (y0_hat, y1_hat), each shape (n,)."""
        y0 = self.h0(u_out).squeeze(-1)
        y1 = self.h1(u_out).squeeze(-1)
        return y0, y1