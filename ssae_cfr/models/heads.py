"""TARNet-style outcome heads h0, h1 over the balanced code z_mod.

    y0_hat, y1_hat = h0(z_mod), h1(z_mod)
    yf_hat = t * y1_hat + (1 - t) * y0_hat

Two *independent* feed-forward heads, one per treatment arm. Keeping them separate
(rather than concatenating t into a single regressor) stops the network from washing
out a weak treatment effect: each potential outcome gets its own capacity. The heads
emit a raw scalar per unit - a real value for continuous outcomes, a logit for binary
ones - and the factual loss picks MSE or BCE-with-logits accordingly, so no activation
is applied here.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from torch import Tensor, nn

from .ssae import build_mlp


class OutcomeHeads(nn.Module):
    """Two independent heads mapping z_mod (R^{k_latent}) to the two potential outcomes."""

    def __init__(
        self,
        k_latent: int,
        hidden: Sequence[int],
        activation: str = "elu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.h0 = build_mlp(k_latent, hidden, 1, activation, batchnorm)
        self.h1 = build_mlp(k_latent, hidden, 1, activation, batchnorm)

    def forward(self, z_mod: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (y0_hat, y1_hat), each shape (n,)."""
        y0 = self.h0(z_mod).squeeze(-1)
        y1 = self.h1(z_mod).squeeze(-1)
        return y0, y1