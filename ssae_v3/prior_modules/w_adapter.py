"""W-Adapter: the semantic correction, from per-feature value x meaning.

    (x_ij, q~_j) -> phi -> tokens -> mean pool -> rho -> s -> A_W -> a_W

`q~` (q_tilde) is frozen: row j is the normalized semantic coordinate of covariate j,
what the LLM's embedding says feature j means. `a_W` is added to u_shared
(`u_out = u_shared + r_W * a_W`), never to the raw covariates.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from ..core.modules import build_mlp


class ValueSemanticTokenizer(nn.Module):
    """Builds one token per (patient, feature) and maps it through a shared phi MLP.

    The token input concatenates [x_ij, q~_j, x_ij * q~_j], width 1 + 2*r_W. The
    interaction term x_ij * q~_j is required: it is what lets the token depend on the
    covariate's value and its meaning jointly, rather than on a linear rewrite of the
    two stacked separately. One phi is shared across all m features.

    `trainable` turns the frozen semantic table into a learned per-feature embedding,
    initialized at whatever `q_tilde` was passed in. It is the ablation that asks what
    the covariate descriptions are worth against a table the outcome loss is free to
    invent for itself; it is not part of the frozen v3.0 core, where q~ is offline and
    fixed. It also adds m * r_W trainable parameters, so an arm using it is not
    capacity-matched to one that does not.
    """

    def __init__(
        self,
        q_tilde: Tensor,
        d_token: int,
        hidden: Sequence[int] = (32,),
        activation: str = "relu",
        batchnorm: bool = False,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if q_tilde.dim() != 2:
            raise ValueError(f"q_tilde must be 2-D (m, r_W); got shape {tuple(q_tilde.shape)}")
        # cloned so the table never aliases the caller's tensor: `.to()` alone is a
        # no-op (same storage) when the dtype already matches
        table = q_tilde.detach().to(dtype=torch.float32).clone()
        self.trainable = bool(trainable)
        if self.trainable:
            self.q_tilde = nn.Parameter(table)
        else:
            self.register_buffer("q_tilde", table)
        m, r_W = q_tilde.shape
        self.m = m
        self.r_W = r_W
        in_dim = 1 + 2 * r_W
        self.phi = build_mlp(in_dim, hidden, d_token, activation, batchnorm)

    def forward(self, x_std: Tensor) -> Tensor:
        """x_std (n, m) -> tokens (n, m, d_token)."""
        n, m = x_std.shape
        q_tilde = self.q_tilde.to(device=x_std.device, dtype=x_std.dtype)
        q_broadcast = q_tilde.unsqueeze(0).expand(n, -1, -1)          # (n, m, r_W)
        value = x_std.unsqueeze(-1)                                    # (n, m, 1)
        interaction = value * q_broadcast                              # (n, m, r_W)
        token_in = torch.cat([value, q_broadcast, interaction], dim=-1)  # (n, m, 1+2*r_W)
        # phi is shared across features; flatten the (patient, feature) axes into one
        # batch dimension for a single call, so BatchNorm1d normalizes over d_token
        # rather than mistaking the feature axis for a channel axis.
        flat_out = self.phi(token_in.reshape(n * m, -1))
        return flat_out.reshape(n, m, -1)


class WSemanticAdapter(nn.Module):
    """Tokenizer -> mean pool over features -> rho -> A_W, giving a_W in R^{d_u}.

    Pooling is a plain mean over the feature axis, no attention or learned set
    aggregation. The final layer of A_W is zero-initialized (weight and bias), so
    a_W is exactly zero at init - including when the semantic table itself is learned.
    """

    def __init__(
        self,
        q_tilde: Tensor,
        d_u: int,
        d_token: int = 32,
        phi_hidden: Sequence[int] = (32,),
        d_s: int = 32,
        rho_hidden: Sequence[int] = (32,),
        activation: str = "relu",
        batchnorm: bool = False,
        trainable_semantics: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = ValueSemanticTokenizer(
            q_tilde, d_token, phi_hidden, activation, batchnorm, trainable_semantics
        )
        self.rho = build_mlp(d_token, rho_hidden, d_s, activation, batchnorm)
        self.a_w_head = build_mlp(d_s, [], d_u, activation, batchnorm)
        self._zero_final_layer()

    def _zero_final_layer(self) -> None:
        final = self.a_w_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x_std: Tensor) -> Tensor:
        tokens = self.tokenizer(x_std)      # (n, m, d_token)
        pooled = tokens.mean(dim=1)          # (n, d_token)
        s = self.rho(pooled)                 # (n, d_s)
        return self.a_w_head(s)              # (n, d_u)
