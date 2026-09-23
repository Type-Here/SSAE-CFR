"""Parameter-free cosine cross-attention over the frozen FeatureCard embeddings.

The patient queries the same frozen Z the direct module uses, and the resulting
weights reweight that patient's covariate values:

    q_i       = mu_emp_i, or (x_std_i @ Z)/sqrt(m) under query="p_direct"
    q_hat_i   = q_i / (||q_i|| + eps)
    k_hat_j   = z_j      / (||z_j||      + eps)
    alpha_i   = softmax_j( (q_hat_i . k_hat_j) / tau )
    p_att_i   = ( (x_std_i * (m * alpha_i)) @ Z ) / sqrt(m)

There are no W_Q / W_K / W_V, no attention MLP, no output projection and no gate: the
module owns zero trainable parameters and `Z` is a buffer. Under `query="mu_emp"` the
query is the learned representation and is deliberately not detached, so the causal
objective can shape it into a better query without a second trainable prior network.
Under `query="p_direct"` the query is the patient's own direct semantic vector, which
depends only on `x` and `Z`: the whole prior path is then fixed given the data, and no
gradient reaches the encoder through it. The two answer different questions - whether
a *learned* query helps, and whether attention helps at all. The query is normalized
before scoring, so the `1/sqrt(m)` in `p_direct` cannot change the weights.

The `m * alpha` factor makes the direct module an exact special case: at uniform
attention `m * alpha_ij = 1` and `p_att` is `(x_std @ Z) / sqrt(m)` - the v0.5 prior.
Attention is therefore an adaptive generalization of direct fusion, not a different
architecture.

Values are the original centered rows of `Z`; the L2-normalized rows are used only to
compute the score. Because cosine similarity is scale-invariant, a global rescaling of
`Z` (the norm-matched control) leaves the attention weights untouched and changes only
the value magnitude.

`norm_match` rescales each patient's semantic vector to the magnitude direct fusion
would have given that patient:

    p* = p_att * ||p_direct|| / (||p_att|| + eps)

Without it, temperature is not a clean knob: concentrating `m * alpha` on fewer
features grows `||p_att||`, so a sharper attention is also a louder prior (measured at
eta=0.5: the prior-to-empirical norm ratio runs 1.19 at tau=1 to 2.09 at tau=0.02) and
a temperature comparison would be reading injection strength again. With it, tau
changes only *which* features the patient's prior is built from. The scale is a
per-patient constant times a differentiable norm, so gradients still reach the
encoder, and at uniform attention the factor is exactly 1 - the direct-fusion identity
survives.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor

from .base import PriorIntegration

_EPS = 1e-8

# What the patient queries the frozen cards with. "mu_emp" is the learned empirical
# representation; "p_direct" is the direct semantic vector (x @ Z)/sqrt(m), which
# depends on nothing trainable - an ablation that removes the learned query entirely.
QUERIES = ("mu_emp", "p_direct")


class ParameterFreeCrossAttentionIntegration(PriorIntegration):
    """Cosine attention from mu_emp over the frozen FeatureCard embeddings."""

    def __init__(
        self,
        Z: Tensor,
        temperature: float = 1.0,
        norm_match: bool = False,
        query: str = "mu_emp",
    ) -> None:
        super().__init__()
        if query not in QUERIES:
            raise ValueError(f"query must be one of {QUERIES}; got {query!r}")
        if Z.dim() != 2:
            raise ValueError(f"Z must be 2-D (m, d_q); got shape {tuple(Z.shape)}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        Z = Z.detach().to(dtype=torch.float32).clone()
        self.register_buffer("Z", Z)
        # The normalized keys are a function of the frozen values, so they are cached
        # once rather than recomputed per batch. The values stay the unnormalized rows.
        self.register_buffer("K_hat", Z / (Z.norm(dim=-1, keepdim=True) + _EPS))
        self.m = int(Z.shape[0])
        self.d_q = int(Z.shape[1])
        self.temperature = float(temperature)
        self.norm_match = bool(norm_match)
        self.query = query
        self.sqrt_m = math.sqrt(self.m)

    def attention(self, q: Tensor) -> Tensor:
        """Attention weights alpha, shape (batch, m), rows summing to 1.

        `q` is whatever the configured query is, already in R^{d_q}; it is normalized
        here, so its scale never reaches the weights.
        """
        q_hat = q / (q.norm(dim=-1, keepdim=True) + _EPS)
        scores = q_hat @ self.K_hat.T
        return torch.softmax(scores / self.temperature, dim=-1)

    def forward(self, x_std: Tensor, mu_emp: Tensor) -> Tuple[Tensor, Dict[str, object]]:
        """Return (p_att, diagnostics)."""
        if x_std.shape[-1] != self.m:
            raise ValueError(f"x has {x_std.shape[-1]} covariates, Z has {self.m} rows")
        if mu_emp.shape[-1] != self.d_q:
            raise ValueError(f"mu_emp has width {mu_emp.shape[-1]}, Z has width {self.d_q}")

        p_direct = (x_std @ self.Z) / self.sqrt_m
        alpha = self.attention(mu_emp if self.query == "mu_emp" else p_direct)
        p = ((x_std * (self.m * alpha)) @ self.Z) / self.sqrt_m

        scale = None
        if self.norm_match:
            scale = p_direct.norm(dim=-1, keepdim=True) / (p.norm(dim=-1, keepdim=True) + _EPS)
            p = p * scale
        return p, self._diagnostics(alpha, p, scale)

    def _diagnostics(self, alpha: Tensor, p: Tensor, scale) -> Dict[str, object]:
        """Descriptive attention statistics. Never added to the objective.

        `alpha` itself is returned so a caller that knows the treatment assignment can
        break the weights down by arm; this module has no notion of treatment.
        """
        with torch.no_grad():
            a = alpha.detach()
            entropy = -(a * torch.log(a + _EPS)).sum(dim=-1)
            top3 = a.topk(min(3, self.m), dim=-1).values.sum(dim=-1)
            return {
                "p_norm": float(p.detach().norm(dim=-1).mean()),
                "attention_entropy_mean": float(entropy.mean()),
                "attention_entropy_median": float(entropy.median()),
                "attention_entropy_normalized": float(entropy.mean() / math.log(self.m)),
                "effective_feature_count_mean": float(torch.exp(entropy).mean()),
                "attention_max_weight_mean": float(a.max(dim=-1).values.mean()),
                "attention_top3_mass_mean": float(top3.mean()),
                "attention_temperature": self.temperature,
                "attention_norm_match": float(self.norm_match),
                "attention_query_is_mu_emp": float(self.query == "mu_emp"),
                "attention_norm_match_scale_mean": (
                    float(scale.detach().mean()) if scale is not None else 1.0
                ),
                "alpha": alpha,
            }
