"""Direct embedding fusion: p = (x_std @ Z) / sqrt(m).

Z is a frozen buffer and this module has no trainable parameter. There is
deliberately no learned projection of `x @ Z`: with Z of (near) full row rank over
the m observed features, a free trainable A would make `Z @ A` an arbitrary
feature-to-latent matrix, and the prior branch would become a generic learned
mapping from x whose behaviour says nothing about the embeddings' content.

The 1/sqrt(m) factor is the only numerical scaling the base model applies. It keeps
the typical magnitude of the semantic vector from growing with the covariate count
when the method moves to another dataset.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor

from .base import PriorIntegration


class DirectEmbeddingIntegration(PriorIntegration):
    """Frozen FeatureCard embeddings fused straight into the latent space."""

    def __init__(self, Z: Tensor) -> None:
        super().__init__()
        if Z.dim() != 2:
            raise ValueError(f"Z must be 2-D (m, d_q); got shape {tuple(Z.shape)}")
        # detach + clone: `.to(dtype=...)` returns the caller's storage when the dtype
        # already matches, which would let a later edit of the artifact mutate the
        # buffer of an already-built model.
        self.register_buffer("Z", Z.detach().to(dtype=torch.float32).clone())
        self.m = int(Z.shape[0])
        self.d_q = int(Z.shape[1])
        self.sqrt_m = math.sqrt(self.m)

    def forward(self, x_std: Tensor, mu_emp: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Return (p, diagnostics). `mu_emp` is unused: the prior is deterministic in x."""
        if x_std.shape[-1] != self.m:
            raise ValueError(f"x has {x_std.shape[-1]} covariates, Z has {self.m} rows")
        # divided, not multiplied by a reciprocal, so the result is bitwise the
        # expression the specification names
        p = (x_std @ self.Z) / self.sqrt_m
        return p, {"p_norm": float(p.detach().norm(dim=-1).mean())}
