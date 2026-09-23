"""Matched false-assignment controls for the FeatureCard embeddings.

The scientific comparison is real assignment against a matched false one, not prior
against no prior: an arm carrying a prior changes the representation pathway whatever
the prior says.

`permuted` keeps exactly the same set of Qwen vectors - and therefore the embedding
dimension, the row norms, the global anisotropy, the singular values, the pairwise
geometry and the numerical scale - and destroys only which FeatureCard is attached to
which observed covariate.

`permuted_norm_matched` is `permuted` rescaled so that the semantic vector it produces
has the same RMS magnitude as the real one on this realization's training split. A row
permutation preserves the row norms of Z but not the norm of `p = (x @ Z)/sqrt(m)`,
which depends on the assignment: measured on the centered IHDP artifact the real
assignment gives 24.6 against 26.7-31.6 across ten permuted draws, so at a shared
eta_prior the plain control injects about 16 percent more prior magnitude than the arm
it controls for. This variant removes that difference, at the cost of no longer
preserving the row norms exactly.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from .artifacts import FeatureEmbeddings

EMBEDDING_VARIANTS = ("none", "real", "permuted", "permuted_norm_matched")


def permute_rows(Z: Tensor, seed: int) -> Tuple[Tensor, Tuple[int, ...]]:
    """Row-permuted copy of Z under a private generator, plus the permutation used.

    Redraws while the permuted matrix equals the original. An identity draw is the
    obvious case, but two identical rows would also let a non-identity permutation be
    a numerical no-op, and a control that is secretly the real assignment is worse
    than no control.
    """
    m = Z.shape[0]
    if m < 2:
        raise ValueError(f"a row permutation needs at least 2 features; got {m}")
    generator = torch.Generator().manual_seed(int(seed))
    for _ in range(64):
        order = torch.randperm(m, generator=generator)
        permuted = Z[order]
        if not torch.equal(permuted, Z):
            return permuted, tuple(int(i) for i in order)
    raise ValueError(f"could not draw a non-identity row permutation for m={m} from seed {seed}")


def prior_rms(Z: Tensor, x_std: Tensor) -> float:
    """RMS magnitude of the semantic vector this Z produces on `x_std`.

    sqrt(mean_i ||p_i||^2) for p = (x_std @ Z) / sqrt(m), the quantity the norm-matched
    control equalizes across arms.
    """
    m = Z.shape[0]
    if x_std.shape[-1] != m:
        raise ValueError(f"x has {x_std.shape[-1]} covariates, Z has {m} rows")
    p = (x_std @ Z) / math.sqrt(m)
    return float(torch.sqrt((p ** 2).sum(dim=-1).mean()))


def apply_embedding_variant(
    embeddings: FeatureEmbeddings,
    variant: str,
    seed: int,
    x_std: Optional[Tensor] = None,
) -> FeatureEmbeddings:
    """The embedding artifact this arm runs on: the real one, or a matched false one.

    `none` returns the artifact untouched; the model is built without a prior module
    in that case, so the matrix is never read.

    `permuted_norm_matched` needs `x_std` - the fitting split's covariates and nothing
    else, since the scale is a property of the arm and may not be read off data the
    model is scored on.
    """
    if variant not in EMBEDDING_VARIANTS:
        raise ValueError(f"unknown embedding variant {variant!r}; choose from {EMBEDDING_VARIANTS}")
    if variant in ("none", "real"):
        return embeddings

    Z, order = permute_rows(embeddings.Z, seed)
    metadata = dict(embeddings.metadata)
    metadata.update({"embedding_variant": variant, "control_seed": int(seed), "permutation": order})

    if variant == "permuted_norm_matched":
        if x_std is None:
            raise ValueError(
                "permuted_norm_matched needs the training split covariates to compute its scale"
            )
        s_real = prior_rms(embeddings.Z, x_std)
        s_perm = prior_rms(Z, x_std)
        if not (s_perm > 0.0):
            raise ValueError("the permuted prior has zero RMS; cannot norm-match it")
        scale = s_real / s_perm
        Z = Z * scale
        metadata.update({"norm_match_scale": scale, "prior_rms_real": s_real, "prior_rms_permuted": s_perm})

    return dataclasses.replace(embeddings, Z=Z, metadata=metadata)
