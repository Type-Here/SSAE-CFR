"""Semantic prior: covariate descriptions -> embeddings V -> projector P_U.

`embeddings` builds V (offline, on the university machine); `projector` turns V into
the fixed analytic projector P_U = U_k U_k^T (Variant A, feature space). Both outputs
are cached per dataset in `artifacts/`.
"""

from .embeddings import (
    build_embeddings,
    cache_embeddings,
    load_embeddings,
    placeholder_embeddings,
)
from .projector import build_projector, choose_k_svd, retention, svd_energy

__all__ = [
    "build_embeddings",
    "cache_embeddings",
    "load_embeddings",
    "placeholder_embeddings",
    "build_projector",
    "choose_k_svd",
    "retention",
    "svd_energy",
]