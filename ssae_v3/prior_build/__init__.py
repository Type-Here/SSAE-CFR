"""Semantic prior build pipeline: covariate descriptions -> V -> prior bundle.

`embeddings` builds V offline (on the university machine); `projector` turns V into
the frozen prior bundle (P_U, W_r, Q, Q_tilde). Outputs are cached per dataset under
`artifacts/`. No torch/transformers import happens at package import time.
"""

from .descriptions import (
    descriptions_for,
    emit_gloss_template,
    load_glosses,
    load_prompt_template,
    missing_glosses,
)
from .embeddings import (
    build_embeddings,
    cache_embeddings,
    load_embeddings,
    placeholder_embeddings,
)
from .projector import (
    PriorBundle,
    build_prior_bundle,
    build_projector,
    choose_k_svd,
    cosine_similarities,
    load_bundle,
    retention,
    save_bundle,
    svd_energy,
)

__all__ = [
    "build_embeddings",
    "cache_embeddings",
    "load_embeddings",
    "placeholder_embeddings",
    "PriorBundle",
    "build_prior_bundle",
    "build_projector",
    "cosine_similarities",
    "choose_k_svd",
    "retention",
    "svd_energy",
    "load_bundle",
    "save_bundle",
    "descriptions_for",
    "emit_gloss_template",
    "load_glosses",
    "load_prompt_template",
    "missing_glosses",
]
