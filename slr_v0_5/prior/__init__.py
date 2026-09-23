"""The frozen prior: artifact loading, integration mechanism, matched controls."""

from .artifacts import (
    EmbeddingArtifactError,
    FeatureEmbeddings,
    center_embeddings,
    embedding_dir_for,
    load_feature_embeddings,
)
from .base import PriorIntegration, build_prior_integration
from .controls import EMBEDDING_VARIANTS, apply_embedding_variant, permute_rows
from .cosine_cross_attention import ParameterFreeCrossAttentionIntegration
from .direct_embedding import DirectEmbeddingIntegration

__all__ = [
    "EmbeddingArtifactError",
    "FeatureEmbeddings",
    "center_embeddings",
    "embedding_dir_for",
    "load_feature_embeddings",
    "PriorIntegration",
    "build_prior_integration",
    "DirectEmbeddingIntegration",
    "ParameterFreeCrossAttentionIntegration",
    "EMBEDDING_VARIANTS",
    "apply_embedding_variant",
    "permute_rows",
]
