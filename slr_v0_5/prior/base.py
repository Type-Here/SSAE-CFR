"""The prior-integration interface and the factory that selects one.

A `PriorIntegration` maps a batch of standardized covariates (and, if it needs
them, the empirical means) to one semantic vector per patient, in latent space:

    p, diagnostics = integration(x_std, mu_emp)          # p: (batch, d_q)

The causal model only ever calls this interface, so a mechanism change - direct
fusion in 0.5, cosine cross-attention in 0.5.1 - drops in without touching the
encoder, the heads or the losses.

A module's second return value carries its own diagnostics. Scalar entries are picked
up by the model and reported per run; a tensor entry (attention weights, say) is
carried through the forward output for a caller that knows the treatment assignment.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, TYPE_CHECKING

from torch import Tensor, nn

from .artifacts import FeatureEmbeddings

if TYPE_CHECKING:
    from ..config import SLRConfig


class PriorIntegration(nn.Module):
    """Base class: frozen prior in, per-patient semantic vector out."""

    def forward(self, x_std: Tensor, mu_emp: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        raise NotImplementedError


def build_prior_integration(
    cfg: "SLRConfig",
    embeddings: Optional[FeatureEmbeddings],
) -> Optional[PriorIntegration]:
    """The integration module this run needs, or None when no prior is used.

    `embedding_variant="none"` is the wide-latent empirical reference: it returns
    None, so the model has no prior object at all rather than one multiplied by a
    zero coefficient. The controls are applied by the caller before this point, so
    whatever `embeddings` holds is what the arm runs on.
    """
    if cfg.embedding_variant == "none":
        return None
    if embeddings is None:
        raise ValueError(
            f"embedding_variant={cfg.embedding_variant!r} needs an embedding artifact; none was given"
        )
    if cfg.prior_integration == "direct_embedding":
        from .direct_embedding import DirectEmbeddingIntegration

        return DirectEmbeddingIntegration(embeddings.Z)
    if cfg.prior_integration == "cosine_cross_attention":
        from .cosine_cross_attention import ParameterFreeCrossAttentionIntegration

        return ParameterFreeCrossAttentionIntegration(
            embeddings.Z,
            temperature=cfg.attention_temperature,
            norm_match=cfg.attention_norm_match,
            query=cfg.attention_query,
        )
    if cfg.prior_integration == "cross_attention":
        raise NotImplementedError(
            "projected cross-attention (W_Q/W_K/W_V) is not implemented; 0.5.1 runs "
            "the parameter-free 'cosine_cross_attention'"
        )
    raise ValueError(f"unknown prior_integration {cfg.prior_integration!r}")
