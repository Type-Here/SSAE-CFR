"""SLR-CFR v0.5 - Sparse Latent Representation for Counterfactual Regression.

A wide empirical encoder (latent width = the FeatureCard embedding width) fused by
addition with a frozen, deterministic semantic vector p = (x_std @ Z) / sqrt(m).
No autoencoder, no decoder, no reconstruction, no SVD, no projectors, no graph, no
guidance loss, no attention, no learned gate, and no trainable parameter anywhere in
the prior path.

    from slr_v0_4 import SLRCFRv05, load_config
    from slr_v0_4.prior import load_feature_embeddings
"""

from .config import SLRConfig, load_config
from .model import SLRCFRv05

__all__ = ["SLRConfig", "load_config", "SLRCFRv05"]
