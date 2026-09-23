"""SLR-CFR v0.5 run configuration.

One flat dataclass per training run. The YAML is flat too: a per-dataset file lists
only the fields it changes and everything else falls back to the default.

    from slr_v0_4.config import load_config
    cfg = load_config("slr_v0_4/configs/ihdp.yaml", eta_prior=0.1)

`d_latent` is left None here and filled in from the embedding artifact by the
training runner, because the latent width is the embedding width (spec section 5)
and hardcoding a model-specific dimension would let the two drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import yaml

# The variant names live with the module that applies them, so the two cannot drift.
from .prior.controls import EMBEDDING_VARIANTS
from .prior.cosine_cross_attention import QUERIES as ATTENTION_QUERIES

MODEL_VERSION = "0.5.1"

# How the frozen prior enters the latent space. "cosine_cross_attention" is the
# parameter-free patient-specific variant added in 0.5.1; "cross_attention" stays
# reserved for a projected (W_Q/W_K/W_V) version and is rejected at model build time,
# not silently ignored.
PRIOR_INTEGRATIONS = ("direct_embedding", "cosine_cross_attention", "cross_attention")

# EMBEDDING_VARIANTS (imported above): which feature-to-embedding assignment an arm
# uses. "none" is the wide-latent empirical reference - no prior tensor is built at all.

NOISE_TYPES = ("gaussian", "laplace")
NOISE_SCALES = ("absolute", "relative")


@dataclass
class SLRConfig:
    """Hyperparameters for one SLR-CFR v0.5 run."""

    # -- identity -------------------------------------------------------------
    model_version: str = MODEL_VERSION
    dataset: str = "ihdp"
    in_channels: int = 25          # covariate count m
    seed: int = 42

    # -- architecture ---------------------------------------------------------
    # None => taken from the embedding artifact (d_latent = d_q).
    d_latent: Optional[int] = None
    encoder_hidden: Sequence[int] = (64,)   # m -> 64 -> d_latent
    head_hidden: Sequence[int] = (32,)      # u_out -> 32 -> 1  (h0, h1)
    activation: str = "relu"
    batchnorm: bool = False

    # -- prior ----------------------------------------------------------------
    prior_integration: str = "direct_embedding"
    embedding_variant: str = "real"
    eta_prior: float = 0.1                  # fixed, never trainable
    # Softmax temperature of the cosine attention. Fixed at 1.0 for the first 0.5.1
    # comparison and never selected on PEHE or true CATE; inert for direct fusion.
    attention_temperature: float = 1.0
    # Rescale each patient's attention prior to the magnitude direct fusion would give
    # it. Off by default (the 0.5.1 plan's base definition); on, it decouples the
    # temperature from the injected magnitude. Recorded per run.
    attention_norm_match: bool = False
    # What the attention queries with: the learned representation, or the patient's own
    # direct semantic vector (an ablation with no learned query at all).
    attention_query: str = "mu_emp"
    # Subtract the mean FeatureCard vector from every row of Z before it is used. Off
    # by default: the base model consumes the stored artifact untouched. On, it is
    # recorded in every run row and printed with the sweep, never applied silently.
    center_embeddings: bool = False

    # -- loss -----------------------------------------------------------------
    alpha_mmd: float = 3.0
    beta_sparse: float = 0.0

    # -- noise (empirical branch only; off for the first semantic experiment) --
    noise_enabled: bool = False
    noise_dist: str = "gaussian"
    noise_scale: str = "absolute"
    alpha_smd: float = 2.0                  # omega = tanh(alpha_smd * SMD)

    # -- optimisation ---------------------------------------------------------
    epochs: int = 400
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: Optional[int] = None        # None => full batch

    # -- early stopping (on the oracle-free validation factual objective) ------
    patience: int = 3                       # 0 disables it
    min_delta: float = 0.0
    es_check_every: int = 5

    def __post_init__(self) -> None:
        self.encoder_hidden = tuple(self.encoder_hidden)
        self.head_hidden = tuple(self.head_hidden)
        self.validate()

    def validate(self) -> None:
        if self.prior_integration not in PRIOR_INTEGRATIONS:
            raise ValueError(
                f"prior_integration must be one of {PRIOR_INTEGRATIONS}; got {self.prior_integration!r}"
            )
        if self.embedding_variant not in EMBEDDING_VARIANTS:
            raise ValueError(
                f"embedding_variant must be one of {EMBEDDING_VARIANTS}; got {self.embedding_variant!r}"
            )
        if self.eta_prior < 0.0:
            raise ValueError(f"eta_prior must be >= 0; got {self.eta_prior}")
        if self.attention_query not in ATTENTION_QUERIES:
            raise ValueError(
                f"attention_query must be one of {ATTENTION_QUERIES}; got {self.attention_query!r}"
            )
        if self.attention_temperature <= 0.0:
            raise ValueError(f"attention_temperature must be > 0; got {self.attention_temperature}")
        if self.embedding_variant == "none" and self.eta_prior != 0.0:
            raise ValueError(
                f"embedding_variant='none' carries no prior tensor, so eta_prior must be 0; "
                f"got {self.eta_prior}"
            )
        if self.noise_dist not in NOISE_TYPES:
            raise ValueError(f"noise_dist must be one of {NOISE_TYPES}; got {self.noise_dist!r}")
        if self.noise_scale not in NOISE_SCALES:
            raise ValueError(f"noise_scale must be one of {NOISE_SCALES}; got {self.noise_scale!r}")
        if self.optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError(f"unknown optimizer {self.optimizer!r}")
        if self.d_latent is not None and self.d_latent < 1:
            raise ValueError("d_latent must be >= 1 or None (read from the artifact)")
        if self.alpha_mmd < 0.0 or self.beta_sparse < 0.0:
            raise ValueError("alpha_mmd and beta_sparse must be >= 0")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.patience < 0:
            raise ValueError("patience must be >= 0 (0 disables early stopping)")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be >= 0")
        if self.es_check_every < 1:
            raise ValueError("es_check_every must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SLRConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        return cls(**dict(data))


def load_config(path: Optional[Union[str, Path]] = None, **overrides: Any) -> SLRConfig:
    """Load an SLRConfig from a YAML file, then apply keyword overrides.

    Precedence (low to high): dataclass defaults < YAML file < overrides.
    """
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
        data.update(loaded)
    data.update(overrides)
    return SLRConfig.from_dict(data)
