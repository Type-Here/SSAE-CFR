""" SSAE-CFR training hyperparameters

This is separate from the repo-root `config.py`.
That file holds per-dataset column roles (which column is treatment/outcome/dropped);
this holds model and training hyperparameters (layer sizes, loss weights, schedules).
Keeping them apart avoids conflating "what the data is" with "how the model is trained".

Usage
-----
    from ssae_cfr.config import load_config
    cfg = load_config("ssae_cfr/config/ihdp.yaml")   # YAML overrides defaults
    cfg = load_config(None, k_latent=16)             # defaults + kwarg overrides

Defaults here.
A per-dataset YAML need only list the fields it changes;
everything else falls back to the default.

Note:
 - `k_latent` is the encoder bottleneck;
 - `k_svd` is the rank of the prior projector ``P_U``.
They are independent.
`k_svd = None` means "choose automatically" via `choose_k_svd` using the
`energy_threshold` / `retention_floor` / `protected` knobs here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import yaml

# Defined here rather than in the modules that consume them: `config` is the leaf every
# other package imports, so this is the one direction that cannot produce a cycle, and it
# keeps the accepted values and the field that carries them in the same file.
B_MODES = ("learned", "one", "zero")
NOISE_SCALES = ("absolute", "relative")


@dataclass
class TrainConfig:
    """Flat hyperparameter bundle for one SSAE-CFR training run."""

    # -- identity ----------------------------------------------------------
    dataset: str = "ihdp"
    seed: int = 0

    # -- architecture (dims; ``m`` and the trailing k_latent are wired at build
    #    time, so hidden layers only list the interior widths) ---------------
    k_latent: int = 32                       # encoder bottleneck
    encoder_hidden: Sequence[int] = (64,)    # m -> 64 -> k_latent
    decoder_hidden: Sequence[int] = (64,)    # k_latent -> 64 -> m
    head_hidden: Sequence[int] = (32,)       # z -> 32 -> 1   (h0, h1)
    gating_hidden: Sequence[int] = (32,)     # x -> 32 -> m   (the admission gate b)
    activation: str = "elu"
    batchnorm: bool = False
    # "learned" gates the residual per covariate and per patient. The two constants are
    # the model's own ablations, on the same architecture and the same weights: "one"
    # admits the whole residual, so x_mod == x and this is the model WITHOUT a prior;
    # "zero" admits none of it, so x_mod == P_U x, the prior-only model.
    b_mode: str = "learned"                  # "learned" | "one" | "zero"

    # -- prior / P_U rank selection (choose_k_svd) -------------
    k_svd: Optional[int] = None              # None => choose automatically
    energy_threshold: float = 0.90           # explained-variance fraction
    retention_floor: float = 0.5             # min diag(P_U) for protected covariates
    protected: Sequence[int] = ()            # indices of clinically critical covariates
    k_svd_min: int = 2
    k_svd_max: Optional[int] = None          # None => m - 1 (never a pass-through)
    embedding_model: str = "BioMistral"      # LLM used for covariate embeddings

    # -- loss weights --------------------------------------------
    alpha_mmd: float = 1.0
    # `lambda_rec` is no longer a regularization detail. The decoder rebuilds the whole
    # x from a code built out of the admitted part of it, so L_rec pushes `b` toward 1
    # while L_pref pushes it toward 0: the ratio of these two weights is the mechanism
    # that decides the gate. Tuning it is its own step, on the new architecture.
    lambda_rec: float = 1.0
    # SET AT 0.0, NOT SETTLED. The L1 "sparsity" for now disabled. 
    # Raised to 1e-1 it collapses the representation (||z_prior||
    # 5.83 -> 1.37, SMD reduction 0.375 -> 0.148, on 0/20 realizations) - the
    # balance-by-collapse, not concept separation. Concept separation needs an
    # overcomplete dictionary; k_latent = 32 against m = 25 is a bottleneck, where an L1
    # can only discard coordinates, not separate them.
    beta_l1: float = 0.0
    # The preference for documented explanations, as a bounded penalty on the admission
    # gate: L_pref = mean_j b_j, which since b >= 0 is ||b||_1 / m. It replaces
    # gamma_align, whose penalty on ||mu_res||^2 was unbounded and, measured, deleted the
    # residual branch rather than down-weighting it.
    gamma_pref: float = 1.0                  # max, reached after warm-up
    l1_target: str = "z"                     # "z" (default) or "mu"

    # -- noise / SMD modulator -----------------------------------
    noise_dist: str = "gaussian"             # v1 = gaussian; laplace variants are v2
    alpha_smd: float = 2.0                   # omega = tanh(alpha_smd * SMD)
    # "absolute" is z = mu + omega*eps, which the encoder escapes by inflating the
    # signal; "relative" scales the noise by ||mu||_detached/sqrt(k_latent) so the
    # noise-to-signal ratio is omega whatever the scale of mu. Default "absolute"
    # to keep metrics comparable to previous version
    noise_scale: str = "absolute"            # "absolute" | "relative"

    # -- schedule / optimisation ---------------------------------
    gamma_warmup: int = 30                   # epochs, linear 0 -> gamma_pref
    epochs: int = 300
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: Optional[int] = None         # None => full-batch (IHDP/ACTG175)

    # -- early stopping ------------------------------------------
    # patience = 0 disables it, which is the default: turning it on changes every
    # number the harness produces, so it has to be an explicit choice. Monitored on
    # the validation split's normalized factual objective - the only criterion that
    # needs no oracle. Needs a validation split; without one it is silently inert.
    patience: int = 0                        # epochs without improvement before stopping
    min_delta: float = 0.0                   # improvement smaller than this doesn't count
    es_check_every: int = 5                  # epochs between validation evaluations

    def __post_init__(self) -> None:
        # tuple-ify sequence fields so a config is safely hashable/immutable-ish
        self.encoder_hidden = tuple(self.encoder_hidden)
        self.decoder_hidden = tuple(self.decoder_hidden)
        self.head_hidden = tuple(self.head_hidden)
        self.gating_hidden = tuple(self.gating_hidden)
        self.protected = tuple(self.protected)
        self.validate()

    def validate(self) -> None:
        if self.l1_target not in ("z", "mu"):
            raise ValueError(f"l1_target must be 'z' or 'mu'; got {self.l1_target!r}")
        if self.noise_dist not in ("gaussian", "laplace", "laplace_directional"):
            raise ValueError(f"unknown noise_dist {self.noise_dist!r}")
        if self.noise_scale not in NOISE_SCALES:
            raise ValueError(f"noise_scale must be one of {NOISE_SCALES}; got {self.noise_scale!r}")
        if self.b_mode not in B_MODES:
            raise ValueError(f"b_mode must be one of {B_MODES}; got {self.b_mode!r}")
        if self.optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError(f"unknown optimizer {self.optimizer!r}")
        if not (0.0 < self.energy_threshold <= 1.0):
            raise ValueError("energy_threshold must be in (0, 1]")
        if self.k_latent < 1:
            raise ValueError("k_latent must be >= 1")
        if self.k_svd is not None and self.k_svd < 1:
            raise ValueError("k_svd must be >= 1 or None (auto)")
        if self.gamma_warmup < 0 or self.epochs < 1:
            raise ValueError("gamma_warmup >= 0 and epochs >= 1 required")
        if self.patience < 0:
            raise ValueError("patience must be >= 0 (0 disables early stopping)")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be >= 0")
        if self.es_check_every < 1:
            raise ValueError("es_check_every must be >= 1")

    # -- (de)serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        return cls(**dict(data))


def load_config(
    path: Optional[Union[str, Path]] = None,
    **overrides: Any,
) -> TrainConfig:
    """Load a :class:`TrainConfig` from a YAML file, then apply keyword overrides.

    Precedence (low → high): dataclass defaults < YAML file < ``**overrides``.
    ``path=None`` yields the plain defaults (optionally overridden by kwargs).
    """
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
        data.update(loaded)
    data.update(overrides)
    return TrainConfig.from_dict(data)
