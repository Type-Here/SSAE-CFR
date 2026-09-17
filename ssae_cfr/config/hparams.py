"""SSAE-CFR training hyperparameters (model config, not dataset column roles).

- Dataset *column roles* live in the repo-root "config.py";
- model/training *hyperparameters* live here.

Usage
-----
    from ssae_cfr.config import load_config
    cfg = load_config("ssae_cfr/config/ihdp.yaml")   # YAML overrides defaults
    cfg = load_config(None, model_variant="M1")       # defaults + kwarg overrides

`TrainConfig` is a single flat dataclass for one training run, on the architecture
described in CLAUDE.md's "v3 architecture" section: an unfiltered empirical host (encoder
-> u -> decoder) plus two optional, additive corrections - U (structural, from P_U x_std)
and W (semantic, from per-feature value x semantics) - fused into `u_shared` (where the
MMD acts) and `u_out` (what the causal heads read). An earlier architecture routed the
prior through an admission gate in covariate space instead (a learned per-covariate gate
deciding how much of `(I - P_U) x` to admit into a single encoder input, with a preference
penalty and a warm-up schedule pushing against it); that mechanism does not exist in this
codebase, which is why no `b_mode`, `gamma_pref`, `gamma_warmup` or `gating_hidden` field
appears below.

Field notes
-----------
- `d_u` is the empirical bottleneck - the size of the empirical code and of the U/W
  corrections added to it, since they are summed (spec section 5.5). Named for the code
  itself (`u`), not for a generic "k".
- `k_U` is the rank of the covariate-space projector `P_U`; `r_W_rank` the rank of the
  semantic-space projector `W_r`, independent of `k_U`. `k_U = None` means "choose
  automatically" via `choose_k_svd` using the `energy_threshold` / `retention_floor` /
  `protected` knobs below; `r_W_rank = None` defaults to `k_U` - the first benchmark
  default, not a claim that the two ranks should always match (spec section 4.3).
- `r_U`, `r_W` are the reliability constants (spec section 6) that scale the U and W
  corrections before they are added to the empirical code. Fixed, not learned, in this
  version (`models.reliability.FixedReliability`); a later patient- or feature-level
  estimator replaces the interface behind these two numbers without touching this
  dataclass.
- `model_variant` is the position on the experiment ladder (spec section 12): M0 = host
  only, M1 = +U, M2 = +W, M3 = +U+W. It is convenience sugar over `use_u_adapter` /
  `use_w_adapter`; see `_resolve_variant` for exactly how the three interact.
- `l1_target`: `"u"` (default) is the empirical code the L1 acts on - named for what it
  actually touches, since the code the model calls `u` also feeds the causal path through
  every correction added on top of it; `"mu"` is its deterministic mean.

A per-dataset YAML need only list the fields it changes; everything else falls back to
the default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import yaml

# Defined here rather than in the modules that consume them: `config` is the leaf every
# other package imports, so this is the one direction that cannot produce a cycle, and it
# keeps the accepted values and the field that carries them in the same file.
NOISE_SCALES = ("absolute", "relative")
MODEL_VARIANTS = ("M0", "M1", "M2", "M3")

# model_variant -> (use_u_adapter, use_w_adapter)
_VARIANT_BRANCHES = {
    "M0": (False, False),
    "M1": (True, False),
    "M2": (False, True),
    "M3": (True, True),
}


@dataclass
class TrainConfig:
    """Flat hyperparameter bundle for one SSAE-CFR training run."""

    # -- identity ------------------------------------------------------------
    dataset: str = "ihdp"
    seed: int = 0

    # -- experiment ladder position (spec section 12) -------------------------
    # "M0" | "M1" | "M2" | "M3". `use_u_adapter` / `use_w_adapter` are derived from this
    # unless set explicitly; see `_resolve_variant`.
    model_variant: str = "M0"
    use_u_adapter: Optional[bool] = None
    use_w_adapter: Optional[bool] = None
    # Reliability constants (spec section 6): how much of each correction is added to the
    # empirical code. Forced to 0.0 when the corresponding branch is off, by
    # `_resolve_variant` - the config-level half of invariant I6 (r=0 leaves the
    # empirical path untouched).
    r_U: float = 1.0
    r_W: float = 1.0

    # -- architecture (dims; ``m`` is wired at build time) ---------------------
    d_u: int = 32                            # the empirical bottleneck
    encoder_hidden: Sequence[int] = (64,)    # m -> 64 -> d_u
    decoder_hidden: Sequence[int] = (64,)    # d_u -> 64 -> m
    head_hidden: Sequence[int] = (32,)       # u_out -> 32 -> 1   (h0, h1)
    activation: str = "elu"
    batchnorm: bool = False

    # -- adapter capacities -----------------------------------------------------
    u_adapter_hidden: Sequence[int] = (32,)  # A_U: P_U x_std -> ... -> d_u
    d_token: int = 32                        # per-feature token width for phi
    phi_hidden: Sequence[int] = (32,)        # phi: [x_ij, q~_j, x_ij*q~_j] -> ... -> d_token
    d_s: int = 32                            # pooled-set summary width
    rho_hidden: Sequence[int] = (32,)        # rho: d_token -> ... -> d_s

    # -- prior ranks (independent, spec section 4.3) ----------------------------
    k_U: Optional[int] = None                # rank of P_U; None => choose_k_svd (auto)
    r_W_rank: Optional[int] = None           # rank of W_r; None => equal to k_U (first default only)
    energy_threshold: float = 0.90           # explained-variance fraction
    retention_floor: float = 0.5             # min diag(P_U) for protected covariates
    protected: Sequence[int] = ()            # indices of clinically critical covariates
    k_min: int = 2
    k_max: Optional[int] = None              # None => m - 1 (never a pass-through)
    embedding_model: str = "BioMistral"      # LLM used for covariate embeddings

    # -- loss weights (spec section 8.2) -----------------------------------------
    alpha_mmd: float = 1.0
    lambda_rec: float = 1.0
    # SET AT 0.0, NOT SETTLED (spec section 8.2 requires it re-selected on M0 and then
    # frozen across M0-M3; that selection has not happened yet - see CLAUDE.md's open
    # questions).
    beta_l1: float = 0.0
    l1_target: str = "u"                     # "u" (default, the empirical code) or "mu"

    # -- noise / SMD modulator (held at the M0 setting, spec section 8.3) --------
    noise_dist: str = "gaussian"             # gaussian; laplace variants are experimental
    alpha_smd: float = 2.0                   # omega = tanh(alpha_smd * SMD)
    # "absolute" is z = mu + omega*eps, which the encoder escapes by inflating the
    # signal; "relative" scales the noise by ||mu||_detached/sqrt(d_u) so the
    # noise-to-signal ratio is omega whatever the scale of mu. Default "absolute" to
    # keep metrics comparable across runs; the noise-inflation issue is real and known,
    # and is audited separately after the semantic core is isolated.
    noise_scale: str = "absolute"            # "absolute" | "relative"

    # -- schedule / optimisation --------------------------------------------------
    epochs: int = 300
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: Optional[int] = None         # None => full-batch (IHDP/ACTG175)

    # -- early stopping -------------------------------------------------------------
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
        self.u_adapter_hidden = tuple(self.u_adapter_hidden)
        self.phi_hidden = tuple(self.phi_hidden)
        self.rho_hidden = tuple(self.rho_hidden)
        self.protected = tuple(self.protected)
        self._resolve_variant()
        self.validate()

    def _resolve_variant(self) -> None:
        """Reconcile `model_variant` with `use_u_adapter` / `use_w_adapter`.

        `model_variant` is convenience sugar: M0/M1/M2/M3 name a point on the experiment
        ladder (spec section 12) and imply which branches are wired in. If the two
        adapter flags are left at their `None` sentinel, they are derived from
        `model_variant`. If either was set explicitly, it must agree with what
        `model_variant` implies - a silent override would let a config claim "M2" while
        actually running M1, which is exactly the ambiguity this dataclass exists to
        prevent. After this method runs, both flags are plain bools (never None).

        The final step - forcing `r_U`/`r_W` to 0.0 when the matching branch is off - is
        what makes invariant I6 (r=0 removes augmentation) hold at the config level: a
        caller cannot accidentally ship a config with the U branch disabled but
        `r_U != 0`, which would make "disabled" a matter of an adapter that happens to be
        `None` rather than a property of the numbers actually multiplying it.
        """
        if self.model_variant not in MODEL_VARIANTS:
            raise ValueError(
                f"model_variant must be one of {MODEL_VARIANTS}; got {self.model_variant!r}"
            )
        implied_u, implied_w = _VARIANT_BRANCHES[self.model_variant]

        if self.use_u_adapter is None:
            self.use_u_adapter = implied_u
        elif self.use_u_adapter != implied_u:
            raise ValueError(
                f"use_u_adapter={self.use_u_adapter!r} conflicts with "
                f"model_variant={self.model_variant!r} (implies use_u_adapter={implied_u!r})"
            )

        if self.use_w_adapter is None:
            self.use_w_adapter = implied_w
        elif self.use_w_adapter != implied_w:
            raise ValueError(
                f"use_w_adapter={self.use_w_adapter!r} conflicts with "
                f"model_variant={self.model_variant!r} (implies use_w_adapter={implied_w!r})"
            )

        if not self.use_u_adapter:
            self.r_U = 0.0
        if not self.use_w_adapter:
            self.r_W = 0.0

    def validate(self) -> None:
        if self.model_variant not in MODEL_VARIANTS:
            raise ValueError(f"model_variant must be one of {MODEL_VARIANTS}; got {self.model_variant!r}")
        if self.l1_target not in ("u", "mu"):
            raise ValueError(f"l1_target must be 'u' or 'mu'; got {self.l1_target!r}")
        if self.noise_dist not in ("gaussian", "laplace", "laplace_directional"):
            raise ValueError(f"unknown noise_dist {self.noise_dist!r}")
        if self.noise_scale not in NOISE_SCALES:
            raise ValueError(f"noise_scale must be one of {NOISE_SCALES}; got {self.noise_scale!r}")
        if self.optimizer not in ("adam", "adamw", "sgd"):
            raise ValueError(f"unknown optimizer {self.optimizer!r}")
        if not (0.0 < self.energy_threshold <= 1.0):
            raise ValueError("energy_threshold must be in (0, 1]")
        if self.d_u < 1:
            raise ValueError("d_u must be >= 1")
        if self.k_U is not None and self.k_U < 1:
            raise ValueError("k_U must be >= 1 or None (auto)")
        if self.r_W_rank is not None and self.r_W_rank < 1:
            raise ValueError("r_W_rank must be >= 1 or None (defaults to k_U)")
        if not (0.0 <= self.r_U <= 1.0):
            raise ValueError(f"r_U must be in [0, 1]; got {self.r_U}")
        if not (0.0 <= self.r_W <= 1.0):
            raise ValueError(f"r_W must be in [0, 1]; got {self.r_W}")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
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

    Precedence (low -> high): dataclass defaults < YAML file < ``**overrides``.
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
