"""SSAE-CFR training hyperparameters (model config, not dataset column roles).

- Dataset column roles live in the repo-root "config.py";
- model/training hyperparameters live here.

Usage
-----
    from ssae_v3.hparams import load_config
    cfg = load_config("ssae_v3/config/ihdp.yaml")   # YAML overrides defaults
    cfg = load_config(None, model_variant="u_adapter")  # defaults + kwarg overrides

`DefaultConfig` is a single flat dataclass for one training run: an unfiltered empirical
host (encoder -> u -> decoder) plus two optional, additive corrections - U (structural,
from U_k^T x_std) and W (semantic, from per-feature value x semantics) - fused into
`u_shared` (where the MMD acts) and `u_out` (what the causal heads read).

Field notes
-----------
- `d_u` is the empirical bottleneck - the size of the empirical code and of the U/W
  corrections added to it, since they are summed.
- `k_U` is the rank of the covariate-space basis `U_k` (equivalently of the projector
  `P_U = U_k U_k^T`, which the U branch no longer forms); `r_W_rank` the rank of the
  semantic-space projector `W_r`, independent of `k_U`. `k_U = None` means "choose
  automatically" via `choose_k_svd` using the `energy_threshold` / `retention_floor` /
  `protected` knobs below; `r_W_rank = None` defaults to `k_U` (first benchmark default,
  not a claim that the two ranks should match).
- `r_U`, `r_W` are the reliability constants that scale the U and W corrections before
  they are added to the empirical code. Fixed, not learned, in this version. A branch
  that is off forces its constant to 0.0; switching `model_variant` on an existing
  config (e.g. `dataclasses.replace`) does NOT restore it, and a live branch left at 0
  is inert - its correction is multiplied by zero, so no gradient reaches the adapter
  and its zero-initialized final layer never moves. Set them explicitly when deriving
  one variant's config from another's.
- `model_variant` is the position on the experiment ladder. It is convenience sugar over
  `use_u_adapter` / `use_w_adapter` / `use_expert_adapter`; see `_resolve_variant`.
- The W adapter and the expert-graph adapter fill the same slot (the semantic correction
  added to `u_shared`) and are mutually exclusive. The expert adapter has no reliability
  constant: `u_out = u_shared + a_expert`. W stays available as a control arm.
- `l1_target`: `"u"` (default) is the empirical code the L1 acts on; `"mu"` is its
  deterministic mean.

A per-dataset YAML need only list the fields it changes; everything else falls back to
the default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import yaml

# Defined here rather than in the modules that consume them: `hparams` is the leaf every
# other package imports, so this is the one direction that cannot produce a cycle.
NOISE_TYPES = ("gaussian", "laplace")
# Which fixed prior projectors guide the encoder input. Read by the guidance module
# as well, the same way core.noise reads the noise vocabularies from here.
PRIOR_MODES = ("none", "embedding", "graph", "both")
NOISE_SCALES = ("absolute", "relative")
MODEL_VARIANTS = (
    "empirical",         # host only
    "u_adapter",         # host + U
    "w_adapter",         # host + W
    "u_w_adapter",       # host + U + W
    "expert_adapter",    # host + expert graph
    "u_expert_adapter",  # host + U + expert graph
)

# model_variant -> (use_u_adapter, use_w_adapter, use_expert_adapter). W and the expert
# adapter occupy the same architectural slot, so no variant enables both.
_VARIANT_BRANCHES = {
    "empirical": (False, False, False),
    "u_adapter": (True, False, False),
    "w_adapter": (False, True, False),
    "u_w_adapter": (True, True, False),
    "expert_adapter": (False, False, True),
    "u_expert_adapter": (True, False, True),
}


@dataclass
class DefaultConfig:
    """Flat hyperparameter bundle for one SSAE-CFR training run."""

    # -- identity ------------------------------------------------------------
    dataset: str = "ihdp"
    in_channels: int = 25  # covariate count m; must match the dataset, model validates
    seed: int = 42

    # -- experiment ladder position -------------------------------------------
    model_variant: str = "empirical"
    use_u_adapter: Optional[bool] = None
    use_w_adapter: Optional[bool] = None
    use_expert_adapter: Optional[bool] = None
    # How much of each correction is added to the empirical code. Forced to 0.0 when the
    # corresponding branch is off, by `_resolve_variant`.
    r_U: float = 1.0
    r_W: float = 1.0

    # -- architecture -----------------------------------------------------------
    d_u: int = 32                            # the empirical bottleneck
    encoder_hidden: Sequence[int] = (64,)    # m -> 64 -> d_u
    decoder_hidden: Sequence[int] = (64,)    # d_u -> 64 -> m
    head_hidden: Sequence[int] = (32,)       # u_out -> 32 -> 1   (h0, h1)
    activation: str = "relu"
    batchnorm: bool = False

    # -- adapter capacities -----------------------------------------------------
    u_adapter_hidden: Sequence[int] = (32,)  # A_U: U_k^T x_std -> ... -> d_u
    d_token: int = 32                        # per-feature token width for phi
    phi_hidden: Sequence[int] = (32,)        # phi: [x_ij, q~_j, x_ij*q~_j] -> ... -> d_token
    d_s: int = 32                            # pooled-set summary width
    rho_hidden: Sequence[int] = (32,)        # rho: d_token -> ... -> d_s
    # Ablation only, NOT part of the frozen core: makes the per-feature semantic table a
    # learned embedding initialized at whatever q~ the run was given, instead of the
    # frozen offline one. Adds m * r_W trainable parameters, so an arm using it is not
    # capacity-matched to one that does not.
    w_semantics_trainable: bool = False
    # Expert-graph adapter. It has no reliability constant: a_expert is added to
    # u_shared directly.
    d_edge: int = 16                         # per-edge representation width
    expert_rho_hidden: Sequence[int] = (32,)  # rho: K*d_edge -> ... -> d_u

    # -- prior ranks (independent) ----------------------------------------------
    k_U: Optional[int] = None                # rank of U_k; None => choose_k_svd (auto)
    r_W_rank: Optional[int] = None           # rank of W_r; None => equal to k_U (first default only)
    energy_threshold: float = 0.90           # explained-variance fraction
    retention_floor: float = 0.5             # min diag(P_U) for protected covariates
    protected: Sequence[int] = ()            # indices of clinically critical covariates
    k_min: int = 2
    k_max: Optional[int] = None              # None => m - 1 (never a pass-through)
    embedding_model: str = "BioMistral"      # LLM used for covariate embeddings

    # -- prior guidance (fixed projectors, zero trainable parameters) -------------
    # `prior_mode` picks which prior subspaces guide the encoder input: the embedding
    # subspace, the expert-graph subspace, both (their union span), or neither. The two
    # strengths are fixed for a run - `gamma_prior` scales the input correction,
    # `lambda_prior` weights the first-layer guidance loss. Guidance and the v3 adapters
    # are mutually exclusive: an arm running both is a control for neither.
    prior_mode: str = "none"
    gamma_prior: float = 0.0
    lambda_prior: float = 0.0

    # -- loss weights -------------------------------------------------------------
    alpha_mmd: float = 1.0
    lambda_rec: float = 1.0
    # NOT SETTLED: must be re-selected on the empirical variant and then frozen across
    # the whole ladder; that selection has not happened yet.
    beta_l1: float = 0.0
    l1_target: str = "u"                     # "u" (default, the empirical code) or "mu"

    # -- noise / SMD modulator (held at the empirical-variant setting) -----------
    # On by default: the SMD-driven noise is what every existing number was produced
    # under, and with it off the MMD sees deterministic codes - the collapse regime the
    # stochastic branch exists to defeat. False is the ablation switch, not the baseline.
    is_noise_active: bool = False             # True => add noise to the empirical code
    noise_dist: str = "gaussian"             # "gaussian" | "laplace"
    alpha_smd: float = 2.0                   # omega = tanh(alpha_smd * SMD)
    # "absolute" (z = mu + omega*eps) is what every existing number was produced under;
    # "relative" scales noise by ||mu||_detached/sqrt(d_u) so the noise-to-signal ratio is
    # omega regardless of the code's scale. Kept "absolute" for comparability; the
    # noise-inflation issue is real and known, and is audited separately.
    noise_scale: str = "absolute"            # "absolute" | "relative"

    # -- schedule / optimisation --------------------------------------------------
    epochs: int = 300
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: Optional[int] = None         # None => full-batch (IHDP/ACTG175)

    # -- early stopping -------------------------------------------------------------
    # patience = 0 disables it (the default): monitored on the validation split's
    # normalized factual objective, the only criterion that needs no oracle. Needs a
    # validation split; without one it is silently inert.
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
        self.expert_rho_hidden = tuple(self.expert_rho_hidden)
        self.protected = tuple(self.protected)
        self._resolve_variant()
        self.validate()

    def _resolve_variant(self) -> None:
        """Reconcile `model_variant` with `use_u_adapter` / `use_w_adapter`.

        `model_variant` is convenience sugar for which branches are wired in. If the two
        adapter flags are left at their `None` sentinel, they are derived from
        `model_variant`; if either was set explicitly, it must agree with what
        `model_variant` implies. After this method runs, both flags are plain bools.

        Also forces `r_U`/`r_W` to 0.0 when the matching branch is off, so a disabled
        branch cannot leave a nonzero reliability behind.
        """
        if self.model_variant not in MODEL_VARIANTS:
            raise ValueError(
                f"model_variant must be one of {MODEL_VARIANTS}; got {self.model_variant!r}"
            )
        implied_u, implied_w, implied_expert = _VARIANT_BRANCHES[self.model_variant]

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

        if self.use_expert_adapter is None:
            self.use_expert_adapter = implied_expert
        elif self.use_expert_adapter != implied_expert:
            raise ValueError(
                f"use_expert_adapter={self.use_expert_adapter!r} conflicts with "
                f"model_variant={self.model_variant!r} (implies "
                f"use_expert_adapter={implied_expert!r})"
            )

        if not self.use_u_adapter:
            self.r_U = 0.0
        if not self.use_w_adapter:
            self.r_W = 0.0

    def validate(self) -> None:
        if self.model_variant not in MODEL_VARIANTS:
            raise ValueError(f"model_variant must be one of {MODEL_VARIANTS}; got {self.model_variant!r}")
        if self.use_w_adapter and self.use_expert_adapter:
            raise ValueError(
                "the W adapter and the expert adapter are mutually exclusive in v1: "
                "they occupy the same architectural slot, so an arm running both would "
                "not be a control for either"
            )
        if self.prior_mode not in PRIOR_MODES:
            raise ValueError(f"prior_mode must be one of {PRIOR_MODES}; got {self.prior_mode!r}")
        if self.prior_mode != "none" and (
            self.use_u_adapter or self.use_w_adapter or self.use_expert_adapter
        ):
            raise ValueError(
                f"prior_mode={self.prior_mode!r} guides the encoder input directly and "
                f"cannot be combined with model_variant={self.model_variant!r}; the "
                "guided arms all run on the unmodified empirical backbone"
            )
        if self.gamma_prior < 0.0:
            raise ValueError(f"gamma_prior must be >= 0; got {self.gamma_prior}")
        if self.lambda_prior < 0.0:
            raise ValueError(f"lambda_prior must be >= 0; got {self.lambda_prior}")
        if self.l1_target not in ("u", "mu"):
            raise ValueError(f"l1_target must be 'u' or 'mu'; got {self.l1_target!r}")
        if self.noise_dist not in NOISE_TYPES:
            raise ValueError(f"noise_dist must be one of {NOISE_TYPES}; got {self.noise_dist!r}")
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
    def from_dict(cls, data: Mapping[str, Any]) -> "DefaultConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        return cls(**dict(data))


def load_config(
    path: Optional[Union[str, Path]] = None,
    **overrides: Any,
) -> DefaultConfig:
    """Load a DefaultConfig from a YAML file, then apply keyword overrides.

    Precedence (low -> high): dataclass defaults < YAML file < overrides.
    path=None yields the plain defaults (optionally overridden by kwargs).
    """
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
        data.update(loaded)
    data.update(overrides)
    return DefaultConfig.from_dict(data)
