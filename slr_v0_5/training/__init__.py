"""Training and evaluation for SLR-CFR v0.5: one realization at a time, and the sweep."""

from .runner import (
    CONTROL_SEED_BASE,
    DEFAULT_VAL_FRACTION,
    aggregate,
    build_model,
    factual_objective,
    fit,
    run_realization,
    score_split,
    standardized_splits,
)

__all__ = [
    "CONTROL_SEED_BASE",
    "DEFAULT_VAL_FRACTION",
    "aggregate",
    "build_model",
    "factual_objective",
    "fit",
    "run_realization",
    "score_split",
    "standardized_splits",
]
