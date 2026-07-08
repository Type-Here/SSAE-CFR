"""Utilities: covariate standardization, SMD modulator, metrics, schedules.

`standardize` (train-fit median-impute + z-score, applied before the prior
projection); `smd` (the detached noise modulator); `metrics` (PEHE, eps_ATE, policy
risk, E-value, SMD reduction); `schedules` (alignment warm-up).
"""

from .schedules import gamma_warmup
from .smd import omega_from_smd, smd_per_covariate
from .standardize import Standardizer, standardize_dataset

__all__ = [
    "Standardizer",
    "standardize_dataset",
    "omega_from_smd",
    "smd_per_covariate",
    "gamma_warmup",
]