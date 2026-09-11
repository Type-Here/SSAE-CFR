"""Utilities: covariate standardization, SMD modulator, metrics, schedules.

`standardize` (train-fit median-impute + z-score, applied before the prior
projection); `smd` (the detached noise modulator); `metrics` (PEHE, eps_ATE, policy
risk, E-value, SMD reduction); `schedules` (the preference warm-up).
"""

from .metrics import (
    approximate_risk_ratio,
    approximate_risk_ratio_ci,
    e_value,
    e_value_ci,
    eps_ate,
    pehe,
    pehe_against_zero,
    policy_risk,
    policy_risk_table,
    policy_value,
    smd_reduction,
)
from .schedules import linear_warmup
from .smd import omega_from_smd, smd_per_covariate
from .standardize import Standardizer, standardize_dataset

__all__ = [
    "Standardizer",
    "standardize_dataset",
    "omega_from_smd",
    "smd_per_covariate",
    "linear_warmup",
    "pehe",
    "pehe_against_zero",
    "eps_ate",
    "policy_value",
    "policy_risk",
    "policy_risk_table",
    "e_value",
    "e_value_ci",
    "approximate_risk_ratio",
    "approximate_risk_ratio_ci",
    "smd_reduction",
]