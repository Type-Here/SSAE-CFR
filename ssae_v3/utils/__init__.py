"""Utilities: standardization, splitting, the SMD noise modulator, and metrics.

`standardize` (train-fit median-impute + z-score); `split` (treatment-stratified
train/val/test, standardizer fit on train only); `smd` (the detached noise
modulator); `metrics` (PEHE, eps_ATE, policy risk, E-value, SMD reduction).
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
from .smd import omega_from_smd, smd_per_covariate
from .split import Splits, split_and_standardize, split_dataset, train_val_test_indices, treated_fraction
from .standardize import Standardizer, standardize_dataset

__all__ = [
    "Standardizer",
    "standardize_dataset",
    "Splits",
    "split_dataset",
    "split_and_standardize",
    "train_val_test_indices",
    "treated_fraction",
    "omega_from_smd",
    "smd_per_covariate",
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
