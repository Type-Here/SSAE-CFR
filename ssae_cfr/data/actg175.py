"""ACTG175 adapters - RCT reference and pseudo-observational stress test.

Two variants, both reading their column roles from the repo-root `config.py`:

- RCT (`AIDS_V1`): the clean randomized trial. Binary outcome `label`, treatment
  `treat`. No oracle counterfactuals.
- Pseudo-obs variant A (`AIDS_V1_BIASED`): a sharp null. The bias tool reassigns the
  observed treatment from the covariates but does not regenerate the outcome, so the
  true effect of the relabeled treatment is zero for every unit (tau(x) = 0 for all
  x). We encode that by attaching `mu0 = mu1 = 0`, which makes `tau_true` identically
  zero and PEHE-against-zero a valid metric (closer to 0 = more bias removed). The
  treatment column is already renamed to `treat` by the upstream harmonization.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .base import Dataset
from .roles import baseline_config, resolve_data_path

RCT_CONFIG = "AIDS_V1"
PSEUDO_OBS_CONFIG = "AIDS_V1_BIASED"


def _load_actg175(config_name: str, dataset_name: str) -> Dataset:
    """Load an ACTG175 variant using the column roles from `config.py`."""
    cfg = baseline_config(config_name)
    df = pd.read_parquet(resolve_data_path(cfg))
    return Dataset.from_frame(
        df,
        name=dataset_name,
        treatment_col=cfg.treatment_col,
        outcome_col=cfg.outcome_col,
        drop_cols=list(cfg.drop_cols or ()),
        outcome_type="binary",
    )


def load_actg175_rct() -> Dataset:
    """Load the ACTG175 randomized reference (`AIDS_V1`)."""
    return _load_actg175(RCT_CONFIG, "aids_v1")


def load_actg175_pseudo_obs() -> Dataset:
    """Load ACTG175 pseudo-obs variant A (`AIDS_V1_BIASED`) as a sharp null.

    Attaches `mu0 = mu1 = 0` so `tau_true` is identically zero, enabling
    PEHE-against-zero (see the module docstring).
    """
    ds = _load_actg175(PSEUDO_OBS_CONFIG, "aids_v1_biased")
    zeros = np.zeros(ds.n, dtype=np.float64)
    return dataclasses.replace(ds, mu0=zeros, mu1=zeros)