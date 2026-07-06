"""MIMIC adapters - real observational subsets (no oracle counterfactuals).

Column roles come from the repo-root `config.py`:

- diur_v1 (`DIUR_V1`): the primary real observational subset, good overlap.
  Treatment `treat_early`, outcome `y_28d_mort_inhosp`.
- sepsis_v2 (`SEPSIS_V2`): the hard case (confounding by indication). Treatment
  `treat_steroid`, outcome `y_28d_mort_inhosp`.

Both outcomes are binary and there is no ground-truth effect, so evaluation relies
on surrogate metrics (SMD reduction, policy risk, E-value) rather than PEHE.

These subsets have mixed-type covariates (Decimal-typed labs, `gender`/`race`
strings, datetime columns), so they are routed through `encode_features`: numeric
coercion, datetime/id-like dropping, and one-hot encoding of categoricals. Numeric
gaps stay as NaN and are imputed by the train-fit standardizer downstream.
"""

from __future__ import annotations

import pandas as pd

from .base import Dataset
from .preprocessing import encode_features
from .roles import baseline_config, resolve_data_path

DIUR_CONFIG = "DIUR_V1"
SEPSIS_CONFIG = "SEPSIS_V2"

# Steroid dosing measured in the 0-24h treatment window: these describe the treatment
# itself (treat_steroid), so they are post-treatment leakage. The raw roles do not
# drop them, and their names contain "id" only by coincidence ("stero-id"), so the
# id-like rule must not be relied on to remove them.
SEPSIS_LEAKAGE = ("steroid_events_0_24h", "steroid_unparsed_0_24h", "steroid_parsed_0_24h")


def _load_mimic(
    config_name: str,
    dataset_name: str,
    extra_drop: tuple = (),
) -> Dataset:
    """Load a MIMIC subset using the column roles from `config.py`."""
    cfg = baseline_config(config_name)
    df = pd.read_parquet(resolve_data_path(cfg))
    encoded, report = encode_features(
        df,
        treatment_col=cfg.treatment_col,
        outcome_col=cfg.outcome_col,
        drop_cols=list(cfg.drop_cols or ()),
        extra_drop=extra_drop,
    )
    return Dataset.from_frame(
        encoded,
        name=dataset_name,
        treatment_col=cfg.treatment_col,
        outcome_col=cfg.outcome_col,
        feature_cols=report.feature_names,
        outcome_type="binary",
    )


def load_diur_v1() -> Dataset:
    """Load MIMIC diur_v1 (`DIUR_V1`)."""
    return _load_mimic(DIUR_CONFIG, "diur_v1")


def load_sepsis_v2() -> Dataset:
    """Load MIMIC sepsis_v2 (`SEPSIS_V2`)."""
    return _load_mimic(SEPSIS_CONFIG, "sepsis_v2", extra_drop=SEPSIS_LEAKAGE)