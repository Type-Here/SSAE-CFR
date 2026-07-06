"""Structural feature preprocessing for adapters with mixed-type columns.

The MIMIC subsets carry columns that are not model-ready as-is: some numeric labs
are stored as `Decimal`/object, there are datetime and id-like columns, string
categoricals (`gender`, `race`), constant columns, and labs with very high missing
rates. This module turns a raw frame into an all-numeric feature matrix and reports
exactly what it did with every column, since tracking drops across the prior work was
error-prone.

Policy:

- object columns that parse as numbers (e.g. Decimal-typed labs) -> numeric;
- datetime and id-like columns -> dropped;
- numeric columns that are constant, or missing above `max_missing_rate`, -> dropped
  (imputing a mostly-absent lab is unreliable; a companion `has_<lab>` indicator, if
  present, keeps the "was it measured" signal);
- categorical columns: dropped if constant, above `max_missing_rate`, or with more
  than `max_cat_cardinality` levels (one-hotting many rare levels is dispersive and
  the sparse indicators can dominate a causal model); binary categoricals become a
  single 0/1 column; the rest are one-hot encoded.

Numeric missing values that survive the drop are left as NaN here on purpose and
imputed later by the train-fit `utils.standardize.Standardizer`, so no test
information leaks into the transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ID_LIKE_EXACT = {"id", "stay_id", "subject_id", "hadm_id"}


def _is_id_like(name: str) -> bool:
    """Token-based id detection (avoids matching `id` inside words like `steroid`)."""
    n = name.lower()
    return n in ID_LIKE_EXACT or n.endswith("_id") or n.startswith("id_")


def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s)


def _is_string_like(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(s)
    )


def _is_object_but_numeric(s: pd.Series, sample_n: int = 2000, min_parse_rate: float = 0.95) -> bool:
    """True for string/object columns whose values parse as numbers (e.g. Decimals)."""
    if not _is_string_like(s):
        return False
    values = s.dropna()
    if len(values) == 0:
        return False
    values = values.sample(n=min(sample_n, len(values)), random_state=0)
    return pd.to_numeric(values, errors="coerce").notna().mean() >= min_parse_rate


@dataclass
class FeatureReport:
    """How each column was routed, for logging and reproducibility.

    `dropped` maps a column name to the reason it was dropped, so the full column
    accounting is inspectable in one place.
    """

    num_cols: List[str]
    binary_cols: List[str]
    onehot_cols: List[str]
    dropped: Dict[str, str]
    feature_names: List[str]


def _as_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.float64)
    return pd.to_numeric(s, errors="coerce").astype(np.float64)


def encode_features(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    drop_cols: Sequence[str] = (),
    extra_drop: Sequence[str] = (),
    max_missing_rate: float = 0.6,
    max_cat_cardinality: int = 15,
) -> Tuple[pd.DataFrame, FeatureReport]:
    """Return an all-numeric frame (features + treatment + outcome) and a report.

    `extra_drop` lists dataset-specific leakage columns the raw roles miss (e.g.
    treatment-window dosing columns). See the module docstring for the policy.
    """
    exclude = set(drop_cols) | set(extra_drop) | {treatment_col, outcome_col}
    dropped: Dict[str, str] = {c: "role/leakage drop" for c in exclude if c in df.columns}

    numeric_series: Dict[str, pd.Series] = {}
    categorical_series: Dict[str, pd.Series] = {}

    for c in df.columns:
        if c in exclude:
            continue
        if _is_id_like(c):
            dropped[c] = "id-like"
            continue
        s = df[c]
        if _is_object_but_numeric(s):
            numeric_series[c] = _as_numeric(s)
        elif _is_datetime(s):
            dropped[c] = "datetime"
        elif _is_string_like(s):
            categorical_series[c] = s
        else:
            numeric_series[c] = _as_numeric(s)

    num_cols: List[str] = []
    binary_cols: List[str] = []
    onehot_cols: List[str] = []
    feature_names: List[str] = []
    frames: List[pd.DataFrame] = []

    # Numeric columns: drop constant or too-sparse, keep the rest (NaN imputed later).
    for c, s in numeric_series.items():
        miss = float(s.isna().mean())
        if miss > max_missing_rate:
            dropped[c] = f"missing {miss:.2f} > {max_missing_rate}"
            continue
        if s.nunique(dropna=True) <= 1:
            dropped[c] = "constant"
            continue
        num_cols.append(c)
        frames.append(s.rename(c).to_frame())
        feature_names.append(c)

    # Categorical columns: cardinality-gated one-hot; binary -> single 0/1.
    for c, s in categorical_series.items():
        miss = float(s.isna().mean())
        if miss > max_missing_rate:
            dropped[c] = f"missing {miss:.2f} > {max_missing_rate}"
            continue
        card = int(s.nunique(dropna=True))
        if card <= 1:
            dropped[c] = "constant"
            continue
        if card > max_cat_cardinality:
            dropped[c] = f"cardinality {card} > {max_cat_cardinality}"
            continue
        if card == 2:
            positive = sorted(s.dropna().unique().tolist())[-1]
            name = f"{c}_{positive}"
            col = (s == positive).astype(np.float64)
            col[s.isna()] = np.nan
            binary_cols.append(c)
            frames.append(col.rename(name).to_frame())
            feature_names.append(name)
        else:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(s.astype(object).to_frame())
            names = list(encoder.get_feature_names_out([c]))
            onehot_cols.append(c)
            frames.append(pd.DataFrame(encoded, columns=names, index=df.index))
            feature_names.extend(names)

    out = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)
    out = out[feature_names]
    out[treatment_col] = df[treatment_col].to_numpy()
    out[outcome_col] = df[outcome_col].to_numpy()

    report = FeatureReport(
        num_cols=num_cols,
        binary_cols=binary_cols,
        onehot_cols=onehot_cols,
        dropped=dropped,
        feature_names=feature_names,
    )
    return out, report
