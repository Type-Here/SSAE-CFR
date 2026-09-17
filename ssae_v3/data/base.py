"""Dataset interface for SSAE-CFR.

The single container every adapter (`ihdp.py`, `actg175.py`, `mimic.py`) produces:
the boundary between where the numbers come from (raw CSV/parquet + per-dataset
column roles from the repo-root `config.py`) and what the model consumes (plain
numpy arrays with a fixed set of roles).

No column names are hard-coded here; adapters read roles from `config.py` and pass
them to `Dataset.from_frame`. Covariate set follows the vendored convention
`X = all columns - drop_cols`. Standardization is deliberately not done here - `x`
must be standardized before the `P_U` projection (which lives in R^m, one axis per
covariate), and that fit/transform belongs to the split/training pipeline
(`utils/`), so a `Dataset` always holds covariates in whatever scale the adapter
handed it and records the fact via `standardized`. Oracle counterfactuals
(`mu0`/`mu1`) are optional - only IHDP carries them; downstream code gates
PEHE/eps_ATE on `has_oracle`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """A CATE-estimation dataset in the form SSAE-CFR consumes.

    name: short identifier (e.g. "ihdp"), used for artifact/cache keys.
    x: covariate matrix, shape (n, m), float.
    t: binary treatment indicator, shape (n,), values in {0, 1}.
    yf: factual (observed) outcome, shape (n,); continuous or binary per outcome_type.
    feature_names: length-m column names of x, in order (must line up with the LLM
        gloss for each covariate, and with SMD diagnostics).
    mu0, mu1: oracle potential-outcome means E[Y(0)|x] / E[Y(1)|x], shape (n,) each,
        when known (semisynthetic data only); None for real observational subsets.
        True ITE is mu1 - mu0 (see tau_true).
    outcome_type: "continuous" or "binary"; selects MSE vs BCE for the factual loss.
    standardized: whether x has already been standardized (False on raw load; the
        split pipeline flips it after fitting train statistics).
    """

    name: str
    x: np.ndarray
    t: np.ndarray
    yf: np.ndarray
    feature_names: Sequence[str]
    mu0: Optional[np.ndarray] = None
    mu1: Optional[np.ndarray] = None
    outcome_type: str = "continuous"
    standardized: bool = False

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.t = np.asarray(self.t).reshape(-1)
        self.yf = np.asarray(self.yf, dtype=np.float64).reshape(-1)
        self.feature_names = list(self.feature_names)

        if self.x.ndim != 2:
            raise ValueError(f"x must be 2-D (n, m); got shape {self.x.shape}")
        n, m = self.x.shape
        if self.t.shape[0] != n:
            raise ValueError(f"t has {self.t.shape[0]} rows, x has {n}")
        if self.yf.shape[0] != n:
            raise ValueError(f"yf has {self.yf.shape[0]} rows, x has {n}")
        if len(self.feature_names) != m:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries, x has {m} columns"
            )

        t_vals = set(np.unique(self.t).tolist())
        if not t_vals <= {0, 1}:
            raise ValueError(f"t must be binary in {{0, 1}}; got values {sorted(t_vals)}")
        self.t = self.t.astype(np.int64)

        if self.outcome_type not in ("continuous", "binary"):
            raise ValueError(
                f"outcome_type must be 'continuous' or 'binary'; got {self.outcome_type!r}"
            )

        if (self.mu0 is None) != (self.mu1 is None):
            raise ValueError("mu0 and mu1 must be provided together or not at all")
        if self.mu0 is not None:
            self.mu0 = np.asarray(self.mu0, dtype=np.float64).reshape(-1)
            self.mu1 = np.asarray(self.mu1, dtype=np.float64).reshape(-1)
            if self.mu0.shape[0] != n or self.mu1.shape[0] != n:
                raise ValueError("mu0/mu1 must have one entry per unit")

    # -- convenience views -------------------------------------------------

    @property
    def n(self) -> int:
        """Number of units."""
        return self.x.shape[0]

    @property
    def m(self) -> int:
        """Number of covariates (the dimension P_U and the SSAE act in)."""
        return self.x.shape[1]

    @property
    def has_oracle(self) -> bool:
        """True when true counterfactuals (mu0/mu1) are available."""
        return self.mu0 is not None

    @property
    def tau_true(self) -> Optional[np.ndarray]:
        """Oracle ITE mu1 - mu0 when known, else None.

        Identically zero for the ACTG175 pseudo-obs sharp null (the bias tool
        relabels treatment without regenerating Y), which is what makes
        PEHE-against-zero valid there.
        """
        if not self.has_oracle:
            return None
        return self.mu1 - self.mu0

    def subset(self, idx: np.ndarray) -> "Dataset":
        """A new Dataset holding only rows in idx (used by the train/test split).

        Metadata (name, feature_names, outcome_type, standardized) carries over
        unchanged, so a split keeps pointing at the covariate order P_U was built for.
        """
        idx = np.asarray(idx)
        return Dataset(
            name=self.name,
            x=self.x[idx],
            t=self.t[idx],
            yf=self.yf[idx],
            feature_names=list(self.feature_names),
            mu0=None if self.mu0 is None else self.mu0[idx],
            mu1=None if self.mu1 is None else self.mu1[idx],
            outcome_type=self.outcome_type,
            standardized=self.standardized,
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        *,
        name: str,
        treatment_col: str,
        outcome_col: str,
        drop_cols: Sequence[str] = (),
        feature_cols: Optional[Sequence[str]] = None,
        mu0_col: Optional[str] = None,
        mu1_col: Optional[str] = None,
        outcome_type: str = "continuous",
    ) -> "Dataset":
        """Build a Dataset from a dataframe plus explicit column roles.

        Covariate set: feature_cols if given, else all columns minus drop_cols and
        the treatment/outcome/mu0/mu1 columns. config.py's drop_cols lists already
        include ids, timestamps, treatment/outcome and known leakage columns, so the
        default branch usually suffices; feature_cols is an explicit-whitelist
        escape hatch. mu0_col/mu1_col name the oracle columns when present (IHDP).
        """
        for col in (treatment_col, outcome_col):
            if col not in df.columns:
                raise KeyError(f"column {col!r} not found in dataframe")

        if feature_cols is None:
            excluded = set(drop_cols) | {treatment_col, outcome_col}
            excluded |= {c for c in (mu0_col, mu1_col) if c is not None}
            feature_cols = [c for c in df.columns if c not in excluded]
        else:
            feature_cols = list(feature_cols)
        if not feature_cols:
            raise ValueError("no covariate columns left after applying drop_cols")

        x = df[feature_cols].to_numpy(dtype=np.float64)
        t = df[treatment_col].to_numpy()
        yf = df[outcome_col].to_numpy(dtype=np.float64)

        mu0 = df[mu0_col].to_numpy(dtype=np.float64) if mu0_col is not None else None
        mu1 = df[mu1_col].to_numpy(dtype=np.float64) if mu1_col is not None else None

        return cls(
            name=name,
            x=x,
            t=t,
            yf=yf,
            feature_names=feature_cols,
            mu0=mu0,
            mu1=mu1,
            outcome_type=outcome_type,
        )
