"""Dataset interface for SSAE-CFR.

This module defines the single container every dataset adapter (`ihdp.py`,
`actg175.py`, `mimic.py`) produces.
It is the boundary between:
 - *where the numbers come from* (raw CSV/parquet + per-dataset column roles)
 - *what the model consumes* (plain numpy arrays with a fixed set of roles).

Design notes
------------
- No column names are hard-coded here. The roles (which column is the
  treatment, the outcome, which columns to drop, where the oracle `mu0`/`mu1`
  live) come from `config.py` at the repo root (`BaselineConfig`).
- Adapters read those roles and pass them into :meth:`Dataset.from_frame`;
  the covariate set follows the vendored convention `X = all columns − drop_cols`.
- Standardization is deliberately *not* done here.
  `x` is required to be standardized before the `P_U` projection,
  because `P_U` lives in `R^m` with one axis per covariate and the projection
  geometry is only meaningful when covariates are comparably scaled.
  That fit/transform pair belongs to the split/training pipeline
  (`utils/`), not to this immutable container, so a :class:`Dataset` always
  holds covariates in whatever scale the adapter handed it and records the fact
  via :attr:`standardized`.
* Oracle counterfactuals are optional.
  For now, only IHDP carries the true response (`mu0`/`mu1`);
  real observational subsets do not.
  Downstream code gates PEHE/eps_ATE on :attr:`has_oracle`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    """A CATE-estimation dataset in the form SSAE-CFR consumes.

    Attributes
    ----------
    name:
        Short dataset identifier (e.g. "ihdp", "aids_v1_biased"). Used
        for artifact/cache keys (the per-dataset P_U cache is keyed on it).
    x:
        Covariate matrix, shape (n, m), float. One row per unit, one column
        per covariate, aligned with :attr:`feature_names`.
    t:
        Binary treatment indicator, shape (n,), values in {0, 1}.
    yf:
        Factual (observed) outcome, shape (n,). Continuous or binary
        depending on :attr:`outcome_type`.
    feature_names:
        Length-m list naming each column of :attr:`x`, in order. Needed by
        the prior module (the LLM description of covariate j must line up
        with column j) and by SMD diagnostics.
    mu0, mu1:
        Oracle potential-outcome means E[Y(0)|x] / E[Y(1)|x],
        shape (n,) each, when known (semisynthetic data only).
        None: for real observational subsets.
        The true ITE is mu1 - mu0 (see :attr:`tau_true`).
    outcome_type:
        "continuous" or "binary": selects MSE vs BCE for the factual
        loss and the metric set at evaluation time.
    standardized:
        Whether :attr:`x` has already been standardized. "False" on raw load;
        the pipeline flips it after fitting train statistics.
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
        """Number of covariates (the dimension ``P_U`` and the SSAE act in)."""
        return self.x.shape[1]

    @property
    def has_oracle(self) -> bool:
        """True when true counterfactuals (``mu0``/``mu1``) are available."""
        return self.mu0 is not None

    @property
    def tau_true(self) -> Optional[np.ndarray]:
        """Oracle ITE ``mu1 - mu0`` when known, else ``None``.

        For the ACTG175 pseudo-obs sharp null this is identically zero (the bias
        tool relabels treatment without regenerating Y), which is what makes
        PEHE-against-zero valid there — see Implementation_plan §2.
        """
        if not self.has_oracle:
            return None
        return self.mu1 - self.mu0

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
        """Build a :class:`Dataset` from a dataframe plus explicit column roles.

        The roles are passed in by the caller (an adapter reading them from the
        repo-root "config.py"); nothing about specific datasets is baked in
        here. The covariate set follows the vendored convention:

            X = feature_cols  if given, else
            X = all columns − drop_cols − {treatment, outcome, mu0, mu1}.

        Note that the "drop_cols" lists in "config.py" already include ids,
        timestamps, the treatment and outcome, and known leakage columns, so the
        default branch usually suffices; "feature_cols" is the escape hatch for
        a dataset that needs an explicit whitelist.

        Parameters mirror the attributes of :class:`Dataset`. "mu0_col" and
        "mu1_col" name the oracle columns when present (IHDP); omit them for
        real data.
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
