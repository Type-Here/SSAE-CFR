"""IHDP adapter - the PEHE ground-truth benchmark.

IHDP is semi-synthetic: the outcome is generated from a known response surface, so
it ships the oracle potential-outcome means `mu0`/`mu1` and PEHE / eps_ATE are
computable. It is not described in the repo-root `config.py` (that file covers only
the MIMIC/ACTG subsets), so the column roles are named here:

    treatment = `treatment`, outcome = `y_factual`, oracle = `mu0` / `mu1`.

The raw file also carries `y_cfactual` (the counterfactual factual value); it is
dropped from the covariates. Covariates are `x1..x25` (m = 25), outcome continuous.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from .base import Dataset
from .roles import repo_root

DEFAULT_PATH = "data/raw/IHDP/ihdp.csv"


def load_ihdp(path: Union[str, Path, None] = None) -> Dataset:
    """Load IHDP into a `Dataset` carrying the oracle `mu0`/`mu1`.

    `path` defaults to `data/raw/IHDP/ihdp.csv` resolved against the repo root.
    """
    csv_path = Path(path) if path is not None else repo_root() / DEFAULT_PATH
    df = pd.read_csv(csv_path)
    return Dataset.from_frame(
        df,
        name="ihdp",
        treatment_col="treatment",
        outcome_col="y_factual",
        drop_cols=["y_cfactual"],
        mu0_col="mu0",
        mu1_col="mu1",
        outcome_type="continuous",
    )