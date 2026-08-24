"""IHDP adapter - the PEHE ground-truth benchmark.

IHDP is semi-synthetic: the covariates and the treatment come from a real trial, but
the outcome is drawn from a known response surface, so the oracle potential-outcome
means `mu0`/`mu1` ship with the data and PEHE / eps_ATE are computable. IHDP is not
described in the repo-root `config.py` (that file covers only the MIMIC/ACTG subsets),
so the column roles are named here:

    treatment = `treatment`, outcome = `y_factual`, oracle = `mu0` / `mu1`.

The 25 covariates are 6 continuous measures and 19 binary indicators, anonymized to
`x1..x25` in the redistributed files; `prior/glosses/ihdp.yaml` carries the recovered
mapping. The outcome is continuous.

Two loaders, because IHDP is really two things:

`load_ihdp` reads the single-realization CSV. It is one draw of the response surface
over the 747 units, convenient for smoke tests and for anything that only needs a
dataset with an oracle.

`load_ihdp_realization` reads one of the 100 realizations shipped in the standard
`.npz` pair. This is what published PEHE numbers are averaged over, and the files also
fix the train/test partition (672 / 75 units), so a run on realization `r` is directly
comparable to the literature rather than to a split we chose ourselves. Both files
describe the same 747 units and the same covariates; only the simulated outcome
changes from realization to realization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .base import Dataset
from .roles import repo_root

DEFAULT_PATH = "data/raw/IHDP/ihdp.csv"
REPLICATION_PATHS = {
    "train": "data/raw/IHDP/ihdp_npci_1-100.train.npz",
    "test": "data/raw/IHDP/ihdp_npci_1-100.test.npz",
}
N_REALIZATIONS = 100
FEATURE_NAMES = [f"x{i}" for i in range(1, 26)]


def load_ihdp(path: Union[str, Path, None] = None) -> Dataset:
    """Load the single-realization IHDP CSV into a `Dataset` with oracle `mu0`/`mu1`.

    `path` defaults to `data/raw/IHDP/ihdp.csv` resolved against the repo root. That
    file is realization 1 of the replication set below, unsplit.
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


def _replication_path(split: str, path: Union[str, Path, None]) -> Path:
    if split not in REPLICATION_PATHS:
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
    return Path(path) if path is not None else repo_root() / REPLICATION_PATHS[split]


def load_ihdp_realization(
    realization: int,
    split: str = "train",
    path: Union[str, Path, None] = None,
) -> Dataset:
    """One realization of one split of the IHDP replication set.

    `realization` is 1-based, matching how the literature refers to them (1..100).
    `split` picks the partition the benchmark files already fixed: "train" (672 units)
    or "test" (75). Reusing that partition rather than redrawing one is the whole point
    - it is what makes a PEHE comparable across papers.

    The covariates are named `x1..x25` here exactly as in the CSV, so a `P_U` built for
    one loader is valid for the other.
    """
    if not 1 <= realization <= N_REALIZATIONS:
        raise ValueError(f"realization must be in 1..{N_REALIZATIONS}; got {realization}")

    with np.load(_replication_path(split, path)) as data:
        index = realization - 1
        x = np.asarray(data["x"][:, :, index], dtype=np.float64)
        t = np.asarray(data["t"][:, index])
        yf = np.asarray(data["yf"][:, index], dtype=np.float64)
        mu0 = np.asarray(data["mu0"][:, index], dtype=np.float64)
        mu1 = np.asarray(data["mu1"][:, index], dtype=np.float64)

    return Dataset(
        name="ihdp",
        x=x,
        t=t.astype(np.int64),
        yf=yf,
        feature_names=list(FEATURE_NAMES),
        mu0=mu0,
        mu1=mu1,
        outcome_type="continuous",
    )


def has_replication_set() -> bool:
    """Whether the 100-realization files are present, so callers can fail helpfully."""
    return all((repo_root() / rel).exists() for rel in REPLICATION_PATHS.values())