"""Train/validation/test split, treatment-stratified, standardizer fit on train only.

Split the *raw* dataset first, then fit the `Standardizer` on the training rows and
reuse it everywhere else - standardizing before splitting would leak held-out
means/scales into the transform, and since `P_U` acts on standardized covariates
that leak would reach the projection geometry too. `split_and_standardize` does both
steps together so a caller cannot get the order wrong.

The validation split is optional (`val_size=0` gives a plain train/test pair) but is
what makes honest hyperparameter selection possible: tuning against test PEHE is
oracle peeking (PEHE needs counterfactuals no real deployment has), so selection must
run on validation with an oracle-free criterion such as the factual loss.

Splits are treatment-stratified by default: these datasets are treatment-imbalanced
(IHDP is about 19 percent treated), and an unstratified draw can leave a split with a
treated arm too thin, or empty, breaking the factual loss and every policy metric.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np

from ..data.base import Dataset
from .standardize import Standardizer, standardize_dataset


class Splits(NamedTuple):
    """Standardized splits of one dataset, plus the train-fit standardizer.

    `val` is None when no validation split was requested.
    """

    train: Dataset
    val: Optional[Dataset]
    test: Dataset
    standardizer: Standardizer


def train_val_test_indices(
    t: np.ndarray,
    test_size: float = 0.25,
    val_size: float = 0.0,
    seed: int = 0,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row indices for a (train, val, test) split, stratified on `t` by default.

    Both sizes are fractions of the whole dataset. Each arm is shuffled and cut
    independently, keeping the sample's treated fraction in every split, and each
    arm keeps at least one row in train and one in test when large enough to allow it.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1); got {test_size}")
    if not 0.0 <= val_size < 1.0:
        raise ValueError(f"val_size must be in [0, 1); got {val_size}")
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size must leave a training split; got {test_size + val_size}")

    t = np.asarray(t).reshape(-1)
    rng = np.random.default_rng(seed)

    groups = [np.where(t == v)[0] for v in (0, 1)] if stratify else [np.arange(t.shape[0])]
    train_parts, val_parts, test_parts = [], [], []
    for group in groups:
        if group.size == 0:
            continue
        shuffled = rng.permutation(group)
        n_test = int(round(test_size * group.size))
        n_val = int(round(val_size * group.size))
        if group.size > 1:
            n_test = min(max(n_test, 1), group.size - 1)
            n_val = min(n_val, group.size - n_test - 1)  # never starve the training split
        else:
            n_test = n_val = 0
        test_parts.append(shuffled[:n_test])
        val_parts.append(shuffled[n_test:n_test + n_val])
        train_parts.append(shuffled[n_test + n_val:])

    def _joined(parts) -> np.ndarray:
        return np.sort(np.concatenate(parts)) if parts else np.array([], dtype=np.int64)

    return _joined(train_parts), _joined(val_parts), _joined(test_parts)


def split_dataset(
    dataset: Dataset,
    test_size: float = 0.25,
    val_size: float = 0.0,
    seed: int = 0,
    stratify: bool = True,
) -> Tuple[Dataset, Optional[Dataset], Dataset]:
    """Split `dataset` into (train, val, test) without touching the covariate scale.

    `val` is None when `val_size` is 0.
    """
    train_idx, val_idx, test_idx = train_val_test_indices(
        dataset.t, test_size, val_size, seed, stratify
    )
    val = dataset.subset(val_idx) if val_idx.size else None
    return dataset.subset(train_idx), val, dataset.subset(test_idx)


def split_and_standardize(
    dataset: Dataset,
    test_size: float = 0.25,
    val_size: float = 0.0,
    seed: int = 0,
    stratify: bool = True,
) -> Splits:
    """Split, then standardize every split with statistics fit on the training split.

    The returned standardizer is the one to reuse for any further data scored by the
    same model.
    """
    train, val, test = split_dataset(dataset, test_size, val_size, seed, stratify)
    train_std, standardizer = standardize_dataset(train)
    val_std = None if val is None else standardize_dataset(val, standardizer)[0]
    test_std, _ = standardize_dataset(test, standardizer)
    return Splits(train=train_std, val=val_std, test=test_std, standardizer=standardizer)


def treated_fraction(dataset: Dataset) -> Optional[float]:
    """Share of treated units, or None when the dataset is empty (a split diagnostic)."""
    if dataset.n == 0:
        return None
    return float(np.mean(dataset.t == 1))
