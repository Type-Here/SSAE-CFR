"""Tests for the train/val/test split and the no-leakage standardization order.

Three properties matter. The splits must be a genuine partition that preserves the
treated fraction (the datasets are imbalanced enough that an unstratified draw can starve
a split's treated arm). The standardizer must be fit on train only: the held-out splits
get the train statistics, so their own columns are *not* forced to zero mean. And the
validation split must be optional, so the plain train/test protocol still works.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssae_cfr.data.base import Dataset
from ssae_cfr.utils.split import (
    split_and_standardize,
    split_dataset,
    train_val_test_indices,
    treated_fraction,
)


def _dataset(n: int = 400, m: int = 6, p_treat: float = 0.2, seed: int = 0) -> Dataset:
    """An imbalanced synthetic dataset carrying an oracle, on a deliberately odd scale."""
    rng = np.random.default_rng(seed)
    t = (rng.random(n) < p_treat).astype(np.int64)
    x = rng.standard_normal((n, m)) * 5.0 + 10.0  # far from standardized
    mu0 = x[:, 0]
    mu1 = mu0 + 2.0
    yf = np.where(t == 1, mu1, mu0) + rng.standard_normal(n) * 0.1
    return Dataset(
        name="synthetic",
        x=x,
        t=t,
        yf=yf,
        feature_names=[f"x{j}" for j in range(m)],
        mu0=mu0,
        mu1=mu1,
    )


def test_indices_partition_the_rows():
    ds = _dataset()
    train_idx, _, test_idx = train_val_test_indices(ds.t, test_size=0.25, seed=0)
    assert set(train_idx) | set(test_idx) == set(range(ds.n))
    assert set(train_idx) & set(test_idx) == set()


def test_test_size_is_respected():
    ds = _dataset(n=1000)
    _, _, test_idx = train_val_test_indices(ds.t, test_size=0.3, seed=0)
    assert abs(test_idx.size / ds.n - 0.3) < 0.02


def test_stratification_preserves_the_treated_fraction():
    ds = _dataset(n=1000, p_treat=0.15)
    train, _, test = split_dataset(ds, test_size=0.25, seed=0)
    assert treated_fraction(train) == pytest.approx(treated_fraction(ds), abs=0.02)
    assert treated_fraction(test) == pytest.approx(treated_fraction(ds), abs=0.02)


def test_both_arms_survive_in_a_small_test_split():
    """A thin treated arm must still put at least one unit on each side."""
    ds = _dataset(n=60, p_treat=0.08)
    train, _, test = split_dataset(ds, test_size=0.2, seed=3)
    for split in (train, test):
        assert (split.t == 1).sum() >= 1
        assert (split.t == 0).sum() >= 1


def test_split_is_reproducible_and_seed_dependent():
    ds = _dataset()
    a, _, _ = train_val_test_indices(ds.t, seed=0)
    b, _, _ = train_val_test_indices(ds.t, seed=0)
    c, _, _ = train_val_test_indices(ds.t, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_subset_carries_the_oracle_and_metadata():
    ds = _dataset()
    train, _, test = split_dataset(ds, test_size=0.25, seed=0)
    for split in (train, test):
        assert split.has_oracle
        assert split.tau_true == pytest.approx(2.0, abs=1e-9)
        assert split.feature_names == list(ds.feature_names)
        assert split.name == ds.name and split.outcome_type == ds.outcome_type
    assert train.n + test.n == ds.n


def test_standardization_is_fit_on_train_only():
    ds = _dataset()
    train, _, test, _ = split_and_standardize(ds, test_size=0.25, seed=0)

    assert train.standardized and test.standardized
    assert train.x.mean(axis=0) == pytest.approx(np.zeros(ds.m), abs=1e-9)
    assert train.x.std(axis=0) == pytest.approx(np.ones(ds.m), abs=1e-9)

    # the test split is transformed, not re-fit, so it does not land exactly on zero
    assert np.abs(test.x.mean(axis=0)).max() > 1e-6
    assert np.abs(test.x.mean(axis=0)).max() < 0.5  # but it is close, same distribution


def test_standardizer_statistics_come_from_the_training_rows():
    ds = _dataset()
    train_idx, _, _ = train_val_test_indices(ds.t, test_size=0.25, seed=0)
    standardizer = split_and_standardize(ds, test_size=0.25, seed=0).standardizer
    assert standardizer.mean_ == pytest.approx(ds.x[train_idx].mean(axis=0))
    assert standardizer.scale_ == pytest.approx(ds.x[train_idx].std(axis=0))


def test_rejects_a_degenerate_test_size():
    ds = _dataset()
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            split_dataset(ds, test_size=bad)


# -- the optional validation split -----------------------------------------


def test_no_validation_split_by_default():
    ds = _dataset()
    assert split_dataset(ds, test_size=0.25)[1] is None
    assert split_and_standardize(ds, test_size=0.25).val is None


def test_three_way_split_partitions_the_rows():
    ds = _dataset(n=1000)
    train_idx, val_idx, test_idx = train_val_test_indices(
        ds.t, test_size=0.2, val_size=0.2, seed=0
    )
    assert set(train_idx) | set(val_idx) | set(test_idx) == set(range(ds.n))
    for a, b in ((train_idx, val_idx), (train_idx, test_idx), (val_idx, test_idx)):
        assert set(a) & set(b) == set()


def test_both_held_out_sizes_are_fractions_of_the_whole():
    ds = _dataset(n=1000)
    _, val_idx, test_idx = train_val_test_indices(ds.t, test_size=0.2, val_size=0.15, seed=0)
    assert abs(val_idx.size / ds.n - 0.15) < 0.02
    assert abs(test_idx.size / ds.n - 0.20) < 0.02


def test_validation_split_is_stratified_and_standardized_with_train_stats():
    ds = _dataset(n=1000, p_treat=0.15)
    splits = split_and_standardize(ds, test_size=0.2, val_size=0.2, seed=0)
    assert splits.val is not None and splits.val.standardized
    assert treated_fraction(splits.val) == pytest.approx(treated_fraction(ds), abs=0.02)
    # transformed with the train statistics, so not centred on its own mean
    assert np.abs(splits.val.x.mean(axis=0)).max() > 1e-6
    assert splits.train.n + splits.val.n + splits.test.n == ds.n


def test_rejects_held_out_sizes_that_leave_no_training_split():
    ds = _dataset()
    with pytest.raises(ValueError):
        split_dataset(ds, test_size=0.6, val_size=0.5)
    with pytest.raises(ValueError):
        split_dataset(ds, test_size=0.25, val_size=-0.1)


def test_treated_fraction_of_an_empty_split_is_none():
    ds = _dataset()
    assert treated_fraction(ds.subset(np.array([], dtype=int))) is None