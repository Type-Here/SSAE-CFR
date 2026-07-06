"""Covariate standardization - fit on train, reuse the same stats on test.

The prior projector `P_U` acts in covariate space with one axis per covariate, so
its geometry is only meaningful when covariates are comparably scaled. Standardizing
per column (zero mean, unit variance) makes the projection and the encoder input
well-conditioned.

The statistics must be fit on the training split only and reused unchanged on
validation/test, otherwise test information leaks into the transform. `Standardizer`
keeps that discipline explicit: `fit` on train `x`, then `transform` any split.

It also fills missing values: numeric gaps (left as NaN by the adapters) are imputed
with the per-column train median before z-scoring. Imputing from train medians only
keeps the same no-leakage guarantee.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np

from ..data.base import Dataset


class Standardizer:
    """Per-column median-impute then z-score, with train-fit statistics.

    Columns with zero variance in the training split are left unscaled (their scale
    is set to 1) so constant covariates pass through instead of producing NaNs.
    Columns that are entirely NaN in the training split get a median of 0.
    """

    def __init__(self) -> None:
        self.median_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    @property
    def fitted(self) -> bool:
        return self.mean_ is not None

    def _impute(self, x: np.ndarray) -> np.ndarray:
        rows, cols = np.where(np.isnan(x))
        if rows.size:
            x = x.copy()
            x[rows, cols] = np.take(self.median_, cols)
        return x

    def fit(self, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=np.float64)
        median = np.nanmedian(x, axis=0)
        self.median_ = np.where(np.isnan(median), 0.0, median)
        x_imp = self._impute(x)
        self.mean_ = x_imp.mean(axis=0)
        scale = x_imp.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Standardizer.transform called before fit")
        x = self._impute(np.asarray(x, dtype=np.float64))
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def standardize_dataset(
    ds: Dataset,
    standardizer: Optional[Standardizer] = None,
) -> Tuple[Dataset, Standardizer]:
    """Return a standardized copy of `ds` plus the `Standardizer` used.

    If `standardizer` is None a fresh one is fit on `ds.x` (single-split
    convenience, e.g. for shape/stat checks). To respect the train/test boundary,
    fit a `Standardizer` on the training `Dataset` and pass it in when standardizing
    the test `Dataset`.
    """
    if standardizer is None:
        standardizer = Standardizer().fit(ds.x)
    x_std = standardizer.transform(ds.x)
    return dataclasses.replace(ds, x=x_std, standardized=True), standardizer