"""Covariate standardization - fit on train, reuse the same stats on test.

`P_U` acts in covariate space with one axis per covariate, so its geometry is only
meaningful when covariates are comparably scaled; standardizing per column also
conditions the encoder input. Stats must be fit on the training split only and
reused unchanged elsewhere, or test information leaks into the transform.

Also imputes: numeric gaps (left as NaN by the adapters) are filled with the
per-column train median before z-scoring, so no leakage there either.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np

from ..data.base import Dataset


class Standardizer:
    """Per-column median-impute then z-score, with train-fit statistics.

    Zero-variance training columns get scale 1 (pass through instead of NaN);
    an all-NaN training column gets median 0.
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

    If `standardizer` is None, one is fit on `ds.x` (single-split convenience). To
    respect the train/test boundary, fit on the training `Dataset` and pass it in
    when standardizing val/test.
    """
    if standardizer is None:
        standardizer = Standardizer().fit(ds.x)
    x_std = standardizer.transform(ds.x)
    return dataclasses.replace(ds, x=x_std, standardized=True), standardizer
