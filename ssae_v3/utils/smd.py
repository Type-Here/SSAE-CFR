"""SMD-based noise modulator omega.

    smd_j = |mean_t(x_j) - mean_c(x_j)| / sqrt((var_t_j + var_c_j) / 2 + eps)
    smd   = mean_j(smd_j)
    omega = tanh(alpha_smd * smd)

Computed on the raw covariates in numpy, so it is a plain scalar with no gradient -
the encoder cannot lower omega by changing its weights, which is what prevents the
induced noise from collapsing to zero to cheat the MMD balancing term. The
per-covariate vector is returned too since it doubles as a diagnostic.
"""

from __future__ import annotations

import numpy as np


def smd_per_covariate(x: np.ndarray, t: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-covariate absolute SMD between treated (t==1) and control (t==0), shape (m,).

    Zeros if either group is empty in this batch (undefined SMD -> no induced noise).
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t).reshape(-1)
    treated = x[t == 1]
    control = x[t == 0]
    if treated.shape[0] == 0 or control.shape[0] == 0:
        return np.zeros(x.shape[1], dtype=np.float64)

    mean_diff = np.abs(treated.mean(axis=0) - control.mean(axis=0))
    pooled_sd = np.sqrt((treated.var(axis=0) + control.var(axis=0)) / 2.0 + eps)
    return mean_diff / pooled_sd


def omega_from_smd(x: np.ndarray, t: np.ndarray, alpha_smd: float = 2.0) -> float:
    """Detached scalar omega = tanh(alpha_smd * mean_j smd_j), range [0, 1)."""
    smd = float(smd_per_covariate(x, t).mean())
    return float(np.tanh(alpha_smd * smd))
