"""SMD-based noise modulator omega.

The encoder's stochastic branch is z = mu + omega * eps. The scale omega is not
learned - it is driven by how imbalanced the batch is, measured by the Standardized
Mean Difference (SMD) between treated and control covariates:

    smd_j = |mean_t(x_j) - mean_c(x_j)| / sqrt((var_t_j + var_c_j) / 2 + eps)   # per covariate
    smd   = mean_j(smd_j)                                                       # aggregate
    omega = tanh(alpha_smd * smd)                                               # scalar in [0, 1)

Two design points make this robust:
  - It is computed on the raw covariates x (the model input), per batch, in numpy - so
    it is a plain scalar with no gradient. The encoder cannot lower omega by changing its
    weights, which is exactly what prevents the noise from collapsing to zero to cheat
    the balancing (MMD) term. Noise is high precisely when the groups are far apart (the
    regime with little overlap) and fades to zero once they are balanced.
  - The per-covariate vector is returned too, because it doubles as a diagnostic (which
    covariates are imbalanced, and the SMD-reduction metric x vs z_mod).
"""

from __future__ import annotations

import numpy as np


def smd_per_covariate(x: np.ndarray, t: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-covariate absolute SMD between treated (t==1) and control (t==0).

    Returns an array of shape (m,). If either group is empty in this batch the SMD is
    undefined, so we return zeros (no computable imbalance -> no induced noise).
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
    """Detached scalar modulator omega = tanh(alpha_smd * mean_j smd_j).

    Computed in numpy on the raw covariates, so the returned float carries no gradient
    - the encoder sees it as a constant. Range is [0, 1): 0 when the batch is balanced,
    approaching 1 as imbalance grows.
    """
    smd = float(smd_per_covariate(x, t).mean())
    return float(np.tanh(alpha_smd * smd))