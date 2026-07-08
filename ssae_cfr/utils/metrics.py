"""Evaluation metrics -- STUB.
* ``pehe`` = sqrt(mean(((y1_hat-y0_hat) - (mu1-mu0))²))     - needs oracle (IHDP);
  for the ACTG175 sharp null, PEHE-against-zero = RMS(tau_hat).
* ``eps_ate`` = |mean(tau_hat) - mean(tau_true)|.
* ``policy_risk`` vs treat-all / treat-none.
* ``e_value`` - strength of unmeasured confounding needed to explain away the effect.
* ``smd_reduction`` - aggregate SMD of raw x vs of z_mod (balancing gain).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def pehe(tau_hat: "np.ndarray", tau_true: "np.ndarray") -> float:
    """sqrt(mean((tau_hat - tau_true)²)). STUB."""
    raise NotImplementedError("PEHE not yet implemented")


def eps_ate(tau_hat: "np.ndarray", tau_true: "np.ndarray") -> float:
    """|mean(tau_hat) - mean(tau_true)|. STUB."""
    raise NotImplementedError("eps_ate not yet implemented")


def policy_risk(tau_hat: "np.ndarray", t: "np.ndarray", yf: "np.ndarray") -> float:
    """Expected loss of the tau_hat-implied treatment policy. STUB."""
    raise NotImplementedError("policy_risk not yet implemented")


def e_value(estimate: float, ci_low: float, ci_high: float) -> float:
    """E-value for an effect estimate and its CI. STUB."""
    raise NotImplementedError("e_value not yet implemented")


def smd_reduction(x: "np.ndarray", z_mod: "np.ndarray", t: "np.ndarray") -> float:
    """Aggregate SMD of raw x vs balanced z_mod (higher = more bias removed). STUB."""
    raise NotImplementedError("smd_reduction not yet implemented")