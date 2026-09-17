"""Evaluation metrics for CATE estimates.

Pure numpy and stateless: each metric takes already-computed arrays (tau_hat, the
observed t/yf, and an oracle where one exists) and returns a scalar. Nothing here
knows about datasets.

Oracle metrics (IHDP; ACTG175's pseudo-obs sharp null where tau_true = 0 everywhere):
`pehe` (individual-level, sqrt(mean((tau_hat - tau_true)^2))) and `eps_ate`
(population-level, |mean(tau_hat) - mean(tau_true)|) - always report both, since a
model can cancel unit-level errors into a good eps_ATE with a terrible PEHE. On the
sharp null, `pehe_against_zero` = RMS(tau_hat) is a valid PEHE since every unit's
true effect is zero, so any spread in tau_hat is residual bias.

Policy metrics (any dataset with a factual outcome, no oracle needed): the
tau_hat-implied policy treats a unit when tau_hat > threshold; its value is
estimated by matching each unit to the arm it was actually observed in (the standard
CFR-literature estimator). Unbiased only under randomization, or with propensity
weights supplied via `weights`; on confounded data (MIMIC) it is a surrogate, read
against treat-all/treat-none rather than absolutely.

Sensitivity: the E-value is the minimum treatment-and-outcome risk-ratio association
an unmeasured confounder would need to fully explain away an observed effect (larger
= more robust); continuous outcomes go through `approximate_risk_ratio` first.

Balance: `smd_reduction` compares the aggregate SMD of the raw covariates x against
that of the representation z - the direct read on whether the MMD term worked.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .smd import smd_per_covariate


def _to_numpy(a) -> np.ndarray:
    """Accept numpy arrays, torch tensors, or anything array-like; return float64."""
    if hasattr(a, "detach"):  # torch tensor, possibly on a device and requiring grad
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64).reshape(-1)

def _to_numpy_2d(a) -> np.ndarray:
    """Like `_to_numpy` but keeps the (n, d) shape."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D array (n, d); got shape {a.shape}")
    return a


# -- oracle metrics --------------------------------------------------------


def pehe(tau_hat, tau_true) -> float:
    """sqrt(mean((tau_hat - tau_true)^2)). `tau_true` may be per-unit or a scalar."""
    tau_hat = _to_numpy(tau_hat)
    tau_true = _to_numpy(tau_true)
    if tau_true.size == 1:
        tau_true = np.full_like(tau_hat, tau_true.item())
    if tau_hat.shape != tau_true.shape:
        raise ValueError(f"tau_hat has {tau_hat.shape[0]} entries, tau_true {tau_true.shape[0]}")
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def pehe_against_zero(tau_hat) -> float:
    """PEHE under a sharp null (tau = 0 everywhere), i.e. RMS(tau_hat)."""
    return pehe(tau_hat, 0.0)


def eps_ate(tau_hat, tau_true) -> float:
    """Absolute ATE error, |mean(tau_hat) - mean(tau_true)|."""
    tau_hat = _to_numpy(tau_hat)
    tau_true = _to_numpy(tau_true)
    return float(abs(tau_hat.mean() - tau_true.mean()))


# -- policy metrics --------------------------------------------------------


def _weighted_mean(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    """Mean of `values` (optionally weighted); nan when the selection is empty."""
    if values.size == 0:
        return float("nan")
    if weights is None:
        return float(values.mean())
    total = float(weights.sum())
    if total <= 0.0:
        return float("nan")
    return float(np.dot(values, weights) / total)


def policy_value(tau_hat, t, yf, threshold: float = 0.0, weights=None) -> float:
    """Expected outcome under the policy "treat iff tau_hat > threshold".

    Estimated by matching: units the policy would treat contribute the observed
    outcomes of the actually-treated, units it would not contribute the observed
    outcomes of the actually-untreated, each weighted by its sample share. Pass
    inverse-propensity `weights` on confounded data; otherwise treatment is assumed
    as good as randomized. Returns nan if a needed arm is empty (value not identified).
    """
    tau_hat = _to_numpy(tau_hat)
    t = _to_numpy(t)
    yf = _to_numpy(yf)
    w = None if weights is None else _to_numpy(weights)
    if not (tau_hat.shape == t.shape == yf.shape):
        raise ValueError("tau_hat, t and yf must have the same length")

    pi = tau_hat > threshold
    p_treat = float(pi.mean())

    value = 0.0
    for select, share in ((pi & (t == 1), p_treat), (~pi & (t == 0), 1.0 - p_treat)):
        if share == 0.0:  # the policy sends nobody here, so the arm cannot contribute
            continue
        value += share * _weighted_mean(yf[select], None if w is None else w[select])
    return float(value)


def policy_risk(
    tau_hat,
    t,
    yf,
    threshold: float = 0.0,
    higher_is_better: bool = True,
    weights=None,
) -> float:
    """Risk (lower = better) of the tau_hat-implied policy.

    `higher_is_better=True`: outcome is a benefit, risk = 1 - value (CFR convention
    for a binary outcome coded 1 = good). Set False when the outcome is itself a
    loss (e.g. MIMIC in-hospital mortality), and risk is the expected outcome directly.
    """
    value = policy_value(tau_hat, t, yf, threshold=threshold, weights=weights)
    return float(1.0 - value) if higher_is_better else float(value)


def policy_risk_table(
    tau_hat,
    t,
    yf,
    threshold: float = 0.0,
    higher_is_better: bool = True,
    weights=None,
) -> Dict[str, float]:
    """Model policy risk next to the treat-all and treat-none reference policies."""
    tau_hat = _to_numpy(tau_hat)
    kw = dict(t=t, yf=yf, higher_is_better=higher_is_better, weights=weights)
    treat_all = np.ones_like(tau_hat)
    treat_none = -np.ones_like(tau_hat)
    return {
        "policy": policy_risk(tau_hat, threshold=threshold, **kw),
        "treat_all": policy_risk(treat_all, threshold=0.0, **kw),
        "treat_none": policy_risk(treat_none, threshold=0.0, **kw),
    }


# -- sensitivity to unmeasured confounding ---------------------------------


def e_value(risk_ratio: float) -> float:
    """E-value of a risk-ratio effect (VanderWeele and Ding, 2017): RR + sqrt(RR*(RR-1)).

    Protective effects are inverted first (RR and 1/RR agree). Result is >= 1; 1
    means a null effect needing no confounding to explain.
    """
    rr = float(risk_ratio)
    if rr <= 0.0:
        raise ValueError(f"risk_ratio must be positive; got {rr}")
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def e_value_ci(ci_low: float, ci_high: float) -> float:
    """E-value for the CI limit nearest the null (1 if the CI covers the null).

    The more honest of the two: how much confounding would move the interval to
    include the null, not just the point estimate.
    """
    lo, hi = float(ci_low), float(ci_high)
    if lo > hi:
        raise ValueError(f"ci_low ({lo}) must not exceed ci_high ({hi})")
    if lo <= 1.0 <= hi:
        return 1.0
    return e_value(lo if lo > 1.0 else hi)


def approximate_risk_ratio(standardized_difference: float) -> float:
    """Approximate risk ratio for a continuous outcome: exp(0.91 * d).

    `d` is the effect divided by the outcome's standard deviation; the standard
    approximation bringing continuous outcomes onto the E-value's risk-ratio scale.
    """
    return float(np.exp(0.91 * float(standardized_difference)))


def approximate_risk_ratio_ci(
    standardized_difference: float, standard_error: float
) -> Tuple[float, float]:
    """Approximate risk-ratio CI: exp(0.91 * d -/+ 1.78 * se). Feeds `e_value_ci`."""
    d = float(standardized_difference)
    se = float(standard_error)
    if se < 0.0:
        raise ValueError(f"standard_error must be non-negative; got {se}")
    half = 1.78 * se
    return float(np.exp(0.91 * d - half)), float(np.exp(0.91 * d + half))


# -- balance ---------------------------------------------------------------


def smd_reduction(x, z, t) -> float:
    """Fraction of aggregate SMD removed going from raw x to representation z.

    1 - mean_j SMD_j(z) / mean_j SMD_j(x): 1.0 = perfectly balanced, 0.0 = no
    improvement, negative = z is less balanced than x. Returns 0.0 when x is already
    balanced. x and z live in different spaces (R^m vs R^k), but the SMD is
    standardized per-dimension before averaging, so the ratio compares average
    per-dimension imbalance, not a distance.
    """
    x = _to_numpy_2d(x)
    z = _to_numpy_2d(z)
    t = _to_numpy(t)
    if x.shape[0] != t.shape[0] or z.shape[0] != t.shape[0]:
        raise ValueError("x, z and t must have the same number of rows")

    before = float(smd_per_covariate(x, t).mean())
    after = float(smd_per_covariate(z, t).mean())
    if before == 0.0:
        return 0.0
    return float(1.0 - after / before)
