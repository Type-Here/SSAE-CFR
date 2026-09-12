"""Evaluation metrics for CATE estimates.

Everything here is pure numpy and stateless: a metric takes already-computed arrays
(tau_hat from `SSAECFR.predict_tau`, the observed t/yf, and an oracle where one
exists) and returns a scalar. Nothing imports torch, and nothing knows about
datasets - `evaluate.py` is what decides which subset of these applies to which
dataset.

Four families, matching what each dataset can actually support:

Oracle metrics (IHDP, and the ACTG175 sharp null where tau_true = 0 everywhere)

    pehe    = sqrt(mean((tau_hat - tau_true)^2))
    eps_ate = |mean(tau_hat) - mean(tau_true)|

  PEHE is the individual-level error and eps_ATE the population-level one; a model
  can have a near-perfect eps_ATE with a terrible PEHE by cancelling errors across
  units, which is why both are always reported together. On the pseudo-observational
  ACTG175 the sharp null makes `pehe_against_zero` = RMS(tau_hat) a valid PEHE:
  every unit's true effect is zero, so any spread in tau_hat is bias the balancing
  failed to remove, and lower is better.

Policy metrics (any dataset with a factual outcome; no oracle needed)

  The tau_hat-implied policy treats a unit when tau_hat > threshold. Its value is
  estimated by matching each unit to the arm it was actually observed in - the
  standard estimator from the CFR literature. It is only unbiased under
  randomization (or with propensity weights supplied via `weights`); on confounded
  data such as the MIMIC subsets it is a surrogate, read against the treat-all and
  treat-none references rather than in absolute terms.

Sensitivity (observational data, where ignorability is the thing in doubt)

  The E-value is the minimum strength of association - on the risk-ratio scale, with
  both treatment and outcome - that an unmeasured confounder would need to fully
  explain away an observed effect. Larger = more robust. It is defined for a risk
  ratio, so continuous outcomes go through the standard approximation first
  (`approximate_risk_ratio`).

Balance

  `smd_reduction` compares the aggregate standardized mean difference of the raw
  covariates x against that of the learned representation z. It is the direct
  read on whether the MMD term did its job, and the one metric that applies
  identically to every dataset.
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
    """Precision in Estimating Heterogeneous Effects: sqrt(mean((tau_hat - tau_true)^2)).

    `tau_true` may be an array of per-unit effects or a scalar (the sharp-null case).
    Lower is better; the units are those of the outcome.
    """
    tau_hat = _to_numpy(tau_hat)
    tau_true = _to_numpy(tau_true)
    if tau_true.size == 1:
        tau_true = np.full_like(tau_hat, tau_true.item())
    if tau_hat.shape != tau_true.shape:
        raise ValueError(f"tau_hat has {tau_hat.shape[0]} entries, tau_true {tau_true.shape[0]}")
    return float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))


def pehe_against_zero(tau_hat) -> float:
    """PEHE under a sharp null (tau = 0 for every unit), i.e. RMS(tau_hat).

    Valid on the ACTG175 pseudo-observational variant, where the biasing tool
    relabels treatment without regenerating the outcome: the true effect is zero
    everywhere, so whatever effect the model still reports is residual bias.
    """
    return pehe(tau_hat, 0.0)


def eps_ate(tau_hat, tau_true) -> float:
    """Absolute error on the average treatment effect, |mean(tau_hat) - mean(tau_true)|."""
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
    outcomes of those actually treated, units it would not contribute the observed
    outcomes of those actually untreated, each group weighted by its share of the
    sample. Pass inverse-propensity `weights` (shape (n,)) on confounded data;
    without them the estimator assumes treatment is as good as randomized.

    Returns nan if a needed arm is empty (e.g. the policy treats units but no unit
    in that group was actually treated), since the value is then not identified.
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

    With `higher_is_better` the outcome is a benefit and the risk is 1 - value, the
    convention from the CFR literature for a binary outcome coded 1 = good. Set it
    False when the outcome is itself a loss - MIMIC in-hospital mortality, say -
    and the risk is the expected outcome directly.
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
    """Policy risk of the model's policy next to the treat-all and treat-none references.

    A CATE model only earns its keep if it beats both constant policies; on small or
    weakly heterogeneous samples it often does not, and that is the number to report.
    """
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
    """E-value of an effect on the risk-ratio scale (VanderWeele and Ding, 2017).

    RR + sqrt(RR * (RR - 1)) for RR >= 1; protective effects are inverted first, so
    RR and 1/RR give the same answer. The result is >= 1, and 1 means a null effect
    that needs no confounding at all to explain. Read it as: an unmeasured confounder
    would have to be associated with both treatment and outcome by a risk ratio of at
    least this much, above and beyond the measured covariates, to explain the effect.
    """
    rr = float(risk_ratio)
    if rr <= 0.0:
        raise ValueError(f"risk_ratio must be positive; got {rr}")
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def e_value_ci(ci_low: float, ci_high: float) -> float:
    """E-value for the confidence limit nearest the null (1 if the CI covers the null).

    This is the more honest of the two numbers: it asks how much confounding would be
    needed to move the *interval* to include the null, not just the point estimate.
    """
    lo, hi = float(ci_low), float(ci_high)
    if lo > hi:
        raise ValueError(f"ci_low ({lo}) must not exceed ci_high ({hi})")
    if lo <= 1.0 <= hi:
        return 1.0
    return e_value(lo if lo > 1.0 else hi)


def approximate_risk_ratio(standardized_difference: float) -> float:
    """Approximate risk ratio for a continuous outcome: exp(0.91 * d).

    `d` is the effect divided by the outcome's standard deviation. The constant is
    the standard approximation used to bring continuous outcomes (ACTG175 CD4 counts,
    IHDP) onto the risk-ratio scale the E-value is defined on.
    """
    return float(np.exp(0.91 * float(standardized_difference)))


def approximate_risk_ratio_ci(
    standardized_difference: float, standard_error: float
) -> Tuple[float, float]:
    """Approximate risk-ratio CI for a continuous outcome: exp(0.91 * d -/+ 1.78 * se).

    `se` is the standard error of `d` on the standardized scale. Feed the pair
    straight into `e_value_ci`.
    """
    d = float(standardized_difference)
    se = float(standard_error)
    if se < 0.0:
        raise ValueError(f"standard_error must be non-negative; got {se}")
    half = 1.78 * se
    return float(np.exp(0.91 * d - half)), float(np.exp(0.91 * d + half))


# -- balance ---------------------------------------------------------------


def smd_reduction(x, z, t) -> float:
    """Fraction of aggregate SMD removed going from raw x to the representation z.

    1 - mean_j SMD_j(z) / mean_j SMD_j(x): 1.0 means perfectly balanced, 0.0 no
    improvement, negative means the representation is *less* balanced than the input.
    Returns 0.0 when x is already balanced (nothing to remove, so no gain to claim).

    The two aggregates live in different spaces (x in R^m, z in R^k_latent), which
    is fine because the SMD is standardized per dimension before averaging - but it
    does mean the ratio compares average per-dimension imbalance, not a distance.
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