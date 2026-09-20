"""Evaluation harness: train one or more runs and report the dataset-appropriate metrics.

`train.py` fits a model; this module decides what "good" means for the dataset at
hand and produces the numbers. One run is: load raw -> stratified split -> standardize
on train -> load the prior (if the variant needs one) -> fit -> score every split.
`evaluate` repeats that over a list of seeds and reports mean/median/std, since a
single run's PEHE is noisy enough to be misleading on its own.

What each dataset supports:

  ihdp             oracle mu0/mu1 -> PEHE and eps_ATE, in- and out-of-sample.
  aids_v1_biased   sharp null (mu0 = mu1 = 0) -> the same oracle metrics, where PEHE
                   is PEHE-against-zero and measures leftover bias.
  aids_v1          the randomized reference: no oracle.
  diur_v1          no oracle -> balance and policy risk against the constant policies.
  sepsis_v2        the same, plus an E-value for how much confounding would undo it.

Balance (SMD reduction from x to the balanced representation `u_shared`) and the
E-value need no oracle, so both are computed everywhere, alongside the model's
representation norms (`u`, `u_shared`, `u_out`, `c`, `a_W`) - how much of the final
code the semantic prior is actually responsible for.

A binary outcome's causal contrast is taken on the probability scale
(`sigmoid(y1) - sigmoid(y0)`, a risk difference), never from a raw head difference,
which would be a log-odds ratio. The E-value needs a risk ratio: binary outcomes give
one directly, continuous ones go through the standardized-difference approximation.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..core.outcome_scale import to_outcome_scale
from ..data import (
    Dataset,
    available,
    load_actg175_pseudo_obs,
    load_actg175_rct,
    load_diur_v1,
    load_ihdp,
    load_sepsis_v2,
)
from ..hparams import DefaultConfig, load_config
from ..model_v3 import SSAECFRv3
from ..prior_modules.expert_bundle import ExpertPriorBundle
from ..utils.metrics import (
    approximate_risk_ratio,
    approximate_risk_ratio_ci,
    e_value,
    e_value_ci,
    eps_ate,
    pehe,
    policy_risk_table,
    policy_value,
    smd_reduction,
)
from ..utils.split import split_and_standardize, treated_fraction
from .train import fit, prior_for

LOADERS: Dict[str, Callable[[], Dataset]] = available({
    "ihdp": load_ihdp,
    "aids_v1": load_actg175_rct,
    "aids_v1_biased": load_actg175_pseudo_obs,
    "diur_v1": load_diur_v1,
    "sepsis_v2": load_sepsis_v2,
})

# Whether a higher outcome is better. IHDP's outcome is a cognitive test score; every
# other dataset here records a harm (AIDS progression, 28-day mortality), so treating
# them all as benefits would invert the policy ranking.
OUTCOME_IS_BENEFIT: Dict[str, bool] = {
    "ihdp": True,
    "aids_v1": False,
    "aids_v1_biased": False,
    "diur_v1": False,
    "sepsis_v2": False,
}

N_BOOTSTRAP = 400


# -- scoring one fitted model on one split ---------------------------------


@torch.no_grad()
def _forward_eval(model: SSAECFRv3, ds: Dataset) -> Dict[str, np.ndarray]:
    """Deterministic (omega = 0) forward pass, returned as numpy."""
    was_training = model.training
    model.eval()
    x = torch.as_tensor(ds.x, dtype=torch.float32)
    t = torch.as_tensor(ds.t, dtype=torch.float32)
    out = model(x, t, omega=0.0)
    if was_training:
        model.train()
    # np.asarray, not .detach(): the reliability entries may be plain python scalars
    # rather than tensors, and the harness should read them either way.
    return {
        key: (value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value))
        for key, value in out.items()
    }


def _potential_outcomes(
    out: Dict[str, np.ndarray],
    outcome_type: str,
    loc: float = 0.0,
    scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """(y0, y1) on the scale the causal contrast should be taken on.

    Delegates to `to_outcome_scale`, the same rule `SSAECFRv3.predict_tau` uses, so
    the harness and the model can never disagree about what a treatment effect is.
    """
    y0 = np.asarray(out["y0_hat"], dtype=np.float64).reshape(-1)
    y1 = np.asarray(out["y1_hat"], dtype=np.float64).reshape(-1)
    return (
        to_outcome_scale(y0, outcome_type, loc, scale),
        to_outcome_scale(y1, outcome_type, loc, scale),
    )


def _factual_from_potentials(
    y0: np.ndarray, y1: np.ndarray, ds: Dataset, y_scale: float, normalized: bool
) -> float:
    """The factual loss given potential outcomes already predicted for `ds`.

    Split out so a caller that has just run a forward pass does not have to run a
    second one; `factual_objective` is the standalone entry point.
    """
    t = np.asarray(ds.t, dtype=np.float64)
    yf_hat = t * y1 + (1.0 - t) * y0
    if ds.outcome_type == "binary":
        p = np.clip(yf_hat, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(ds.yf * np.log(p) + (1.0 - ds.yf) * np.log(1.0 - p)))
    mse = float(np.mean((yf_hat - ds.yf) ** 2))
    if not normalized:
        return mse
    return mse / (y_scale ** 2) if y_scale > 0.0 else float("nan")


def factual_objective(model: SSAECFRv3, ds: Dataset, normalized: bool = False) -> float:
    """Held-out factual loss - the selection criterion a real deployment could compute.

    PEHE needs counterfactuals, so tuning against it is oracle peeking. This is the
    honest alternative: how well the model predicts the outcome it actually observed,
    on data it was not fit on. Lower is better.

    A filter, not a ranking: it scores each unit only on the arm that unit
    actually received, so one term of `tau = y1 - y0` is left completely
    unconstrained per unit, and the counterfactual head is least constrained exactly
    where overlap is worst - which on IHDP is where the difficulty was manufactured.

    `normalized` divides by `y_scale ** 2` (continuous outcomes), giving the loss on
    the scale the model actually optimizes. Averaging the raw version across IHDP
    realizations - whose outcome scale varies about fifteenfold - averages squared
    quantities in incompatible units. A binary cross-entropy is already scale-free,
    so the flag is a no-op there.
    """
    if ds.n == 0:
        return float("nan")
    out = _forward_eval(model, ds)
    y0, y1 = _potential_outcomes(out, ds.outcome_type, *model.outcome_affine)
    _, y_scale = model.outcome_affine
    return _factual_from_potentials(y0, y1, ds, y_scale, normalized)


_FINAL_DIAGNOSTIC_KEYS = (
    "share_L_fact", "share_L_mmd", "share_L_sparse", "share_L_rec",
    "L_total", "u_norm", "u_shared_norm", "u_out_norm",
    "c_norm", "a_W_norm", "a_expert_norm", "r_U", "r_W", "omega",
)


def final_training_diagnostics(history: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """The last training epoch's loss shares and representation norms, as score keys.

    Carried out of `fit` so a benchmark summary can report what the optimizer was
    actually working on, aggregated across runs, rather than only what the model
    scored - the shares in particular decide whether a nominal weight is doing
    anything. Keys are prefixed `train_` since they describe the fit, not a split.

    History entries appended after the loop (the early-stopping record) carry none
    of these keys, so the last entry that does is the one read.
    """
    for entry in reversed(list(history)):
        if "share_L_fact" in entry:
            return {f"train_{k}": entry[k] for k in _FINAL_DIAGNOSTIC_KEYS if k in entry}
    return {}


def _bootstrap_ci(
    statistic: Callable[[np.ndarray], float],
    n: int,
    seed: int = 0,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for a statistic of the units, resampling row indices.

    Captures sampling variability of the fitted model's predictions only - it does
    not refit, so it says nothing about the model's own estimation uncertainty.
    """
    rng = np.random.default_rng(seed)
    draws = np.array(
        [statistic(rng.integers(0, n, size=n)) for _ in range(n_boot)], dtype=np.float64
    )
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(draws, alpha / 2.0)), float(np.quantile(draws, 1.0 - alpha / 2.0))


def _sensitivity(y0: np.ndarray, y1: np.ndarray, yf: np.ndarray, outcome_type: str, seed: int) -> Dict[str, float]:
    """Risk ratio and E-values (point + nearest-the-null CI limit) for the average effect."""
    if outcome_type == "binary":
        def rr_of(idx: np.ndarray) -> float:
            p0 = float(y0[idx].mean())
            return float(y1[idx].mean() / p0) if p0 > 0.0 else float("nan")

        rr = rr_of(np.arange(y0.shape[0]))
        lo, hi = _bootstrap_ci(rr_of, y0.shape[0], seed=seed)
    else:
        sd = float(np.std(yf))
        if sd == 0.0:
            return {"risk_ratio": float("nan"), "e_value": float("nan"), "e_value_ci": float("nan")}

        def d_of(idx: np.ndarray) -> float:
            return float((y1[idx] - y0[idx]).mean() / sd)

        d = d_of(np.arange(y0.shape[0]))
        d_lo, d_hi = _bootstrap_ci(d_of, y0.shape[0], seed=seed)
        rr = approximate_risk_ratio(d)
        half_width = (d_hi - d_lo) / 2.0
        lo, hi = approximate_risk_ratio_ci(d, half_width / 1.96 if half_width > 0 else 0.0)

    if not np.isfinite(rr) or rr <= 0.0:
        return {"risk_ratio": float("nan"), "e_value": float("nan"), "e_value_ci": float("nan")}
    ci = float("nan")
    if np.isfinite(lo) and np.isfinite(hi) and lo > 0.0:
        ci = e_value_ci(min(lo, hi), max(lo, hi))
    return {"risk_ratio": float(rr), "e_value": e_value(rr), "e_value_ci": ci}


def score_split(model: SSAECFRv3, ds: Dataset, benefit: bool, seed: int = 0) -> Dict[str, float]:
    """Every metric the split can support, as a flat dict."""
    if ds.n == 0:
        return {}
    out = _forward_eval(model, ds)
    y0, y1 = _potential_outcomes(out, ds.outcome_type, *model.outcome_affine)
    tau_hat = y1 - y0
    # The balance metric has to be read off the representation the MMD actually
    # acted on - scoring any other code measures nothing.
    balance_code = model.balance_representation(out)
    _, y_scale = model.outcome_affine

    scores: Dict[str, float] = {
        "n": float(ds.n),
        "ate_hat": float(tau_hat.mean()),
        "tau_sd": float(tau_hat.std()),
        "smd_reduction": smd_reduction(ds.x, balance_code, ds.t),
        # Guard against balance-by-collapse: a representation shrunk to zero scores
        # perfectly here while carrying no information, so the norm qualifies the score.
        "balance_norm": float(np.linalg.norm(balance_code, axis=-1).mean()),
        # Both readings are kept: the raw MSE is in the outcome's own units and is the
        # meaningful one on a single dataset, the normalized one is what survives
        # averaging across splits whose outcome scales differ. Selection uses the
        # normalized key. A binary cross-entropy is already scale-free, so the two
        # coincide there and only the raw key is emitted.
        "factual_objective": _factual_from_potentials(y0, y1, ds, y_scale, False),
    }
    if ds.outcome_type != "binary":
        scores["factual_objective_normalized"] = _factual_from_potentials(
            y0, y1, ds, y_scale, True
        )
    scores.update(model.diagnostics(out))

    if ds.has_oracle:
        tau_true = ds.tau_true
        scores["pehe"] = pehe(tau_hat, tau_true)
        scores["eps_ate"] = eps_ate(tau_hat, tau_true)

    if ds.outcome_type == "binary":
        table = policy_risk_table(tau_hat, ds.t, ds.yf, higher_is_better=benefit)
        scores.update({f"policy_risk_{key}": value for key, value in table.items()})
    else:
        scores["policy_value"] = policy_value(tau_hat, ds.t, ds.yf)
        scores["policy_value_treat_all"] = policy_value(np.ones(ds.n), ds.t, ds.yf)
        scores["policy_value_treat_none"] = policy_value(-np.ones(ds.n), ds.t, ds.yf)

    scores.update(_sensitivity(y0, y1, ds.yf, ds.outcome_type, seed))
    return scores


# -- one run, and a set of runs --------------------------------------------


def fit_and_score(
    train: Dataset,
    test: Dataset,
    cfg: DefaultConfig,
    val: Optional[Dataset] = None,
    prior_path: Optional[str] = None,
    benefit: bool = False,
    seed: int = 0,
    verbose: bool = False,
    control: str = "none",
    control_seed: Optional[int] = None,
    expert_bundle: Optional[ExpertPriorBundle] = None,
) -> Tuple[SSAECFRv3, Dict[str, float]]:
    """Fit one model on already-split, already-standardized data and score every split.

    `control` (see `prior_modules.controls`) swaps in a negative-control transform
    of the loaded prior before the model is built; `"none"` uses the real prior.

    `expert_bundle` is the expert feature-to-concept graph, required by the variants
    that use the expert branch. Its own negative controls transform the graph, so the
    caller passes an already-controlled bundle rather than naming a control here;
    `control` above stays the SVD prior's.

    Returns the fitted model alongside the scores so a caller can keep probing it.
    """
    run_cfg = dataclasses.replace(cfg, in_channels=train.m)
    prior = prior_for(train, run_cfg, prior_path, control=control, control_seed=control_seed)

    # The outcome scale is fit on the training split only, exactly like the
    # covariate standardizer - a property of the data the model was shown, not of
    # the data it is scored on. A degenerate (constant) training outcome falls back
    # to 0/1.
    y_loc, y_scale = 0.0, 1.0
    if train.outcome_type != "binary":
        yf_train = np.asarray(train.yf, dtype=np.float64)
        y_loc = float(yf_train.mean())
        y_scale = float(yf_train.std())
        if not np.isfinite(y_scale) or y_scale <= 0.0:
            y_loc, y_scale = 0.0, 1.0

    model = SSAECFRv3(
        run_cfg,
        outcome_type=train.outcome_type,
        y_loc=y_loc,
        y_scale=y_scale,
        U_k=None if prior is None else prior.U_k,
        q_tilde=None if prior is None else prior.q_tilde,
        expert_bundle=expert_bundle,
    )
    history = fit(model, train, run_cfg, verbose=verbose, val=val)

    scores: Dict[str, float] = {}
    for prefix, split in (("in", train), ("val", val), ("out", test)):
        if split is None:
            continue
        scores.update(
            {f"{prefix}_{k}": v for k, v in score_split(model, split, benefit, seed).items()}
        )
    scores["treated_fraction_train"] = treated_fraction(train) or float("nan")
    scores["treated_fraction_test"] = treated_fraction(test) or float("nan")
    scores.update(final_training_diagnostics(history))
    if history and "early_stopped_to_epoch" in history[-1]:
        scores["stopped_at_epoch"] = float(history[-1]["early_stopped_to_epoch"])
    else:
        scores["stopped_at_epoch"] = float(run_cfg.epochs - 1)
    return model, scores


def run_once(
    dataset_name: str,
    cfg: DefaultConfig,
    prior_path: Optional[str] = None,
    test_size: float = 0.25,
    seed: int = 0,
    verbose: bool = False,
    val_size: float = 0.0,
) -> Dict[str, float]:
    """Fit one model on `dataset_name` and score the splits.

    Keys are prefixed `in_` (train), `val_` (validation, when `val_size > 0`) and
    `out_` (test). `val_factual_objective_normalized` is the oracle-free number to
    select hyperparameters on; nothing that feeds a reported result should ever be
    selected on an `out_` key.
    """
    if dataset_name not in LOADERS:
        if dataset_name in OUTCOME_IS_BENEFIT:
            raise KeyError(
                f"dataset {dataset_name!r} is known but its adapter is not in this "
                f"checkout; available here: {sorted(LOADERS)}"
            )
        raise KeyError(f"unknown dataset {dataset_name!r}; known: {sorted(LOADERS)}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = LOADERS[dataset_name]()
    splits = split_and_standardize(ds, test_size=test_size, val_size=val_size, seed=seed)

    _, scores = fit_and_score(
        splits.train,
        splits.test,
        cfg,
        val=splits.val,
        prior_path=prior_path,
        benefit=OUTCOME_IS_BENEFIT.get(dataset_name, False),
        seed=seed,
        verbose=verbose,
    )
    return scores


def aggregate(runs: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean, median and std of every metric across runs, ignoring nan.

    The median is carried alongside the mean because PEHE across IHDP realizations
    is heavy-tailed: a handful of realizations draw outcomes an order of magnitude
    larger than the rest and decide the mean on their own.
    """
    keys = sorted({key for run in runs for key in run})
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.array([run[key] for run in runs if key in run], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[key] = {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "n_runs": 0}
            continue
        summary[key] = {
            "mean": float(finite.mean()),
            "median": float(np.median(finite)),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "n_runs": int(finite.size),
        }
    return summary


def evaluate(
    dataset_name: str,
    cfg: DefaultConfig,
    seeds: Sequence[int] = (0,),
    prior_path: Optional[str] = None,
    test_size: float = 0.25,
    verbose: bool = False,
    val_size: float = 0.0,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    """Run `dataset_name` once per seed; return (aggregate summary, per-run scores).

    Each seed redraws the splits as well as the model init, so the spread across
    seeds covers both sources of variability.
    """
    runs = [
        run_once(dataset_name, cfg, prior_path, test_size, seed, verbose, val_size)
        for seed in seeds
    ]
    return aggregate(runs), runs


def format_summary(dataset_name: str, summary: Dict[str, Dict[str, float]], seeds: Sequence[int]) -> str:
    """Human-readable report, in-sample and out-of-sample side by side."""
    prefixes = [("in_", "in-sample"), ("out_", "out-of-sample")]
    if any(key.startswith("val_") for key in summary):
        prefixes.insert(1, ("val_", "validation"))

    width = 24 + 19 * len(prefixes)
    lines = [f"{dataset_name}  ({len(seeds)} run(s), seeds {list(seeds)})", "-" * width]
    bare = sorted(k for k in summary if not k.startswith(("in_", "val_", "out_")))
    metrics = sorted({k[3:] for k in summary if k.startswith("in_")})

    header = "".join(f"{label:>19}" for _, label in prefixes)
    lines.append(f"{'metric':<24}{header}")
    for metric in metrics:
        cells = []
        for prefix, _ in prefixes:
            stat = summary.get(prefix + metric)
            if stat is None or not np.isfinite(stat["mean"]):
                cells.append("-")
            elif len(seeds) > 1:
                cells.append(f"{stat['mean']:.4f} +- {stat['std']:.4f}")
            else:
                cells.append(f"{stat['mean']:.4f}")
        lines.append(f"{metric:<24}" + "".join(f"{cell:>19}" for cell in cells))

    if bare:
        lines.append("-" * width)
        for key in bare:
            lines.append(f"{key:<24}{summary[key]['mean']:>19.4f}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    """Evaluate a dataset and emit its metric set."""
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate SSAE-CFR v3 on one dataset.")
    parser.add_argument("--dataset", default="ihdp", choices=sorted(LOADERS))
    parser.add_argument("--config", default=None, help="path to a per-dataset YAML config")
    parser.add_argument("--prior", default=None, help="path to a prior_bundle.npz (default: artifacts/<dataset>/)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.0,
        help="fraction held out for hyperparameter selection (0 = no validation split)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--json-out", default=None, help="write the full summary as JSON")
    parser.add_argument("--verbose", action="store_true", help="log the training curve")
    args = parser.parse_args(argv)

    overrides = {"dataset": args.dataset}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    cfg = load_config(args.config, **overrides)

    summary, runs = evaluate(
        args.dataset, cfg, args.seeds, args.prior, args.test_size, args.verbose, args.val_size
    )
    print(format_summary(args.dataset, summary, args.seeds))

    if args.json_out is not None:
        payload = {"dataset": args.dataset, "seeds": list(args.seeds), "summary": summary, "runs": runs}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
