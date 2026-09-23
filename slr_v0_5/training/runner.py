"""Fit one SLR-CFR v0.5 model and score it on the IHDP realization protocol.

One realization is: load the shipped 672/75 partition -> hold out a validation
fraction of the 672 -> standardize on the fitting subset -> build the model at the
artifact's embedding width -> fit -> score the four scopes `in_` / `val_` / `pool_`
/ `out_`. `pool_` is all 672 units, which is the literature's "within-sample".

Dataset loading, splitting, standardization, the SMD modulator and the metrics are
imported from the common utilities; nothing here imports an older model.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ssae_v3.data.base import Dataset
from ssae_v3.data.ihdp import load_ihdp_realization
from ssae_v3.utils.metrics import eps_ate, pehe, smd_reduction
from ssae_v3.utils.smd import omega_from_smd
from ssae_v3.utils.split import train_val_test_indices, treated_fraction
from ssae_v3.utils.standardize import standardize_dataset

from ..config import SLRConfig
from ..model import SLRCFRv05
from ..outcome_scale import to_outcome_scale
from ..prior.artifacts import FeatureEmbeddings, center_embeddings
from ..prior.base import build_prior_integration
from ..prior.controls import apply_embedding_variant, prior_rms

DEFAULT_VAL_FRACTION = 0.3  # of the 672 training units => 63/27/10 of the whole
# Each realization draws its own control: a single permutation scored N times is one
# draw, not a control distribution.
CONTROL_SEED_BASE = 1000

_OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}

_FINAL_DIAGNOSTIC_KEYS = (
    "share_L_fact", "share_L_mmd", "share_L_sparse", "L_total",
    "mu_emp_norm", "u_emp_norm", "u_out_norm", "p_norm",
    "prior_contribution_norm", "prior_to_empirical_norm_ratio",
    "u_out_l1", "u_out_active_fraction", "omega",
    # present only under the attention mechanism
    "attention_entropy_mean", "attention_entropy_normalized",
    "effective_feature_count_mean", "attention_max_weight_mean",
    "attention_top3_mass_mean", "attention_norm_match_scale_mean",
)


# -- splits ----------------------------------------------------------------


def standardized_splits(
    realization: int, val_fraction: float, seed: int
) -> Tuple[Dataset, Optional[Dataset], Dataset, Dataset]:
    """(fit, val, pool, test) for one realization, standardized on the fitting subset.

    `pool` is the whole 672-unit training partition, a scoring view of data the model
    has partly seen, not a second training set.
    """
    train_raw = load_ihdp_realization(realization, "train")
    test_raw = load_ihdp_realization(realization, "test")

    if val_fraction > 0.0:
        fit_idx, _, val_idx = train_val_test_indices(
            train_raw.t, test_size=val_fraction, val_size=0.0, seed=seed
        )
    else:
        fit_idx, val_idx = np.arange(train_raw.n), np.array([], dtype=np.int64)

    fit_split, standardizer = standardize_dataset(train_raw.subset(fit_idx))
    val_split = None
    if val_idx.size:
        val_split = standardize_dataset(train_raw.subset(val_idx), standardizer)[0]
    pool_split = standardize_dataset(train_raw, standardizer)[0]
    test_split = standardize_dataset(test_raw, standardizer)[0]
    return fit_split, val_split, pool_split, test_split


# -- scoring ---------------------------------------------------------------


@torch.no_grad()
def _forward_eval(model: SLRCFRv05, ds: Dataset) -> Dict[str, np.ndarray]:
    """Deterministic (omega = 0) forward pass, returned as numpy."""
    was_training = model.training
    model.eval()
    x = torch.as_tensor(ds.x, dtype=torch.float32)
    t = torch.as_tensor(ds.t, dtype=torch.float32)
    out = model(x, t, omega=0.0)
    if was_training:
        model.train()
    return {
        key: (value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
        for key, value in out.items()
    }


def _potential_outcomes(
    out: Dict[str, np.ndarray], outcome_type: str, loc: float, scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    y0 = np.asarray(out["y0_hat"], dtype=np.float64).reshape(-1)
    y1 = np.asarray(out["y1_hat"], dtype=np.float64).reshape(-1)
    return (
        to_outcome_scale(y0, outcome_type, loc, scale),
        to_outcome_scale(y1, outcome_type, loc, scale),
    )


def _factual_from_potentials(
    y0: np.ndarray, y1: np.ndarray, ds: Dataset, y_scale: float, normalized: bool
) -> float:
    t = np.asarray(ds.t, dtype=np.float64)
    yf_hat = t * y1 + (1.0 - t) * y0
    if ds.outcome_type == "binary":
        p = np.clip(yf_hat, 1e-7, 1.0 - 1e-7)
        return float(-np.mean(ds.yf * np.log(p) + (1.0 - ds.yf) * np.log(1.0 - p)))
    mse = float(np.mean((yf_hat - ds.yf) ** 2))
    if not normalized:
        return mse
    return mse / (y_scale ** 2) if y_scale > 0.0 else float("nan")


def factual_objective(model: SLRCFRv05, ds: Dataset, normalized: bool = False) -> float:
    """Held-out factual loss - the selection criterion a real deployment could compute.

    `normalized` divides by y_scale**2, putting the loss on the scale the model
    optimizes, which is what makes it comparable across IHDP realizations (their
    outcome scale varies about fifteenfold). Lower is better.
    """
    if ds.n == 0:
        return float("nan")
    out = _forward_eval(model, ds)
    _, y_scale = model.outcome_affine
    y0, y1 = _potential_outcomes(out, ds.outcome_type, *model.outcome_affine)
    return _factual_from_potentials(y0, y1, ds, y_scale, normalized)


def attention_by_arm(alpha: np.ndarray, t: np.ndarray) -> Dict[str, object]:
    """Per-feature mean attention, and how far the two arms' mean weights differ.

    Descriptive only. A difference between the arms' attention says the query - the
    empirical representation - separates them, not that the prior caused it.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    t = np.asarray(t).reshape(-1)
    scores: Dict[str, object] = {"alpha_mean_per_feature": alpha.mean(axis=0).tolist()}
    treated, control = alpha[t == 1], alpha[t == 0]
    if treated.shape[0] and control.shape[0]:
        scores["alpha_mean_treated"] = treated.mean(axis=0).tolist()
        scores["alpha_mean_control"] = control.mean(axis=0).tolist()
        scores["alpha_treated_control_l1"] = float(
            np.abs(treated.mean(axis=0) - control.mean(axis=0)).sum()
        )
    return scores


def score_split(model: SLRCFRv05, ds: Dataset) -> Dict[str, float]:
    """Every metric this split supports, as a flat dict."""
    if ds.n == 0:
        return {}
    out = _forward_eval(model, ds)
    y0, y1 = _potential_outcomes(out, ds.outcome_type, *model.outcome_affine)
    tau_hat = y1 - y0
    _, y_scale = model.outcome_affine
    balance_code = np.asarray(out["u_out"])

    scores: Dict[str, float] = {
        "n": float(ds.n),
        "ate_hat": float(tau_hat.mean()),
        "tau_sd": float(tau_hat.std()),
        # Read off u_out, the representation the MMD actually acted on.
        "smd_reduction": smd_reduction(ds.x, balance_code, ds.t),
        # A representation shrunk to zero balances perfectly while carrying nothing,
        # so the norm qualifies the balance score.
        "balance_norm": float(np.linalg.norm(balance_code, axis=-1).mean()),
        "factual_objective": _factual_from_potentials(y0, y1, ds, y_scale, False),
    }
    if ds.outcome_type != "binary":
        scores["factual_objective_normalized"] = _factual_from_potentials(y0, y1, ds, y_scale, True)
    if ds.has_oracle:
        scores["pehe"] = pehe(tau_hat, ds.tau_true)
        scores["eps_ate"] = eps_ate(tau_hat, ds.tau_true)
    scores.update(model.diagnostics(out))
    if out.get("alpha") is not None:
        scores.update(attention_by_arm(out["alpha"], ds.t))
    return scores


def final_training_diagnostics(history: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """The last training epoch's loss shares and norms, prefixed `train_`."""
    for entry in reversed(list(history)):
        if "share_L_fact" in entry:
            return {f"train_{k}": entry[k] for k in _FINAL_DIAGNOSTIC_KEYS if k in entry}
    return {}


# -- training --------------------------------------------------------------


def _make_optimizer(model: nn.Module, cfg: SLRConfig) -> torch.optim.Optimizer:
    return _OPTIMIZERS[cfg.optimizer](model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def fit(
    model: SLRCFRv05,
    dataset: Dataset,
    cfg: SLRConfig,
    val: Optional[Dataset] = None,
    log_every: int = 25,
    verbose: bool = True,
) -> List[dict]:
    """Train `model` on a standardized `dataset`; return the per-epoch history.

    With `cfg.patience > 0` and a validation split, training stops once the validation
    factual objective has not improved by `min_delta` for `patience` consecutive
    checks and the model is rewound to the best-scoring weights. Once an epoch is
    chosen this way the `val_` scores stop being an unbiased estimate.
    """
    x = torch.from_numpy(np.asarray(dataset.x, dtype=np.float32))
    t = torch.from_numpy(np.asarray(dataset.t).astype(np.float32))
    yf = torch.from_numpy(np.asarray(dataset.yf, dtype=np.float32))

    opt = _make_optimizer(model, cfg)
    n = dataset.n
    batch = cfg.batch_size or n
    history: List[dict] = []

    stopping = cfg.patience > 0 and val is not None and val.n > 0
    best_score, best_state, best_epoch, since_best = float("inf"), None, -1, 0

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n) if batch < n else torch.arange(n)
        last: dict = {}
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            xb, tb, yb = x[idx], t[idx], yf[idx]
            omega = omega_from_smd(xb.numpy(), tb.numpy(), cfg.alpha_smd) if cfg.noise_enabled else 0.0

            out = model(xb, tb, omega)
            terms = model.loss_terms(out, tb, yb, dataset.outcome_type)
            loss, breakdown = model.total_loss(terms, cfg)

            opt.zero_grad()
            loss.backward()
            opt.step()
            last = {**breakdown, **model.diagnostics(out, omega)}

        last["epoch"] = epoch
        history.append(last)
        if verbose and (epoch % log_every == 0 or epoch == cfg.epochs - 1):
            print(
                f"epoch {epoch:4d} | L {last['L_total']:.4f} "
                f"(fact {last['L_fact']:.4f} mmd {last['L_mmd']:.4f} "
                f"sparse {last['L_sparse']:.3f}) | {model.format_epoch(last)}"
            )

        if not stopping:
            continue
        if epoch % cfg.es_check_every and epoch != cfg.epochs - 1:
            continue
        score = factual_objective(model, val, normalized=True)
        last["val_objective"] = score
        if score < best_score - cfg.min_delta:
            best_score, best_epoch, since_best = score, epoch, 0
            # detached CPU copy: the live tensors keep training under us otherwise
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= cfg.patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (best {best_score:.4f} @ {best_epoch})")
                break

    if stopping and best_state is not None:
        model.load_state_dict(best_state)
        history.append({"early_stopped_to_epoch": best_epoch, "best_val_objective": best_score})
    return history


# -- one realization -------------------------------------------------------


def build_model(
    cfg: SLRConfig,
    train: Dataset,
    embeddings: Optional[FeatureEmbeddings],
    control_seed: int,
) -> SLRCFRv05:
    """Build the model for one arm: transform the artifact, control it, then wire it in.

    Centering, when the config asks for it, happens before the row control. The mean
    row is permutation-invariant, so the order does not change the matrix; doing it
    first keeps the mean the real artifact's own in every arm.

    The outcome scale is fit on the training split only, exactly like the covariate
    standardizer - a property of the data the model was shown, not of the data it is
    scored on. A degenerate (constant) training outcome falls back to 0/1.
    """
    arm_embeddings = None
    if embeddings is not None:
        arm_embeddings = embeddings
        if cfg.center_embeddings:
            arm_embeddings = center_embeddings(arm_embeddings)
        # The norm-matched control reads its scale off the fitting split only, which
        # is the same rule the covariate standardizer and the outcome scale follow.
        x_train = torch.as_tensor(train.x, dtype=torch.float32)
        arm_embeddings = apply_embedding_variant(
            arm_embeddings, cfg.embedding_variant, control_seed, x_std=x_train
        )
    prior_integration = build_prior_integration(cfg, arm_embeddings)

    y_loc, y_scale = 0.0, 1.0
    if train.outcome_type != "binary":
        yf_train = np.asarray(train.yf, dtype=np.float64)
        y_loc = float(yf_train.mean())
        y_scale = float(yf_train.std())
        if not np.isfinite(y_scale) or y_scale <= 0.0:
            y_loc, y_scale = 0.0, 1.0

    return SLRCFRv05(
        cfg,
        prior_integration=prior_integration,
        outcome_type=train.outcome_type,
        y_loc=y_loc,
        y_scale=y_scale,
    )


def run_realization(
    realization: int,
    cfg: SLRConfig,
    embeddings: Optional[FeatureEmbeddings] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
    control_seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Fit and score one realization; keys prefixed `in_` / `val_` / `pool_` / `out_`.

    `embeddings` is the real artifact; the arm's control is applied here, so every arm
    reads the same file and a difference between arms is a difference in the
    assignment alone. Seeding happens before the splits and the model are built, in
    that fixed order, so arms sharing a `seed` share the split and the initial weights.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if control_seed is None:
        control_seed = CONTROL_SEED_BASE + realization

    fit_split, val_split, pool_split, test_split = standardized_splits(
        realization, val_fraction, seed
    )
    run_cfg = dataclasses.replace(cfg, in_channels=fit_split.m)
    model = build_model(run_cfg, fit_split, embeddings, control_seed)
    history = fit(model, fit_split, run_cfg, val=val_split, verbose=verbose)

    scores: Dict[str, float] = {}
    for prefix, split in (("in", fit_split), ("val", val_split), ("pool", pool_split), ("out", test_split)):
        if split is None:
            continue
        scores.update({f"{prefix}_{k}": v for k, v in score_split(model, split).items()})

    scores.update(final_training_diagnostics(history))
    scores["realization"] = float(realization)
    scores["control_seed"] = float(control_seed)
    scores["eta_prior"] = float(run_cfg.eta_prior)
    scores["center_embeddings"] = float(run_cfg.center_embeddings)
    scores["prior_integration"] = run_cfg.prior_integration
    scores["attention_norm_match"] = float(run_cfg.attention_norm_match)
    scores["attention_query"] = run_cfg.attention_query
    if model.prior_integration is not None:
        scores["prior_rms_train"] = prior_rms(model.prior_integration.Z, torch.as_tensor(fit_split.x, dtype=torch.float32))
    scores["d_latent"] = float(run_cfg.d_latent)
    scores["treated_fraction_train"] = treated_fraction(fit_split) or float("nan")
    scores["treated_fraction_test"] = treated_fraction(test_split) or float("nan")
    scores["trainable_parameter_count"] = float(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    if history and "early_stopped_to_epoch" in history[-1]:
        scores["stopped_at_epoch"] = float(history[-1]["early_stopped_to_epoch"])
    else:
        scores["stopped_at_epoch"] = float(run_cfg.epochs - 1)
    return scores


def aggregate(runs: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Mean, median and std of every numeric metric across runs, ignoring nan.

    Non-numeric entries (an arm label, for instance) are skipped rather than coerced.

    The median is carried alongside the mean because PEHE across IHDP realizations is
    heavy-tailed: a few realizations draw outcomes an order of magnitude larger than
    the rest and decide the mean on their own.
    """
    keys = sorted({
        key for run in runs for key, value in run.items() if isinstance(value, (int, float))
    })
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
