"""Training entry point for one SSAE-CFR v3 run.

Pipeline: load config -> standardize the dataset (train stats reused on any other
split) -> load the dataset's real semantic prior, if the variant needs one -> build
the model -> optimize the four-term objective, logging per-epoch diagnostics.

There is no placeholder prior here: a variant that needs P_U/q_tilde and finds no
built bundle is a configuration error, not something to paper over with random
tensors, so `prior_for` raises and names the command that builds the real thing.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from ..data.base import Dataset
from ..data.ihdp import load_ihdp
from ..hparams import DefaultConfig, MODEL_VARIANTS, load_config
from ..model_v3 import SSAECFRv3
from ..prior_modules.controls import apply_control
from ..prior_modules.loader import PriorTensors, load_prior_tensors
from ..utils.smd import omega_from_smd
from ..utils.standardize import standardize_dataset

_OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


def prior_for(
    dataset: Dataset,
    cfg: DefaultConfig,
    prior_path: Optional[str] = None,
    control: str = "none",
    control_seed: Optional[int] = None,
) -> Optional[PriorTensors]:
    """The prior tensors this run actually needs, or None when no branch consumes them.

    A run with both adapters off never touches the prior, so returning None makes
    "this run used no prior" a fact about the object graph rather than a claim in a
    log line. When a branch is on and no bundle exists, `load_prior_tensors` raises
    FileNotFoundError; this re-raises it with the rebuild command attached, since a
    silent placeholder here would make the run's result meaningless without saying so.

    `control != "none"` swaps in a negative-control transform of the loaded prior
    (see `prior_modules.controls`) with the same shape and capacity, so a caller can
    ask what fraction of a semantic gain is really just capacity.
    """
    if not (cfg.use_u_adapter or cfg.use_w_adapter):
        return None
    try:
        prior = load_prior_tensors(dataset.feature_names, dataset=cfg.dataset, path=prior_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc} Build it with: "
            f"python -m ssae_v3.prior_build.build build --dataset {cfg.dataset} --reuse-V"
        ) from exc
    if control == "none":
        return prior
    seed = cfg.seed if control_seed is None else control_seed
    P_U, q_tilde = apply_control(prior.P_U, prior.q_tilde, control, seed)
    return dataclasses.replace(prior, P_U=P_U, q_tilde=q_tilde)


def _make_optimizer(model: nn.Module, cfg: DefaultConfig) -> torch.optim.Optimizer:
    factory = _OPTIMIZERS[cfg.optimizer]
    return factory(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


_SHARE_LABELS = (("L_fact", "fac"), ("L_mmd", "mmd"), ("L_sparse", "spa"), ("L_rec", "rec"))


def format_shares(breakdown: dict) -> str:
    """The four share_* entries as one compact `fac 35% rec 58% ...` string.

    Printed next to the raw term values because those cannot be compared to each
    other directly: they carry different weights and live on different natural scales.
    """
    parts = []
    for key, label in _SHARE_LABELS:
        share = breakdown.get(f"share_{key}")
        if share is not None and np.isfinite(share):
            parts.append(f"{label} {100.0 * share:.0f}%")
    return " ".join(parts)


def fit(
    model: SSAECFRv3,
    dataset: Dataset,
    cfg: DefaultConfig,
    log_every: int = 25,
    verbose: bool = True,
    val: Optional[Dataset] = None,
) -> List[dict]:
    """Train `model` on a standardized `dataset`; return the per-epoch history.

    When `cfg.patience > 0` and a `val` split is supplied, training stops once the
    validation factual objective has not improved by `cfg.min_delta` for `patience`
    consecutive checks, and the model is rewound to the best-scoring weights. This
    is the only stopping rule available on data with no oracle - it is also the
    criterion selection uses - but it constrains only the factual head, so the epoch
    that minimizes it is not necessarily the epoch that minimizes PEHE, and once an
    epoch is chosen this way the `val_` scores stop being an unbiased estimate.
    Measured to help the criterion it monitors and hurt the causal ones; off by
    default (`cfg.patience = 0`).
    """
    x_np = np.asarray(dataset.x, dtype=np.float32)
    t_np = np.asarray(dataset.t)
    x = torch.from_numpy(x_np)
    t = torch.from_numpy(t_np.astype(np.float32))
    yf = torch.from_numpy(np.asarray(dataset.yf, dtype=np.float32))

    opt = _make_optimizer(model, cfg)
    n = dataset.n
    batch = cfg.batch_size or n
    history: List[dict] = []

    stopping = cfg.patience > 0 and val is not None and val.n > 0
    best_score, best_state, best_epoch, since_best = float("inf"), None, -1, 0
    if stopping:
        # local import: evaluate imports fit from this module, so a top-level import
        # here would close the cycle
        from .evaluate import factual_objective

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n) if batch < n else torch.arange(n)
        last: dict = {}
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            xb, tb, yb = x[idx], t[idx], yf[idx]
            # omega on the raw (standardized) covariates of this batch, detached
            omega = omega_from_smd(xb.numpy(), tb.numpy(), cfg.alpha_smd)

            out = model(xb, tb, omega)
            terms = model.loss_terms(out, xb, tb, yb, dataset.outcome_type)
            loss, breakdown = model.total_loss(terms, cfg)

            opt.zero_grad()
            loss.backward()
            opt.step()
            last = {**breakdown, **model.train_diagnostics(out, omega)}

        last["epoch"] = epoch
        history.append(last)
        if verbose and (epoch % log_every == 0 or epoch == cfg.epochs - 1):
            print(
                f"epoch {epoch:4d} | L {last['L_total']:.4f} "
                f"(fact {last['L_fact']:.4f} mmd {last['L_mmd']:.4f} "
                f"rec {last['L_rec']:.4f} sparse {last['L_sparse']:.3f}) | "
                f"share {format_shares(last)} | "
                f"{model.format_epoch(last)}"
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


def main(argv: Optional[List[str]] = None) -> None:
    """Train one SSAE-CFR v3 run on IHDP.

    `--variant` picks the rung of the experiment ladder. `empirical` needs no prior
    at all; the other three load one from the dataset's real bundle (or `--prior`,
    an explicit bundle path) - there is no placeholder fallback.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train one SSAE-CFR v3 run.")
    parser.add_argument("--config", default=None, help="path to a per-dataset YAML config")
    parser.add_argument("--prior", default=None, help="path to a prior_bundle.npz (default: artifacts/<dataset>/)")
    parser.add_argument(
        "--variant",
        default=None,
        choices=MODEL_VARIANTS,
        help="rung of the experiment ladder (default: whatever the config says)",
    )
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args(argv)

    overrides = {"epochs": args.epochs}
    if args.variant is not None:
        overrides["model_variant"] = args.variant
        overrides["use_u_adapter"] = None
        overrides["use_w_adapter"] = None
    cfg = load_config(args.config, **overrides)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset = load_ihdp()
    dataset, _ = standardize_dataset(dataset)
    run_cfg = dataclasses.replace(cfg, in_channels=dataset.m)

    prior = prior_for(dataset, run_cfg, args.prior)
    prior_note = "no prior (empirical model)" if prior is None else f"prior for {run_cfg.dataset!r}"

    model = SSAECFRv3(
        run_cfg,
        outcome_type=dataset.outcome_type,
        P_U=None if prior is None else prior.P_U,
        q_tilde=None if prior is None else prior.q_tilde,
    )
    print(
        f"IHDP: n={dataset.n} m={dataset.m} | variant {run_cfg.model_variant} "
        f"(r_U={run_cfg.r_U} r_W={run_cfg.r_W}) | {prior_note}"
    )
    fit(model, dataset, run_cfg)


if __name__ == "__main__":
    main()
