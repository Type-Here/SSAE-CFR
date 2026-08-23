"""Training entry point for one SSAE-CFR run.

Pipeline: load config -> load and standardize the dataset (train stats reused on any
other split) -> build (or load) the fixed P_U for the dataset -> build the model ->
optimize the total loss with the gamma warm-up, logging the per-run diagnostics.

Until the real covariate embeddings exist (they are computed on the university machine),
`build_projector_for` falls back to a placeholder V, so the whole pipeline runs and can
be verified end to end locally. Swapping in the real cached V is a one-line change.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from .config import TrainConfig, load_config
from .data import load_ihdp
from .data.base import Dataset
from .losses import total_loss
from .models import SSAECFR
from .prior import build_projector, choose_k_svd, load_projector, placeholder_embeddings
from .utils.smd import omega_from_smd
from .utils.standardize import standardize_dataset

_OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


def load_prior_for(dataset: Dataset, prior_path: str) -> np.ndarray:
    """Load a cached P_U and assert it was built against this dataset's covariates.

    The sidecar records the covariate order P_U was built with; if it disagrees with the
    dataset's `feature_names` the projection geometry would be silently wrong, so we
    refuse rather than train on a mismatched prior.
    """
    P_U, meta = load_projector(prior_path)
    names = meta.get("feature_names")
    if names is not None and list(names) != list(dataset.feature_names):
        raise ValueError(
            f"cached P_U at {prior_path} was built for a different covariate set "
            f"(m={len(names)}) than dataset {dataset.name!r} (m={dataset.m}); rebuild the prior"
        )
    if P_U.shape != (dataset.m, dataset.m):
        raise ValueError(f"P_U is {P_U.shape}, expected ({dataset.m}, {dataset.m})")
    return P_U


def build_projector_for(dataset: Dataset, cfg: TrainConfig, V: Optional[np.ndarray] = None) -> np.ndarray:
    """Return P_U for `dataset`. With no real V, use a reproducible placeholder."""
    if V is None:
        V = placeholder_embeddings(dataset.m, d_LLM=64, seed=cfg.seed)
    k = cfg.k_svd if cfg.k_svd is not None else choose_k_svd(
        V,
        energy_threshold=cfg.energy_threshold,
        protected=cfg.protected or None,
        retention_floor=cfg.retention_floor,
        k_min=cfg.k_svd_min,
        k_max=cfg.k_svd_max,
    )
    return build_projector(V, k)


def _make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    factory = _OPTIMIZERS[cfg.optimizer]
    return factory(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _diagnostics(out: dict, omega: float) -> dict:
    """Representation-level watch numbers logged alongside the loss breakdown."""
    lam = out["lam"].detach()
    return {
        "omega": omega,
        "lam_mean": float(lam.mean()),
        "lam_std": float(lam.std()),
        "z_prior_norm": float(out["z_prior"].detach().norm(dim=-1).mean()),
        "z_res_norm": float(out["z_res"].detach().norm(dim=-1).mean()),
    }


def fit(
    model: SSAECFR,
    dataset: Dataset,
    cfg: TrainConfig,
    log_every: int = 25,
    verbose: bool = True,
) -> List[dict]:
    """Train `model` on a standardized `dataset`; return the per-epoch history."""
    x_np = np.asarray(dataset.x, dtype=np.float32)
    t_np = np.asarray(dataset.t)
    x = torch.from_numpy(x_np)
    t = torch.from_numpy(t_np.astype(np.float32))
    yf = torch.from_numpy(np.asarray(dataset.yf, dtype=np.float32))

    opt = _make_optimizer(model, cfg)
    n = dataset.n
    batch = cfg.batch_size or n
    history: List[dict] = []

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n) if batch < n else torch.arange(n)
        last = {}
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            xb, tb, yb = x[idx], t[idx], yf[idx]
            # omega on the raw (standardized) covariates of this batch, detached
            omega = omega_from_smd(xb.numpy(), tb.numpy(), cfg.alpha_smd)

            out = model(xb, tb, omega)
            terms = model.loss_terms(out, xb, tb, yb, dataset.outcome_type)
            loss, breakdown = total_loss(terms, cfg, epoch)

            opt.zero_grad()
            loss.backward()
            opt.step()
            last = {**breakdown, **_diagnostics(out, omega)}

        last["epoch"] = epoch
        history.append(last)
        if verbose and (epoch % log_every == 0 or epoch == cfg.epochs - 1):
            print(
                f"epoch {epoch:4d} | L {last['L_total']:.4f} "
                f"(fact {last['L_fact']:.4f} mmd {last['L_mmd']:.4f} "
                f"rec {last['L_rec']:.4f} sparse {last['L_sparse']:.3f} "
                f"align {last['L_align']:.4f} g {last['gamma']:.2f}) | "
                f"lam {last['lam_mean']:.2f}+-{last['lam_std']:.2f} omega {last['omega']:.2f}"
            )
    return history


def main(argv: Optional[List[str]] = None) -> None:
    """Train one SSAE-CFR run on IHDP.

    With `--prior artifacts/ihdp/P_U.npz` it uses the real cached projector; without it,
    a reproducible placeholder P_U so the pipeline runs before the embeddings exist.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train one SSAE-CFR run.")
    parser.add_argument("--config", default=None, help="path to a per-dataset YAML config")
    parser.add_argument("--prior", default=None, help="path to a cached P_U.npz (real prior)")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, epochs=args.epochs)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset = load_ihdp()
    dataset, _ = standardize_dataset(dataset)

    if args.prior is not None:
        P_U = load_prior_for(dataset, args.prior)
        print(f"loaded real prior from {args.prior}")
    else:
        P_U = build_projector_for(dataset, cfg)
        print("using PLACEHOLDER P_U (no real embeddings yet) - results are not meaningful")

    model = SSAECFR(
        m=dataset.m,
        P_U=torch.as_tensor(P_U, dtype=torch.float32),
        cfg=cfg,
        outcome_type=dataset.outcome_type,
    )
    print(f"IHDP: n={dataset.n} m={dataset.m} | P_U rank={int(round(float(np.trace(P_U))))}")
    fit(model, dataset, cfg)


if __name__ == "__main__":
    main()