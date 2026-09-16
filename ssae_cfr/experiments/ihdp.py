"""The IHDP benchmark, run the way the literature runs it.

Every PEHE number published on IHDP is an average over many simulated realizations of
the response surface, not a single fit. Averaging over seeds of one realization - which
is what `evaluate.py --seeds` does - measures a different quantity: the variability of
our optimizer and our split, rather than the variability of the benchmark. So a number
produced that way cannot be put in a table next to TARNet's or BCAUSS's, however
carefully it was computed.

This module runs the published protocol instead:

  - 100 realizations from `ihdp_npci_1-100.{train,test}.npz`. Each realization covers
    the same 747 units, but resimulates the outcome *and* redraws the partition, so the
    spread across realizations already covers both sources of variation.
  - The train/test partition is the one the files ship (672 / 75 units), not one we
    draw. Redrawing it would silently change the estimand's difficulty, since the
    partition interacts with which region of covariate space is held out.
  - Within the 672 training units, a stratified validation split (30 percent by
    default, so 63/27/10 of the whole - the split Shalit et al. used).
  - Standardization fit on the fitting subset only, reused on validation and test.
  - Mean and standard error across realizations, which is what the "+-" in published
    IHDP tables denotes.

Four scopes are reported, and the two that matter for comparison are `pool` and `out`:

  in    the 63 percent the model was fit on
  val   the 27 percent held out for selection (never used to fit, never to report)
  pool  all 672 training units - this is what papers call *within-sample* PEHE
  out   the 75 test units - *out-of-sample* PEHE

A note on which response surface this is. The distributed files implement the NPCI
"B" (log-linear) surface: mu0 is exponential in the covariates, mu1 is linear, and the
offset is chosen so the true ATT is exactly 4. Shalit et al.'s text says setting "A",
but the data released with that paper - and reused by essentially everyone since - is
the surface described here, which is worth stating explicitly in any write-up.

The imbalance is also simulated, and in a way that matters for this project: the
treated units whose mother was not white were deleted from the original trial, and
maternal race is *not* among the 25 covariates. IHDP therefore has a deliberate,
documented hidden confounder - the exact failure mode the semantic prior is meant to
guard against.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import B_MODES, NOISE_SCALES, TrainConfig, load_config
from ..data import Dataset, N_REALIZATIONS, has_replication_set, load_ihdp_realization
from ..data.ihdp import FEATURE_NAMES
from ..prior import load_projector, retention
from ..evaluate import aggregate, factual_objective, fit_and_score, score_split
from ..utils.split import train_val_test_indices
from ..utils.standardize import standardize_dataset

DEFAULT_VAL_FRACTION = 0.3  # of the 672 training units => 63/27/10 of the whole

# The per-dataset YAML is the benchmark's config, so it loads by default. Passing
# --config None-by-accident used to fall back to the bare dataclass defaults, which
# silently ignored anything set in the YAML.
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "ihdp.yaml"

# Published results under this protocol, for orientation while reading our own output.
# Shalit, Johansson and Sontag (2017), "Estimating individual treatment effect:
# generalization bounds and algorithms", Table 1: 1000 realizations, 63/27/10 splits.
# Values are mean +- standard error of sqrt(PEHE) and of eps_ATE.
REFERENCE_RESULTS: Dict[str, Dict[str, float]] = {
    "BART":     {"pehe_in": 2.10, "pehe_out": 2.30, "eps_ate_in": 0.23, "eps_ate_out": 0.34},
    "Caus.For": {"pehe_in": 3.80, "pehe_out": 3.80, "eps_ate_in": 0.18, "eps_ate_out": 0.40},
    "BNN":      {"pehe_in": 2.20, "pehe_out": 2.10, "eps_ate_in": 0.37, "eps_ate_out": 0.42},
    "TARNet":   {"pehe_in": 0.88, "pehe_out": 0.95, "eps_ate_in": 0.26, "eps_ate_out": 0.28},
    "CFR MMD":  {"pehe_in": 0.73, "pehe_out": 0.78, "eps_ate_in": 0.30, "eps_ate_out": 0.31},
    "CFR Wass": {"pehe_in": 0.71, "pehe_out": 0.76, "eps_ate_in": 0.25, "eps_ate_out": 0.27},
}
REFERENCE_SOURCE = "Shalit et al. 2017, Table 1 (1000 realizations, 63/27/10)"

# Printed under the reference rows so the table is never mistaken for a like-for-like
# comparison. The published numbers come from 1000 realizations of tuned models in
# another codebase; ours from however many were asked for, untuned. PEHE across
# realizations is heavy-tailed (Curth et al.), so an unpaired difference of means
# between codebases is decided largely by which realizations each side happened to
# draw. The band orients a run. It does not support a claim.
REFERENCE_CAVEAT = (
    "  ^ orientation only, not a like-for-like comparison: 1000 tuned realizations in\n"
    "    another codebase vs ours, untuned. PEHE is heavy-tailed across realizations,\n"
    "    so unpaired cross-codebase means do not support a claim."
)


def _standardized_splits(
    realization: int, val_fraction: float, seed: int
) -> Tuple[Dataset, Optional[Dataset], Dataset, Dataset]:
    """(fit, val, pool, test) for one realization, standardized on the fitting subset.

    `pool` is the whole 672-unit training partition, kept so within-sample PEHE can be
    reported on the same units the literature reports it on. It is standardized with the
    fitting subset's statistics like everything else - it is a scoring view of data the
    model has partly seen, not a second training set.
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


def run_realization(
    realization: int,
    cfg: TrainConfig,
    prior_path: Optional[str] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, float]:
    """Fit and score one realization; keys prefixed `in_` / `val_` / `pool_` / `out_`."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    fit_split, val_split, pool_split, test_split = _standardized_splits(
        realization, val_fraction, seed
    )
    model, scores = fit_and_score(
        fit_split,
        test_split,
        cfg,
        val=val_split,
        prior_path=prior_path,
        benefit=True,  # IHDP's outcome is a cognitive test score: higher is better
        seed=seed,
        verbose=verbose,
    )
    scores.update({f"pool_{k}": v for k, v in score_split(model, pool_split, True, seed).items()})
    scores["pool_factual_objective"] = factual_objective(model, pool_split)
    scores["pool_factual_objective_normalized"] = factual_objective(
        model, pool_split, normalized=True
    )
    scores["realization"] = float(realization)
    return scores


def run_benchmark(
    cfg: TrainConfig,
    realizations: Sequence[int] = tuple(range(1, N_REALIZATIONS + 1)),
    prior_path: Optional[str] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    """Run one fit per realization; return (aggregate summary, per-realization scores).

    `seed` is held fixed across realizations on purpose: the spread we report should be
    the benchmark's own variability, not ours added on top of it.
    """
    if not has_replication_set():
        raise FileNotFoundError(
            "the IHDP replication set is missing; download "
            "ihdp_npci_1-100.train.npz and ihdp_npci_1-100.test.npz from "
            "https://www.fredjo.com/ into data/raw/IHDP/"
        )
    runs = [
        run_realization(r, cfg, prior_path, val_fraction, seed, verbose) for r in realizations
    ]
    return aggregate(runs), runs


def format_benchmark(summary: Dict[str, Dict[str, float]], n_realizations: int) -> str:
    """The comparison table: our numbers over the published ones, same protocol."""
    lines = [
        f"IHDP benchmark - {n_realizations} realization(s), fixed 672/75 partition",
        "-" * 74,
        f"{'':<12}{'sqrt(PEHE) in':>16}{'sqrt(PEHE) out':>16}{'eps_ATE in':>15}{'eps_ATE out':>15}",
    ]

    def cell(key: str) -> str:
        stat = summary.get(key)
        if stat is None or not np.isfinite(stat["mean"]):
            return "-"
        sem = stat["std"] / np.sqrt(stat["n_runs"]) if stat["n_runs"] > 1 else 0.0
        return f"{stat['mean']:.2f} +- {sem:.2f}"

    def median_cell(key: str) -> str:
        stat = summary.get(key)
        if stat is None or not np.isfinite(stat.get("median", float("nan"))):
            return "-"
        return f"{stat['median']:.2f}"

    # The mean first, in the outcome's own units, because that is the statistic the
    # published rows report and the table has to be readable straight down. The median
    # sits under it because on this benchmark the two disagree by a factor of two: IHDP
    # redraws the response surface per realization, so PEHE is heavy-tailed and the mean
    # describes the handful of realizations that drew the largest outcomes. Report both.
    lines.append(
        f"{'SSAE-CFR':<12}{cell('pool_pehe'):>16}{cell('out_pehe'):>16}"
        f"{cell('pool_eps_ate'):>15}{cell('out_eps_ate'):>15}"
    )
    lines.append(
        f"{'  (median)':<12}{median_cell('pool_pehe'):>16}{median_cell('out_pehe'):>16}"
        f"{median_cell('pool_eps_ate'):>15}{median_cell('out_eps_ate'):>15}"
    )
    lines.append("")
    lines.append(f"published, for reference ({REFERENCE_SOURCE}):")
    for name, ref in REFERENCE_RESULTS.items():
        lines.append(
            f"{name:<12}{ref['pehe_in']:>16.2f}{ref['pehe_out']:>16.2f}"
            f"{ref['eps_ate_in']:>15.2f}{ref['eps_ate_out']:>15.2f}"
        )
    lines.append(REFERENCE_CAVEAT)

    lines.append("-" * 74)
    # The normalized factual loss is the selection number: the raw MSE is in squared
    # outcome units, so across realizations it is dominated by which ones drew big
    # outcomes rather than by how well anything fit.
    for key, label in (
        ("val_factual_objective_normalized", "val factual MSE (norm)"),
        ("pool_smd_reduction", "SMD reduction (pool)"),
        ("out_smd_reduction", "SMD reduction (out)"),
        ("pool_z_norm", "|z| (pool)"),
        # b_mean alone cannot distinguish a per-patient decision from a gate that has
        # collapsed to one uniform value. The within-covariate variance share is the tell
        # for that; the per-patient sd underneath answers the narrower question of how
        # much the admission budget varies per patient. Under those, the admitted
        # fraction of residual energy - a high mean admission spent on covariates
        # carrying little residual admits little.
        ("pool_b_mean", "b mean (pool)"),
        ("pool_b_within_cov_share", "b within-cov var share"),
        ("pool_b_patient_std", "b per-patient sd (pool)"),
        ("pool_residual_admitted", "residual admitted (pool)"),
        ("pool_ate_hat", "ATE hat (pool)"),
        ("stopped_at_epoch", "stopped at epoch"),
    ):
        stat = summary.get(key)
        if stat is not None and np.isfinite(stat["mean"]):
            lines.append(
                f"{label:<24}{stat['mean']:>10.4f} +- {stat['std']:.4f}"
                f"   median {stat['median']:.4f}"
            )
    lines.append("true ATE is 4.0 by construction; true ATT is exactly 4.0")
    lines.append(format_loss_budget(summary))
    return "\n".join(lines)


def format_loss_budget(summary: Dict[str, Dict[str, float]]) -> str:
    """What fraction of the objective each loss term commanded at the last epoch.

    Printed with every benchmark because a nominal weight is not readable on its own:
    the five terms have natural magnitudes that differ by orders of magnitude, so equal
    weights do not mean equal influence. Averaged across realizations, since the shares
    are dimensionless and therefore comparable across outcome scales - unlike the raw
    term values, which are not.
    """
    rows = [
        ("L_fact", "factual"),
        ("L_mmd", "balancing (MMD)"),
        ("L_sparse", "sparsity (L1)"),
        ("L_rec", "reconstruction"),
        ("L_pref", "preference (L_pref)"),
    ]
    lines = ["", "loss budget at the last epoch (share of the objective, mean over realizations):"]
    any_row = False
    for key, label in rows:
        stat = summary.get(f"train_share_{key}")
        if stat is None or not np.isfinite(stat["mean"]):
            continue
        any_row = True
        lines.append(f"  {label:<22}{100.0 * stat['mean']:>7.1f}%   +- {100.0 * stat['std']:.1f}")
    if not any_row:
        return ""
    return "\n".join(lines)


def format_admission_table(
    summary: Dict[str, Dict[str, float]],
    feature_names: Sequence[str],
    retention_j: Optional[np.ndarray] = None,
    scope: str = "pool",
) -> str:
    """Mean admission per covariate, next to what the prior claimed to keep of it.

    This is the readable output of the decomposition. `b_j` is what the model asked back
    from the residual of covariate j; `retention_j = diag(P_U)_j` is the fraction of that
    covariate the prior kept in the first place. A covariate with low retention and high
    admission is one the truncation discarded and the model needed anyway - the direct,
    per-covariate reading of whether the SVD cut in the right place. Sorted by admission,
    so the covariates the prior serves worst come first.

    Read on the `pool` scope by default: all 672 training units, so the table describes
    the data the gate was fit on rather than the 75 test units.
    """
    rows = []
    for j, name in enumerate(feature_names):
        stat = summary.get(f"{scope}_b_cov_{j:02d}")
        if stat is None or not np.isfinite(stat["mean"]):
            continue
        keep = float(retention_j[j]) if retention_j is not None else float("nan")
        rows.append((stat["mean"], name, keep))
    if not rows:
        return ""

    lines = [
        "",
        f"admission per covariate ({scope}: mean b_j over units and realizations), "
        "highest first:",
        f"  {'covariate':<12}{'b_j':>8}{'retention_j':>14}",
    ]
    for value, name, keep in sorted(rows, reverse=True):
        keep_str = f"{keep:.3f}" if np.isfinite(keep) else "-"
        lines.append(f"  {name:<12}{value:>8.3f}{keep_str:>14}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    """Run the IHDP benchmark and print the comparison table."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the IHDP benchmark protocol.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="path to a per-dataset YAML config (default: the shipped ihdp.yaml)",
    )
    parser.add_argument("--prior", default=None, help="path to a cached P_U.npz (real prior)")
    parser.add_argument(
        "--realizations",
        type=int,
        default=N_REALIZATIONS,
        help=f"how many realizations to run, from 1 (max {N_REALIZATIONS})",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help="share of the 672 training units held out for selection (0 = fit on all)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--alpha-mmd",
        type=float,
        default=None,
        help="weight on the MMD balancing term (overrides the config)",
    )
    parser.add_argument(
        "--gamma-pref",
        type=float,
        default=None,
        help="weight on the preference term L_pref = mean_j b_j (overrides the config)",
    )
    parser.add_argument(
        "--lambda-rec",
        type=float,
        default=None,
        help="weight on the reconstruction; with --gamma-pref this ratio decides b",
    )
    parser.add_argument("--beta-l1", type=float, default=None, help="weight on the L1 term")
    parser.add_argument(
        "--b-mode",
        choices=B_MODES,
        default=None,
        help="'learned' gates the residual; 'one' is the model WITHOUT a prior "
             "(x_mod == x) and 'zero' the prior-only model - the nested ablations",
    )
    parser.add_argument(
        "--noise-scale",
        choices=NOISE_SCALES,
        default=None,
        help="'absolute' (default) or 'relative', which scales the noise by ||mu||",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="early-stopping patience in validation checks (0 = off, the default); "
             "needs --val-fraction > 0",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None, help="write the full summary as JSON")
    parser.add_argument("--verbose", action="store_true", help="log every training curve")
    args = parser.parse_args(argv)

    overrides: Dict[str, object] = {"dataset": "ihdp"}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.alpha_mmd is not None:
        overrides["alpha_mmd"] = args.alpha_mmd
    for name in ("gamma_pref", "lambda_rec", "beta_l1", "b_mode", "noise_scale"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    if args.patience is not None:
        overrides["patience"] = args.patience
    cfg = load_config(args.config, **overrides)

    if cfg.patience > 0 and args.val_fraction <= 0.0:
        raise SystemExit("--patience needs a validation split; pass --val-fraction > 0")

    if args.prior is None:
        print("using PLACEHOLDER P_U (no real embeddings yet) - results are not reportable")

    realizations = list(range(1, min(args.realizations, N_REALIZATIONS) + 1))
    summary, runs = run_benchmark(
        cfg, realizations, args.prior, args.val_fraction, args.seed, args.verbose
    )
    print(
        f"alpha_mmd={cfg.alpha_mmd} gamma_pref={cfg.gamma_pref} "
        f"lambda_rec={cfg.lambda_rec} beta_l1={cfg.beta_l1} "
        f"b_mode={cfg.b_mode} noise_scale={cfg.noise_scale} "
        f"epochs={cfg.epochs} k_latent={cfg.k_latent} patience={cfg.patience}"
    )
    print(format_benchmark(summary, len(realizations)))

    retention_j = retention(load_projector(args.prior)[0]) if args.prior else None
    print(format_admission_table(summary, FEATURE_NAMES, retention_j))

    if args.json_out is not None:
        payload = {
            "protocol": "ihdp_npci_1-100, fixed 672/75 partition",
            "realizations": realizations,
            "val_fraction": args.val_fraction,
            "prior": args.prior,
            "config": cfg.to_dict(),
            "summary": summary,
            "runs": runs,
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()