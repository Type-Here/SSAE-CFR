"""The IHDP benchmark protocol and the paired variant ladder.

One fit per realization on the shipped 672/75 partition, scored on four scopes:
`in_` (the fitting subset), `val_`, `pool_` (all 672 - the literature's
"within-sample") and `out_` (the shipped 75-unit test partition).

`run_ladder` runs several variants, and their negative controls, on the SAME
realizations with the SAME splits and seed, so the arms are paired and a difference
between them is not a difference between draws.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.base import Dataset
from ..data.ihdp import N_REALIZATIONS, has_replication_set, load_ihdp_realization
from ..hparams import DefaultConfig, MODEL_VARIANTS, load_config
from ..prior_modules.controls import CONTROLS
from ..utils.split import train_val_test_indices
from ..utils.standardize import standardize_dataset
from .evaluate import aggregate, fit_and_score, score_split

DEFAULT_VAL_FRACTION = 0.3  # of the 672 training units => 63/27/10 of the whole

# The per-dataset YAML is the benchmark's config, so it loads by default. Defaulting
# --config to None instead would silently fall back to the bare dataclass defaults.
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "ihdp.yaml"

# Published results under this protocol, for orientation only.
# Shalit, Johansson and Sontag (2017), Table 1: 1000 realizations, 63/27/10 splits.
# Values are mean of sqrt(PEHE) and of eps_ATE.
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
# comparison: 1000 tuned realizations in another codebase against this one, untuned. PEHE
# is heavy-tailed across realizations, so an unpaired difference of means between
# codebases is decided largely by which realizations each side drew.
REFERENCE_CAVEAT = (
    "  ^ orientation only, not a like-for-like comparison: 1000 tuned realizations in\n"
    "    another codebase vs this one, untuned. PEHE is heavy-tailed across realizations,\n"
    "    so unpaired cross-codebase means do not support a claim."
)


def _standardized_splits(
    realization: int, val_fraction: float, seed: int
) -> Tuple[Dataset, Optional[Dataset], Dataset, Dataset]:
    """(fit, val, pool, test) for one realization, standardized on the fitting subset.

    `pool` is the whole 672-unit training partition, kept so within-sample PEHE is
    reported on the units the literature reports it on. It is a scoring view of data
    the model has partly seen, not a second training set.
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
    cfg: DefaultConfig,
    prior_path: Optional[str] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
    verbose: bool = False,
    control: str = "none",
    control_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Fit and score one realization; keys prefixed `in_` / `val_` / `pool_` / `out_`.

    `control_seed` picks which draw of a negative control this run uses; left None it
    is `cfg.seed`, which is held fixed across realizations, so a control arm is one
    random draw measured 30 times rather than 30 random draws. That is the right
    default for pairing but it means a single arm cannot separate the control's
    distribution from the particular subspace it drew - vary this to do that.
    """
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
        control=control,
        control_seed=control_seed,
    )
    scores.update({f"pool_{k}": v for k, v in score_split(model, pool_split, True, seed).items()})
    scores["realization"] = float(realization)
    if control_seed is not None:
        scores["control_seed"] = float(control_seed)
    return scores


def run_benchmark(
    cfg: DefaultConfig,
    realizations: Sequence[int] = tuple(range(1, N_REALIZATIONS + 1)),
    prior_path: Optional[str] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
    verbose: bool = False,
    control: str = "none",
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]]]:
    """Run one fit per realization; return (aggregate summary, per-realization scores).

    `seed` is held fixed across realizations on purpose: the reported spread is then
    the benchmark's own variability, with no extra run-to-run variance added.
    """
    if not has_replication_set():
        raise FileNotFoundError(
            "the IHDP replication set is missing; download "
            "ihdp_npci_1-100.train.npz and ihdp_npci_1-100.test.npz from "
            "https://www.fredjo.com/ into data/raw/IHDP/"
        )
    runs = [
        run_realization(r, cfg, prior_path, val_fraction, seed, verbose, control)
        for r in realizations
    ]
    return aggregate(runs), runs


# -- the paired ladder ------------------------------------------------------


def _arm_label(variant: str, control: str) -> str:
    return variant if control == "none" else f"{variant} [{control}]"


def _arm_config(base: DefaultConfig, variant: str) -> DefaultConfig:
    """`base` retargeted at one ladder rung, with the enabled branches' reliability live.

    A config built for a variant whose branches are off carries `r_U = r_W = 0`, and
    switching the variant does not restore them. Left at zero the correction is
    multiplied by zero, so no gradient ever reaches the adapter and its zero-initialized
    final layer never moves: the arm trains as the empirical model while claiming to be
    a semantic one. Reviving a zeroed constant here is what keeps a ladder rung from
    being silently inert; `_resolve_variant` re-zeroes whichever branch this rung does
    not use, so nothing turns a branch on by accident either.
    """
    return dataclasses.replace(
        base,
        model_variant=variant,
        use_u_adapter=None,
        use_w_adapter=None,
        r_U=base.r_U if base.r_U > 0.0 else 1.0,
        r_W=base.r_W if base.r_W > 0.0 else 1.0,
    )


def run_ladder(
    variants: Sequence[str] = MODEL_VARIANTS,
    realizations: Sequence[int] = tuple(range(1, 11)),
    controls: Sequence[str] = ("none",),
    cfg: Optional[DefaultConfig] = None,
    prior_path: Optional[str] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    """Run every (variant, control) arm on the same realizations, paired.

    Returns (per-arm run lists, per-arm aggregate summaries), keyed by arm label.
    Control arms are skipped for a variant that consumes no prior, since they would
    only duplicate it.
    """
    base = cfg if cfg is not None else load_config(str(DEFAULT_CONFIG))
    for variant in variants:
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {MODEL_VARIANTS}")
    for control in controls:
        if control not in CONTROLS:
            raise ValueError(f"unknown control {control!r}; choose from {CONTROLS}")

    # The real arm always runs: controls are additional arms, never a replacement. A
    # table of control arms without the arms they control, or without a baseline, is
    # uninterpretable.
    controls = ("none",) + tuple(c for c in controls if c != "none")

    rows: Dict[str, List[Dict[str, float]]] = {}
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variant in variants:
        arm_cfg = _arm_config(base, variant)
        uses_prior = arm_cfg.use_u_adapter or arm_cfg.use_w_adapter
        for control in controls:
            if control != "none" and not uses_prior:
                continue
            label = _arm_label(variant, control)
            runs = [
                run_realization(r, arm_cfg, prior_path, val_fraction, seed, verbose, control)
                for r in realizations
            ]
            rows[label] = runs
            summaries[label] = aggregate(runs)
    return rows, summaries


def _paired_wins(arm: Sequence[Dict[str, float]], base: Sequence[Dict[str, float]], key: str) -> str:
    """How often `arm` beats `base` (lower is better) on the same realization."""
    by_real = {run["realization"]: run for run in base}
    wins = total = 0
    for run in arm:
        other = by_real.get(run["realization"])
        if other is None or key not in run or key not in other:
            continue
        if np.isfinite(run[key]) and np.isfinite(other[key]):
            total += 1
            wins += int(run[key] < other[key])
    return f"{wins}/{total}" if total else "-"


def format_ladder(
    rows: Dict[str, List[Dict[str, float]]],
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    baseline: str = "empirical",
) -> str:
    """The ladder table: one row per arm, each control directly under what it controls.

    The median leads because 10 of 100 realizations carry 46% of the summed out-PEHE
    and std(y) runs 1.81-39.11 across them, so the mean describes the few realizations
    that drew the largest outcomes. The win column is paired against the baseline arm
    on the same realizations. `|u_shared|` sits beside the SMD reduction because a
    representation is perfectly balanced once shrunk to zero, so a balance gain with a
    collapsing norm is not one.
    """
    n = len(next(iter(rows.values()))) if rows else 0
    lines = [
        f"variant ladder - {n} paired realization(s), fixed 672/75 partition",
        "-" * 96,
        f"{'arm':<26}{'out PEHE med':>13}{'mean':>8}{'win':>7}"
        f"{'out epsATE med':>15}{'SMD red':>10}{'|u_shared|':>12}",
    ]

    def med(label: str, key: str) -> float:
        stat = summaries[label].get(key)
        return float("nan") if stat is None else stat["median"]

    def mean(label: str, key: str) -> float:
        stat = summaries[label].get(key)
        return float("nan") if stat is None else stat["mean"]

    base_runs = rows.get(baseline)
    inert: List[str] = []
    for label in rows:
        win = "-"
        if base_runs is not None and label != baseline:
            win = _paired_wins(rows[label], base_runs, "out_pehe")
        lines.append(
            f"{label:<26}{med(label, 'out_pehe'):>13.3f}{mean(label, 'out_pehe'):>8.3f}{win:>7}"
            f"{med(label, 'out_eps_ate'):>15.3f}{med(label, 'pool_smd_reduction'):>10.3f}"
            f"{med(label, 'pool_balance_norm'):>12.3f}"
        )
        # An arm whose branch is on but whose correction never left zero trained as the
        # empirical model, so its scores carry no semantic content.
        if not label.startswith("empirical"):
            corrections = med(label, "pool_c_norm") + med(label, "pool_a_W_norm")
            if np.isfinite(corrections) and corrections == 0.0:
                inert.append(label)

    lines.append("-" * 96)
    for label in inert:
        lines.append(f"WARNING {label}: both corrections are exactly 0 - this arm is the empirical model")
    lines.append(f"win = realizations where the arm beats {baseline!r} on out PEHE (paired, lower is better)")
    control_arms = [label for label in rows if "[" in label]
    if control_arms:
        lines.append(
            "a control arm matching its variant means the gain is capacity, not semantics"
        )
    else:
        lines.append(
            "NO CONTROL ARMS RUN - a variant-vs-baseline gap alone does not separate "
            "semantic knowledge from extra capacity"
        )
    return "\n".join(lines)


# -- the W-semantics sweep ---------------------------------------------------

# What each arm feeds the W branch's per-feature table, as (control, trainable). The
# question the sweep asks is not "does W help" (that is the ladder's job) but "does the
# LLM's geometry of meaning matter, or would any per-feature code do".
#
#   W_real        the built prior, q_j = W_r^T v_j from the BioMistral glosses
#   W_random      a random table at the same shape and the same RMS: the same capacity,
#                 the same per-feature distinctness, none of the relations between
#                 features
#   W_onehot      a scaled identity: feature j means "I am feature j" and nothing more -
#                 the floor, the adapter knowing only which column it is reading
#   W_learned_id  the same random table, but trainable: what the outcome loss can invent
#                 for itself when nothing is given to it
#
# phi, rho, pooling and d_u are identical across all four. Two things cannot be matched
# and are reported instead of being hidden: W_onehot needs m columns where the others
# need r_W (m one-hots do not fit in R^{r_W}), which widens phi's first layer; and
# W_learned_id adds m * r_W trainable parameters by construction.
W_SEMANTICS_ARMS: Dict[str, Tuple[str, bool]] = {
    "W_real": ("none", False),
    "W_random": ("random_semantics", False),
    "W_onehot": ("onehot_semantics", False),
    "W_learned_id": ("random_semantics", True),
}

# Offset applied to the realization index to pick each realization's control draw. Any
# fixed value would do; what matters is that it varies with the realization.
CONTROL_SEED_BASE = 1000


def _w_arm_config(base: DefaultConfig, trainable: bool) -> DefaultConfig:
    """`base` retargeted at the w_adapter rung, with the semantic table frozen or learned."""
    arm = _arm_config(base, "w_adapter")
    return dataclasses.replace(arm, w_semantics_trainable=trainable)


def w_adapter_param_counts(
    base: DefaultConfig,
    prior_path: Optional[str] = None,
    arms: Optional[Dict[str, Tuple[str, bool]]] = None,
) -> Dict[str, int]:
    """Trainable parameters in the W branch alone, per arm.

    Built rather than derived, so the number is the one the optimizer actually sees.
    Printed with the sweep because two of the four arms cannot be capacity-matched to
    the real one and a reader has to be able to see by how much.
    """
    from ..prior_modules.controls import apply_control
    from ..prior_modules.loader import load_prior_tensors
    from ..prior_modules.w_adapter import WSemanticAdapter

    arms = W_SEMANTICS_ARMS if arms is None else arms
    reference = load_ihdp_realization(1, "train")
    prior = load_prior_tensors(reference.feature_names, dataset=base.dataset, path=prior_path)

    counts: Dict[str, int] = {}
    for name, (control, trainable) in arms.items():
        _, q_tilde = apply_control(None, prior.q_tilde, control, CONTROL_SEED_BASE)
        cfg = _w_arm_config(base, trainable)
        adapter = WSemanticAdapter(
            q_tilde, cfg.d_u, cfg.d_token, cfg.phi_hidden, cfg.d_s, cfg.rho_hidden,
            cfg.activation, cfg.batchnorm, cfg.w_semantics_trainable,
        )
        counts[name] = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    return counts


def run_w_semantics_sweep(
    realizations: Sequence[int] = tuple(range(1, 31)),
    arms: Optional[Dict[str, Tuple[str, bool]]] = None,
    cfg: Optional[DefaultConfig] = None,
    prior_path: Optional[str] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
    verbose: bool = False,
    include_empirical: bool = True,
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    """Run every W-semantics arm on the same realizations, paired.

    The empirical host runs first as the reference the whole sweep is read against: an
    arm that does not beat it has nothing to explain.

    Each realization draws its own control seed (`CONTROL_SEED_BASE + realization`), so
    the stochastic arms are an ensemble of draws rather than one draw scored once per
    realization. That is a deliberate departure from `run_ladder`, where the control seed
    is `cfg.seed` and therefore fixed: a single subspace measured N times cannot tell its
    own luck from its arm's behaviour, and the spread between draws was measured to be
    wider than the effects being reported.
    """
    base = cfg if cfg is not None else load_config(str(DEFAULT_CONFIG))
    arms = W_SEMANTICS_ARMS if arms is None else arms

    plan: List[Tuple[str, DefaultConfig, str]] = []
    if include_empirical:
        plan.append(("empirical", _arm_config(base, "empirical"), "none"))
    for name, (control, trainable) in arms.items():
        plan.append((name, _w_arm_config(base, trainable), control))

    rows: Dict[str, List[Dict[str, float]]] = {}
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for label, arm_cfg, control in plan:
        runs = [
            run_realization(
                r, arm_cfg, prior_path, val_fraction, seed, verbose, control,
                control_seed=CONTROL_SEED_BASE + r,
            )
            for r in realizations
        ]
        rows[label] = runs
        summaries[label] = aggregate(runs)
    return rows, summaries


def format_w_sweep(
    rows: Dict[str, List[Dict[str, float]]],
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    param_counts: Optional[Dict[str, int]] = None,
    baseline: str = "empirical",
    reference: str = "W_real",
) -> str:
    """The W-semantics table: one row per arm, with both paired win columns.

    Two comparisons, because one alone is not enough to conclude anything. `vs host` is
    whether the arm buys anything over the empirical model, and `vs real` is whether the
    real prior's geometry is doing the buying - an arm that matches W_real means the LLM
    contributed nothing that a table of that shape could not.
    """
    n = len(next(iter(rows.values()))) if rows else 0
    lines = [
        f"W semantics sweep - {n} paired realization(s), fixed 672/75 partition",
        "-" * 103,
        f"{'arm':<15}{'out PEHE med':>13}{'mean':>8}{'vs host':>9}{'vs real':>9}"
        f"{'out epsATE':>12}{'SMD red':>9}{'|a_W|':>9}{'W params':>10}",
    ]

    def stat(label: str, key: str, field: str) -> float:
        entry = summaries[label].get(key)
        return float("nan") if entry is None else entry[field]

    base_runs = rows.get(baseline)
    real_runs = rows.get(reference)
    inert: List[str] = []
    for label in rows:
        vs_host = "-"
        if base_runs is not None and label != baseline:
            vs_host = _paired_wins(rows[label], base_runs, "out_pehe")
        vs_real = "-"
        if real_runs is not None and label not in (baseline, reference):
            vs_real = _paired_wins(rows[label], real_runs, "out_pehe")
        params = "-" if param_counts is None else str(param_counts.get(label, "-"))
        lines.append(
            f"{label:<15}{stat(label, 'out_pehe', 'median'):>13.3f}"
            f"{stat(label, 'out_pehe', 'mean'):>8.3f}{vs_host:>9}{vs_real:>9}"
            f"{stat(label, 'out_eps_ate', 'median'):>12.3f}"
            f"{stat(label, 'pool_smd_reduction', 'median'):>9.3f}"
            f"{stat(label, 'pool_a_W_norm', 'median'):>9.3f}{params:>10}"
        )
        if label != baseline and stat(label, "pool_a_W_norm", "median") == 0.0:
            inert.append(label)

    lines.append("-" * 103)
    for label in inert:
        lines.append(f"WARNING {label}: |a_W| is exactly 0 - this arm trained as the empirical model")
    lines.append(f"vs host = realizations where the arm beats {baseline!r} on out PEHE (lower is better)")
    lines.append(f"vs real = the same against {reference!r}; a tie there means the LLM geometry is not the cause")
    lines.append(
        "W params = trainable parameters in the W branch alone. W_onehot needs m columns\n"
        "  where the others need r_W, which widens phi's first layer; W_learned_id adds the\n"
        "  m x r_W table itself. Neither can be matched away - read the win columns against them."
    )
    return "\n".join(lines)


# -- the single-variant benchmark table -------------------------------------


def format_loss_budget(summary: Dict[str, Dict[str, float]]) -> str:
    """What fraction of the objective each loss term commanded at the last epoch.

    Printed with every benchmark because a nominal weight is not readable on its own:
    the terms have natural magnitudes differing by orders of magnitude, so equal
    weights do not mean equal influence. Shares are dimensionless and therefore
    comparable across realizations whose outcome scales differ.
    """
    rows = [
        ("L_fact", "factual"),
        ("L_mmd", "balancing (MMD)"),
        ("L_sparse", "sparsity (L1)"),
        ("L_rec", "reconstruction"),
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


def format_benchmark(summary: Dict[str, Dict[str, float]], n_realizations: int) -> str:
    """The comparison table: this run over the published rows, same protocol."""
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

    # The mean first, in the outcome's own units, because that is what the published
    # rows report. The median sits under it because on this benchmark the two disagree
    # by about a factor of two.
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
        # Qualifies the SMD reduction above it: balance is free at zero norm.
        ("pool_balance_norm", "|u_shared| (pool)"),
        # Exactly zero on the empirical variant by construction, so printing them on
        # every rung makes an inert semantic branch visible.
        ("pool_c_norm", "|c| structural (pool)"),
        ("pool_a_W_norm", "|a_W| semantic (pool)"),
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


def main(argv: Optional[List[str]] = None) -> None:
    """Run the IHDP benchmark, or the paired ladder when several variants are asked for."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the IHDP benchmark protocol.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="path to a per-dataset YAML config (default: the shipped ihdp.yaml)",
    )
    parser.add_argument("--prior", default=None, help="path to a prior_bundle.npz")
    parser.add_argument(
        "--realizations",
        type=int,
        default=N_REALIZATIONS,
        help=f"how many realizations to run, from 1 (max {N_REALIZATIONS})",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="comma-separated ladder arms (default: the config's single variant)",
    )
    parser.add_argument(
        "--controls",
        default="none",
        help="comma-separated negative controls to run beside each prior-using arm",
    )
    parser.add_argument(
        "--w-sweep",
        action="store_true",
        help="run the four W-semantics arms (real / random / one-hot / learned) instead "
             "of the ladder, paired on the same realizations",
    )
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    overrides = {} if args.epochs is None else {"epochs": args.epochs}
    cfg = load_config(args.config, **overrides)
    realizations = tuple(range(1, args.realizations + 1))

    if args.w_sweep:
        rows, summaries = run_w_semantics_sweep(
            realizations, None, cfg, args.prior, args.val_fraction, args.seed, args.verbose
        )
        counts = w_adapter_param_counts(cfg, args.prior)
        print(format_w_sweep(rows, summaries, counts))
        return

    if args.variants is None:
        summary, _ = run_benchmark(
            cfg, realizations, args.prior, args.val_fraction, args.seed, args.verbose
        )
        print(format_benchmark(summary, len(realizations)))
        return

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    controls = tuple(c.strip() for c in args.controls.split(",") if c.strip())
    rows, summaries = run_ladder(
        variants, realizations, controls, cfg, args.prior,
        args.val_fraction, args.seed, args.verbose,
    )
    print(format_ladder(rows, summaries, baseline=variants[0]))


if __name__ == "__main__":
    main()
