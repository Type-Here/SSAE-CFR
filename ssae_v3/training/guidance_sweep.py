"""The paired prior-guidance sweep: real embedding/graph priors against matched false
ones, on the unmodified empirical backbone (`SSAECFRv4`).

Ten arms, each a (embedding variant, graph variant) pair. Every arm shares the same
realizations, split, standardizer and model seed, so a difference between arms is a
difference in the prior alone - the real embedding subspace and the real expert graph
are loaded once with `load_guidance_bundle` and controlled per realization, never
reloaded per arm.

    empirical                    no guidance at all - the plain empirical host
    u_real / u_random            embedding subspace only, real vs a random rank-matched one
    graph_real / graph_permuted / graph_degree_matched
                                  graph subspace only, real vs two degree/rank-matched controls
    both_real                    both real sources together
    u_real_graph_degree_matched  real embedding, false graph
    u_random_graph_real          false embedding, real graph
    both_random                  both false

This module runs the sweep and prints paired win counts and sign-test p values. It
does not decide anything: no line here declares a winner.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..data.ihdp import N_REALIZATIONS, load_ihdp_realization
from ..hparams import DefaultConfig, load_config
from ..model_v4 import SSAECFRv4
from ..prior_modules.guidance_bundle import PriorGuidanceBundle, load_guidance_bundle
from ..prior_modules.guidance_controls import (
    EMBEDDING_VARIANTS,
    GRAPH_VARIANTS,
    apply_guidance_controls,
)
from ..prior_modules.prior_guidance import PriorGuidance
from . import experiments
from .evaluate import aggregate

DEFAULT_GAMMA_PRIOR = 0.5
DEFAULT_LAMBDA_PRIOR = 0.1

# (embedding variant, graph variant) per arm. Both "none" is the plain empirical host;
# every other arm is parameter-matched to it, since guidance owns no trainable weight.
GUIDANCE_ARMS: Dict[str, Tuple[str, str]] = {
    "empirical":                   ("none",   "none"),
    "u_real":                      ("real",   "none"),
    "u_random":                    ("random", "none"),
    "graph_real":                  ("none",   "real"),
    "graph_permuted":              ("none",   "permuted"),
    "graph_degree_matched":        ("none",   "degree_matched"),
    "both_real":                   ("real",   "real"),
    "u_real_graph_degree_matched": ("real",   "degree_matched"),
    "u_random_graph_real":         ("random", "real"),
    "both_random":                 ("random", "degree_matched"),
}

# The paired comparisons that actually separate a real prior from a matched false one.
# Each entry is (arm, base, question); a comparison whose arms were not both run is
# skipped by the formatter rather than raising.
_COMPARISONS: Tuple[Tuple[str, str, str], ...] = (
    ("u_real", "u_random",
     "does the real embedding subspace beat a random one of the same rank"),
    ("graph_real", "graph_permuted",
     "does putting the observed features in the right graph positions matter"),
    ("graph_real", "graph_degree_matched",
     "does the expert grouping beat a graph of the same degree structure"),
    ("both_real", "u_real",
     "does the graph add anything beyond the embedding subspace"),
    ("both_real", "graph_real",
     "does the embedding geometry add anything beyond the graph"),
    ("both_real", "u_real_graph_degree_matched",
     "which source carries a combined gain"),
    ("both_real", "u_random_graph_real",
     "the same question from the other side"),
)


def prior_mode_for(embedding_variant: str, graph_variant: str) -> str:
    """The `PriorGuidance` mode implied by one (embedding, graph) control pair.

    "both" when both sources are active, "embedding"/"graph" when only one is, "none"
    when neither is. Raises on a variant name neither controls module recognizes.
    """
    if embedding_variant not in EMBEDDING_VARIANTS:
        raise ValueError(
            f"unknown embedding variant {embedding_variant!r}; choose from {EMBEDDING_VARIANTS}"
        )
    if graph_variant not in GRAPH_VARIANTS:
        raise ValueError(
            f"unknown graph variant {graph_variant!r}; choose from {GRAPH_VARIANTS}"
        )
    embedding_active = embedding_variant != "none"
    graph_active = graph_variant != "none"
    if embedding_active and graph_active:
        return "both"
    if embedding_active:
        return "embedding"
    if graph_active:
        return "graph"
    return "none"


def build_arm_guidance(
    bundle: PriorGuidanceBundle,
    embedding_variant: str,
    graph_variant: str,
    gamma_prior: float,
    control_seed: int,
) -> PriorGuidance:
    """The `PriorGuidance` for one arm: apply the requested controls, then wire in only
    the source(s) that arm's mode actually uses.

    `PriorGuidance` rejects a prior its mode does not use, so passing an inactive
    source would be a construction error, not a silent no-op - the "none" branches of
    each source are left out of the keyword call entirely rather than passed as None.
    """
    controlled = apply_guidance_controls(bundle, embedding_variant, graph_variant, control_seed)
    mode = prior_mode_for(embedding_variant, graph_variant)
    kwargs: dict = {}
    if embedding_variant != "none":
        kwargs["p_u"] = controlled.P_U
        kwargs["q_u"] = controlled.U_k
    if graph_variant != "none":
        kwargs["p_c"] = controlled.P_C
        kwargs["q_c"] = controlled.Q_C
    return PriorGuidance(mode=mode, gamma_prior=gamma_prior, **kwargs)


def _check_base_cfg(base: DefaultConfig) -> None:
    if base.model_variant != "empirical" or base.use_u_adapter or base.use_w_adapter or base.use_expert_adapter:
        raise ValueError(
            "the guidance sweep runs on the unmodified empirical backbone: "
            f"model_variant={base.model_variant!r} use_u_adapter={base.use_u_adapter!r} "
            f"use_w_adapter={base.use_w_adapter!r} use_expert_adapter={base.use_expert_adapter!r} "
            "would leave the arms not parameter-matched"
        )


def _load_bundle(base: DefaultConfig, expert_dir: Optional[Path]) -> PriorGuidanceBundle:
    feature_names = load_ihdp_realization(1, "train").feature_names
    return load_guidance_bundle(feature_names, dataset=base.dataset, expert_dir=expert_dir)


def arm_parameter_counts(
    cfg: DefaultConfig,
    bundle: PriorGuidanceBundle,
    arms: Optional[Dict[str, Tuple[str, str]]] = None,
    gamma_prior: float = 0.0,
) -> Dict[str, int]:
    """Trainable parameters in one untrained `SSAECFRv4` per arm.

    Built rather than derived, so the number is the one the optimizer actually sees.
    Guidance owns zero trainable parameters by construction, so every arm should come
    out identical; `run_guidance_sweep` asserts this rather than assuming it.
    """
    _check_base_cfg(cfg)
    arms = GUIDANCE_ARMS if arms is None else arms
    base = dataclasses.replace(cfg, in_channels=bundle.m)

    counts: Dict[str, int] = {}
    for name, (embedding_variant, graph_variant) in arms.items():
        mode = prior_mode_for(embedding_variant, graph_variant)
        arm_cfg = dataclasses.replace(base, prior_mode=mode, gamma_prior=gamma_prior)
        guidance = build_arm_guidance(bundle, embedding_variant, graph_variant, gamma_prior, control_seed=0)
        model = SSAECFRv4(arm_cfg, guidance=guidance)
        counts[name] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


def run_guidance_sweep(
    realizations: Sequence[int] = tuple(range(1, 31)),
    arms: Optional[Dict[str, Tuple[str, str]]] = None,
    cfg: Optional[DefaultConfig] = None,
    expert_dir: Optional[Path] = None,
    gamma_prior: float = DEFAULT_GAMMA_PRIOR,
    lambda_prior: float = DEFAULT_LAMBDA_PRIOR,
    val_fraction: float = experiments.DEFAULT_VAL_FRACTION,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]]]:
    """Run every guidance arm on the same realizations, paired.

    Returns (per-arm run lists, per-arm aggregate summaries), keyed by arm name. The
    real embedding subspace and the real expert graph are loaded once and controlled
    per realization, so every arm reads the same artifact and a difference between
    arms is a difference in the prior alone. Each realization draws its own control
    seed (`CONTROL_SEED_BASE + realization`), never a single control measured
    repeatedly - the spread between control draws has been measured to be wider than
    the effects this sweep is trying to detect.
    """
    base = cfg if cfg is not None else load_config(str(experiments.DEFAULT_CONFIG))
    _check_base_cfg(base)
    arms = GUIDANCE_ARMS if arms is None else arms
    for embedding_variant, graph_variant in arms.values():
        prior_mode_for(embedding_variant, graph_variant)  # validates the variant names

    bundle = _load_bundle(base, expert_dir)

    param_counts = arm_parameter_counts(base, bundle, arms, gamma_prior)
    distinct_counts = set(param_counts.values())
    if len(distinct_counts) != 1:
        majority = max(distinct_counts, key=lambda n: sum(1 for c in param_counts.values() if c == n))
        offenders = {name: n for name, n in param_counts.items() if n != majority}
        raise ValueError(
            f"guidance arms are not parameter-matched (expected {majority} everywhere): {offenders}"
        )

    rows: Dict[str, List[Dict[str, float]]] = {}
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, (embedding_variant, graph_variant) in arms.items():
        mode = prior_mode_for(embedding_variant, graph_variant)
        arm_cfg = dataclasses.replace(base, prior_mode=mode, gamma_prior=gamma_prior, lambda_prior=lambda_prior)

        runs: List[Dict[str, float]] = []
        for r in realizations:
            control_seed = experiments.CONTROL_SEED_BASE + r
            guidance = build_arm_guidance(bundle, embedding_variant, graph_variant, gamma_prior, control_seed)
            run = experiments.run_realization(
                r, arm_cfg, val_fraction=val_fraction, seed=seed, verbose=verbose, guidance=guidance,
            )
            run["gamma_prior"] = float(gamma_prior)
            run["lambda_prior"] = float(lambda_prior)
            run["control_seed"] = float(control_seed)
            run["embedding_rank"] = float(guidance.embedding_rank)
            run["graph_rank"] = float(guidance.graph_rank)
            run["active_rank"] = float(guidance.active_rank)
            run["trainable_parameter_count"] = float(param_counts[name])
            runs.append(run)
        rows[name] = runs
        summaries[name] = aggregate(runs)
    return rows, summaries


def sign_test_p(wins: int, total: int) -> float:
    """Two-sided exact binomial sign-test p-value against p = 0.5. No scipy.

    `total == 0` returns 1.0: no data can never reject the null.
    """
    if total == 0:
        return 1.0
    if not (0 <= wins <= total):
        raise ValueError(f"wins must be in [0, total={total}]; got {wins}")
    k = min(wins, total - wins)
    tail = sum(math.comb(total, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail * (0.5 ** total))


def _wins_and_p(
    arm_runs: Sequence[Dict[str, float]],
    base_runs: Sequence[Dict[str, float]],
    key: str = "out_pehe",
) -> Tuple[int, int, float]:
    """(wins, total, sign-test p) of `arm_runs` against `base_runs`, paired by realization.

    Parses `experiments._paired_wins`'s own "wins/total" string rather than
    re-walking the pairing logic a second time.
    """
    label = experiments._paired_wins(arm_runs, base_runs, key)
    if label == "-":
        return 0, 0, 1.0
    wins_str, total_str = label.split("/")
    wins, total = int(wins_str), int(total_str)
    return wins, total, sign_test_p(wins, total)


def format_guidance_sweep(
    rows: Dict[str, List[Dict[str, float]]],
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    gamma_prior: float,
    lambda_prior: float,
    param_counts: Optional[Dict[str, int]] = None,
    bundle: Optional[PriorGuidanceBundle] = None,
) -> str:
    """The guidance-sweep table, followed by the paired comparisons that answer the
    question this sweep exists to ask.

    Prints paired win counts and sign-test p values only - it does not declare a
    winner, interpret a result, or print an encouraging remark.
    """
    n = len(next(iter(rows.values()))) if rows else 0
    lines = [
        f"prior guidance sweep - {n} paired realization(s), "
        f"gamma_prior={gamma_prior}, lambda_prior={lambda_prior}",
        "-" * 120,
        f"{'arm':<28}{'out PEHE med':>13}{'mean':>8}{'vs host':>9}{'out epsATE':>12}"
        f"{'val obj':>10}{'SMD red':>9}{'align':>8}{'guide L':>9}{'delta':>8}",
    ]

    def stat(label: str, key: str, field: str) -> float:
        entry = summaries.get(label, {}).get(key)
        return float("nan") if entry is None else entry[field]

    base_runs = rows.get("empirical")
    for label in rows:
        vs_host = "-"
        if base_runs is not None and label != "empirical":
            wins, total, _ = _wins_and_p(rows[label], base_runs)
            vs_host = f"{wins}/{total}" if total else "-"
        lines.append(
            f"{label:<28}{stat(label, 'out_pehe', 'median'):>13.3f}"
            f"{stat(label, 'out_pehe', 'mean'):>8.3f}{vs_host:>9}"
            f"{stat(label, 'out_eps_ate', 'median'):>12.3f}"
            f"{stat(label, 'val_factual_objective_normalized', 'median'):>10.4f}"
            f"{stat(label, 'pool_smd_reduction', 'median'):>9.3f}"
            f"{stat(label, 'train_alignment_active', 'median'):>8.3f}"
            f"{stat(label, 'train_guidance_loss', 'median'):>9.4f}"
            f"{stat(label, 'train_x_guided_delta_ratio', 'median'):>8.3f}"
        )
    lines.append("-" * 120)

    if bundle is not None:
        lines.append(f"embedding rank {bundle.embedding_rank}, graph rank {bundle.graph_rank}")
    active_ranks = sorted({
        int(run["active_rank"]) for runs in rows.values() for run in runs if "active_rank" in run
    })
    if active_ranks:
        lines.append(f"active rank(s) across arms: {active_ranks}")
    if param_counts is not None:
        distinct = set(param_counts.values())
        if len(distinct) == 1:
            lines.append(f"trainable parameters: {next(iter(distinct))} (identical across every arm)")
        else:
            lines.append(f"trainable parameters differ across arms: {param_counts}")

    if bundle is not None:
        lines.append(f"source hashes: {bundle.source_hashes}")
        meta = bundle.graph_source_metadata
        if meta:
            lines.append(
                f"graph provenance: {meta.get('n_concepts')} concepts, {meta.get('n_edges')} edges, "
                f"degrees {meta.get('degrees')}, source {meta.get('source')}"
            )
        if meta.get("response_is_generated") is not True:
            lines.append(
                "NOTE the expert document was authored, not produced by a generation run; "
                "no number here is evidence about what a frozen model can produce on its own"
            )

    lines.append("")
    lines.append("paired comparisons (wins/total against the row on the left, sign-test p):")
    for arm, base, question in _COMPARISONS:
        if arm not in rows or base not in rows:
            continue
        wins, total, p = _wins_and_p(rows[arm], rows[base])
        lines.append(f"  {arm:<16} vs {base:<28} {wins}/{total}  p={p:.4f}  ({question})")

    lines.append("")
    lines.append(
        "this table was not selected on PEHE; selection must use the oracle-free "
        "val_factual_objective_normalized column"
    )
    return "\n".join(lines)


def _parse_arms(spec: Optional[str]) -> Dict[str, Tuple[str, str]]:
    if spec is None:
        return GUIDANCE_ARMS
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in names if name not in GUIDANCE_ARMS]
    if unknown:
        raise ValueError(f"unknown guidance arm(s) {unknown}; choose from {sorted(GUIDANCE_ARMS)}")
    return {name: GUIDANCE_ARMS[name] for name in names}


def _parse_grid(spec: Optional[str], single: float) -> List[float]:
    if spec is None:
        return [single]
    return [float(v) for v in spec.split(",") if v.strip()]


def main(argv: Optional[List[str]] = None) -> None:
    """Run the paired prior-guidance sweep, one table per (gamma_prior, lambda_prior) pair."""
    parser = argparse.ArgumentParser(description="Run the paired prior-guidance sweep.")
    parser.add_argument(
        "--config",
        default=str(experiments.DEFAULT_CONFIG),
        help="path to a per-dataset YAML config (default: the shipped ihdp.yaml)",
    )
    parser.add_argument(
        "--realizations", type=int, default=30, help="how many realizations to run, from 1"
    )
    parser.add_argument(
        "--arms", default=None, help="comma-separated arm names (default: all ten)"
    )
    parser.add_argument(
        "--expert-dir", default=None,
        help="directory holding feature_embeddings.pt and expert_prior.yaml",
    )
    parser.add_argument("--gamma-prior", type=float, default=DEFAULT_GAMMA_PRIOR)
    parser.add_argument("--lambda-prior", type=float, default=DEFAULT_LAMBDA_PRIOR)
    parser.add_argument(
        "--gamma-grid", default=None, help="comma-separated gamma_prior values, overrides --gamma-prior"
    )
    parser.add_argument(
        "--lambda-grid", default=None, help="comma-separated lambda_prior values, overrides --lambda-prior"
    )
    parser.add_argument("--val-fraction", type=float, default=experiments.DEFAULT_VAL_FRACTION)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    overrides = {} if args.epochs is None else {"epochs": args.epochs}
    cfg = load_config(args.config, **overrides)
    realizations = tuple(range(1, min(args.realizations, N_REALIZATIONS) + 1))
    arms = _parse_arms(args.arms)
    expert_dir = None if args.expert_dir is None else Path(args.expert_dir)

    # A run defines its own prior_mode per arm; a value left in the config file is
    # never carried through unmodified.
    gammas = _parse_grid(args.gamma_grid, args.gamma_prior)
    lambdas = _parse_grid(args.lambda_grid, args.lambda_prior)

    bundle = _load_bundle(cfg, expert_dir)
    for gamma_prior in gammas:
        for lambda_prior in lambdas:
            rows, summaries = run_guidance_sweep(
                realizations, arms, cfg, expert_dir, gamma_prior, lambda_prior,
                args.val_fraction, args.seed, args.verbose,
            )
            param_counts = arm_parameter_counts(cfg, bundle, arms, gamma_prior)
            print(format_guidance_sweep(rows, summaries, gamma_prior, lambda_prior, param_counts, bundle))
            print()


if __name__ == "__main__":
    main()
