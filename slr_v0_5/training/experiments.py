"""The paired v0.5.1 sweep: two prior mechanisms, each against its matched controls.

Seven arms over the same IHDP realizations, sharing split, standardizer, seed,
architecture and trainable parameter count:

    wide_empirical                   eta_prior = 0, no prior tensor - the wide-latent reference
    direct_*                         p = (x @ Z) / sqrt(m), the 0.5 mechanism
    attention_*                      p = (x * m*alpha) @ Z / sqrt(m), the 0.5.1 mechanism
    *_real                           the real feature-to-embedding assignment
    *_permuted                       the same vectors, rows permuted per realization
    *_permuted_norm_matched          the same permutation, rescaled on the training split
                                     so its semantic vector has the real arm's RMS magnitude

Two questions are being asked at once and are kept apart: whether attention is a
better integration mechanism (attention_real vs direct_real) and whether the semantic
assignment matters inside each mechanism (*_real vs *_permuted_norm_matched). A
mechanism gain with no assignment separation is a fact about attention, not about the
embeddings.

One table per eta_prior. The same realizations recur at every eta, so the tables are
not independent observations of the same effect and are never pooled.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ssae_v3.data.ihdp import N_REALIZATIONS, has_replication_set, load_ihdp_realization

from ..config import SLRConfig, load_config
from ..model import SLRCFRv05
from ..prior.artifacts import FeatureEmbeddings, load_feature_embeddings
from .runner import (
    DEFAULT_VAL_FRACTION,
    aggregate,
    build_model,
    run_realization,
    standardized_splits,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ihdp.yaml"
DEFAULT_ETA_PRIOR = 0.1

# arm name -> (prior integration, embedding variant). The host's integration is
# irrelevant - with no embedding variant no prior module is built at all - but it is
# named rather than left blank so every arm reads the same way.
ARMS: Dict[str, Tuple[str, str]] = {
    "wide_empirical":                  ("direct_embedding", "none"),
    "direct_real":                     ("direct_embedding", "real"),
    "direct_permuted":                 ("direct_embedding", "permuted"),
    "direct_permuted_norm_matched":    ("direct_embedding", "permuted_norm_matched"),
    "attention_real":                  ("cosine_cross_attention", "real"),
    "attention_permuted":              ("cosine_cross_attention", "permuted"),
    "attention_permuted_norm_matched": ("cosine_cross_attention", "permuted_norm_matched"),
}

# The 0.5 sweep named its direct arms differently. A stored table can still be paired
# with a new one, so the old names are translated on load rather than stranding those
# runs - the direct path itself is unchanged.
_LEGACY_ARM_NAMES = {
    "qwen_real": "direct_real",
    "qwen_permuted": "direct_permuted",
    "qwen_permuted_norm_matched": "direct_permuted_norm_matched",
}

# (arm, base, question), in the order the 0.5.1 plan asks them. The formatter skips a
# comparison whose arms were not both run.
_COMPARISONS: Tuple[Tuple[str, str, str], ...] = (
    ("attention_real", "direct_real",
     "A: does attention improve the integration mechanism"),
    ("attention_real", "attention_permuted_norm_matched",
     "B: does the semantic assignment matter inside attention"),
    ("direct_real", "direct_permuted_norm_matched",
     "C: does the semantic assignment matter inside direct fusion"),
    ("direct_real", "wide_empirical",
     "D: does the real direct prior beat the wide-latent host"),
    ("attention_real", "wide_empirical",
     "D: does the real attention prior beat the wide-latent host"),
    ("attention_real", "attention_permuted",
     "the unmatched control, kept because it is free"),
    ("direct_real", "direct_permuted",
     "the unmatched control, kept because it is free"),
)

# Reported paired keys: the causal metric and the oracle-free criterion. Selection may
# only ever use the second.
_PAIRED_KEYS = ("out_pehe", "val_factual_objective_normalized")


def arm_config(cfg: SLRConfig, arm: str, eta_prior: float, d_latent: int) -> SLRConfig:
    """`cfg` retargeted at one arm: its mechanism, embedding variant, eta and width.

    The reference arm carries no prior tensor, so its eta is 0 whatever the sweep
    value is; every other field is shared with the other arms.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    integration, variant = ARMS[arm]
    return dataclasses.replace(
        cfg,
        prior_integration=integration,
        embedding_variant=variant,
        eta_prior=0.0 if variant == "none" else float(eta_prior),
        d_latent=int(d_latent),
    )


def arm_parameter_counts(
    cfg: SLRConfig,
    embeddings: FeatureEmbeddings,
    arms: Sequence[str],
    eta_prior: float,
) -> Dict[str, int]:
    """Trainable parameters in one untrained model per arm.

    Built rather than derived, so the number is the one the optimizer would see. The
    prior owns no trainable parameter, so every arm should come out identical;
    `run_sweep` checks this rather than assuming it.
    """
    train_split, _, _, _ = standardized_splits(1, DEFAULT_VAL_FRACTION, seed=0)
    counts: Dict[str, int] = {}
    for arm in arms:
        model = build_model(
            arm_config(cfg, arm, eta_prior, embeddings.d_q), train_split, embeddings, control_seed=0
        )
        counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


def load_embeddings(cfg: SLRConfig, embedding_dir: Optional[Path]) -> FeatureEmbeddings:
    feature_names = load_ihdp_realization(1, "train").feature_names
    return load_feature_embeddings(feature_names, dataset=cfg.dataset, embedding_dir=embedding_dir)


def run_sweep(
    realizations: Sequence[int] = tuple(range(1, 31)),
    arms: Sequence[str] = tuple(ARMS),
    cfg: Optional[SLRConfig] = None,
    embeddings: Optional[FeatureEmbeddings] = None,
    eta_prior: float = DEFAULT_ETA_PRIOR,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]], Dict[str, int]]:
    """Run every arm on the same realizations, paired.

    Returns (per-arm run lists, per-arm aggregate summaries, per-arm parameter counts).
    The embedding artifact is loaded once and controlled per realization.
    """
    if not has_replication_set():
        raise FileNotFoundError(
            "the IHDP replication set is missing; download ihdp_npci_1-100.train.npz "
            "and ihdp_npci_1-100.test.npz from https://www.fredjo.com/ into data/raw/IHDP/"
        )
    base = cfg if cfg is not None else load_config(str(DEFAULT_CONFIG))
    if embeddings is None:
        embeddings = load_embeddings(base, None)

    param_counts = arm_parameter_counts(base, embeddings, arms, eta_prior)
    distinct = set(param_counts.values())
    if len(distinct) != 1:
        raise ValueError(f"arms are not parameter-matched: {param_counts}")

    rows: Dict[str, List[Dict[str, float]]] = {}
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for arm in arms:
        cfg_arm = arm_config(base, arm, eta_prior, embeddings.d_q)
        runs = []
        for r in realizations:
            run = run_realization(
                r, cfg_arm, embeddings=embeddings, val_fraction=val_fraction,
                seed=seed, verbose=verbose,
            )
            run["embedding_variant"] = arm
            run["trainable_parameter_count"] = float(param_counts[arm])
            runs.append(run)
        rows[arm] = runs
        summaries[arm] = aggregate(runs)
    return rows, summaries, param_counts


# -- paired statistics -----------------------------------------------------


def sign_test_p(wins: int, total: int) -> float:
    """Two-sided exact binomial sign-test p-value against p = 0.5. No scipy."""
    if total == 0:
        return 1.0
    if not (0 <= wins <= total):
        raise ValueError(f"wins must be in [0, total={total}]; got {wins}")
    k = min(wins, total - wins)
    tail = sum(math.comb(total, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail * (0.5 ** total))


def paired_wins(
    arm: Sequence[Dict[str, float]], base: Sequence[Dict[str, float]], key: str
) -> Tuple[int, int, float]:
    """(wins, total, sign-test p) of `arm` against `base` on `key`, lower is better."""
    by_real = {run["realization"]: run for run in base}
    wins = total = 0
    for run in arm:
        other = by_real.get(run["realization"])
        if other is None or key not in run or key not in other:
            continue
        if math.isfinite(run[key]) and math.isfinite(other[key]):
            total += 1
            wins += int(run[key] < other[key])
    return wins, total, sign_test_p(wins, total)


def format_sweep(
    rows: Dict[str, List[Dict[str, float]]],
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    eta_prior: float,
    param_counts: Dict[str, int],
    embeddings: FeatureEmbeddings,
) -> str:
    """The sweep table and the paired comparisons. Nothing here declares a winner."""
    n = len(next(iter(rows.values()))) if rows else 0
    meta = embeddings.metadata
    lines = [
        f"SLR-CFR v0.5.1 prior integration - {n} paired realization(s), eta_prior={eta_prior}",
        "-" * 139,
        f"{'arm':<33}{'out PEHE med':>13}{'mean':>8}{'vs host':>9}{'out epsATE':>12}"
        f"{'pool PEHE':>11}{'val obj':>10}{'SMD red':>9}{'|u_out|':>9}{'prior/emp':>11}"
        f"{'att H/logm':>12}{'eff feat':>10}{'stop':>7}",
    ]

    def stat(arm: str, key: str, field: str) -> float:
        entry = summaries.get(arm, {}).get(key)
        return float("nan") if entry is None else entry[field]

    base_runs = rows.get("wide_empirical")
    for arm in rows:
        vs_host = "-"
        if base_runs is not None and arm != "wide_empirical":
            wins, total, _ = paired_wins(rows[arm], base_runs, "out_pehe")
            vs_host = f"{wins}/{total}" if total else "-"
        lines.append(
            f"{arm:<33}{stat(arm, 'out_pehe', 'median'):>13.3f}{stat(arm, 'out_pehe', 'mean'):>8.3f}"
            f"{vs_host:>9}{stat(arm, 'out_eps_ate', 'median'):>12.3f}"
            f"{stat(arm, 'pool_pehe', 'median'):>11.3f}"
            f"{stat(arm, 'val_factual_objective_normalized', 'median'):>10.4f}"
            f"{stat(arm, 'pool_smd_reduction', 'median'):>9.3f}"
            f"{stat(arm, 'out_u_out_norm', 'median'):>9.2f}"
            f"{stat(arm, 'out_prior_to_empirical_norm_ratio', 'median'):>11.3f}"
            f"{stat(arm, 'out_attention_entropy_normalized', 'median'):>12.4f}"
            f"{stat(arm, 'out_effective_feature_count_mean', 'median'):>10.2f}"
            f"{stat(arm, 'stopped_at_epoch', 'median'):>7.0f}"
        )
    lines.append("-" * 139)

    distinct = set(param_counts.values())
    if len(distinct) == 1:
        lines.append(f"trainable parameters: {next(iter(distinct))} (identical across every arm)")
    else:
        lines.append(f"trainable parameters differ across arms: {param_counts}")
    centered = {bool(run.get("center_embeddings", 0.0)) for runs in rows.values() for run in runs}
    lines.append(
        f"embeddings: m={meta['m']} d_q={meta['d_q']} model={meta['model_name']} "
        f"pooling={meta['pooling']} dtype={meta['dtype']} "
        f"centered={sorted(centered) if len(centered) != 1 else next(iter(centered))}"
    )
    lines.append(f"artifact: {meta['source']} sha256={meta['artifact_sha256'][:16]}")
    taus = sorted({
        run["out_attention_temperature"] for runs in rows.values() for run in runs
        if "out_attention_temperature" in run
    })
    if taus:
        lines.append(f"attention temperature: {taus if len(taus) != 1 else taus[0]}")

    lines.append("")
    lines.append("paired comparisons (wins/total for the left arm, lower is better, sign-test p):")
    for arm, base, question in _COMPARISONS:
        if arm not in rows or base not in rows:
            continue
        cells = []
        for key in _PAIRED_KEYS:
            wins, total, p = paired_wins(rows[arm], rows[base], key)
            cells.append(f"{key} {wins}/{total} p={p:.4f}")
        lines.append(f"  {arm:<32} vs {base:<32} " + "  ".join(cells))
        lines.append(f"      ({question})")

    lines.append("")
    lines.append(
        "the same realizations recur at every eta_prior, so tables from different eta "
        "values are not independent observations and are not pooled; any operating-point "
        "selection must use val_factual_objective_normalized, never PEHE or true CATE"
    )
    return "\n".join(lines)


# -- merging a previous run ------------------------------------------------


def load_previous_tables(path: Path) -> Dict[float, Dict[str, List[Dict[str, float]]]]:
    """Per-eta arm rows from a JSON file a previous sweep wrote.

    Lets a new arm be added to an experiment without re-running the arms that are
    already measured: the pairing is by realization, and the arms share the split, the
    seed and the initial weights by construction, so rows from two processes pair
    exactly as rows from one do.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        float(table["eta_prior"]): {
            _LEGACY_ARM_NAMES.get(arm, arm): runs for arm, runs in table["runs"].items()
        }
        for table in payload["tables"]
    }


def merge_previous_arms(
    rows: Dict[str, List[Dict[str, float]]],
    summaries: Dict[str, Dict[str, Dict[str, float]]],
    param_counts: Dict[str, int],
    previous: Dict[str, List[Dict[str, float]]],
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, Dict[str, Dict[str, float]]], Dict[str, int], List[str]]:
    """Fold previously measured arms into this run's tables, in canonical arm order.

    An arm that was just run wins over a stored copy of itself, and the caller is told
    which ones were dropped. Merged arms must cover the same realizations and the same
    centering decision as the fresh ones, or they are not paired observations and the
    merge is refused.
    """
    fresh_realizations = {
        frozenset(run["realization"] for run in runs) for runs in rows.values()
    }
    fresh_centered = {run.get("center_embeddings") for runs in rows.values() for run in runs}
    dropped = []

    merged_rows: Dict[str, List[Dict[str, float]]] = {}
    merged_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    merged_counts = dict(param_counts)
    for arm in ARMS:
        if arm in rows:
            if arm in previous:
                dropped.append(arm)
            merged_rows[arm] = rows[arm]
            merged_summaries[arm] = summaries[arm]
            continue
        if arm not in previous:
            continue
        runs = previous[arm]
        realizations = frozenset(run["realization"] for run in runs)
        if fresh_realizations and realizations not in fresh_realizations:
            raise ValueError(
                f"merged arm {arm!r} covers different realizations than this run; "
                "the rows would not be paired"
            )
        centered = {run.get("center_embeddings") for run in runs}
        if fresh_centered and centered != fresh_centered:
            raise ValueError(
                f"merged arm {arm!r} was run with center_embeddings={centered} but this "
                f"run used {fresh_centered}"
            )
        merged_rows[arm] = runs
        merged_summaries[arm] = aggregate(runs)
        counts = {int(run["trainable_parameter_count"]) for run in runs if "trainable_parameter_count" in run}
        if len(counts) == 1:
            merged_counts[arm] = next(iter(counts))
    return merged_rows, merged_summaries, merged_counts, dropped


# -- CLI -------------------------------------------------------------------


def _parse_arms(spec: Optional[str]) -> List[str]:
    if spec is None:
        return list(ARMS)
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in names if name not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; choose from {sorted(ARMS)}")
    return names


def _parse_grid(spec: Optional[str], single: float) -> List[float]:
    if spec is None:
        return [single]
    return [float(v) for v in spec.split(",") if v.strip()]


def main(argv: Optional[List[str]] = None) -> None:
    """Run the paired v0.5 sweep, one table per eta_prior."""
    parser = argparse.ArgumentParser(description="Run the paired SLR-CFR v0.5 sweep.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--realizations", type=int, default=30, help="how many realizations, from 1")
    parser.add_argument("--arms", default=None, help="comma-separated arm names (default: all three)")
    parser.add_argument(
        "--embedding-dir", default=None, help="directory holding feature_embeddings.pt"
    )
    parser.add_argument("--eta-prior", type=float, default=DEFAULT_ETA_PRIOR)
    parser.add_argument(
        "--attention-query",
        default=None,
        choices=("mu_emp", "p_direct"),
        help="what the attention queries with (default: whatever the config says)",
    )
    parser.add_argument(
        "--attention-norm-match",
        dest="attention_norm_match",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="rescale each patient's attention prior to the direct prior's magnitude",
    )
    parser.add_argument(
        "--attention-temperature",
        type=float,
        default=None,
        help="softmax temperature of the cosine attention (default: whatever the config says)",
    )
    parser.add_argument(
        "--center-embeddings",
        dest="center_embeddings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="subtract the mean FeatureCard vector from Z (default: whatever the config says)",
    )
    parser.add_argument(
        "--eta-grid", default=None, help="comma-separated eta_prior values, overrides --eta-prior"
    )
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default=None, help="write the runs and summaries as JSON")
    parser.add_argument(
        "--merge-json",
        default=None,
        help="a previous sweep's JSON; its arms are shown and paired beside the ones run now",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    overrides = {} if args.epochs is None else {"epochs": args.epochs}
    if args.center_embeddings is not None:
        overrides["center_embeddings"] = args.center_embeddings
    if args.attention_temperature is not None:
        overrides["attention_temperature"] = args.attention_temperature
    if args.attention_norm_match is not None:
        overrides["attention_norm_match"] = args.attention_norm_match
    if args.attention_query is not None:
        overrides["attention_query"] = args.attention_query
    cfg = load_config(args.config, **overrides)
    realizations = tuple(range(1, min(args.realizations, N_REALIZATIONS) + 1))
    arms = _parse_arms(args.arms)
    embedding_dir = None if args.embedding_dir is None else Path(args.embedding_dir)
    embeddings = load_embeddings(cfg, embedding_dir)

    previous = {} if args.merge_json is None else load_previous_tables(Path(args.merge_json))

    payload = {"config": cfg.to_dict(), "embeddings": dict(embeddings.metadata), "tables": []}
    for eta_prior in _parse_grid(args.eta_grid, args.eta_prior):
        rows, summaries, param_counts = run_sweep(
            realizations, arms, cfg, embeddings, eta_prior, args.val_fraction, args.seed, args.verbose
        )
        if eta_prior in previous:
            rows, summaries, param_counts, dropped = merge_previous_arms(
                rows, summaries, param_counts, previous[eta_prior]
            )
            if dropped:
                print(f"note: {dropped} were run now, so the stored copies were not merged")
        elif previous:
            print(f"note: the merge file holds no table at eta_prior={eta_prior}")
        print(format_sweep(rows, summaries, eta_prior, param_counts, embeddings))
        print()
        payload["tables"].append({
            "eta_prior": eta_prior,
            "runs": rows,
            "summaries": summaries,
            "parameter_counts": param_counts,
        })
        # Written after every table rather than once at the end: a grid is long enough
        # that an interrupted run should still leave the etas it did finish on disk.
        if args.json_out is not None:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            print(f"wrote {args.json_out} ({len(payload['tables'])} table(s) so far)")


if __name__ == "__main__":
    main()
