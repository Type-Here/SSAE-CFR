"""IHDP with a concept-structured response surface.

The shipped IHDP replication set simulates its outcome with NPCI setting B:

    mu1 = x @ beta - omega        mu0 = exp((x + 0.5) @ beta)        y = mu(t) + N(0, 1)

where beta is drawn feature by feature, independently, from {0, .1, .2, .3, .4} with
probabilities (.6, .1, .1, .1, .1), afresh for every realization. Which covariates drive
the effect is therefore unrelated to what they mean or to how they relate to each other,
so no prior about the covariates can carry information about tau on that benchmark.

This module keeps everything the shipped files contain except the outcome: the same
units, covariates, treatment, train/test partition and realizations. Only beta changes.
With `alignment = 1` the coefficient is drawn per CONCEPT of the expert graph and shared
by every feature in that concept, so the outcome depends on how strongly each concept is
present in a unit - the situation in which knowing which features belong together is
genuinely useful. A feature in several concepts receives the sum of their coefficients;
a feature in none keeps an individual coefficient. `alignment = 0` redraws the original,
unstructured surface. Intermediate values mix the two coefficient vectors.

The output is a pair of .npz files in exactly the shipped format and under the shipped
names, so any model version that reads the replication set can be pointed at it by path
without a code change.

    python -m benchmarks.ihdp_structured --alignment 1.0 --out data/raw/IHDP_structured_a1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from ssae_v3.data.ihdp import FEATURE_NAMES, REPLICATION_PATHS
from ssae_v3.prior_modules.expert_bundle import load_expert_bundle
from ssae_v3.data.roles import repo_root

BETA_VALUES = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
BETA_PROBS = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
TARGET_ATT = 4.0
X_OFFSET = 0.5
SEED_BASE = 20_260_923
DEFAULT_GRAPH = "artifacts/ihdp/expert_prior_2/expert_prior.yaml"


def concept_membership(graph_path: Path) -> np.ndarray:
    """(m, K) 0/1 feature-to-concept incidence of the expert graph, in covariate order."""
    bundle = load_expert_bundle(list(FEATURE_NAMES), dataset="ihdp", path=graph_path)
    return bundle.relation_mask.numpy().astype(np.float64)


def draw_beta(membership: np.ndarray, alignment: float, rng: np.random.Generator) -> np.ndarray:
    """One realization's coefficient vector.

    Both the unstructured and the concept-level vector are always drawn, whatever the
    alignment, so the random stream - and hence every other quantity of a realization -
    does not depend on the alignment value.
    """
    m, K = membership.shape
    beta_free = rng.choice(BETA_VALUES, size=m, p=BETA_PROBS)
    b_concept = rng.choice(BETA_VALUES, size=K, p=BETA_PROBS)
    while not b_concept.any():
        # A surface with every concept switched off has a constant effect; redraw.
        b_concept = rng.choice(BETA_VALUES, size=K, p=BETA_PROBS)
    beta_group = membership @ b_concept
    orphan = membership.sum(axis=1) == 0
    beta_group[orphan] = beta_free[orphan]
    return alignment * beta_group + (1.0 - alignment) * beta_free


def response_surface(x: np.ndarray, t: np.ndarray, beta: np.ndarray):
    """NPCI setting B potential outcomes, with omega set so the ATT is exactly 4."""
    index = x @ beta
    mu0 = np.exp((x + X_OFFSET) @ beta)
    omega = np.mean((index - mu0)[t == 1]) - TARGET_ATT
    mu1 = index - omega
    return mu0, mu1


def generate(
    alignment: float,
    graph_path: Path,
    source: Optional[Dict[str, Path]] = None,
    seed_base: int = SEED_BASE,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Both splits' arrays for every realization of the shipped replication set."""
    if not 0.0 <= alignment <= 1.0:
        raise ValueError(f"alignment must lie in [0, 1]; got {alignment}")
    source = source or {s: repo_root() / p for s, p in REPLICATION_PATHS.items()}
    shipped = {split: dict(np.load(path)) for split, path in source.items()}
    membership = concept_membership(graph_path)

    n_real = shipped["train"]["x"].shape[2]
    out = {
        split: {k: v.copy() for k, v in arrays.items()} for split, arrays in shipped.items()
    }
    betas = np.zeros((n_real, membership.shape[0]))
    for r in range(n_real):
        rng = np.random.default_rng(seed_base + r)
        beta = draw_beta(membership, alignment, rng)
        betas[r] = beta
        # omega is fixed on the pooled units, as in the shipped files.
        x_all = np.vstack([shipped[s]["x"][:, :, r] for s in ("train", "test")])
        t_all = np.concatenate([shipped[s]["t"][:, r] for s in ("train", "test")])
        mu0_all, mu1_all = response_surface(x_all, t_all, beta)
        n_train = shipped["train"]["x"].shape[0]
        for split, rows in (("train", slice(0, n_train)), ("test", slice(n_train, None))):
            mu0, mu1 = mu0_all[rows], mu1_all[rows]
            t = shipped[split]["t"][:, r]
            noise = rng.standard_normal((2, len(t)))
            y1 = mu1 + noise[0]
            y0 = mu0 + noise[1]
            out[split]["mu0"][:, r] = mu0
            out[split]["mu1"][:, r] = mu1
            out[split]["yf"][:, r] = np.where(t == 1, y1, y0)
            out[split]["ycf"][:, r] = np.where(t == 1, y0, y1)
    out["_meta"] = {"beta": betas, "membership": membership}
    return out


def write(out_dir: Path, arrays: Dict[str, Dict[str, np.ndarray]], manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rel in REPLICATION_PATHS.items():
        np.savez(out_dir / Path(rel).name, **arrays[split])
    np.savez(out_dir / "surface_meta.npz", **arrays["_meta"])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--alignment", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=Path(DEFAULT_GRAPH))
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    args = parser.parse_args(argv)

    graph = args.graph if args.graph.is_absolute() else repo_root() / args.graph
    arrays = generate(args.alignment, graph, seed_base=args.seed_base)
    manifest = {
        "alignment": args.alignment,
        "seed_base": args.seed_base,
        "graph": str(args.graph),
        "graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
        "surface": "NPCI setting B, beta per concept (alignment) mixed with beta per feature",
        "beta_values": BETA_VALUES.tolist(),
        "beta_probs": BETA_PROBS.tolist(),
        "target_att": TARGET_ATT,
    }
    write(args.out, arrays, manifest)
    print(f"wrote {args.out} (alignment={args.alignment})")


if __name__ == "__main__":
    main()
