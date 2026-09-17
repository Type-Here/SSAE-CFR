"""Build and cache the semantic prior bundle (V, P_U, W_r, Q) for a dataset.

Entry point run on the university machine (the embedding step). Modes:

  Emit a gloss template (run locally, needs the dataset for the column order):
      python -m ssae_v3.prior_build.build emit --dataset diur_v1

  Build the prior (run on the uni machine; needs only the gloss YAML, not the data):
      python -m ssae_v3.prior_build.build build --dataset diur_v1 --model BioMistral/BioMistral-7B

The build reads the covariate order and glosses from the committed YAML, turns them
into prompts, embeds them, caches V and the prior bundle under `artifacts/<dataset>/`
(gitignored), and updates the versioned `artifacts/manifest.json`. `--reuse-V`
rebuilds the bundle from the cached V with no GPU and no model download.
`--placeholder` gives a data-free, LLM-free dry run with a random V.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..data.roles import repo_root
from .descriptions import (
    descriptions_for,
    emit_gloss_template,
    load_glosses,
    load_prompt_template,
    missing_glosses,
)
from .embeddings import build_embeddings, cache_embeddings, load_embeddings, placeholder_embeddings
from .projector import (
    build_prior_bundle,
    choose_k_svd,
    cosine_similarities,
    retention,
    save_bundle,
    svd_energy,
)

DEFAULT_MODEL = "BioMistral/BioMistral-7B"
BUNDLE_NAME = "prior_bundle.npz"


def _loaders():
    # imported lazily so `emit` does not require every adapter's raw file to be present
    from ..data import (
        load_actg175_pseudo_obs,
        load_actg175_rct,
        load_diur_v1,
        load_ihdp,
        load_sepsis_v2,
    )

    return {
        "ihdp": load_ihdp,
        "aids_v1": load_actg175_rct,
        "aids_v1_biased": load_actg175_pseudo_obs,
        "diur_v1": load_diur_v1,
        "sepsis_v2": load_sepsis_v2,
    }


def glosses_path(dataset: str) -> Path:
    """Committed gloss YAML for a dataset: `ssae_v3/prior_build/glosses/<dataset>.yaml`."""
    return Path(__file__).resolve().parent / "glosses" / f"{dataset}.yaml"


def artifacts_dir(dataset: str) -> Path:
    return repo_root() / "artifacts" / dataset


def _resolve_protected(names: Sequence[str], feature_names: Sequence[str]) -> List[int]:
    idx = []
    for name in names:
        if name not in feature_names:
            raise SystemExit(f"protected covariate {name!r} not among feature names")
        idx.append(list(feature_names).index(name))
    return idx


def _sha256(V: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(V, dtype=np.float32).tobytes()).hexdigest()


def _update_manifest(entry: dict) -> Path:
    """Merge one dataset entry into the versioned artifacts/manifest.json."""
    path = repo_root() / "artifacts" / "manifest.json"
    manifest = {}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8") or "{}")
    manifest.setdefault("priors", {})[entry["dataset"]] = entry
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def emit(dataset: str) -> None:
    loader = _loaders().get(dataset)
    if loader is None:
        raise SystemExit(f"unknown dataset {dataset!r}; choose from {sorted(_loaders())}")
    ds = loader()
    out = emit_gloss_template(ds.feature_names, glosses_path(dataset))
    print(f"wrote {ds.m}-covariate gloss template -> {out}")
    print("edit the glosses (especially opaque columns), then run the build.")


def build(
    dataset: str,
    model: str = DEFAULT_MODEL,
    k_u: Optional[int] = None,
    r_w_rank: Optional[int] = None,
    energy_threshold: float = 0.90,
    protected: Sequence[str] = (),
    retention_floor: float = 0.5,
    placeholder: bool = False,
    d_llm: int = 64,
    seed: int = 0,
    dtype: str = "auto",
    batch_size: int = 8,
    center: bool = True,
    reuse_V: bool = False,
) -> None:
    gpath = glosses_path(dataset)
    if not gpath.exists():
        raise SystemExit(f"no gloss file at {gpath}; run `emit --dataset {dataset}` first")

    glosses = load_glosses(gpath)
    template = load_prompt_template(gpath)
    feature_names = list(glosses.keys())
    m = len(feature_names)
    prompts = descriptions_for(feature_names, glosses, template)

    blank = missing_glosses(feature_names, glosses)
    if blank:
        print(f"WARNING: {len(blank)} covariates have no gloss (using prettified names): {blank}")
    print(f"prompt template: {template}")
    print(f"example prompt : {prompts[0]}")

    out = artifacts_dir(dataset)
    if reuse_V:
        # rebuilding the bundle from a cached V is pure numpy on an (m x d_LLM)
        # matrix: no GPU, no download. Iterating on ranks/centering costs nothing.
        V = load_embeddings(out / "V.npz")
        model_used = json.loads((out / "V.json").read_text(encoding="utf-8"))["model_name"]
        print(f"[reuse] loaded cached V from {out}/V.npz (model {model_used})")
        if V.shape[0] != m:
            raise SystemExit(f"cached V has {V.shape[0]} rows but the gloss file lists {m}")
    elif placeholder:
        print(f"[placeholder] random V, d_LLM={d_llm} - dry run, not a real prior")
        V = placeholder_embeddings(m, d_LLM=d_llm, seed=seed)
        model_used = f"placeholder(seed={seed})"
    else:
        print(f"embedding {m} covariate prompts with {model} (dtype={dtype}) ...")
        V = build_embeddings(prompts, model_name=model, dtype=dtype, batch_size=batch_size)
        model_used = model
    if not reuse_V:
        cache_embeddings(V, out / "V.npz", model_name=model_used, feature_names=feature_names)

    protected_idx = _resolve_protected(protected, feature_names)
    bundle = build_prior_bundle(
        V,
        k_U=k_u,
        r_W_rank=r_w_rank,
        center=center,
        feature_names=feature_names,
        energy_threshold=energy_threshold,
        protected=protected_idx or None,
        retention_floor=retention_floor,
    )

    ret = retention(bundle.P_U)
    meta = {
        "dataset": dataset,
        "model_name": model_used,
        "prompt_template": template,
        "glosses": str(gpath.relative_to(repo_root())),
        "d_LLM": bundle.d_LLM,
        "m": m,
        "k_U": bundle.k_U,
        "r_W_rank": bundle.r_W_rank,
        "s_Q": bundle.s_Q,
        "energy_threshold": energy_threshold,
        "energy_at_k": bundle.meta["energy_at_k"],
        "protected": list(protected),
        "feature_names": feature_names,
        "V_sha256": _sha256(V),
        "dtype": dtype,
        "centered": center,
        "placeholder": placeholder,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    bundle.meta.update(meta)
    save_bundle(bundle, out / BUNDLE_NAME)
    _update_manifest(meta)

    print(
        f"k_U={bundle.k_U} (rank of P_U), r_W_rank={bundle.r_W_rank}, "
        f"energy captured={bundle.meta['energy_at_k']:.3f}, s_Q={bundle.s_Q:.4f}, "
        f"m={m}, d_LLM={bundle.d_LLM}, centered={center}"
    )
    if protected_idx:
        print("protected retention:", {n: round(float(ret[i]), 3) for n, i in zip(protected, protected_idx)})
    if bundle.k_U <= 2 and not placeholder:
        print(
            "WARNING: k_U <= 2 means the embeddings are nearly collinear and P_U is "
            "degenerate. Run `diagnose` before using this prior."
        )
    print(f"cached V + prior bundle -> {out}/  | manifest updated")


def diagnose(dataset: str, energy_threshold: float = 0.90, top_n: int = 3) -> None:
    """Report whether a cached V carries usable semantic structure.

    Runs entirely on `artifacts/<dataset>/V.npz` - no GPU, no model, no dataset - so
    it is the cheap check to run before spending anything on a prior. Checks, in the
    order they invalidate each other: (1) is V anisotropic (near-1 mean off-diagonal
    cosine means every covariate embedded to nearly the same place); (2) what does
    centering do to the spectrum (a large rank gap at the same energy threshold is
    the signature of a shared offset dominating the SVD); (3) do the nearest
    neighbours make sense (the only check on meaning rather than geometry).
    """
    out = artifacts_dir(dataset)
    vpath = out / "V.npz"
    if not vpath.exists():
        raise SystemExit(f"no cached V at {vpath}; run `build --dataset {dataset}` first")

    V = load_embeddings(vpath)
    meta = json.loads((out / "V.json").read_text(encoding="utf-8"))
    names = meta["feature_names"]
    m = V.shape[0]
    print(f"{dataset}: V is {V.shape} from {meta['model_name']}\n")

    for label, center in (("uncentered", False), ("centered", True)):
        energy = svd_energy(V, center=center)
        k = choose_k_svd(V, energy_threshold=energy_threshold, center=center)
        cos = cosine_similarities(V, center=center)
        off = cos[~np.eye(m, dtype=bool)]
        curve = ", ".join(f"k={i + 1}:{energy[i]:.3f}" for i in range(min(6, energy.size)))
        print(f"{label:<11} energy  {curve}")
        print(f"{'':<11} k at {energy_threshold:.0%} energy = {k}")
        print(f"{'':<11} off-diagonal cosine  mean {off.mean():+.3f}  "
              f"min {off.min():+.3f}  max {off.max():+.3f}\n")

    print(f"nearest neighbours by centered cosine (top {top_n}):")
    cos = cosine_similarities(V, center=True)
    np.fill_diagonal(cos, -np.inf)
    for j, name in enumerate(names):
        order = np.argsort(cos[j])[::-1][:top_n]
        neighbours = ", ".join(f"{names[i]} {cos[j, i]:+.2f}" for i in order)
        print(f"  {name:<12} {neighbours}")
    print(
        "\nRead the neighbours, not the numbers: covariates that mean similar things "
        "should sit together. If they do not, the model or the prompts are wrong, and "
        "no k or threshold will fix it."
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the SSAE-CFR semantic prior bundle.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("emit", help="write an editable gloss template for a dataset")
    pe.add_argument("--dataset", required=True)

    pb = sub.add_parser("build", help="embed glosses and cache V + the prior bundle")
    pb.add_argument("--dataset", required=True)
    pb.add_argument("--model", default=DEFAULT_MODEL)
    pb.add_argument("--k-u", type=int, default=None, help="rank of P_U (None = choose_k_svd)")
    pb.add_argument("--r-w-rank", type=int, default=None, help="rank of W_r (None = same as k-u)")
    pb.add_argument("--energy", type=float, default=0.90)
    pb.add_argument("--protected", default="", help="comma-separated covariate names")
    pb.add_argument("--retention-floor", type=float, default=0.5)
    pb.add_argument("--placeholder", action="store_true", help="dry run: random V, no LLM")
    pb.add_argument("--d-llm", type=int, default=64, help="placeholder V width")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="model precision; auto = float16 on GPU, float32 on CPU",
    )
    pb.add_argument("--batch-size", type=int, default=8, help="prompts per forward pass")
    pb.add_argument(
        "--no-center",
        dest="center",
        action="store_false",
        help="skip removing the mean embedding before the SVD (not recommended: the "
             "shared offset then dominates the spectrum and P_U collapses)",
    )
    pb.add_argument(
        "--reuse-V",
        action="store_true",
        help="rebuild the bundle from the cached V instead of re-embedding (no GPU needed)",
    )

    pd_ = sub.add_parser("diagnose", help="check a cached V for usable semantic structure")
    pd_.add_argument("--dataset", required=True)
    pd_.add_argument("--energy", type=float, default=0.90)
    pd_.add_argument("--top-n", type=int, default=3, help="neighbours to list per covariate")

    args = parser.parse_args(argv)
    if args.cmd == "emit":
        emit(args.dataset)
    elif args.cmd == "diagnose":
        diagnose(args.dataset, energy_threshold=args.energy, top_n=args.top_n)
    else:
        protected = [s.strip() for s in args.protected.split(",") if s.strip()]
        build(
            dataset=args.dataset, model=args.model, k_u=args.k_u, r_w_rank=args.r_w_rank,
            energy_threshold=args.energy, protected=protected,
            retention_floor=args.retention_floor, placeholder=args.placeholder,
            d_llm=args.d_llm, seed=args.seed, dtype=args.dtype,
            batch_size=args.batch_size, center=args.center, reuse_V=args.reuse_V,
        )


if __name__ == "__main__":
    main()
