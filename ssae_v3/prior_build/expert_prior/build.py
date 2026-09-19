"""CLI for the expert-prior artifact. Staged, because one stage costs a GPU.

    python -m ssae_v3.prior_build.expert_prior.build compose  --dataset ihdp
    python -m ssae_v3.prior_build.expert_prior.build generate --dataset ihdp
    python -m ssae_v3.prior_build.expert_prior.build validate --dataset ihdp
    python -m ssae_v3.prior_build.expert_prior.build embed    --dataset ihdp
    python -m ssae_v3.prior_build.expert_prior.build all      --dataset ihdp

`compose` and `validate` need no model and no GPU: the first shows exactly what will
be sent, the second re-checks a response already on disk. Iterating on the contract or
the validator therefore costs nothing, in the same spirit as the SVD build's
`--reuse-V`. Only `generate` and `embed` need weights.

Everything lands in `artifacts/<dataset>/expert_prior/`, a directory of its own. The
SVD prior's `V.npz`, `P_U.npz` and `prior_bundle.npz` sit beside it and are never read
or written here, and the versioned `artifacts/manifest.json` is left alone: this
artifact's provenance lives entirely in its own manifest.

The manifest accumulates across stages rather than being written once, so a run that
fails at validation still leaves behind the prompt and generation records that explain
what was attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import yaml

from ...data.roles import repo_root
from .cards import concept_embedding_texts, feature_embedding_texts
from .embed import DEFAULT_MAX_LENGTH, embed_texts, save_embeddings
from .generate import (
    DEFAULT_CONTEXT_MARGIN,
    DEFAULT_MIN_BUDGET,
    DEFAULT_MODEL,
    generate_expert_prior,
    plan_generation,
)
from .inputs import load_inputs
from .prompt import compose_prompt, load_template, sha256_text
from .schema import SchemaError, parse_response, validate_expert_prior

ARTIFACT_SUBDIR = "expert_prior"
ARTIFACT_VERSION = "expert_v1"

PROMPT_NAME = "prompt.txt"
RESPONSE_NAME = "response_raw.txt"
EXPERT_PRIOR_NAME = "expert_prior.yaml"
FEATURE_EMB_NAME = "feature_embeddings.pt"
CONCEPT_EMB_NAME = "concept_embeddings.pt"
MANIFEST_NAME = "expert_prior_manifest.json"
REPORT_NAME = "validation_report.json"

# Pinned, not exposed: one artifact must never mix two pooling rules, and expert_v1
# matches V's rule so the two are comparable. `embed.py` implements `last` for a later
# ablation, which will be a separate artifact.
POOLING = "mean"


def artifacts_dir(dataset: str) -> Path:
    return repo_root() / "artifacts" / dataset / ARTIFACT_SUBDIR


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def _versions() -> Dict[str, str]:
    out: Dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "not installed"
    return out


def _update_manifest(dataset: str, section: str, entry: Dict[str, Any]) -> Path:
    """Merge one stage's record into this artifact's own manifest."""
    path = artifacts_dir(dataset) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8") or "{}")
    manifest.setdefault("dataset", dataset)
    manifest.setdefault("artifact_version", ARTIFACT_VERSION)
    manifest[section] = entry
    manifest["updated"] = _now()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    return path


def _read_manifest(dataset: str) -> Dict[str, Any]:
    path = artifacts_dir(dataset) / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8") or "{}")


# -- stages ----------------------------------------------------------------


def compose(dataset: str, show: bool = False) -> Path:
    """Load and validate the inputs, render the prompt, write it before any inference."""
    inputs = load_inputs(dataset)
    prompt = compose_prompt(inputs)

    out = artifacts_dir(dataset)
    out.mkdir(parents=True, exist_ok=True)
    prompt_path = out / PROMPT_NAME
    prompt_path.write_text(prompt, encoding="utf-8")

    _update_manifest(dataset, "prompt", {
        "prompt_file": PROMPT_NAME,
        "prompt_sha256": sha256_text(prompt),
        "prompt_chars": len(prompt),
        "template_sha256": sha256_text(load_template()),
        "input_sha256": inputs.input_hashes,
        "feature_ids": list(inputs.feature_ids),
        "m": len(inputs.feature_ids),
        "composed": _now(),
    })

    print(f"{dataset}: {len(inputs.feature_ids)} features, prompt {len(prompt)} chars")
    print(f"prompt sha256 {sha256_text(prompt)}")
    print(f"wrote {prompt_path}")
    if show:
        print("\n" + prompt)
    return prompt_path


def generate(
    dataset: str,
    model: str = DEFAULT_MODEL,
    *,
    dtype: str = "auto",
    device: str = "auto",
    max_new_tokens: Optional[int] = None,
    min_budget: int = DEFAULT_MIN_BUDGET,
    margin: int = DEFAULT_CONTEXT_MARGIN,
    context_limit: Optional[int] = None,
    do_sample: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: int = 0,
    dry_run: bool = False,
) -> Optional[Path]:
    """The single global expert-generation call. Sizes the budget before loading weights."""
    out = artifacts_dir(dataset)
    prompt_path = out / PROMPT_NAME
    if not prompt_path.exists():
        raise SystemExit(f"no prompt at {prompt_path}; run `compose --dataset {dataset}` first")
    prompt = prompt_path.read_text(encoding="utf-8")

    plan = plan_generation(
        prompt,
        model,
        max_new_tokens=max_new_tokens,
        min_budget=min_budget,
        margin=margin,
        context_limit=context_limit,
    )
    print(f"budget: {plan.describe()}")
    for note in plan.notes:
        print(f"NOTE: {note}")
    if dry_run:
        print("[dry-run] budget fits; no weights loaded, nothing generated")
        return None

    print(f"generating with {model} (dtype={dtype}, greedy={not do_sample}) ...")
    raw, params = generate_expert_prior(
        prompt,
        model,
        plan=plan,
        dtype=dtype,
        device=device,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    # written before anything tries to parse it, so a failure downstream is diagnosable
    response_path = out / RESPONSE_NAME
    response_path.write_text(raw, encoding="utf-8")

    params.update({
        "response_file": RESPONSE_NAME,
        "response_sha256": sha256_text(raw),
        "response_chars": len(raw),
        "prompt_sha256": sha256_text(prompt),
        "versions": _versions(),
        "generated": _now(),
    })
    _update_manifest(dataset, "generation", params)

    print(f"wrote {response_path} ({params['generated_tokens']} tokens)")
    if params["hit_token_cap"]:
        print(
            "WARNING: generation stopped at the token cap rather than at an end-of-sequence "
            "token, so the document is probably truncated and will fail validation."
        )
    return response_path


def validate(dataset: str, response_path: Optional[Path] = None) -> Path:
    """Parse and validate a saved response. No model, no GPU, free to re-run."""
    inputs = load_inputs(dataset)
    out = artifacts_dir(dataset)
    response_path = Path(response_path) if response_path else out / RESPONSE_NAME
    if not response_path.exists():
        raise SystemExit(f"no response at {response_path}; run `generate --dataset {dataset}` first")

    raw = response_path.read_text(encoding="utf-8")
    doc, fence_stripped = parse_response(raw)
    report = validate_expert_prior(
        doc,
        inputs.request.data,
        inputs.feature_ids,
        raw_text=raw,
        fence_stripped=fence_stripped,
    )

    # canonical re-dump: what downstream reads, hashed and recorded
    validated = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)
    validated_path = out / EXPERT_PRIOR_NAME
    validated_path.write_text(validated, encoding="utf-8")
    (out / REPORT_NAME).write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    _update_manifest(dataset, "validation", {
        "expert_prior_file": EXPERT_PRIOR_NAME,
        "expert_prior_sha256": sha256_text(validated),
        "response_sha256": sha256_text(raw),
        "fence_stripped": report.fence_stripped,
        "n_feature_cards": report.n_feature_cards,
        "n_concepts": report.n_concepts,
        "n_warnings": len(report.warnings),
        "warnings": report.warnings,
        "validated": _now(),
    })

    print(f"valid: {report.n_feature_cards} feature cards, {report.n_concepts} concepts")
    if report.fence_stripped:
        print("NOTE: an outer Markdown fence was removed; response_raw.txt is unmodified")
    if report.warnings:
        print(f"{len(report.warnings)} advisory warning(s), none of them repaired:")
        for w in report.warnings:
            print(f"  - {w}")
    print(f"wrote {validated_path}")
    return validated_path


def embed(
    dataset: str,
    model: Optional[str] = None,
    *,
    dtype: str = "auto",
    device: str = "auto",
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 4,
) -> Path:
    """Embed the expert's own paragraphs, features and concepts into separate files."""
    inputs = load_inputs(dataset)
    out = artifacts_dir(dataset)
    validated_path = out / EXPERT_PRIOR_NAME
    if not validated_path.exists():
        raise SystemExit(
            f"no validated prior at {validated_path}; run `validate --dataset {dataset}` first"
        )

    text = validated_path.read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    source_sha = sha256_text(text)

    # the same model that wrote the document, unless deliberately overridden
    manifest = _read_manifest(dataset)
    model = model or manifest.get("generation", {}).get("model_name") or DEFAULT_MODEL

    features = feature_embedding_texts(doc, inputs.feature_ids)
    concepts = concept_embedding_texts(doc)
    print(f"embedding {len(features)} feature texts + {len(concepts)} concept texts "
          f"with {model} (pooling={POOLING}, max_length={max_length})")

    records: Dict[str, Any] = {}
    for texts, filename in ((features, FEATURE_EMB_NAME), (concepts, CONCEPT_EMB_NAME)):
        result = embed_texts(
            texts.texts,
            model,
            pooling=POOLING,
            max_length=max_length,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        path = save_embeddings(
            out / filename,
            kind=texts.kind,
            ids=texts.ids,
            texts=texts.texts,
            order=texts.order,
            result=result,
            source_sha256=source_sha,
        )
        records[texts.kind] = {
            "file": filename,
            "order": texts.order,
            "ids": list(texts.ids),
            "embeddings_sha256": _sha256_array(result.vectors),
            **result.as_dict(),
        }
        truncated = [texts.ids[i] for i in result.truncated]
        print(f"  {texts.kind:9} {result.vectors.shape} -> {path.name}"
              + (f"  TRUNCATED: {truncated}" if truncated else ""))
        if truncated:
            print(f"  WARNING: {len(truncated)} text(s) exceeded max_length={max_length} and "
                  "were clipped; their vectors describe less than the expert wrote")

    records["source_sha256"] = source_sha
    records["versions"] = _versions()
    records["embedded"] = _now()
    manifest_path = _update_manifest(dataset, "embeddings", records)
    print(f"manifest -> {manifest_path}")
    return out


def run_all(dataset: str, **kwargs: Any) -> None:
    compose(dataset)
    generate(
        dataset,
        kwargs.pop("model", DEFAULT_MODEL),
        dtype=kwargs.get("dtype", "auto"),
        device=kwargs.get("device", "auto"),
        max_new_tokens=kwargs.get("max_new_tokens"),
        min_budget=kwargs.get("min_budget", DEFAULT_MIN_BUDGET),
        margin=kwargs.get("margin", DEFAULT_CONTEXT_MARGIN),
        context_limit=kwargs.get("context_limit"),
        do_sample=kwargs.get("do_sample", False),
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
        seed=kwargs.get("seed", 0),
    )
    validate(dataset)
    embed(
        dataset,
        dtype=kwargs.get("dtype", "auto"),
        device=kwargs.get("device", "auto"),
        max_length=kwargs.get("max_length", DEFAULT_MAX_LENGTH),
        batch_size=kwargs.get("batch_size", 4),
    )


# -- argument parsing ------------------------------------------------------


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dtype", default="auto",
                   choices=("auto", "float16", "bfloat16", "float32"),
                   help="model precision; auto = float16 on GPU, float32 on CPU")
    p.add_argument("--device", default="auto")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the expert-prior artifact (FeatureCards, ConceptBank, embeddings)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compose", help="validate the inputs and write the exact prompt")
    pc.add_argument("--dataset", required=True)
    pc.add_argument("--show", action="store_true", help="print the composed prompt")

    pg = sub.add_parser("generate", help="the single expert-generation call")
    pg.add_argument("--dataset", required=True)
    _add_model_args(pg)
    pg.add_argument("--max-new-tokens", type=int, default=None,
                    help="default: everything the context leaves after the prompt")
    pg.add_argument("--min-budget", type=int, default=DEFAULT_MIN_BUDGET,
                    help="refuse to generate if fewer tokens than this are left")
    pg.add_argument("--context-margin", type=int, default=DEFAULT_CONTEXT_MARGIN)
    pg.add_argument("--context-limit", type=int, default=None,
                    help="override the limit read from the model config")
    pg.add_argument("--sample", dest="do_sample", action="store_true",
                    help="sample instead of greedy decoding (greedy is reproducible)")
    pg.add_argument("--temperature", type=float, default=None)
    pg.add_argument("--top-p", type=float, default=None)
    pg.add_argument("--seed", type=int, default=0)
    pg.add_argument("--dry-run", action="store_true",
                    help="size the budget and stop; loads no weights")

    pv = sub.add_parser("validate", help="parse and validate a saved response; no model needed")
    pv.add_argument("--dataset", required=True)
    pv.add_argument("--response", default=None, help="path to a response other than the saved one")

    pe = sub.add_parser("embed", help="embed the expert's embedding_text paragraphs")
    pe.add_argument("--dataset", required=True)
    _add_model_args(pe)
    pe.set_defaults(model=None)
    pe.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    pe.add_argument("--batch-size", type=int, default=4)

    pa = sub.add_parser("all", help="compose, generate, validate and embed in one run")
    pa.add_argument("--dataset", required=True)
    _add_model_args(pa)
    pa.add_argument("--max-new-tokens", type=int, default=None)
    pa.add_argument("--min-budget", type=int, default=DEFAULT_MIN_BUDGET)
    pa.add_argument("--context-margin", type=int, default=DEFAULT_CONTEXT_MARGIN)
    pa.add_argument("--context-limit", type=int, default=None)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    pa.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "compose":
            compose(args.dataset, show=args.show)
        elif args.cmd == "generate":
            generate(
                args.dataset, args.model, dtype=args.dtype, device=args.device,
                max_new_tokens=args.max_new_tokens, min_budget=args.min_budget,
                margin=args.context_margin, context_limit=args.context_limit,
                do_sample=args.do_sample, temperature=args.temperature,
                top_p=args.top_p, seed=args.seed, dry_run=args.dry_run,
            )
        elif args.cmd == "validate":
            validate(args.dataset, Path(args.response) if args.response else None)
        elif args.cmd == "embed":
            embed(
                args.dataset, args.model, dtype=args.dtype, device=args.device,
                max_length=args.max_length, batch_size=args.batch_size,
            )
        else:
            run_all(
                args.dataset, model=args.model, dtype=args.dtype, device=args.device,
                max_new_tokens=args.max_new_tokens, min_budget=args.min_budget,
                margin=args.context_margin, context_limit=args.context_limit,
                seed=args.seed, max_length=args.max_length, batch_size=args.batch_size,
            )
    except SchemaError as exc:
        # the raw response stays on disk; nothing was repaired and nothing was written
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
