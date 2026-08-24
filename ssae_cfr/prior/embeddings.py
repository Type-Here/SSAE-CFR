"""Covariate embeddings V (m x d_LLM), computed offline from a frozen domain LLM.

Each covariate gets a short standardized text description (dataset-specific: an HIV
trial variable reads differently from an ICU vital). We feed each description through
a frozen clinical language model, mean-pool its last hidden state over the real
(non-padding) tokens, and stack the m pooled vectors into V. The projector module then
turns V into the fixed prior `P_U`.

Design points:
  - The embedding model is a config field (`TrainConfig.embedding_model`), and `d_LLM`
    is read from the model at run time - nothing downstream hard-codes a width. Swap
    the model freely; just record which one produced a cached V.
  - The heavy forward runs on the university machine, not the local PC. The transformers
    / torch import is therefore lazy (inside `build_embeddings`) so importing this module
    for anything else - caching, the placeholder below - stays cheap.
  - V is cached to disk (an .npz alongside its metadata) and referenced, not the model.

For local development of the rest of the pipeline before the real V exists, use
`placeholder_embeddings`: a reproducible random V that yields a geometrically valid
(but semantically meaningless) `P_U`. It is clearly not the real prior and must never
be used for reported results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Union

import numpy as np

PathLike = Union[str, Path]


def _load_model(model_name: str, device: str, dtype: str):
    """Load the frozen embedding model, at a width that fits the device.

    Loading a 7B model at the HuggingFace default of float32 costs about 28 GB of
    weights, which no free-tier GPU host can hold - and the failure happens in system
    RAM, during the load, before anything reaches the accelerator. On a GPU we therefore
    default to float16, which is the width these models were released at anyway; on CPU
    we stay in float32, since float16 there is slow and unsupported for some ops.

    `low_cpu_mem_usage` streams the checkpoint shard by shard instead of materializing a
    full copy first, which is what keeps the load inside a 12 GB RAM allowance.
    """
    import torch
    from transformers import AutoModel

    if dtype == "auto":
        dtype = "float16" if device.startswith("cuda") else "float32"
    torch_dtype = getattr(torch, dtype)

    kwargs = {"low_cpu_mem_usage": True}
    try:
        model = AutoModel.from_pretrained(model_name, dtype=torch_dtype, **kwargs)
    except TypeError:
        # older transformers spell it `torch_dtype`
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, **kwargs)
    return model.to(device).eval()


def build_embeddings(
    descriptions: Sequence[str],
    model_name: str = "BioMistral/BioMistral-7B",
    batch_size: int = 8,
    max_length: int = 128,
    device: str = "auto",
    dtype: str = "auto",
) -> np.ndarray:
    """Embed `m` covariate descriptions into V of shape (m, d_LLM).

    Runs the frozen model once per description (batched), takes the last hidden state,
    and mean-pools over the attention mask so padding tokens do not dilute the vector.
    `model_name` is any HuggingFace causal/encoder LM id. Returns a float32 array
    whatever width the model ran at, since everything downstream of here is float32.
    """
    # lazy: keep torch/transformers out of the import graph for the light paths
    import torch
    from transformers import AutoTokenizer

    descriptions = list(descriptions)
    if not descriptions:
        raise ValueError("descriptions is empty; need one per covariate")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(model_name, device, dtype)

    vectors = []
    with torch.no_grad():
        for start in range(0, len(descriptions), batch_size):
            batch = descriptions[start:start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            out = model(**enc)
            # pool in float32: the model may be running in half precision, and a
            # half-precision sum over the sequence loses precision for no gain here
            hidden = out.last_hidden_state.float()      # (b, seq, d_LLM)
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (b, seq, 1)
            summed = (hidden * mask).sum(dim=1)         # mask out padding
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts                    # mean over real tokens
            vectors.append(pooled.cpu().numpy())

    V = np.concatenate(vectors, axis=0).astype(np.float32)
    if V.shape[0] != len(descriptions):
        raise RuntimeError(f"got {V.shape[0]} vectors for {len(descriptions)} descriptions")
    return V


def placeholder_embeddings(m: int, d_LLM: int = 64, seed: int = 0) -> np.ndarray:
    """A reproducible random V (m x d_LLM) for local development only.

    Yields a valid `P_U` (the projector math only needs left singular vectors) but
    carries no clinical meaning. Never use for reported results - swap in the real,
    cached V from the university machine before evaluating.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, d_LLM)).astype(np.float32)


def cache_embeddings(
    V: np.ndarray,
    path: PathLike,
    *,
    model_name: str,
    feature_names: Sequence[str],
) -> Path:
    """Persist V to `path` (.npz) plus a sidecar .json recording model and columns.

    The sidecar keeps V reproducible and self-describing: which model produced it, its
    width `d_LLM`, and the covariate order the rows correspond to (row j must line up
    with covariate j downstream).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    V = np.asarray(V, dtype=np.float32)
    if V.shape[0] != len(feature_names):
        raise ValueError(f"V has {V.shape[0]} rows but {len(feature_names)} feature names")
    np.savez_compressed(path, V=V)
    meta = {
        "model_name": model_name,
        "d_LLM": int(V.shape[1]),
        "m": int(V.shape[0]),
        "feature_names": list(feature_names),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_embeddings(path: PathLike) -> np.ndarray:
    """Load a cached V from an .npz written by `cache_embeddings`."""
    with np.load(Path(path)) as data:
        return data["V"]