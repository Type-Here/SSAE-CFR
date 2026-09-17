"""Covariate embeddings V (m x d_LLM), computed offline from a frozen domain LLM.

Each covariate's text description is fed through a frozen clinical language model;
its last hidden state is mean-pooled over the real (non-padding) tokens, and the m
pooled vectors stack into V. `projector.py` turns V into the prior bundle.

The transformers/torch import is lazy (inside `build_embeddings`) so importing this
module for anything else - caching, the placeholder below - never requires them.
V is cached to disk (.npz plus a metadata sidecar) and referenced, not the model.

`placeholder_embeddings` gives a reproducible random V for exercising the rest of
the pipeline without a model or GPU. It is not a real prior and must never be used
for reported results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Union

import numpy as np

PathLike = Union[str, Path]


def _load_model(model_name: str, device: str, dtype: str):
    """Load the frozen embedding model without ever holding it whole in system RAM.

    Returns `(model, input_device)` - the device to put the tokenized batch on, which
    is not always the one asked for once the weights have been dispatched.

    Two memory problems, only one of them obvious. Width: a 7B model at float32 is
    about 28 GB of weights, so on GPU we default to float16 (the release width); on
    CPU we stay float32, where float16 is slow and unsupported for some ops.

    Placement: `from_pretrained(...).to(device)` still materializes the whole model
    in CPU RAM before moving it, which OOMs a host whose RAM is smaller than the
    model even though the target GPU had room. Passing `device_map="auto"` makes
    accelerate place each tensor as the checkpoint is read, so peak host memory is
    one shard rather than one model - and then `.to()` must not be called afterwards.
    """
    import torch
    from transformers import AutoModel

    if dtype == "auto":
        dtype = "float16" if device.startswith("cuda") else "float32"
    torch_dtype = getattr(torch, dtype)

    kwargs = {"low_cpu_mem_usage": True}
    if device.startswith("cuda"):
        # "auto" rather than a fixed device: spills to CPU instead of failing if the
        # weights do not quite fit in VRAM.
        kwargs["device_map"] = "auto"

    try:
        model = AutoModel.from_pretrained(model_name, dtype=torch_dtype, **kwargs)
    except TypeError:
        # older transformers spell it `torch_dtype`
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, **kwargs)

    if "device_map" not in kwargs:
        model = model.to(device)
    model.eval()

    # a dispatched model has no single device; feed it wherever its first weights landed
    input_device = next(model.parameters()).device
    return model, input_device


def build_embeddings(
    descriptions: Sequence[str],
    model_name: str = "BioMistral/BioMistral-7B",
    batch_size: int = 8,
    max_length: int = 128,
    device: str = "auto",
    dtype: str = "auto",
) -> np.ndarray:
    """Embed `m` covariate descriptions into V of shape (m, d_LLM).

    Runs the frozen model once per description (batched), mean-pools the last hidden
    state over the attention mask so padding tokens do not dilute the vector.
    `model_name` is any HuggingFace causal/encoder LM id. Returns float32.
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
    model, input_device = _load_model(model_name, device, dtype)

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
            ).to(input_device)
            out = model(**enc)
            # pool in float32: the model may run in half precision, and a
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

    Yields a geometrically valid but semantically meaningless prior. Never use for
    reported results - swap in the real, cached V before evaluating.
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

    The sidecar keeps V reproducible and self-describing: which model produced it,
    its width `d_LLM`, and the covariate order the rows correspond to.
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
