"""Embed the expert's paragraphs with the same frozen model, under one pooling rule.

Pooling
-------
`expert_v1` uses attention-masked mean pooling of the last hidden state, accumulated
in float32 - byte-for-byte the rule the covariate matrix V already uses. The reason is
comparability, not inertia: same model, same pooling, only the text differs, so any
later comparison between these vectors and V is about what was written rather than
about how it was pooled. `build.py` pins the choice; it is not a CLI flag, because one
artifact must never mix two rules.

`last` (the final real token's hidden state) is implemented beside it and deliberately
unused here. It is the more standard choice for a decoder-only model, where the causal
mask means only the final position has attended to the whole sequence while mean
pooling averages in states that saw a prefix. That makes it the natural ablation, and
the reason it is written now is so the ablation costs nothing later - but mixing it
into this build would confound the one comparison worth making.

Truncation
----------
An embedding_text is a paragraph, not a one-line gloss, so `max_length` is far above
V's 128. Any text that still truncates is counted, named and recorded rather than
quietly clipped: a clipped paragraph is a vector for a sentence the expert did not
write, and this project has been bitten too often by losses that never printed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..embeddings import _load_model

DEFAULT_MODEL = "BioMistral/BioMistral-7B"
DEFAULT_MAX_LENGTH = 512
POOLING_STRATEGIES = ("mean", "last")


class EmbeddingError(ValueError):
    """The texts could not be embedded as asked."""


@dataclass(frozen=True)
class EmbeddingResult:
    """Embeddings plus everything needed to describe how they were produced."""

    vectors: np.ndarray
    pooling: str
    model_name: str
    dtype: str
    max_length: int
    token_counts: Tuple[int, ...]
    truncated: Tuple[int, ...]

    @property
    def d_LLM(self) -> int:
        return int(self.vectors.shape[1])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pooling": self.pooling,
            "model_name": self.model_name,
            "dtype": self.dtype,
            "max_length": self.max_length,
            "n": int(self.vectors.shape[0]),
            "d_LLM": self.d_LLM,
            "max_token_count": max(self.token_counts) if self.token_counts else 0,
            "n_truncated": len(self.truncated),
            "truncated_rows": list(self.truncated),
        }


def _pool(hidden, mask, pooling: str):
    """Reduce (b, seq, d) to (b, d) under the named strategy. Accumulates in float32."""
    import torch

    hidden = hidden.float()
    mask = mask.unsqueeze(-1).float()
    if pooling == "mean":
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts
    if pooling == "last":
        # index of the final real token per row; right padding is assumed, and a row
        # of pure padding would otherwise silently take row 0
        lengths = mask.squeeze(-1).sum(dim=1).long().clamp(min=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
    raise EmbeddingError(f"unknown pooling {pooling!r}; choose from {POOLING_STRATEGIES}")


def embed_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    *,
    pooling: str = "mean",
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 4,
    device: str = "auto",
    dtype: str = "auto",
) -> EmbeddingResult:
    """Embed `texts` into (n, d_LLM) float32, recording token counts and truncation."""
    import torch
    from transformers import AutoTokenizer

    texts = [str(t) for t in texts]
    if not texts:
        raise EmbeddingError("no texts to embed")
    if pooling not in POOLING_STRATEGIES:
        raise EmbeddingError(f"unknown pooling {pooling!r}; choose from {POOLING_STRATEGIES}")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        resolved_dtype = "float16" if device.startswith("cuda") else "float32"
    else:
        resolved_dtype = dtype

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # right padding, so `last` indexes the final real token rather than a pad
    tokenizer.padding_side = "right"

    # measure before truncating, so the report is about the text and not about the cap
    token_counts = [len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]
    truncated = tuple(i for i, n in enumerate(token_counts) if n > max_length)

    model, input_device = _load_model(model_name, device, dtype)

    vectors: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(input_device)
            out = model(**enc)
            pooled = _pool(out.last_hidden_state, enc["attention_mask"], pooling)
            vectors.append(pooled.cpu().numpy())

    matrix = np.concatenate(vectors, axis=0).astype(np.float32)
    if matrix.shape[0] != len(texts):
        raise EmbeddingError(f"got {matrix.shape[0]} vectors for {len(texts)} texts")

    return EmbeddingResult(
        vectors=matrix,
        pooling=pooling,
        model_name=model_name,
        dtype=resolved_dtype,
        max_length=max_length,
        token_counts=tuple(token_counts),
        truncated=truncated,
    )


def save_embeddings(
    path,
    *,
    kind: str,
    ids: Sequence[str],
    texts: Sequence[str],
    order: str,
    result: EmbeddingResult,
    source_sha256: str,
) -> Any:
    """Write one `.pt` carrying the vectors and a stable id -> row mapping.

    The payload is self-describing on purpose: an embedding matrix whose row order
    lives only in the code that wrote it is one refactor away from being silently
    misaligned with the covariates it is supposed to describe.
    """
    from pathlib import Path

    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(ids) != result.vectors.shape[0]:
        raise EmbeddingError(f"{len(ids)} ids but {result.vectors.shape[0]} vectors")

    payload = {
        "kind": kind,
        "ids": list(ids),
        "texts": list(texts),
        "embeddings": torch.from_numpy(np.ascontiguousarray(result.vectors)),
        "row_of": {str(i): row for row, i in enumerate(ids)},
        "order": order,
        "pooling": result.pooling,
        "model_name": result.model_name,
        "dtype": result.dtype,
        "max_length": result.max_length,
        "token_counts": list(result.token_counts),
        "truncated_ids": [ids[i] for i in result.truncated],
        "source_sha256": source_sha256,
    }
    torch.save(payload, path)
    return path
