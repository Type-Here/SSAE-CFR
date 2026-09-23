"""Strict loader for the frozen FeatureCard embedding matrix Z.

The model consumes the stored artifact and nothing else: no centering, no row
normalization, no whitening, no PCA, no re-pooling, no reordering, no regeneration.
The loader's whole job is to refuse an artifact that does not line up with the
model's covariate order, and to carry the provenance that a result table has to
report (model name, pooling, dtype, embedding dimension, file hash).
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ssae_v3.data.roles import repo_root


class EmbeddingArtifactError(ValueError):
    """The embedding artifact is missing, malformed, or does not match the covariates."""


@dataclass(frozen=True)
class FeatureEmbeddings:
    """Frozen per-feature embedding matrix Z (m, d_q) plus its provenance.

    `feature_ids` is the covariate order the rows of Z are expressed in; it has
    already been checked against the model's own feature order by the loader.
    """

    feature_ids: Tuple[str, ...]
    Z: Tensor
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.Z.dim() != 2:
            raise EmbeddingArtifactError(f"Z must be 2-D (m, d_q); got shape {tuple(self.Z.shape)}")
        if self.Z.shape[0] != len(self.feature_ids):
            raise EmbeddingArtifactError(
                f"Z has {self.Z.shape[0]} rows, feature_ids has {len(self.feature_ids)}"
            )
        if not torch.isfinite(self.Z).all():
            raise EmbeddingArtifactError("Z has non-finite entries")

    @property
    def m(self) -> int:
        return self.Z.shape[0]

    @property
    def d_q(self) -> int:
        return self.Z.shape[1]


def center_embeddings(embeddings: FeatureEmbeddings) -> FeatureEmbeddings:
    """Subtract the mean FeatureCard vector from every row of Z.

    The stored card embeddings are strongly anisotropic (mean off-diagonal cosine
    +0.965 on the IHDP artifact), so `x @ Z` is dominated by the shared offset
    direction. Splitting Z into `mu + Z_c` splits the semantic vector into

        p = (sum_j x_j) * mu / sqrt(m)  +  (x @ Z_c) / sqrt(m)

    whose first term is invariant under any row permutation: it is identical in the
    real and the control arm and carries no assignment information at all. Centering
    removes it and leaves only the part the real-vs-permuted comparison is about.

    The mean row is itself permutation-invariant, so centering before or after a row
    control gives the same matrix.

    This is a transformation of the artifact, not of the model: it is requested by
    `center_embeddings` in the config, recorded in the metadata, and reported per run.
    """
    if embeddings.metadata.get("centered"):
        return embeddings
    Z = embeddings.Z - embeddings.Z.mean(dim=0, keepdim=True)
    metadata = dict(embeddings.metadata)
    metadata["centered"] = True
    return dataclasses.replace(embeddings, Z=Z, metadata=metadata)


def embedding_dir_for(dataset: str) -> Path:
    """Default directory holding one dataset's FeatureCard embeddings."""
    return repo_root() / "artifacts" / dataset / "expert_prior"


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _missing_source_message(path: Path) -> str:
    """Name the missing file, and any sibling directory that does hold one like it.

    Several expert-prior directories can sit side by side for one dataset and only
    some carry embeddings.
    """
    message = f"no embedding artifact at {path}"
    parent = path.parent.parent
    if parent.is_dir():
        siblings = sorted(str(d) for d in parent.iterdir() if d.is_dir() and (d / path.name).exists())
        if siblings:
            message += f"; these directories hold a {path.name}: {siblings}"
    return message


def load_feature_embeddings(
    feature_names: Sequence[str],
    dataset: str = "ihdp",
    embedding_dir: Optional[Path] = None,
) -> FeatureEmbeddings:
    """Load Z for `dataset`, asserting its row order equals `feature_names` exactly.

    A mismatch is a hard failure naming the first offending index: silently
    reordering the rows would attach every FeatureCard to the wrong covariate, which
    is precisely the permuted control this experiment runs on purpose.
    """
    directory = Path(embedding_dir) if embedding_dir is not None else embedding_dir_for(dataset)
    path = directory / "feature_embeddings.pt"
    if not path.exists():
        raise EmbeddingArtifactError(_missing_source_message(path))

    payload = torch.load(path, map_location="cpu", weights_only=False)
    ids = list(payload["ids"])
    expected = list(feature_names)
    if ids != expected:
        for j, (got, want) in enumerate(zip(ids, expected)):
            if got != want:
                raise EmbeddingArtifactError(
                    f"embedding feature order does not match the model input order at index "
                    f"{j}: artifact has {got!r}, dataset has {want!r}"
                )
        raise EmbeddingArtifactError(
            f"embedding artifact has {len(ids)} feature ids, dataset has {len(expected)}"
        )

    Z = torch.as_tensor(payload["embeddings"], dtype=torch.float32).clone()
    metadata = {
        "model_name": payload.get("model_name"),
        "pooling": payload.get("pooling"),
        "dtype": payload.get("dtype"),
        "d_q": int(Z.shape[1]),
        "m": int(Z.shape[0]),
        "centered": False,
        "max_length": payload.get("max_length"),
        "source": str(path),
        "artifact_sha256": _sha256_of(path),
        "source_sha256": payload.get("source_sha256"),
    }
    return FeatureEmbeddings(feature_ids=tuple(ids), Z=Z, metadata=metadata)
