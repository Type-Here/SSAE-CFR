"""Offline prior-guidance bundle: embedding subspace and graph subspace, in covariate space.

Two independent prior sources, both reduced to the same kind of object - an
orthonormal basis and its orthogonal projector over R^m, m the number of covariates:

    embedding subspace: centered SVD of the per-feature LLM text embeddings V (m, d)
    graph subspace:     column space of the expert feature-to-concept incidence C (m, K)

Either source may be absent; a bundle with neither is the legal no-prior arm. Nothing
here touches a patient's data - both sources are built from feature metadata only, so
this module has no notion of treatment, outcome or split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..data.roles import repo_root
from .expert_bundle import load_expert_bundle


class GuidanceBundleError(ValueError):
    """A prior-guidance source or bundle is missing, malformed or inconsistent."""


def orthonormal_basis(a: Tensor) -> Tuple[Tensor, int]:
    """An orthonormal basis for the column space of a 2-D tensor, via SVD.

    Computed in float64. Rank is the count of singular values above
    `max(a.shape) * eps * s_max`, the standard numerical-rank threshold. Returns the
    basis as float32 with `rank` columns.
    """
    if a.dim() != 2:
        raise GuidanceBundleError(f"orthonormal_basis needs a 2-D tensor; got shape {tuple(a.shape)}")
    a64 = a.to(torch.float64)
    if not torch.isfinite(a64).all():
        raise GuidanceBundleError("orthonormal_basis input has non-finite entries")

    u, s, _ = torch.linalg.svd(a64, full_matrices=False)
    if s.numel() == 0:
        raise GuidanceBundleError("orthonormal_basis input has no singular values")
    s_max = float(s[0])
    eps = torch.finfo(torch.float64).eps
    threshold = max(a.shape) * eps * s_max
    rank = int((s > threshold).sum())
    if rank == 0:
        raise GuidanceBundleError("orthonormal_basis input has rank 0")

    basis = u[:, :rank]
    return basis.to(torch.float32), rank


def build_embedding_subspace(
    V: Tensor,
    k: Optional[int] = None,
    energy_threshold: float = 0.90,
) -> Tuple[Tensor, int, float]:
    """Centered SVD of the per-feature embedding matrix V (m, d) into a k-dim basis.

    Centering across features is unconditional: LLM embedding spaces are strongly
    anisotropic, and uncentered the leading singular direction is the shared offset
    rather than a semantic direction (measured on the real artifact: uncentered k at
    90% energy = 1, mean off-diagonal cosine 0.965; centered k = 13, cosine -0.040).

    `k=None` picks the smallest k whose cumulative squared-singular-value energy
    reaches `energy_threshold`, clamped to [1, m - 1] so the returned basis can never
    span all of R^m (the projector could never be the identity). Sign of each U
    column is fixed so its largest-magnitude entry is positive, for a deterministic
    result across re-builds.

    Returns (U_k float32 (m, k), k, energy captured at k).
    """
    if V.dim() != 2:
        raise GuidanceBundleError(f"V must be 2-D (m, d); got shape {tuple(V.shape)}")
    m = V.shape[0]
    if m < 2:
        raise GuidanceBundleError(f"V must have at least 2 rows (features); got {m}")
    V64 = V.to(torch.float64)
    if not torch.isfinite(V64).all():
        raise GuidanceBundleError("V has non-finite entries")

    V_c = V64 - V64.mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(V_c, full_matrices=False)

    for i in range(u.shape[1]):
        j = int(torch.argmax(torch.abs(u[:, i])))
        if u[j, i] < 0:
            u[:, i] = -u[:, i]

    total = float(torch.sum(s ** 2))
    if total == 0.0:
        raise GuidanceBundleError("V has zero singular-value mass (all-zero embeddings?)")
    energy = torch.cumsum(s ** 2, dim=0) / total

    k_max = m - 1
    if k is None:
        k = int(torch.searchsorted(energy, torch.tensor(energy_threshold, dtype=energy.dtype)).item()) + 1
        k = max(1, min(k, k_max))
    else:
        if not (1 <= k <= k_max):
            raise GuidanceBundleError(f"k must be in [1, m - 1={k_max}]; got {k}")

    U_k = u[:, :k].to(torch.float32)
    energy_at_k = float(energy[k - 1])
    return U_k, k, energy_at_k


def build_graph_subspace(C: Tensor) -> Tuple[Tensor, int]:
    """Orthonormal basis of the column space of the (m, K) incidence matrix C."""
    return orthonormal_basis(C)


@dataclass(frozen=True)
class PriorGuidanceBundle:
    """Both prior-guidance sources, reduced to (basis, projector) pairs over R^m.

    Either source may be absent (its tensors None, its rank 0); a bundle with neither
    is the legal no-prior arm. `feature_ids` fixes the row order every tensor here is
    expressed in.
    """

    feature_ids: Tuple[str, ...]
    U_k: Optional[Tensor]
    P_U: Optional[Tensor]
    C: Optional[Tensor]
    Q_C: Optional[Tensor]
    P_C: Optional[Tensor]
    embedding_rank: int
    graph_rank: int
    embedding_source_metadata: Mapping[str, Any]
    graph_source_metadata: Mapping[str, Any]
    source_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.feature_ids:
            raise GuidanceBundleError("feature_ids is empty")
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise GuidanceBundleError("feature_ids contains duplicates")
        m = len(self.feature_ids)

        for name in ("U_k", "P_U", "C", "Q_C", "P_C"):
            tensor = getattr(self, name)
            if tensor is None:
                continue
            if tensor.dim() != 2:
                raise GuidanceBundleError(f"{name} must be 2-D; got shape {tuple(tensor.shape)}")
            if tensor.shape[0] != m:
                raise GuidanceBundleError(
                    f"{name} has {tensor.shape[0]} rows, feature_ids has {m}"
                )
            if not torch.isfinite(tensor).all():
                raise GuidanceBundleError(f"{name} has non-finite entries")

        embedding_present = (self.U_k is not None, self.P_U is not None, self.embedding_rank > 0)
        if len(set(embedding_present)) != 1:
            raise GuidanceBundleError(
                "U_k, P_U and embedding_rank > 0 must all hold together or none at all"
            )
        graph_present = (self.C is not None, self.Q_C is not None, self.P_C is not None, self.graph_rank > 0)
        if len(set(graph_present)) != 1:
            raise GuidanceBundleError(
                "C, Q_C, P_C and graph_rank > 0 must all hold together or none at all"
            )

        if self.U_k is not None:
            _check_orthonormal_columns("U_k", self.U_k, self.embedding_rank)
            _check_projector("P_U", self.P_U, self.embedding_rank)
        if self.C is not None:
            unique_vals = torch.unique(self.C)
            if not torch.all((unique_vals == 0.0) | (unique_vals == 1.0)):
                raise GuidanceBundleError("C must contain only 0.0/1.0 entries")
            empty_concepts = [c for c in range(self.C.shape[1]) if float(self.C[:, c].sum()) == 0.0]
            if empty_concepts:
                raise GuidanceBundleError(f"C has all-zero concept columns at indices {empty_concepts}")
            _check_orthonormal_columns("Q_C", self.Q_C, self.graph_rank)
            _check_projector("P_C", self.P_C, self.graph_rank)

    @property
    def m(self) -> int:
        return len(self.feature_ids)

    @property
    def n_concepts(self) -> int:
        return 0 if self.C is None else self.C.shape[1]

    @property
    def n_edges(self) -> int:
        return 0 if self.C is None else int(self.C.sum())


def _check_orthonormal_columns(name: str, basis: Tensor, rank: int) -> None:
    if basis.shape[1] != rank:
        raise GuidanceBundleError(f"{name} has {basis.shape[1]} columns, expected rank {rank}")
    gram = basis.T.to(torch.float64) @ basis.to(torch.float64)
    identity = torch.eye(rank, dtype=torch.float64)
    if not torch.allclose(gram, identity, atol=1e-4):
        raise GuidanceBundleError(f"{name} does not have orthonormal columns")


def _check_projector(name: str, projector: Tensor, rank: int) -> None:
    p64 = projector.to(torch.float64)
    if not torch.allclose(p64, p64.T, atol=1e-4):
        raise GuidanceBundleError(f"{name} is not symmetric")
    if not torch.allclose(p64 @ p64, p64, atol=1e-4):
        raise GuidanceBundleError(f"{name} is not idempotent")
    actual_rank = int(torch.linalg.matrix_rank(p64, atol=1e-4))
    if actual_rank != rank:
        raise GuidanceBundleError(f"{name} has rank {actual_rank}, expected {rank}")


def expert_dir_for(dataset: str) -> Path:
    """Directory holding one dataset's expert-prior artifacts (embeddings + graph)."""
    return repo_root() / "artifacts" / dataset / "expert_prior"


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _missing_source_message(path: Path) -> str:
    """Name the missing file, and any sibling directory that does hold one like it.

    Several expert-prior directories can sit side by side for one dataset and only
    some carry embeddings. Naming the alternatives is not a fallback - the caller
    still has to choose - but it turns a dead end into one command.
    """
    message = f"no source at {path}"
    parent = path.parent.parent
    if parent.is_dir():
        siblings = sorted(
            str(d) for d in parent.iterdir() if d.is_dir() and (d / path.name).exists()
        )
        if siblings:
            message += f"; these directories hold a {path.name}: {siblings}"
    return message


def load_guidance_bundle(
    feature_names: Sequence[str],
    dataset: str = "ihdp",
    expert_dir: Optional[Path] = None,
    k: Optional[int] = None,
    energy_threshold: float = 0.90,
    use_embedding: bool = True,
    use_graph: bool = True,
) -> PriorGuidanceBundle:
    """Assemble a PriorGuidanceBundle from artifacts already on disk.

    `expert_dir` defaults to `expert_dir_for(dataset)`. Missing a requested source is
    a hard failure naming the missing file; there is no fallback to another directory
    and the covariate order is never silently reordered to match.
    """
    feature_ids = tuple(feature_names)
    directory = expert_dir if expert_dir is not None else expert_dir_for(dataset)
    source_hashes = {}

    U_k = P_U = None
    embedding_rank = 0
    embedding_source_metadata: dict = {}
    if use_embedding:
        embeddings_path = Path(directory) / "feature_embeddings.pt"
        if not embeddings_path.exists():
            raise GuidanceBundleError(_missing_source_message(embeddings_path))
        payload = torch.load(embeddings_path, map_location="cpu", weights_only=False)
        ids = list(payload["ids"])
        expected = list(feature_ids)
        if ids != expected:
            for j, (a, b) in enumerate(zip(ids, expected)):
                if a != b:
                    raise GuidanceBundleError(
                        f"embedding source feature order does not match feature_names at "
                        f"index {j}: source has {a!r}, expected {b!r}"
                    )
            raise GuidanceBundleError(
                f"embedding source has {len(ids)} feature ids, expected {len(expected)}"
            )
        V = torch.as_tensor(payload["embeddings"], dtype=torch.float32)
        U_k, embedding_rank, energy_at_k = build_embedding_subspace(V, k=k, energy_threshold=energy_threshold)
        P_U = U_k @ U_k.T
        embedding_source_metadata = {
            "model_name": payload.get("model_name"),
            "pooling": payload.get("pooling"),
            "dtype": payload.get("dtype"),
            "d": int(V.shape[1]),
            "centered": True,
            "k": embedding_rank,
            "energy_at_k": energy_at_k,
            "source": str(embeddings_path),
        }
        source_hashes["feature_embeddings.pt"] = _sha256_of(embeddings_path)

    C = Q_C = P_C = None
    graph_rank = 0
    graph_source_metadata: dict = {}
    if use_graph:
        graph_path = Path(directory) / "expert_prior.yaml"
        if not graph_path.exists():
            raise GuidanceBundleError(_missing_source_message(graph_path))
        expert_bundle = load_expert_bundle(feature_ids, dataset=dataset, path=graph_path)
        C = expert_bundle.relation_mask.to(torch.float32)
        Q_C, graph_rank = build_graph_subspace(C)
        P_C = Q_C @ Q_C.T
        graph_source_metadata = {
            "n_concepts": expert_bundle.n_concepts,
            "n_edges": expert_bundle.n_active_edges,
            "degrees": [int(d) for d in expert_bundle.edges_per_concept.tolist()],
            "graph_rank": graph_rank,
            "source": str(graph_path),
        }
        manifest_path = Path(directory) / "expert_prior_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validation = manifest.get("validation", {})
            if "response_source" in validation:
                graph_source_metadata["response_source"] = validation["response_source"]
            if "response_is_generated" in validation:
                graph_source_metadata["response_is_generated"] = validation["response_is_generated"]
        source_hashes["expert_prior.yaml"] = _sha256_of(graph_path)

    return PriorGuidanceBundle(
        feature_ids=feature_ids,
        U_k=U_k,
        P_U=P_U,
        C=C,
        Q_C=Q_C,
        P_C=P_C,
        embedding_rank=embedding_rank,
        graph_rank=graph_rank,
        embedding_source_metadata=embedding_source_metadata,
        graph_source_metadata=graph_source_metadata,
        source_hashes=source_hashes,
    )
