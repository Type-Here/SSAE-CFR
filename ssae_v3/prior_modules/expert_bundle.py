"""Offline reader for the expert-prior feature-to-concept graph.

Turns a validated expert_prior.yaml document into the tensors the ExpertPriorAdapter
consumes: a binary adjacency over (feature, concept) pairs and, for each active pair,
the categorical relation type the expert attached to it.

Only the graph is a neural input. Confidence, expected direction and the card
embeddings are loaded on request and kept as diagnostics, because a per-edge scalar
or a per-feature vector can act as an identity code, which is the failure mode this
adapter exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import yaml
from torch import Tensor

from ..data.roles import repo_root
from ..prior_build.expert_prior.inputs import inputs_dir

PathLike = Union[str, Path]

# index stored at (j, c) pairs the expert did not link, so "no edge" is never
# confused with relation type 0
NO_EDGE = -1


class ExpertBundleError(ValueError):
    """The expert graph could not be read, or is not a usable graph."""


@dataclass(frozen=True)
class ExpertPriorBundle:
    """The expert graph as tensors, plus optional diagnostic fields.

    `relation_mask[j, c]` is True when the expert stated that feature j is informative
    about concept c. `relation_type_index[j, c]` indexes `relation_type_vocabulary`
    there, and is NO_EDGE elsewhere.

    `feature_ids` is the model's covariate order and `concept_ids` the artifact's
    concept order; both are stored so a downstream reorder is a detectable mismatch.
    """

    feature_ids: Tuple[str, ...]
    concept_ids: Tuple[str, ...]
    relation_mask: Tensor
    relation_type_index: Tensor
    relation_type_vocabulary: Tuple[str, ...]
    relation_confidence: Optional[Dict[Tuple[str, str], str]] = None
    relation_direction: Optional[Dict[Tuple[str, str], str]] = None
    feature_embeddings: Optional[Tensor] = None
    concept_embeddings: Optional[Tensor] = None

    def __post_init__(self) -> None:
        mask, index = self.relation_mask, self.relation_type_index
        if mask.dim() != 2:
            raise ExpertBundleError(f"relation_mask must be 2-D; got {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            raise ExpertBundleError(f"relation_mask must be bool; got {mask.dtype}")
        if index.shape != mask.shape:
            raise ExpertBundleError(
                f"relation_type_index {tuple(index.shape)} does not match relation_mask "
                f"{tuple(mask.shape)}"
            )
        if index.dtype != torch.long:
            raise ExpertBundleError(f"relation_type_index must be long; got {index.dtype}")

        m, K = mask.shape
        if len(self.feature_ids) != m:
            raise ExpertBundleError(
                f"{len(self.feature_ids)} feature ids for a mask with {m} rows"
            )
        if len(self.concept_ids) != K:
            raise ExpertBundleError(
                f"{len(self.concept_ids)} concept ids for a mask with {K} columns"
            )
        duplicates = sorted({c for c in self.concept_ids if self.concept_ids.count(c) > 1})
        if duplicates:
            raise ExpertBundleError(f"duplicate concept ids {duplicates}")

        R = len(self.relation_type_vocabulary)
        if not torch.equal(index != NO_EDGE, mask):
            raise ExpertBundleError(
                "relation_type_index is set exactly where relation_mask is True; the two "
                "disagree"
            )
        active = index[mask]
        if active.numel() and (int(active.min()) < 0 or int(active.max()) >= R):
            raise ExpertBundleError(
                f"relation type index out of range for a vocabulary of {R} types"
            )

        empty = [self.concept_ids[c] for c in range(K) if not bool(mask[:, c].any())]
        if empty:
            raise ExpertBundleError(f"concepts with no active feature relations: {empty}")

        for name, tensor, rows in (
            ("feature_embeddings", self.feature_embeddings, m),
            ("concept_embeddings", self.concept_embeddings, K),
        ):
            if tensor is not None and (tensor.dim() != 2 or tensor.shape[0] != rows):
                raise ExpertBundleError(
                    f"{name} must be 2-D with {rows} rows; got {tuple(tensor.shape)}"
                )

    @property
    def m(self) -> int:
        return self.relation_mask.shape[0]

    @property
    def n_concepts(self) -> int:
        return self.relation_mask.shape[1]

    @property
    def n_relation_types(self) -> int:
        return len(self.relation_type_vocabulary)

    @property
    def n_active_edges(self) -> int:
        return int(self.relation_mask.sum())

    @property
    def edges_per_concept(self) -> Tensor:
        return self.relation_mask.sum(dim=0).to(torch.long)

    @property
    def semantic_similarity(self) -> Optional[Tensor]:
        """Cosine similarity between feature and concept embeddings, (m, K).

        A diagnostic. Not a forward input in v1.
        """
        if self.feature_embeddings is None or self.concept_embeddings is None:
            return None
        f = torch.nn.functional.normalize(self.feature_embeddings.float(), dim=1)
        c = torch.nn.functional.normalize(self.concept_embeddings.float(), dim=1)
        return f @ c.T

    @classmethod
    def from_document(
        cls,
        doc: Mapping[str, Any],
        feature_ids: Sequence[str],
        relation_vocabulary: Sequence[str],
        *,
        with_diagnostics: bool = False,
        feature_embeddings: Optional[Tensor] = None,
        concept_embeddings: Optional[Tensor] = None,
    ) -> "ExpertPriorBundle":
        """Build the graph from a validated expert-prior document.

        Edges come from each ConceptCard's `feature_relations`. `feature_ids` is the
        model's covariate order and fixes the row order of the mask.
        """
        feature_ids = tuple(str(f) for f in feature_ids)
        vocabulary = tuple(str(r) for r in relation_vocabulary)
        row_of = {fid: j for j, fid in enumerate(feature_ids)}
        type_of = {name: i for i, name in enumerate(vocabulary)}

        concepts = doc.get("concept_bank")
        if not isinstance(concepts, Sequence) or isinstance(concepts, str) or not concepts:
            raise ExpertBundleError("the document has no non-empty concept_bank")

        concept_ids = tuple(str(card["concept_id"]).strip() for card in concepts)
        m, K = len(feature_ids), len(concept_ids)
        mask = torch.zeros(m, K, dtype=torch.bool)
        index = torch.full((m, K), NO_EDGE, dtype=torch.long)
        confidence: Dict[Tuple[str, str], str] = {}
        direction: Dict[Tuple[str, str], str] = {}

        for c, card in enumerate(concepts):
            cid = concept_ids[c]
            relations = card.get("feature_relations")
            if not isinstance(relations, Sequence) or isinstance(relations, str) or not relations:
                raise ExpertBundleError(f"concept {cid!r} has no feature_relations")
            for rel in relations:
                fid = str(rel["feature_id"]).strip()
                if fid not in row_of:
                    raise ExpertBundleError(
                        f"concept {cid!r} relates to unknown feature {fid!r}"
                    )
                name = str(rel["relation"]).strip()
                if name not in type_of:
                    raise ExpertBundleError(
                        f"concept {cid!r} uses unknown relation type {name!r}; the "
                        f"contract allows {list(vocabulary)}"
                    )
                j = row_of[fid]
                if bool(mask[j, c]):
                    raise ExpertBundleError(
                        f"concept {cid!r} relates to feature {fid!r} more than once"
                    )
                mask[j, c] = True
                index[j, c] = type_of[name]
                if with_diagnostics:
                    if rel.get("confidence") is not None:
                        confidence[(fid, cid)] = str(rel["confidence"])
                    if rel.get("expected_direction") is not None:
                        direction[(fid, cid)] = str(rel["expected_direction"])

        return cls(
            feature_ids=feature_ids,
            concept_ids=concept_ids,
            relation_mask=mask,
            relation_type_index=index,
            relation_type_vocabulary=vocabulary,
            relation_confidence=confidence if with_diagnostics else None,
            relation_direction=direction if with_diagnostics else None,
            feature_embeddings=feature_embeddings,
            concept_embeddings=concept_embeddings,
        )


def expert_prior_dir(dataset: str) -> Path:
    """Directory holding one dataset's expert-prior artifacts."""
    return repo_root() / "artifacts" / dataset / "expert_prior"


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ExpertBundleError(f"no file at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ExpertBundleError(f"{path.name} must contain a YAML mapping at the top level")
    return dict(data)


def relation_vocabulary(dataset: str, request_path: Optional[PathLike] = None) -> Tuple[str, ...]:
    """The contract's relation vocabulary, which fixes the one-hot width.

    Read from the request file rather than from the response, so the width does not
    depend on which relation types one generation happened to use and a control graph
    stays comparable to the real one.
    """
    path = (
        Path(request_path)
        if request_path is not None
        else inputs_dir(dataset) / f"{dataset}_expert_prior_request.yaml"
    )
    values = _read_yaml(path).get("allowed_feature_concept_relations")
    if not isinstance(values, Sequence) or isinstance(values, str) or not values:
        raise ExpertBundleError(
            f"{path.name} has no non-empty allowed_feature_concept_relations"
        )
    return tuple(str(v) for v in values)


def _check_feature_order(doc_ids: Sequence[str], dataset_ids: Sequence[str]) -> None:
    """Raise naming the first mismatch between the document and the model input order."""
    doc_ids, dataset_ids = list(doc_ids), list(dataset_ids)
    if len(doc_ids) != len(dataset_ids):
        raise ExpertBundleError(
            f"expert prior has {len(doc_ids)} features, model input has {len(dataset_ids)}"
        )
    for j, (a, b) in enumerate(zip(doc_ids, dataset_ids)):
        if a != b:
            raise ExpertBundleError(
                f"expert prior feature order does not match the model input at index {j}: "
                f"document has {a!r}, model has {b!r}"
            )


def load_expert_bundle(
    feature_names: Sequence[str],
    dataset: Optional[str] = None,
    path: Optional[PathLike] = None,
    request_path: Optional[PathLike] = None,
    with_diagnostics: bool = False,
) -> ExpertPriorBundle:
    """Load a dataset's expert graph and check it against `feature_names`.

    Either `dataset` (resolved to the default artifact location) or an explicit `path`
    to expert_prior.yaml must be given. `with_diagnostics` additionally loads the card
    embeddings and the per-edge confidence and direction; none of them enter forward().
    """
    if path is None:
        if dataset is None:
            raise ExpertBundleError("load_expert_bundle needs either dataset or path")
        path = expert_prior_dir(dataset) / "expert_prior.yaml"
    path = Path(path)

    doc = _read_yaml(path)
    cards = doc.get("feature_cards")
    if not isinstance(cards, Sequence) or isinstance(cards, str) or not cards:
        raise ExpertBundleError(f"{path.name} has no non-empty feature_cards")
    _check_feature_order([str(c["feature_id"]).strip() for c in cards], feature_names)

    if dataset is None and request_path is None:
        raise ExpertBundleError("load_expert_bundle needs either dataset or request_path")
    vocabulary = relation_vocabulary(dataset or "", request_path)

    features_emb = concepts_emb = None
    if with_diagnostics:
        directory = path.parent
        features_emb = _load_embeddings(directory / "feature_embeddings.pt")
        concepts_emb = _load_embeddings(directory / "concept_embeddings.pt")

    return ExpertPriorBundle.from_document(
        doc,
        feature_names,
        vocabulary,
        with_diagnostics=with_diagnostics,
        feature_embeddings=features_emb,
        concept_embeddings=concepts_emb,
    )


def _load_embeddings(path: Path) -> Optional[Tensor]:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return torch.as_tensor(payload["embeddings"], dtype=torch.float32)
