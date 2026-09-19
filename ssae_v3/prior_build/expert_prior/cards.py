"""Collect the texts that get embedded. Verbatim, always.

Both FeatureCards and ConceptCards carry their own `embedding_text`, written by the
expert, and that string is what is embedded - unchanged, with no template, no
concatenation of neighbouring fields and no fallback. If a card's text is weak, the
fix is the contract or the generation, never a repair here: a text assembled in
Python would be an embedding of our own phrasing wearing the model's authority.

The only decision this module makes is row order, and it is fixed rather than
inherited from the response:

  features  the order of `<dataset>_features.yaml`, which is the dataset's covariate
            order - the same order V's rows follow and the online loader asserts.
  concepts  the order the expert emitted them in, which is the only order they have.

Both are returned explicitly alongside the texts so a downstream reorder is a
detectable mismatch rather than a silent misalignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple


class CardError(ValueError):
    """A validated document still did not yield the texts to embed."""


@dataclass(frozen=True)
class EmbeddingTexts:
    """Aligned ids and texts for one card kind, in a fixed, recorded order."""

    kind: str
    ids: Tuple[str, ...]
    texts: Tuple[str, ...]
    order: str

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.texts):
            raise CardError(f"{self.kind}: {len(self.ids)} ids but {len(self.texts)} texts")

    @property
    def row_of(self) -> Dict[str, int]:
        return {cid: i for i, cid in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)


def _text_of(card: Mapping[str, Any], label: str) -> str:
    text = card.get("embedding_text")
    if not isinstance(text, str) or not text.strip():
        raise CardError(f"{label} has no usable embedding_text")
    return text.strip()


def feature_embedding_texts(
    doc: Mapping[str, Any],
    feature_ids: Sequence[str],
) -> EmbeddingTexts:
    """FeatureCard texts in features-file order. The document supplies every text."""
    cards = {str(c["feature_id"]).strip(): c for c in doc["feature_cards"]}
    missing = [fid for fid in feature_ids if fid not in cards]
    if missing:
        # validation enforces coverage, so reaching here means the two disagree
        raise CardError(f"no FeatureCard for {missing}")

    ids: List[str] = []
    texts: List[str] = []
    for fid in feature_ids:
        ids.append(fid)
        texts.append(_text_of(cards[fid], f"feature_cards[{fid}]"))
    return EmbeddingTexts("features", tuple(ids), tuple(texts), order="features_yaml")


def concept_embedding_texts(doc: Mapping[str, Any]) -> EmbeddingTexts:
    """ConceptCard texts in the order the expert emitted them."""
    ids: List[str] = []
    texts: List[str] = []
    for i, card in enumerate(doc["concept_bank"]):
        cid = str(card["concept_id"]).strip()
        ids.append(cid)
        texts.append(_text_of(card, f"concept_bank[{i}] ({cid})"))
    return EmbeddingTexts("concepts", tuple(ids), tuple(texts), order="response_order")
