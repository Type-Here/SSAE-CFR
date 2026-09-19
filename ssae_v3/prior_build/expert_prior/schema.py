"""Validate the expert response against the contract, repairing nothing.

The contract is the request YAML's `output` block, and this module is an interpreter
for it rather than a second copy of it. Field names, container shapes, enums,
reference targets and cardinalities are all read from that file at runtime, so
tightening the contract needs no change here. What lives in code is only the meaning
of the type language:

    type: string           scalar string, non-empty after stripping
    type: list             YAML sequence; `items` gives the element type
    items: string          every element is a non-empty scalar string
    items: <schema_name>   every element is a mapping checked against that schema
    enum: <dotted.path>    the value, or every element, must appear in that list
    references: <target>   every value must resolve against a reference target
    min_items: n           minimum sequence length, default 0
    required: false        the field may be absent, default true
    unique_key: <field>    that field is distinct across the sequence
    coverage: {target,key} exactly one element per member of the target
    count_from: <block>    length bounded by that block's target_min/max_concepts

Two severities, and the difference is the whole point of the module. A schema
violation is an error: the build stops, no `expert_prior.yaml` is written, and the
raw response is kept for inspection. A `quality_thresholds` violation is a warning:
it is recorded and printed, and nothing is changed on its account. Python never fills
a missing field, never coerces a scalar into a list, never renames a value to the
nearest allowed one and never drops an offending element.

The single exception to "nothing is modified" is at parse time: one outer Markdown
fence wrapping the whole response is removed, because it is a formatting wrapper
rather than content. It is recorded as `fence_stripped` in the report and the
manifest, `response_raw.txt` keeps the unmodified text, and no other repair exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import yaml

# A fence that opens on the first line and closes on the last, e.g. ```yaml ... ```
_FENCE_OPEN = re.compile(r"\A\s*(?:```+|~~~+)[ \t]*[A-Za-z0-9_+-]*[ \t]*\n")
_FENCE_CLOSE = re.compile(r"\n[ \t]*(?:```+|~~~+)[ \t]*\Z")


class SchemaError(ValueError):
    """The response does not conform to the contract. Carries every failure found."""

    def __init__(self, failures: Sequence[str]):
        self.failures = list(failures)
        head = f"expert prior response failed validation ({len(self.failures)} problems):"
        super().__init__("\n  ".join([head, *self.failures]))


@dataclass
class ValidationReport:
    """Outcome of one validation pass. `errors` non-empty means nothing was written."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fence_stripped: bool = False
    n_feature_cards: int = 0
    n_concepts: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "fence_stripped": self.fence_stripped,
            "n_feature_cards": self.n_feature_cards,
            "n_concepts": self.n_concepts,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def strip_outer_fence(raw: str) -> Tuple[str, bool]:
    """Remove one Markdown fence wrapping the whole text. Returns (text, stripped)."""
    text = raw.replace("\r\n", "\n").strip("\n")
    if not _FENCE_OPEN.search(text):
        return text, False
    inner = _FENCE_OPEN.sub("", text, count=1)
    inner = _FENCE_CLOSE.sub("", inner, count=1)
    return inner.strip("\n"), True


def parse_response(raw: str) -> Tuple[Any, bool]:
    """Parse the model's text into a document. The only normalization is the fence."""
    text, fence_stripped = strip_outer_fence(raw)
    if not text.strip():
        raise SchemaError(["response is empty"])
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SchemaError([f"response is not valid YAML: {exc}"]) from exc
    return doc, fence_stripped


def _resolve(root: Mapping[str, Any], dotted: str) -> Any:
    node: Any = root
    for part in str(dotted).split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def _is_seq(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


class _ContractValidator:
    """Checks one document against one contract. Accumulates every failure."""

    def __init__(self, request: Mapping[str, Any], feature_ids: Sequence[str]):
        self.request = request
        self.output = request["output"]
        self.feature_ids = [str(i) for i in feature_ids]
        self.errors: List[str] = []
        self.refs: Dict[str, Set[str]] = {"features_file.id": set(self.feature_ids)}

    def fail(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}")

    # -- the type language -------------------------------------------------

    def check_mapping(self, path: str, value: Any, schema_name: str) -> None:
        schema = self.output[schema_name]
        if not isinstance(value, Mapping):
            self.fail(path, f"expected a mapping ({schema_name}), got {type(value).__name__}")
            return

        declared = set(schema)
        for key in value:
            if key not in declared:
                self.fail(f"{path}.{key}", f"field is not declared in {schema_name}")

        for name, spec in schema.items():
            at = f"{path}.{name}"
            if name not in value:
                if spec.get("required", True):
                    self.fail(at, f"required field missing from {schema_name}")
                continue
            self.check_field(at, value[name], spec)

    def check_field(self, path: str, value: Any, spec: Mapping[str, Any]) -> None:
        if spec.get("type") == "string":
            self.check_scalar(path, value, spec)
        else:
            self.check_list(path, value, spec)

    def check_scalar(self, path: str, value: Any, spec: Mapping[str, Any]) -> None:
        if _is_seq(value) or isinstance(value, Mapping):
            self.fail(path, "expected a scalar string, got a sequence or mapping")
            return
        if not isinstance(value, str):
            self.fail(path, f"expected a string, got {type(value).__name__}")
            return
        if not value.strip():
            self.fail(path, "string is empty")
            return
        self.check_enum(path, value, spec)
        self.check_reference(path, value, spec)

    def check_list(self, path: str, value: Any, spec: Mapping[str, Any]) -> None:
        if not _is_seq(value):
            self.fail(path, f"expected a YAML sequence, got {type(value).__name__}")
            return

        items = spec.get("items")
        minimum = int(spec.get("min_items", 0))
        if len(value) < minimum:
            self.fail(path, f"needs at least {minimum} entries, has {len(value)}")

        if "count_from" in spec:
            self.check_count_from(path, value, str(spec["count_from"]))

        for i, element in enumerate(value):
            at = f"{path}[{i}]"
            if items == "string":
                self.check_scalar(at, element, spec)
            else:
                self.check_mapping(at, element, str(items))

        if "unique_key" in spec:
            self.check_unique(path, value, str(spec["unique_key"]), items)
        if "coverage" in spec:
            self.check_coverage(path, value, spec["coverage"])

    def check_enum(self, path: str, value: str, spec: Mapping[str, Any]) -> None:
        if "enum" not in spec:
            return
        try:
            allowed = _resolve(self.request, str(spec["enum"]))
        except KeyError:
            self.fail(path, f"contract enum path {spec['enum']!r} does not resolve")
            return
        if value.strip() not in {str(a) for a in allowed}:
            self.fail(path, f"{value!r} is not in {spec['enum']} ({sorted(map(str, allowed))})")

    def check_reference(self, path: str, value: str, spec: Mapping[str, Any]) -> None:
        if "references" not in spec:
            return
        target = str(spec["references"])
        known = self.refs.get(target)
        if known is None:
            self.fail(path, f"contract reference target {target!r} is unknown")
            return
        if value.strip() not in known:
            self.fail(path, f"{value!r} does not resolve against {target}")

    def check_count_from(self, path: str, value: Sequence[Any], block: str) -> None:
        bounds = self.request.get(block)
        if not isinstance(bounds, Mapping):
            self.fail(path, f"contract count_from block {block!r} is missing")
            return
        lo = int(bounds["target_min_concepts"])
        hi = int(bounds["target_max_concepts"])
        if not lo <= len(value) <= hi:
            self.fail(path, f"holds {len(value)} entries, contract allows {lo}..{hi}")

    def check_unique(self, path: str, value: Sequence[Any], key: str, items: Any) -> None:
        seen: Dict[str, int] = {}
        for i, element in enumerate(value):
            if items == "string":
                got = element if isinstance(element, str) else None
            else:
                got = element.get(key) if isinstance(element, Mapping) else None
            if not isinstance(got, str):
                continue
            got = got.strip()
            if got in seen:
                self.fail(f"{path}[{i}].{key}", f"{got!r} already used at index {seen[got]}")
            else:
                seen[got] = i

    def check_coverage(self, path: str, value: Sequence[Any], spec: Mapping[str, Any]) -> None:
        target_name = str(spec["target"])
        key = str(spec["key"])
        target = self.refs.get(target_name)
        if target is None:
            self.fail(path, f"contract coverage target {target_name!r} is unknown")
            return
        present = [
            str(e[key]).strip()
            for e in value
            if isinstance(e, Mapping) and isinstance(e.get(key), str)
        ]
        missing = [i for i in sorted(target, key=self._target_order) if i not in set(present)]
        if missing:
            self.fail(path, f"no entry for {len(missing)} member(s) of {target_name}: {missing}")
        duplicates = sorted({i for i in present if present.count(i) > 1})
        if duplicates:
            self.fail(path, f"more than one entry for {duplicates}")

    def _target_order(self, value: str) -> int:
        return self.feature_ids.index(value) if value in self.feature_ids else len(self.feature_ids)

    # -- entry point -------------------------------------------------------

    def run(self, doc: Any) -> List[str]:
        if not isinstance(doc, Mapping):
            self.fail("<root>", f"expected a YAML mapping, got {type(doc).__name__}")
            return self.errors
        # concept ids must be known before feature_cards.candidate_concepts is checked
        self.refs["concept_bank.concept_id"] = _collect_concept_ids(doc)
        self.check_mapping("<root>", doc, "top_level_schema")
        return self.errors


def _collect_concept_ids(doc: Mapping[str, Any]) -> Set[str]:
    bank = doc.get("concept_bank")
    if not _is_seq(bank):
        return set()
    out: Set[str] = set()
    for entry in bank:
        if isinstance(entry, Mapping) and isinstance(entry.get("concept_id"), str):
            out.add(entry["concept_id"].strip())
    return out


# -- advisory checks -------------------------------------------------------


def _words(text: str) -> List[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def _feature_id_fraction(text: str, feature_ids: Iterable[str]) -> float:
    ids = {i.lower() for i in feature_ids}
    words = _words(text)
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.strip(".,;:()[]{}'\"").lower() in ids)
    return hits / len(words)


def quality_warnings(
    doc: Mapping[str, Any],
    request: Mapping[str, Any],
    feature_ids: Sequence[str],
    raw_text: str,
) -> List[str]:
    """Advisory findings. Recorded and printed; they never change or block anything."""
    thresholds = request.get("quality_thresholds") or {}
    min_words = int(thresholds.get("embedding_text_min_words", 0))
    max_id_fraction = float(thresholds.get("embedding_text_max_feature_id_fraction", 1.0))
    min_features = int(thresholds.get("concept_min_supporting_features", 0))
    leakage_terms = [str(t) for t in (thresholds.get("leakage_terms") or [])]

    out: List[str] = []

    cards: List[Tuple[str, Mapping[str, Any]]] = []
    for key, id_field in (("feature_cards", "feature_id"), ("concept_bank", "concept_id")):
        entries = doc.get(key)
        if _is_seq(entries):
            for entry in entries:
                if isinstance(entry, Mapping):
                    cards.append((f"{key}[{entry.get(id_field, '?')}]", entry))

    for label, card in cards:
        text = card.get("embedding_text")
        if not isinstance(text, str):
            continue
        n = len(_words(text))
        if n < min_words:
            out.append(f"{label}.embedding_text is {n} words, below the advisory floor of {min_words}")
        fraction = _feature_id_fraction(text, feature_ids)
        if fraction > max_id_fraction:
            out.append(
                f"{label}.embedding_text is {fraction:.0%} bare feature IDs, above the "
                f"advisory ceiling of {max_id_fraction:.0%}"
            )

    bank = doc.get("concept_bank")
    if _is_seq(bank):
        for entry in bank:
            if not isinstance(entry, Mapping):
                continue
            relations = entry.get("feature_relations")
            n = len(relations) if _is_seq(relations) else 0
            if n < min_features:
                out.append(
                    f"concept_bank[{entry.get('concept_id', '?')}] rests on {n} feature(s); "
                    f"the contract prefers at least {min_features} where sensible"
                )

    for term in leakage_terms:
        if re.search(rf"\b{re.escape(term)}\b", raw_text, flags=re.IGNORECASE):
            out.append(
                f"response mentions {term!r}, which is listed as forbidden_information - "
                "check by hand that no outcome or simulation knowledge leaked in"
            )
    return out


def validate_expert_prior(
    doc: Any,
    request: Mapping[str, Any],
    feature_ids: Sequence[str],
    *,
    raw_text: str = "",
    fence_stripped: bool = False,
) -> ValidationReport:
    """Check `doc` against the contract. Raises SchemaError listing every failure."""
    validator = _ContractValidator(request, feature_ids)
    errors = validator.run(doc)
    if errors:
        raise SchemaError(errors)

    report = ValidationReport(
        fence_stripped=fence_stripped,
        n_feature_cards=len(doc["feature_cards"]),
        n_concepts=len(doc["concept_bank"]),
    )
    report.warnings = quality_warnings(doc, request, feature_ids, raw_text or yaml.safe_dump(doc))
    return report
