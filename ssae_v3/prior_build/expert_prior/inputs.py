"""The three researcher-authored YAML inputs, loaded and validated independently.

`<dataset>_task_context.yaml` describes the estimation problem, `<dataset>_features.yaml`
lists the observed covariates, `<dataset>_expert_prior_request.yaml` is the output
contract. They are separate files because they answer to different authorities: the
first two describe the study, the third describes what we want back. Each is checked
on its own terms here, before anything is composed, so a malformed file is reported
against its own path rather than surfacing later as a confusing prompt.

Both the parsed structure and the exact file text are kept. The prompt embeds the raw
text, never a re-dump: `yaml.safe_dump` would reorder keys and reflow block scalars,
which would silently break the correspondence between the recorded input hashes and
the recorded prompt hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import yaml

PathLike = Union[str, Path]

SEMANTIC_INPUTS = Path(__file__).resolve().parents[1] / "semantic_inputs"

# The contract fields the rest of the pipeline dereferences by name. Everything else
# in the request file is consumed generically by the validator, so it is not listed.
_REQUEST_SCHEMAS = (
    "top_level_schema",
    "feature_card_schema",
    "concept_card_schema",
    "feature_relation_schema",
)


class InputError(ValueError):
    """A researcher-authored input file is missing or malformed."""


@dataclass(frozen=True)
class InputFile:
    """One YAML input: where it came from, what it says, and what it hashes to."""

    path: Path
    text: str
    data: Dict[str, Any]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExpertPriorInputs:
    """The validated trio for one dataset, plus the derived feature order."""

    dataset: str
    task_context: InputFile
    features: InputFile
    request: InputFile
    feature_ids: Tuple[str, ...]

    @property
    def input_hashes(self) -> Dict[str, str]:
        return {
            self.task_context.path.name: self.task_context.sha256,
            self.features.path.name: self.features.sha256,
            self.request.path.name: self.request.sha256,
        }


def inputs_dir(dataset: str) -> Path:
    """Directory holding one dataset's three expert-prior inputs."""
    return SEMANTIC_INPUTS / dataset


def _read_yaml(path: Path) -> InputFile:
    if not path.exists():
        raise InputError(f"no input file at {path}")
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise InputError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise InputError(f"{path.name} must contain a YAML mapping at the top level")
    return InputFile(path=path, text=text, data=dict(data))


def _require(data: Mapping[str, Any], dotted: str, where: str) -> Any:
    """Fetch a dotted path, raising an InputError naming the file and the path."""
    node: Any = data
    walked: List[str] = []
    for part in dotted.split("."):
        walked.append(part)
        if not isinstance(node, Mapping) or part not in node:
            raise InputError(f"{where}: missing required key {'.'.join(walked)}")
        node = node[part]
    if node is None or (isinstance(node, str) and not node.strip()):
        raise InputError(f"{where}: key {dotted} is empty")
    return node


def validate_task_context(f: InputFile, dataset: str) -> None:
    """Check the study description carries every field the prompt binds."""
    where = f.path.name
    for dotted in (
        "dataset.id",
        "dataset.name",
        "dataset.domain",
        "population.description",
        "treatment.name",
        "treatment.description",
        "outcome.name",
        "outcome.description",
        "estimand.primary",
        "estimand.definition",
    ):
        _require(f.data, dotted, where)

    declared = str(f.data["dataset"]["id"]).strip()
    if declared != dataset:
        raise InputError(f"{where}: dataset.id is {declared!r} but the build asked for {dataset!r}")


def validate_features(f: InputFile, dataset: str) -> Tuple[str, ...]:
    """Check the covariate list and return the ids in file order.

    File order is load-bearing downstream: it is the row order of the feature
    embedding matrix, and it is the order the online prior loader asserts against the
    dataset's own covariate order.
    """
    where = f.path.name
    declared = str(_require(f.data, "dataset", where)).strip()
    if declared != dataset:
        raise InputError(f"{where}: dataset is {declared!r} but the build asked for {dataset!r}")

    features = f.data.get("features")
    if not isinstance(features, Sequence) or isinstance(features, str) or not features:
        raise InputError(f"{where}: 'features' must be a non-empty YAML sequence")

    ids: List[str] = []
    for i, entry in enumerate(features):
        at = f"{where}: features[{i}]"
        if not isinstance(entry, Mapping):
            raise InputError(f"{at} must be a mapping")
        for key in ("id", "name", "description", "data_type"):
            if not str(entry.get(key, "")).strip():
                raise InputError(f"{at} is missing a non-empty {key!r}")
        ids.append(str(entry["id"]).strip())

    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise InputError(f"{where}: duplicate feature ids {duplicates}")
    return tuple(ids)


def validate_request(f: InputFile) -> None:
    """Check the contract is complete and that every enum path in it resolves.

    The validator is generated from this file, so an enum pointing at a list that
    does not exist would otherwise only fail once a response came back - after the
    expensive part. Resolving every path here makes that a load-time error.
    """
    where = f.path.name
    _require(f.data, "concept_bank.target_min_concepts", where)
    _require(f.data, "concept_bank.target_max_concepts", where)
    lo = int(f.data["concept_bank"]["target_min_concepts"])
    hi = int(f.data["concept_bank"]["target_max_concepts"])
    if lo > hi:
        raise InputError(f"{where}: target_min_concepts {lo} exceeds target_max_concepts {hi}")

    output = _require(f.data, "output", where)
    if not isinstance(output, Mapping):
        raise InputError(f"{where}: 'output' must be a mapping")
    for name in _REQUEST_SCHEMAS:
        schema = output.get(name)
        if not isinstance(schema, Mapping) or not schema:
            raise InputError(f"{where}: output.{name} must be a non-empty mapping of field -> spec")

    for name in _REQUEST_SCHEMAS:
        for field, spec in output[name].items():
            at = f"{where}: output.{name}.{field}"
            if not isinstance(spec, Mapping):
                raise InputError(f"{at} must be a mapping of type/enum/... keys")
            if spec.get("type") not in ("string", "list"):
                raise InputError(f"{at} declares unsupported type {spec.get('type')!r}")
            if spec["type"] == "list" and "items" not in spec:
                raise InputError(f"{at} is a list but does not declare 'items'")
            items = spec.get("items")
            if items is not None and items != "string" and items not in output:
                raise InputError(f"{at} refers to unknown item schema {items!r}")
            if "enum" in spec:
                try:
                    values = _require(f.data, str(spec["enum"]), where)
                except InputError as exc:
                    raise InputError(f"{at} enum path does not resolve: {exc}") from exc
                if not isinstance(values, Sequence) or isinstance(values, str):
                    raise InputError(f"{at} enum path {spec['enum']!r} is not a list")


def load_inputs(dataset: str, directory: PathLike | None = None) -> ExpertPriorInputs:
    """Load and validate the three inputs for `dataset`, each on its own terms."""
    base = Path(directory) if directory is not None else inputs_dir(dataset)
    task_context = _read_yaml(base / f"{dataset}_task_context.yaml")
    features = _read_yaml(base / f"{dataset}_features.yaml")
    request = _read_yaml(base / f"{dataset}_expert_prior_request.yaml")

    validate_task_context(task_context, dataset)
    feature_ids = validate_features(features, dataset)
    validate_request(request)

    return ExpertPriorInputs(
        dataset=dataset,
        task_context=task_context,
        features=features,
        request=request,
        feature_ids=feature_ids,
    )
