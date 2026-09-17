"""Covariate descriptions: turn feature names into the text the LLM embeds.

The embedding model does not see a bare column name; it sees a standardized prompt
with a short human-readable gloss slotted in. The gloss for each covariate lives in
a per-dataset YAML (feature_name -> gloss) under `prior_build/glosses/`.

A dataset's covariates may not fit the default clinical prompt template (IHDP's are
obstetric/socioeconomic), so a gloss file may set its own template under the reserved
key `_template`. Reserved keys start with an underscore and are never covariates.

Workflow: `emit_gloss_template` writes an editable YAML stub in the dataset's column
order; `descriptions_for` turns the edited glosses into the ordered prompt list that
`build_embeddings` consumes. Keeping the covariate order identical between the gloss
file and the dataset is what guarantees row j of V lines up with covariate j.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Union

import yaml

PathLike = Union[str, Path]

PROMPT_TEMPLATE = "Physiological role and prognostic impact of {gloss} in the clinical context."

TEMPLATE_KEY = "_template"


def prettify(name: str) -> str:
    """A readable default gloss from a column name: underscores to spaces."""
    return name.replace("_", " ").strip()


def _read_yaml_mapping(path: PathLike) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must contain a YAML mapping of feature_name -> gloss")
    return {str(k): ("" if v is None else str(v)) for k, v in data.items()}


def load_glosses(path: PathLike) -> Dict[str, str]:
    """Load a feature_name -> gloss mapping from YAML (order preserved).

    Reserved keys (anything starting with an underscore) are settings rather than
    covariates and are excluded from the result.
    """
    return {k: v for k, v in _read_yaml_mapping(path).items() if not k.startswith("_")}


def load_prompt_template(path: PathLike, default: str = PROMPT_TEMPLATE) -> str:
    """The gloss file's own prompt template, or `default` when it does not set one.

    The template must contain a `{gloss}` field; without it every prompt would be
    identical, giving a rank-1 V silently.
    """
    template = str(_read_yaml_mapping(path).get(TEMPLATE_KEY, "")).strip() or default
    if "{gloss}" not in template:
        raise ValueError(f"prompt template must contain '{{gloss}}'; got {template!r}")
    return template


def missing_glosses(feature_names: Sequence[str], glosses: Mapping[str, str]) -> List[str]:
    """Feature names with no (non-empty) gloss - they will use the prettified fallback."""
    return [name for name in feature_names if not str(glosses.get(name, "")).strip()]


def descriptions_for(
    feature_names: Sequence[str],
    glosses: Mapping[str, str],
    template: str = PROMPT_TEMPLATE,
) -> List[str]:
    """Ordered list of prompt strings, one per feature name.

    A missing/blank gloss falls back to the prettified column name so the build is
    never blocked; use `missing_glosses` to report those before an expensive run.
    """
    prompts: List[str] = []
    for name in feature_names:
        gloss = str(glosses.get(name, "")).strip() or prettify(name)
        prompts.append(template.format(gloss=gloss))
    return prompts


def emit_gloss_template(
    feature_names: Sequence[str],
    path: PathLike,
    default_to_name: bool = True,
) -> Path:
    """Write an editable YAML gloss stub in covariate order.

    Defaults each gloss to the prettified column name (`default_to_name`) so
    datasets with meaningful columns work out of the box.
    """
    data = {name: (prettify(name) if default_to_name else "") for name in feature_names}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return path
