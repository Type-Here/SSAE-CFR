"""Covariate descriptions: turn feature names into the text the LLM embeds.

The embedding model does not see a bare column name; it sees a standardized prompt with
a short human-readable gloss slotted in, so that `bun` becomes a sentence about blood
urea nitrogen rather than the token "bun". The gloss for each covariate lives in a
per-dataset YAML (feature_name -> gloss) under `prior/glosses/`, authored and versioned
with the code.

The prompt template defaults to a clinical one, which suits the ICU and trial datasets
but not every dataset here: IHDP's covariates are obstetric and socioeconomic, and
asking a model for the "physiological role" of a mother's education level pushes every
one of those vectors toward the same unhelpful region. A gloss file may therefore set
its own template under the reserved key `_template`. Reserved keys start with an
underscore and are never treated as covariates.

Workflow:
  1. `emit_gloss_template` writes a YAML stub for a dataset, one line per covariate in
     the dataset's column order, pre-filled with a prettified column name as the default
     gloss. You then edit the glosses that need domain wording (IHDP especially, whose
     columns are anonymized x1..x25).
  2. `descriptions_for` reads that YAML and produces the ordered list of prompts to feed
     `build_embeddings`. A covariate left blank falls back to its prettified name, so the
     build never breaks - but `missing_glosses` flags which ones used the fallback.

Keeping the covariate order identical between the gloss file and the dataset is what
guarantees row j of V lines up with covariate j; the build stores that order in V's
sidecar and training asserts it.
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

    Reserved keys (anything starting with an underscore, currently just `_template`) are
    settings rather than covariates and are excluded, so the caller can keep treating
    the returned keys as the covariate order.
    """
    return {k: v for k, v in _read_yaml_mapping(path).items() if not k.startswith("_")}


def load_prompt_template(path: PathLike, default: str = PROMPT_TEMPLATE) -> str:
    """The gloss file's own prompt template, or `default` when it does not set one.

    The template must contain a `{gloss}` field; anything else would silently produce m
    identical prompts and therefore a rank-1 V.
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

    A missing/blank gloss falls back to the prettified column name so the build is never
    blocked; use `missing_glosses` to report those before an expensive run.
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

    Defaults each gloss to the prettified column name (`default_to_name`) so datasets
    with meaningful columns work out of the box and only the opaque ones need editing.
    """
    data = {name: (prettify(name) if default_to_name else "") for name in feature_names}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return path
