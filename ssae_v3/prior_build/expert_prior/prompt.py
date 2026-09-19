"""Deterministic composition of the three inputs into one expert-generation prompt.

The template is `semantic_inputs/prompt.ninja2`, shared across datasets, with four
top-level sections: ROLE, TASK CONTEXT, OBSERVED FEATURES, OUTPUT CONTRACT. Only the
dataset-specific values and the three raw YAML blocks are substituted in.

Two properties matter more than convenience here.

Determinism: the same inputs must give a byte-identical prompt, because the manifest
records `prompt_sha256` beside the input hashes and that chain is the reproducibility
claim. Jinja runs with StrictUndefined, so a template placeholder with no binding
raises instead of rendering an empty string; line endings are normalized and the
result carries exactly one trailing newline.

Fidelity: the three `*_yaml` blocks are the raw file text. A `yaml.safe_dump` round
trip would reorder keys and reflow block scalars, so the text the model reads would
no longer be the text the input hash was taken over.

Every binding below is a literal lookup into a validated input. Nothing here infers,
defaults or summarizes; `estimand_type` performs the one join, of two adjacent fields.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Union

from .inputs import SEMANTIC_INPUTS, ExpertPriorInputs

PathLike = Union[str, Path]

TEMPLATE_PATH = SEMANTIC_INPUTS / "prompt.ninja2"

# Sections the composed prompt must contain. Checked after rendering: the four are
# what the pipeline promises, so a template edit that drops one should fail loudly
# rather than quietly produce a prompt with a missing contract.
REQUIRED_SECTIONS = ("ROLE", "TASK CONTEXT", "OBSERVED FEATURES", "OUTPUT CONTRACT")


class PromptError(ValueError):
    """The prompt could not be composed from the inputs and the template."""


def _normalize(text: str) -> str:
    """LF line endings, exactly one trailing newline, nothing else touched."""
    return text.replace("\r\n", "\n").rstrip("\n") + "\n"


def bindings(inputs: ExpertPriorInputs) -> Dict[str, str]:
    """Template variable -> value. Every entry is a literal lookup."""
    ctx = inputs.task_context.data
    req = inputs.request.data
    estimand = ctx["estimand"]
    return {
        "dataset_name": str(ctx["dataset"]["name"]).strip(),
        "dataset_domain": str(ctx["dataset"]["domain"]).strip(),
        "population_description": str(ctx["population"]["description"]).strip(),
        "treatment_name": str(ctx["treatment"]["name"]).strip(),
        "treatment_description": str(ctx["treatment"]["description"]).strip(),
        "outcome_name": str(ctx["outcome"]["name"]).strip(),
        "outcome_description": str(ctx["outcome"]["description"]).strip(),
        # the one join: the estimand's name and its definition, which the template
        # presents on a single line
        "estimand_type": f"{str(estimand['primary']).strip()} = {str(estimand['definition']).strip()}",
        "concept_min": str(req["concept_bank"]["target_min_concepts"]),
        "concept_max": str(req["concept_bank"]["target_max_concepts"]),
        # raw file text, deliberately not re-dumped
        "task_context_yaml": inputs.task_context.text.rstrip("\n"),
        "features_yaml": inputs.features.text.rstrip("\n"),
        "expert_request_yaml": inputs.request.text.rstrip("\n"),
    }


def load_template(path: PathLike | None = None) -> str:
    """Read the shared prompt template."""
    path = Path(path) if path is not None else TEMPLATE_PATH
    if not path.exists():
        raise PromptError(f"no prompt template at {path}")
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def compose_prompt(inputs: ExpertPriorInputs, template_path: PathLike | None = None) -> str:
    """Render the final prompt. Raises rather than leaving a placeholder unbound."""
    from jinja2 import Environment, StrictUndefined, TemplateError

    template_text = load_template(template_path)
    env = Environment(
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    try:
        rendered = env.from_string(template_text).render(**bindings(inputs))
    except TemplateError as exc:
        raise PromptError(f"prompt template failed to render: {exc}") from exc

    prompt = _normalize(rendered)
    missing = [s for s in REQUIRED_SECTIONS if s not in prompt]
    if missing:
        raise PromptError(f"composed prompt is missing required sections: {missing}")
    if "{{" in prompt:
        raise PromptError("composed prompt still contains an unrendered '{{' placeholder")
    return prompt


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
