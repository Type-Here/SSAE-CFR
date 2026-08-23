"""Tests for the covariate-description plumbing (glosses -> prompts).

The build's correctness hinges on prompt order matching the covariate order and on the
blank-gloss fallback being visible rather than silent.
"""

from __future__ import annotations

from ssae_cfr.prior import descriptions_for, emit_gloss_template, load_glosses, missing_glosses
from ssae_cfr.prior.descriptions import PROMPT_TEMPLATE, prettify


def test_descriptions_follow_feature_order():
    names = ["bun", "creatinine", "age"]
    glosses = {"bun": "blood urea nitrogen", "creatinine": "serum creatinine", "age": "patient age"}
    prompts = descriptions_for(names, glosses)
    assert len(prompts) == 3
    assert "blood urea nitrogen" in prompts[0]
    assert "serum creatinine" in prompts[1]
    assert prompts[0] == PROMPT_TEMPLATE.format(gloss="blood urea nitrogen")


def test_blank_gloss_falls_back_to_prettified_name():
    names = ["vaso_max_rate", "bun"]
    glosses = {"vaso_max_rate": "", "bun": "blood urea nitrogen"}
    prompts = descriptions_for(names, glosses)
    assert "vaso max rate" in prompts[0]  # underscores -> spaces
    assert missing_glosses(names, glosses) == ["vaso_max_rate"]


def test_missing_glosses_reports_absent_keys():
    names = ["a", "b", "c"]
    glosses = {"a": "alpha"}  # b, c absent entirely
    assert missing_glosses(names, glosses) == ["b", "c"]


def test_prettify():
    assert prettify("vaso_any_baseline") == "vaso any baseline"
    assert prettify("age") == "age"


def test_emit_and_reload_round_trip(tmp_path):
    names = ["x1", "bun", "age"]
    path = tmp_path / "g.yaml"
    emit_gloss_template(names, path)
    glosses = load_glosses(path)
    assert list(glosses.keys()) == names, "gloss file must preserve covariate order"
    # default_to_name pre-fills the prettified column name
    assert glosses["bun"] == "bun"


def test_emit_blank_when_default_off(tmp_path):
    path = tmp_path / "g.yaml"
    emit_gloss_template(["a", "b"], path, default_to_name=False)
    glosses = load_glosses(path)
    assert all(v == "" for v in glosses.values())