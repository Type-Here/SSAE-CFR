"""The expert-prior pipeline: inputs, prompt composition, the contract validator.

Everything here runs without transformers, a GPU or a built artifact: the two stages
that need weights are `generate` and `embed`, and the rest - which is where the
contract is actually enforced - is pure Python over YAML. The synthetic response below
is deliberately minimal but conformant, and most tests mutate one field of it, so a
failure names the rule that broke rather than a whole document.

The negative cases matter more than the positive one. The pipeline's central promise
is that it repairs nothing, and the way to keep that honest is to check that each
malformation is rejected rather than quietly absorbed.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import yaml

from ssae_v3.prior_build.expert_prior.build import POOLING as BUILD_POOLING
from ssae_v3.prior_build.expert_prior.cards import (
    CardError,
    concept_embedding_texts,
    feature_embedding_texts,
)
from ssae_v3.prior_build.expert_prior.embed import (
    EmbeddingError,
    EmbeddingResult,
    _pool,
    save_embeddings,
)
from ssae_v3.prior_build.expert_prior.generate import (
    GenerationError,
    GenerationPlan,
    _context_limit,
    chat_wrap,
    quantization_record,
    resolve_dtype,
    size_budget,
)
from ssae_v3.prior_build.expert_prior.inputs import InputError, load_inputs, validate_features
from ssae_v3.prior_build.expert_prior.prompt import (
    PromptError,
    REQUIRED_SECTIONS,
    bindings,
    compose_prompt,
    sha256_text,
)
from ssae_v3.prior_build.expert_prior.schema import (
    SchemaError,
    parse_response,
    quality_warnings,
    strip_outer_fence,
    validate_expert_prior,
)

DATASET = "ihdp"

LONG_TEXT = (
    "This describes a pre-treatment characteristic of the infant and mother recorded "
    "at enrollment, expressed in ordinary clinical language so that it remains "
    "meaningful on its own. It is a plausible prognostic marker in this population of "
    "low birth weight premature infants, and it may also relate to how families were "
    "selected into the intervention arm, though neither relation is established here."
)


@pytest.fixture(scope="module")
def inputs():
    return load_inputs(DATASET)


def _feature_card(fid: str) -> Dict[str, Any]:
    return {
        "feature_id": fid,
        "canonical_name": f"canonical name for {fid}",
        "information_type": "clinical",
        "expert_summary": f"What {fid} represents in the current task.",
        "candidate_task_roles": ["prognostic"],
        "candidate_concepts": ["c1"],
        "warnings": [],
        "embedding_text": LONG_TEXT,
    }


def _concept_card(cid: str, feature_ids: List[str]) -> Dict[str, Any]:
    return {
        "concept_id": cid,
        "name": f"concept {cid}",
        "description": f"A description of concept {cid}.",
        "information_type": "clinical",
        "concept_type": "latent_state_or_construct",
        "temporal_status": "pretreatment",
        "task_relevance": ["prognostic"],
        "feature_relations": [
            {
                "feature_id": fid,
                "relation": "indicator",
                "confidence": "medium",
                "rationale": f"Why {fid} carries evidence about {cid}.",
            }
            for fid in feature_ids
        ],
        "expert_confidence": "medium",
        "caveats": [],
        "embedding_text": LONG_TEXT,
    }


@pytest.fixture
def doc(inputs) -> Dict[str, Any]:
    ids = list(inputs.feature_ids)
    return {
        "schema_version": "1.0",
        "task_summary": "A summary of the estimation task.",
        "feature_cards": [_feature_card(f) for f in ids],
        "concept_bank": [
            _concept_card("c1", ids[:2]),
            _concept_card("c2", ids[2:4]),
            _concept_card("c3", ids[4:6]),
            _concept_card("c4", ids[6:8]),
        ],
        "global_caveats": ["Everything here is a hypothesis."],
    }


def _expect_failure(doc, inputs, fragment: str):
    with pytest.raises(SchemaError) as exc:
        validate_expert_prior(doc, inputs.request.data, inputs.feature_ids)
    joined = "\n".join(exc.value.failures)
    assert fragment in joined, joined


# -- inputs ----------------------------------------------------------------


def test_inputs_load_and_expose_feature_order(inputs):
    assert inputs.dataset == DATASET
    assert inputs.feature_ids[:3] == ("x1", "x2", "x3")
    assert len(inputs.feature_ids) == 25
    assert len(inputs.input_hashes) == 3


def test_input_hashes_are_over_the_raw_text(inputs):
    assert inputs.features.sha256 == sha256_text(inputs.features.text)


def test_features_file_must_match_the_requested_dataset(inputs):
    with pytest.raises(InputError, match="but the build asked for"):
        validate_features(inputs.features, "not_ihdp")


def test_missing_dataset_directory_is_reported(tmp_path):
    with pytest.raises(InputError, match="no input file at"):
        load_inputs(DATASET, directory=tmp_path)


# -- prompt ----------------------------------------------------------------


def test_prompt_has_the_four_sections_and_no_placeholders(inputs):
    prompt = compose_prompt(inputs)
    for section in REQUIRED_SECTIONS:
        assert section in prompt
    assert "{{" not in prompt


def test_prompt_is_deterministic(inputs):
    assert compose_prompt(inputs) == compose_prompt(inputs)


def test_prompt_embeds_the_raw_input_text_not_a_redump(inputs):
    prompt = compose_prompt(inputs)
    # a yaml.safe_dump round trip would reflow this block scalar
    assert inputs.features.text.rstrip("\n") in prompt
    assert inputs.request.text.rstrip("\n") in prompt


def test_every_binding_is_a_nonempty_string(inputs):
    for key, value in bindings(inputs).items():
        assert isinstance(value, str) and value.strip(), key


def test_unbound_placeholder_raises_rather_than_rendering_empty(inputs, tmp_path):
    bad = tmp_path / "bad.ninja2"
    bad.write_text("ROLE TASK CONTEXT OBSERVED FEATURES OUTPUT CONTRACT {{ nope }}", "utf-8")
    with pytest.raises(PromptError, match="failed to render"):
        compose_prompt(inputs, template_path=bad)


def test_template_missing_a_section_is_rejected(inputs, tmp_path):
    bad = tmp_path / "bad.ninja2"
    bad.write_text("ROLE TASK CONTEXT OBSERVED FEATURES only", "utf-8")
    with pytest.raises(PromptError, match="missing required sections"):
        compose_prompt(inputs, template_path=bad)


# -- parsing ---------------------------------------------------------------


@pytest.mark.parametrize("fence", ["```yaml", "```", "~~~yaml"])
def test_one_outer_fence_is_stripped_and_reported(fence):
    text, stripped = strip_outer_fence(f"{fence}\nkey: value\n```")
    assert stripped is True
    assert text == "key: value"


def test_unfenced_text_is_untouched():
    text, stripped = strip_outer_fence("key: value\n")
    assert (text, stripped) == ("key: value", False)


def test_fence_inside_the_body_is_not_treated_as_a_wrapper():
    raw = "key: value\nother: |\n  ```\n"
    text, stripped = strip_outer_fence(raw)
    assert stripped is False
    assert "```" in text


def test_invalid_yaml_raises_schema_error():
    with pytest.raises(SchemaError, match="not valid YAML"):
        parse_response("key: [unclosed\n")


def test_empty_response_raises():
    with pytest.raises(SchemaError, match="empty"):
        parse_response("   \n")


# -- the contract ----------------------------------------------------------


def test_conformant_document_validates(doc, inputs):
    report = validate_expert_prior(doc, inputs.request.data, inputs.feature_ids)
    assert report.ok
    assert report.n_feature_cards == 25
    assert report.n_concepts == 4
    assert report.warnings == []


def test_missing_required_field_is_rejected(doc, inputs):
    del doc["feature_cards"][3]["embedding_text"]
    _expect_failure(doc, inputs, "required field missing")


def test_undeclared_field_is_rejected_not_ignored(doc, inputs):
    doc["feature_cards"][0]["extra_thoughts"] = "unasked for"
    _expect_failure(doc, inputs, "not declared in feature_card_schema")


def test_scalar_where_a_list_is_declared_is_rejected(doc, inputs):
    doc["feature_cards"][0]["candidate_task_roles"] = "prognostic"
    _expect_failure(doc, inputs, "expected a YAML sequence")


def test_list_where_a_scalar_is_declared_is_rejected(doc, inputs):
    doc["feature_cards"][0]["information_type"] = ["clinical"]
    _expect_failure(doc, inputs, "expected a scalar string")


def test_value_outside_an_enum_is_rejected(doc, inputs):
    doc["feature_cards"][0]["information_type"] = "vibes"
    _expect_failure(doc, inputs, "allowed_information_types")


def test_concept_task_role_vocabulary_excludes_the_feature_only_role(doc, inputs):
    doc["concept_bank"][0]["task_relevance"] = ["proxy_for_unobserved_construct"]
    _expect_failure(doc, inputs, "allowed_concept_task_roles")


def test_feature_task_role_vocabulary_still_allows_it(doc, inputs):
    doc["feature_cards"][0]["candidate_task_roles"] = ["proxy_for_unobserved_construct"]
    assert validate_expert_prior(doc, inputs.request.data, inputs.feature_ids).ok


def test_concept_type_and_information_type_are_separate_axes(doc, inputs):
    # a clinical construct and a study-design construct are both well formed
    doc["concept_bank"][1]["information_type"] = "study_design"
    doc["concept_bank"][1]["concept_type"] = "study_design_construct"
    assert validate_expert_prior(doc, inputs.request.data, inputs.feature_ids).ok
    # but the two vocabularies do not cross
    doc["concept_bank"][1]["concept_type"] = "study_design"
    _expect_failure(doc, inputs, "allowed_concept_types")


def test_temporal_status_is_constrained(doc, inputs):
    doc["concept_bank"][0]["temporal_status"] = "whenever"
    _expect_failure(doc, inputs, "allowed_temporal_statuses")


def test_empty_role_list_is_rejected_because_unclear_exists(doc, inputs):
    doc["feature_cards"][0]["candidate_task_roles"] = []
    _expect_failure(doc, inputs, "needs at least 1")


def test_empty_candidate_concepts_is_allowed(doc, inputs):
    doc["feature_cards"][0]["candidate_concepts"] = []
    assert validate_expert_prior(doc, inputs.request.data, inputs.feature_ids).ok


def test_missing_feature_card_is_rejected(doc, inputs):
    doc["feature_cards"].pop()
    _expect_failure(doc, inputs, "no entry for 1 member(s)")


def test_duplicate_feature_card_is_rejected(doc, inputs):
    doc["feature_cards"].append(_feature_card("x1"))
    _expect_failure(doc, inputs, "more than one entry")


def test_card_for_an_unknown_feature_is_rejected(doc, inputs):
    doc["feature_cards"][0]["feature_id"] = "x99"
    _expect_failure(doc, inputs, "does not resolve against features_file.id")


def test_reference_to_an_unknown_concept_is_rejected(doc, inputs):
    doc["feature_cards"][0]["candidate_concepts"] = ["c_nonexistent"]
    _expect_failure(doc, inputs, "does not resolve against concept_bank.concept_id")


def test_relation_to_an_unknown_feature_is_rejected(doc, inputs):
    doc["concept_bank"][0]["feature_relations"][0]["feature_id"] = "x99"
    _expect_failure(doc, inputs, "does not resolve against features_file.id")


def test_same_feature_twice_in_one_concept_is_rejected(doc, inputs):
    relations = doc["concept_bank"][0]["feature_relations"]
    relations.append(copy.deepcopy(relations[0]))
    _expect_failure(doc, inputs, "already used at index")


def test_duplicate_concept_id_is_rejected(doc, inputs):
    doc["concept_bank"][1]["concept_id"] = "c1"
    _expect_failure(doc, inputs, "already used at index")


def test_concept_count_below_the_contract_minimum_is_rejected(doc, inputs):
    doc["concept_bank"] = doc["concept_bank"][:2]
    _expect_failure(doc, inputs, "contract allows")


def test_concept_count_above_the_contract_maximum_is_rejected(doc, inputs):
    ids = list(inputs.feature_ids)
    doc["concept_bank"] = [_concept_card(f"c{i}", ids[:2]) for i in range(20)]
    _expect_failure(doc, inputs, "contract allows")


def test_extra_top_level_key_is_rejected(doc, inputs):
    doc["appendix"] = "unasked for"
    _expect_failure(doc, inputs, "not declared in top_level_schema")


def test_missing_top_level_key_is_rejected(doc, inputs):
    del doc["task_summary"]
    _expect_failure(doc, inputs, "required field missing")


def test_optional_field_may_be_present_or_absent(doc, inputs):
    doc["concept_bank"][0]["feature_relations"][0]["expected_direction"] = "higher"
    assert validate_expert_prior(doc, inputs.request.data, inputs.feature_ids).ok


def test_every_failure_is_reported_not_just_the_first(doc, inputs):
    doc["feature_cards"][0]["information_type"] = "vibes"
    doc["feature_cards"][1]["information_type"] = "vibes"
    del doc["feature_cards"][2]["warnings"]
    with pytest.raises(SchemaError) as exc:
        validate_expert_prior(doc, inputs.request.data, inputs.feature_ids)
    assert len(exc.value.failures) >= 3


def test_a_non_mapping_response_is_rejected(inputs):
    _expect_failure(["not", "a", "mapping"], inputs, "expected a YAML mapping")


# -- advisory warnings -----------------------------------------------------


def test_short_embedding_text_warns_but_does_not_fail(doc, inputs):
    doc["feature_cards"][0]["embedding_text"] = "Too short."
    report = validate_expert_prior(doc, inputs.request.data, inputs.feature_ids)
    assert report.ok
    assert any("below the advisory floor" in w for w in report.warnings)


def test_embedding_text_of_bare_ids_warns(doc, inputs):
    doc["concept_bank"][0]["embedding_text"] = "x1 x2 x3 x4 x5 and also x6"
    report = validate_expert_prior(doc, inputs.request.data, inputs.feature_ids)
    assert any("bare feature IDs" in w for w in report.warnings)


def test_single_feature_concept_warns(doc, inputs):
    doc["concept_bank"][0]["feature_relations"] = doc["concept_bank"][0]["feature_relations"][:1]
    report = validate_expert_prior(doc, inputs.request.data, inputs.feature_ids)
    assert any("rests on 1 feature" in w for w in report.warnings)


def test_leakage_term_is_surfaced_for_review(doc, inputs):
    raw = yaml.safe_dump(doc) + "\nnote: derived from mu1\n"
    warnings = quality_warnings(doc, inputs.request.data, inputs.feature_ids, raw)
    assert any("mu1" in w for w in warnings)


def test_leakage_scan_matches_whole_words_only(doc, inputs):
    raw = "the study site and the mother was white"
    assert quality_warnings(doc, inputs.request.data, inputs.feature_ids, raw) == []


# -- embedding texts -------------------------------------------------------


def test_feature_texts_follow_the_features_file_order(doc, inputs):
    doc["feature_cards"].reverse()
    texts = feature_embedding_texts(doc, inputs.feature_ids)
    assert texts.ids == tuple(inputs.feature_ids)
    assert texts.order == "features_yaml"
    assert texts.row_of["x1"] == 0


def test_concept_texts_follow_the_emitted_order(doc):
    texts = concept_embedding_texts(doc)
    assert texts.ids == ("c1", "c2", "c3", "c4")
    assert texts.order == "response_order"


def test_texts_are_the_experts_own_verbatim(doc, inputs):
    doc["feature_cards"][0]["embedding_text"] = "  A distinctive sentence written by the expert.  "
    texts = feature_embedding_texts(doc, inputs.feature_ids)
    # stripped of surrounding whitespace, otherwise untouched, and assembled from
    # nothing else on the card
    assert texts.texts[0] == "A distinctive sentence written by the expert."
    assert doc["feature_cards"][0]["canonical_name"] not in texts.texts[0]


def test_blank_embedding_text_is_never_substituted_for(doc, inputs):
    doc["feature_cards"][0]["embedding_text"] = "   "
    with pytest.raises(CardError, match="no usable embedding_text"):
        feature_embedding_texts(doc, inputs.feature_ids)


# -- generation budget -----------------------------------------------------
# Pure arithmetic, so it runs without transformers. This is the check that decides
# whether a run is worth starting, and it must fail before any weight is loaded.


def test_budget_defaults_to_whatever_the_context_leaves():
    assert size_budget(1000, 32768, margin=64) == 32768 - 1000 - 64


def test_budget_refuses_a_prompt_that_leaves_too_little():
    with pytest.raises(GenerationError, match="below the floor"):
        size_budget(31000, 32768, margin=64, min_budget=3000)


def test_budget_refuses_a_request_that_does_not_fit():
    with pytest.raises(GenerationError, match="does not fit"):
        size_budget(1000, 4096, margin=64, min_budget=100, max_new_tokens=4000)


def test_budget_refuses_a_request_below_the_floor():
    with pytest.raises(GenerationError, match="below the floor"):
        size_budget(1000, 32768, margin=64, min_budget=3000, max_new_tokens=500)


def test_budget_honours_an_explicit_request_that_fits():
    assert size_budget(1000, 32768, margin=64, max_new_tokens=6000) == 6000


def test_context_limit_takes_the_smallest_credible_source():
    config = SimpleNamespace(max_position_embeddings=4096)
    tokenizer = SimpleNamespace(model_max_length=32768)
    assert _context_limit(config, tokenizer) == (4096, "config.max_position_embeddings")


def test_context_limit_ignores_the_no_limit_sentinel():
    config = SimpleNamespace(max_position_embeddings=32768)
    tokenizer = SimpleNamespace(model_max_length=int(1e30))
    limit, source = _context_limit(config, tokenizer)
    assert (limit, source) == (32768, "config.max_position_embeddings")


def test_no_declared_context_limit_is_an_error_not_a_guess():
    config = SimpleNamespace()
    tokenizer = SimpleNamespace(model_max_length=int(1e30))
    with pytest.raises(GenerationError, match="no usable context limit"):
        _context_limit(config, tokenizer)


# -- load width ------------------------------------------------------------
# Also pure: whether a run was quantized is a fact about the artifact and has to be
# decidable, and checkable, without the library that would perform the quantization.


def test_dtype_auto_follows_the_device():
    assert resolve_dtype("auto", "cuda:0") == "float16"
    assert resolve_dtype("auto", "cpu") == "float32"


def test_an_explicit_dtype_is_never_overridden():
    assert resolve_dtype("bfloat16", "cuda:0") == "bfloat16"
    assert resolve_dtype("float32", "cuda:0") == "float32"


def test_full_width_records_no_quantization_rather_than_an_empty_one():
    # None, not {}: a manifest must distinguish "not quantized" from "quantized and
    # not written down"
    assert quantization_record(False, "cuda:0", "auto") is None


def test_4bit_records_everything_needed_to_reproduce_it():
    record = quantization_record(True, "cuda:0", "auto")
    assert record == {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "float16",
        "double_quant": True,
    }


def test_4bit_compute_dtype_follows_the_requested_width():
    assert quantization_record(True, "cuda:0", "bfloat16")["compute_dtype"] == "bfloat16"


def test_4bit_on_cpu_is_refused_before_anything_is_downloaded():
    with pytest.raises(GenerationError, match="needs a CUDA device"):
        quantization_record(True, "cpu", "auto")


# -- addressing the model --------------------------------------------------
# The first real run answered with more output-format rules instead of a YAML
# document: an instruct model handed a long document with no instruction turn
# continues it. Exercised against a stub, so no weights and no transformers.


class _StubTokenizer:
    """Just enough tokenizer to exercise the wrapping decision."""

    def __init__(self, chat_template=None):
        self.chat_template = chat_template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False, "the plan needs the rendered text, not token ids"
        body = "".join(m["content"] for m in messages)
        tail = "[/INST]" if add_generation_prompt else ""
        return f"<s>[INST]{body}{tail}"


def test_chat_template_wraps_the_prompt_and_asks_for_a_generation():
    text, note = chat_wrap("CONTRACT", _StubTokenizer("a template"))
    assert note is None
    assert "CONTRACT" in text
    # without the trailing generation prompt the model has not been handed the turn
    assert text.endswith("[/INST]")


def test_no_chat_template_falls_back_to_raw_and_says_why():
    text, note = chat_wrap("CONTRACT", _StubTokenizer(None))
    assert text == "CONTRACT"
    assert "no chat template" in note


def test_chat_template_can_be_declined_for_a_base_model():
    text, note = chat_wrap("CONTRACT", _StubTokenizer("a template"), False)
    assert text == "CONTRACT"
    assert "disabled by request" in note


def test_the_plan_records_what_was_actually_fed_to_the_model():
    plan = GenerationPlan(
        model_name="m", prompt_tokens=10, context_limit=4096, margin=64,
        max_new_tokens=3000, context_source="config.max_position_embeddings",
        chat_template_applied=True, model_input="<s>[INST]CONTRACT[/INST]",
    )
    record = plan.as_dict()
    assert record["chat_template_applied"] is True
    # the wrapped text is hashed, not stored: the manifest says which string was fed
    assert record["model_input_sha256"] == sha256_text("<s>[INST]CONTRACT[/INST]")
    assert "chat template" in plan.describe()


def test_a_plan_with_no_model_input_hashes_to_nothing_rather_than_to_an_empty_string():
    plan = GenerationPlan(
        model_name="m", prompt_tokens=10, context_limit=4096, margin=64,
        max_new_tokens=3000, context_source="config.max_position_embeddings",
    )
    assert plan.as_dict()["model_input_sha256"] is None


# -- pooling and persistence -----------------------------------------------


def test_mean_pooling_ignores_padding():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(_pool(hidden, mask, "mean"), torch.tensor([[2.0, 2.0]]))


def test_last_pooling_takes_the_final_real_token():
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(_pool(hidden, mask, "last"), torch.tensor([[3.0, 3.0]]))


def test_unknown_pooling_is_rejected():
    torch = pytest.importorskip("torch")
    with pytest.raises(EmbeddingError, match="unknown pooling"):
        _pool(torch.zeros(1, 2, 2), torch.ones(1, 2), "cls")


def test_saved_embeddings_carry_a_stable_id_to_row_mapping(tmp_path):
    torch = pytest.importorskip("torch")
    vectors = np.arange(6, dtype=np.float32).reshape(3, 2)
    result = EmbeddingResult(
        vectors=vectors, pooling="mean", model_name="fake", dtype="float32",
        max_length=512, token_counts=(10, 20, 900), truncated=(2,),
    )
    path = save_embeddings(
        tmp_path / "f.pt", kind="features", ids=["x1", "x2", "x3"],
        texts=["a", "b", "c"], order="features_yaml", result=result,
        source_sha256="deadbeef",
    )
    payload = torch.load(path, weights_only=False)
    assert payload["row_of"] == {"x1": 0, "x2": 1, "x3": 2}
    assert payload["truncated_ids"] == ["x3"]
    assert payload["pooling"] == "mean"
    assert payload["order"] == "features_yaml"
    assert torch.allclose(payload["embeddings"], torch.from_numpy(vectors))


def test_saving_refuses_a_mismatched_id_count(tmp_path):
    pytest.importorskip("torch")
    result = EmbeddingResult(
        vectors=np.zeros((3, 2), dtype=np.float32), pooling="mean", model_name="fake",
        dtype="float32", max_length=512, token_counts=(1, 1, 1), truncated=(),
    )
    with pytest.raises(EmbeddingError, match="2 ids but 3 vectors"):
        save_embeddings(
            tmp_path / "f.pt", kind="features", ids=["x1", "x2"], texts=["a", "b"],
            order="features_yaml", result=result, source_sha256="deadbeef",
        )


def test_expert_v1_pins_mean_pooling():
    # one artifact, one pooling rule: `last` exists for a later, separate ablation
    assert BUILD_POOLING == "mean"
