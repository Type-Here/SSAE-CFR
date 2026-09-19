"""Tests for the expert-graph bundle, adapter and matched controls.

The adapter's central promise is negative: nothing that identifies a feature or a
concept reaches the forward pass, and the expert graph is used exactly as stated. Most
of what follows checks that a malformed graph is rejected rather than quietly accepted,
and that renaming or enriching the graph's metadata changes nothing numerically.

Nothing here needs transformers, a GPU or a built artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from ssae_v3.prior_modules.expert_adapter import ExpertPriorAdapter
from ssae_v3.prior_modules.expert_bundle import (
    NO_EDGE,
    ExpertBundleError,
    ExpertPriorBundle,
    load_expert_bundle,
)
from ssae_v3.prior_modules.expert_graph_controls import (
    EXPERT_GRAPH_CONTROLS,
    apply_graph_control,
    permuted_graph,
    random_matched_graph,
)

FEATURE_IDS = ("x1", "x2", "x3", "x4", "x5", "x6")
VOCABULARY = ("direct_measure", "indicator", "proxy", "contextual_factor")

# concept -> [(feature, relation)]. x6 is deliberately in no concept.
GRAPH = {
    "neonatal_health": [("x1", "direct_measure"), ("x2", "indicator"), ("x3", "indicator")],
    "prenatal_exposure": [("x3", "proxy"), ("x4", "direct_measure")],
    "site_context": [("x5", "contextual_factor")],
}


def make_document(graph=None, feature_ids=FEATURE_IDS):
    """A minimal document carrying only what the bundle reads."""
    graph = GRAPH if graph is None else graph
    return {
        "feature_cards": [{"feature_id": f} for f in feature_ids],
        "concept_bank": [
            {
                "concept_id": cid,
                "feature_relations": [
                    {
                        "feature_id": f,
                        "relation": r,
                        "confidence": "medium",
                        "expected_direction": "increases",
                    }
                    for f, r in relations
                ],
            }
            for cid, relations in graph.items()
        ],
    }


def make_bundle(**kwargs):
    return ExpertPriorBundle.from_document(make_document(), FEATURE_IDS, VOCABULARY, **kwargs)


def make_adapter(bundle=None, d_u=5, d_edge=3, rho_hidden=(7,)):
    bundle = bundle if bundle is not None else make_bundle()
    return ExpertPriorAdapter(
        bundle.relation_mask,
        bundle.relation_type_index,
        bundle.n_relation_types,
        d_u=d_u,
        d_edge=d_edge,
        rho_hidden=rho_hidden,
    )


# --------------------------------------------------------------------------- bundle


def test_bundle_encodes_the_stated_graph():
    """The mask and the type index reproduce the edges the document declares."""
    bundle = make_bundle()
    assert bundle.feature_ids == FEATURE_IDS
    assert bundle.concept_ids == tuple(GRAPH)
    assert bundle.relation_mask.shape == (6, 3)
    assert bundle.n_active_edges == 6

    row = {f: j for j, f in enumerate(FEATURE_IDS)}
    for c, (cid, relations) in enumerate(GRAPH.items()):
        for f, r in relations:
            assert bool(bundle.relation_mask[row[f], c])
            assert int(bundle.relation_type_index[row[f], c]) == VOCABULARY.index(r)
    assert torch.equal(bundle.edges_per_concept, torch.tensor([3, 2, 1]))


def test_inactive_pairs_carry_the_no_edge_sentinel():
    """A pair with no relation is NO_EDGE, never relation type 0."""
    bundle = make_bundle()
    inactive = bundle.relation_type_index[~bundle.relation_mask]
    assert inactive.numel() == 6 * 3 - 6
    assert bool((inactive == NO_EDGE).all())


def test_unknown_feature_reference_is_rejected():
    graph = dict(GRAPH, extra=[("x99", "proxy")])
    with pytest.raises(ExpertBundleError, match="unknown feature"):
        ExpertPriorBundle.from_document(make_document(graph), FEATURE_IDS, VOCABULARY)


def test_unknown_relation_type_is_rejected():
    graph = dict(GRAPH, extra=[("x6", "telepathy")])
    with pytest.raises(ExpertBundleError, match="unknown relation type"):
        ExpertPriorBundle.from_document(make_document(graph), FEATURE_IDS, VOCABULARY)


def test_concept_with_no_relations_is_rejected():
    graph = dict(GRAPH, empty=[])
    with pytest.raises(ExpertBundleError, match="no feature_relations"):
        ExpertPriorBundle.from_document(make_document(graph), FEATURE_IDS, VOCABULARY)


def test_concept_with_no_active_edge_is_rejected_at_construction():
    """A hand-built or control-produced bundle gets the same check."""
    mask = torch.tensor([[True, False], [True, False]])
    index = torch.tensor([[0, NO_EDGE], [1, NO_EDGE]])
    with pytest.raises(ExpertBundleError, match="no active feature relations"):
        ExpertPriorBundle(("x1", "x2"), ("a", "b"), mask, index, VOCABULARY)


def test_duplicate_concept_id_is_rejected():
    bundle = make_bundle()
    with pytest.raises(ExpertBundleError, match="duplicate concept ids"):
        ExpertPriorBundle(
            bundle.feature_ids,
            ("a", "a", "b"),
            bundle.relation_mask,
            bundle.relation_type_index,
            VOCABULARY,
        )


def test_repeated_feature_within_one_concept_is_rejected():
    graph = dict(GRAPH, neonatal_health=[("x1", "direct_measure"), ("x1", "indicator")])
    with pytest.raises(ExpertBundleError, match="more than once"):
        ExpertPriorBundle.from_document(make_document(graph), FEATURE_IDS, VOCABULARY)


def test_inconsistent_tensor_shapes_are_rejected():
    bundle = make_bundle()
    with pytest.raises(ExpertBundleError, match="does not match relation_mask"):
        ExpertPriorBundle(
            bundle.feature_ids,
            bundle.concept_ids,
            bundle.relation_mask,
            bundle.relation_type_index[:, :2],
            VOCABULARY,
        )


def test_mask_and_index_must_agree_on_which_edges_exist():
    bundle = make_bundle()
    index = bundle.relation_type_index.clone()
    index[5, 0] = 0  # an index where the mask says there is no edge
    with pytest.raises(ExpertBundleError, match="disagree"):
        ExpertPriorBundle(
            bundle.feature_ids, bundle.concept_ids, bundle.relation_mask, index, VOCABULARY
        )


def test_relation_index_outside_the_vocabulary_is_rejected():
    bundle = make_bundle()
    index = bundle.relation_type_index.clone()
    index[0, 0] = len(VOCABULARY)
    with pytest.raises(ExpertBundleError, match="out of range"):
        ExpertPriorBundle(
            bundle.feature_ids, bundle.concept_ids, bundle.relation_mask, index, VOCABULARY
        )


def test_feature_order_mismatch_hard_fails(tmp_path: Path):
    """The document's feature order must equal the model's covariate order."""
    doc_path = tmp_path / "expert_prior.yaml"
    doc_path.write_text(yaml.safe_dump(make_document()), encoding="utf-8")
    request_path = tmp_path / "request.yaml"
    request_path.write_text(
        yaml.safe_dump({"allowed_feature_concept_relations": list(VOCABULARY)}), encoding="utf-8"
    )

    swapped = ("x2", "x1", "x3", "x4", "x5", "x6")
    with pytest.raises(ExpertBundleError, match="does not match the model input at index 0"):
        load_expert_bundle(swapped, path=doc_path, request_path=request_path)

    shorter = FEATURE_IDS[:-1]
    with pytest.raises(ExpertBundleError, match="6 features, model input has 5"):
        load_expert_bundle(shorter, path=doc_path, request_path=request_path)

    bundle = load_expert_bundle(FEATURE_IDS, path=doc_path, request_path=request_path)
    assert bundle.feature_ids == FEATURE_IDS


def test_diagnostics_are_off_by_default():
    """Confidence, direction and embeddings are absent unless explicitly requested."""
    plain = make_bundle()
    assert plain.relation_confidence is None
    assert plain.relation_direction is None
    assert plain.feature_embeddings is None
    assert plain.semantic_similarity is None

    rich = make_bundle(with_diagnostics=True)
    assert rich.relation_confidence[("x1", "neonatal_health")] == "medium"
    assert rich.relation_direction[("x1", "neonatal_health")] == "increases"


def test_semantic_similarity_is_available_but_only_as_a_diagnostic():
    bundle = make_bundle(
        feature_embeddings=torch.randn(6, 8), concept_embeddings=torch.randn(3, 8)
    )
    sim = bundle.semantic_similarity
    assert sim.shape == (6, 3)
    assert bool((sim.abs() <= 1.0 + 1e-5).all())


# -------------------------------------------------------------------------- adapter


def test_forward_shapes():
    adapter = make_adapter()
    x = torch.randn(4, 6)
    a_expert, diagnostics = adapter(x)
    assert a_expert.shape == (4, 5)
    assert adapter.concept_representations(x).shape == (4, 3, 3)
    assert diagnostics["n_active_edges"] == 6
    assert torch.equal(diagnostics["edges_per_concept"], torch.tensor([3, 2, 1]))
    assert diagnostics["concept_repr_norms"].shape == (3,)
    assert diagnostics["a_expert_norm"].shape == ()


def test_active_edges_are_gathered_in_stable_concept_major_order():
    """Edge order is concept-major with ascending feature index inside each concept."""
    adapter = make_adapter()
    assert adapter.edge_concept_idx.tolist() == [0, 0, 0, 1, 1, 2]
    assert adapter.edge_feature_idx.tolist() == [0, 1, 2, 2, 3, 4]
    assert adapter.edge_relation_type.tolist() == [
        VOCABULARY.index("direct_measure"),
        VOCABULARY.index("indicator"),
        VOCABULARY.index("indicator"),
        VOCABULARY.index("proxy"),
        VOCABULARY.index("direct_measure"),
        VOCABULARY.index("contextual_factor"),
    ]
    onehot = adapter.edge_onehot
    assert onehot.shape == (6, len(VOCABULARY))
    assert torch.equal(onehot.sum(dim=1), torch.ones(6))


def test_concept_order_is_the_artifact_order_and_is_reproducible():
    bundle = make_bundle()
    first, second = make_adapter(bundle), make_adapter(bundle)
    assert bundle.concept_ids == ("neonatal_health", "prenatal_exposure", "site_context")
    assert torch.equal(first.edge_concept_idx, second.edge_concept_idx)
    assert torch.equal(first.edge_feature_idx, second.edge_feature_idx)


def test_concept_representation_is_the_mean_over_that_concept_s_edges():
    """h_ic equals an explicit mean of phi_edge over the concept's own edges."""
    torch.manual_seed(0)
    adapter = make_adapter()
    x = torch.randn(3, 6)
    h = adapter.concept_representations(x)

    for c in range(3):
        edges = (adapter.edge_concept_idx == c).nonzero(as_tuple=True)[0]
        per_edge = []
        for e in edges:
            value = x[:, adapter.edge_feature_idx[e]].unsqueeze(-1)
            onehot = adapter.edge_onehot[e].unsqueeze(0).expand(x.shape[0], -1)
            per_edge.append(adapter.phi_edge(torch.cat([value, onehot], dim=-1)))
        expected = torch.stack(per_edge).mean(dim=0)
        assert torch.allclose(h[:, c], expected, atol=1e-6)


def test_a_feature_in_no_concept_never_reaches_the_adapter():
    """x6 is in no concept, so changing it cannot move the output."""
    torch.manual_seed(0)
    adapter = make_adapter()
    x = torch.randn(4, 6)
    other = x.clone()
    other[:, 5] = 99.0
    assert torch.equal(
        adapter.concept_representations(x), adapter.concept_representations(other)
    )


def test_output_is_exactly_zero_at_initialization():
    """Zero, not nearly zero: the branch must be an exact no-op before training."""
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        adapter = make_adapter()
        a_expert, _ = adapter(torch.randn(8, 6) * 10.0)
        assert torch.equal(a_expert, torch.zeros(8, 5))


def test_output_stops_being_zero_once_the_head_moves():
    """The zero is the head's initialization, not a dead path."""
    torch.manual_seed(0)
    adapter = make_adapter()
    torch.nn.init.normal_(adapter.head.weight)
    a_expert, _ = adapter(torch.randn(4, 6))
    assert not torch.equal(a_expert, torch.zeros(4, 5))


def test_forward_does_not_depend_on_feature_or_concept_ids():
    """Renaming every id leaves the output bitwise identical."""
    renamed_graph = {
        f"concept_{i}": [(f"feat_{FEATURE_IDS.index(f)}", r) for f, r in relations]
        for i, relations in enumerate(GRAPH.values())
    }
    renamed_ids = tuple(f"feat_{i}" for i in range(6))
    renamed = ExpertPriorBundle.from_document(
        make_document(renamed_graph, renamed_ids), renamed_ids, VOCABULARY
    )

    x = torch.randn(4, 6)
    torch.manual_seed(0)
    original_out, _ = make_adapter(make_bundle())(x)
    torch.manual_seed(0)
    renamed_out, _ = make_adapter(renamed)(x)
    assert torch.equal(original_out, renamed_out)


def test_diagnostic_fields_do_not_change_the_forward_pass():
    """Loading confidence, direction and embeddings leaves the output bitwise identical."""
    plain = make_bundle()
    rich = make_bundle(
        with_diagnostics=True,
        feature_embeddings=torch.randn(6, 8),
        concept_embeddings=torch.randn(3, 8),
    )
    x = torch.randn(4, 6)
    torch.manual_seed(0)
    plain_out, _ = make_adapter(plain)(x)
    torch.manual_seed(0)
    rich_out, _ = make_adapter(rich)(x)
    assert torch.equal(plain_out, rich_out)


def test_the_graph_is_not_trainable():
    adapter = make_adapter()
    names = {name for name, _ in adapter.named_parameters()}
    assert not any("edge_" in name or "relation_" in name for name in names)
    assert adapter.edge_onehot.requires_grad is False


def test_parameter_count_is_independent_of_the_number_of_edges():
    """Capacity depends on K, not on how many edges the expert drew."""
    sparse = make_bundle()
    dense_graph = {
        cid: [(f, "indicator") for f in FEATURE_IDS] for cid in GRAPH
    }
    dense = ExpertPriorBundle.from_document(
        make_document(dense_graph), FEATURE_IDS, VOCABULARY
    )
    assert dense.n_active_edges > sparse.n_active_edges

    count = lambda a: sum(p.numel() for p in a.parameters())
    assert count(make_adapter(sparse)) == count(make_adapter(dense))


# ------------------------------------------------------------------------- controls


def _statistics(bundle):
    counts = torch.bincount(
        bundle.relation_type_index[bundle.relation_mask], minlength=bundle.n_relation_types
    )
    return bundle.n_concepts, bundle.n_active_edges, bundle.edges_per_concept, counts


def _per_concept_composition(bundle):
    return [
        sorted(bundle.relation_type_index[bundle.relation_mask[:, c], c].tolist())
        for c in range(bundle.n_concepts)
    ]


@pytest.mark.parametrize("control", ("permuted", "random_matched"))
def test_controls_preserve_the_matched_statistics(control):
    """Concept count, edge count, per-concept degree and relation-type counts survive."""
    real = make_bundle()
    other = apply_graph_control(real, control, seed=0)

    K, E, degrees, counts = _statistics(real)
    other_K, other_E, other_degrees, other_counts = _statistics(other)
    assert (other_K, other_E) == (K, E)
    assert torch.equal(other_degrees, degrees)
    assert torch.equal(other_counts, counts)
    assert other.concept_ids == real.concept_ids
    assert not torch.equal(other.relation_mask, real.relation_mask)


def test_permuted_graph_also_preserves_each_concept_s_relation_composition():
    """A bijection over features cannot change what kinds of evidence a concept gets."""
    real = make_bundle()
    permuted = permuted_graph(real, seed=0)
    assert _per_concept_composition(permuted) == _per_concept_composition(real)


def test_permuted_graph_is_a_true_permutation_of_the_feature_axis():
    real = make_bundle()
    permuted = permuted_graph(real, seed=0)
    rows = {tuple(r.tolist()) for r in real.relation_mask}
    assert {tuple(r.tolist()) for r in permuted.relation_mask} == rows


def test_controls_are_seed_reproducible_and_seed_sensitive():
    real = make_bundle()
    for control in ("permuted", "random_matched"):
        a = apply_graph_control(real, control, seed=0)
        b = apply_graph_control(real, control, seed=0)
        c = apply_graph_control(real, control, seed=7)
        assert torch.equal(a.relation_mask, b.relation_mask)
        assert torch.equal(a.relation_type_index, b.relation_type_index)
        assert not torch.equal(a.relation_mask, c.relation_mask)


def test_controls_do_not_disturb_the_global_rng():
    real = make_bundle()
    torch.manual_seed(0)
    expected = torch.randn(3)
    torch.manual_seed(0)
    permuted_graph(real, seed=1)
    random_matched_graph(real, seed=1)
    assert torch.equal(torch.randn(3), expected)


def test_controls_drop_diagnostic_fields():
    """Confidence and direction are keyed by feature id, which a control reassigns."""
    rich = make_bundle(with_diagnostics=True, feature_embeddings=torch.randn(6, 8))
    permuted = permuted_graph(rich, seed=0)
    assert permuted.relation_confidence is None
    assert permuted.feature_embeddings is None


def test_real_control_is_the_bundle_itself():
    real = make_bundle()
    assert apply_graph_control(real, "real", seed=0) is real
    assert "real" in EXPERT_GRAPH_CONTROLS


def test_unknown_control_is_rejected():
    with pytest.raises(ValueError, match="unknown control"):
        apply_graph_control(make_bundle(), "attention", seed=0)


def test_a_control_graph_feeds_a_parameter_matched_adapter():
    real = make_bundle()
    for control in ("permuted", "random_matched"):
        other = apply_graph_control(real, control, seed=0)
        count = lambda a: sum(p.numel() for p in a.parameters())
        assert count(make_adapter(other)) == count(make_adapter(real))
