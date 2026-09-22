"""Tests for the paired prior-guidance sweep.

Fast tests only: small synthetic `PriorGuidanceBundle` objects built directly, the way
`tests/test_guidance_bundle.py` does, and a monkeypatched `run_realization` where a
sweep needs to be exercised end to end without training anything. One test at the
bottom trains a handful of real, tiny models on one IHDP realization and is skipped
unless both the replication set and the real expert-prior artifact are present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ssae_v3.data.ihdp import has_replication_set
from ssae_v3.hparams import load_config
from ssae_v3.prior_modules.guidance_bundle import (
    PriorGuidanceBundle,
    build_embedding_subspace,
    build_graph_subspace,
)
from ssae_v3.training import experiments, guidance_sweep
from ssae_v3.training.evaluate import aggregate
from ssae_v3.training.guidance_sweep import (
    DEFAULT_GAMMA_PRIOR,
    DEFAULT_LAMBDA_PRIOR,
    GUIDANCE_ARMS,
    arm_parameter_counts,
    build_arm_guidance,
    format_guidance_sweep,
    prior_mode_for,
    run_guidance_sweep,
    sign_test_p,
)

FEATURE_IDS = tuple(f"x{i}" for i in range(1, 9))  # m = 8


def _low_rank_embedding(m: int = 8, d: int = 5, rank: int = 3, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    basis = torch.randn(m, rank, generator=gen)
    coeffs = torch.randn(rank, d, generator=gen)
    signal = basis @ coeffs
    noise = 1e-3 * torch.randn(m, d, generator=gen)
    return signal + noise


def _synthetic_graph() -> torch.Tensor:
    """An (8, 3) incidence matrix with overlapping concepts."""
    m, K = 8, 3
    C = torch.zeros(m, K)
    edges = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (4, 1), (4, 2), (5, 2), (6, 2), (7, 2)]
    for f, c in edges:
        C[f, c] = 1.0
    return C


def _bundle(U_k=None, C=None, graph_source_metadata=None) -> PriorGuidanceBundle:
    P_U = None
    embedding_rank = 0
    if U_k is not None:
        P_U = U_k @ U_k.T
        embedding_rank = U_k.shape[1]
    Q_C = P_C = None
    graph_rank = 0
    if C is not None:
        Q_C, graph_rank = build_graph_subspace(C)
        P_C = Q_C @ Q_C.T
    return PriorGuidanceBundle(
        feature_ids=FEATURE_IDS,
        U_k=U_k,
        P_U=P_U,
        C=C,
        Q_C=Q_C,
        P_C=P_C,
        embedding_rank=embedding_rank,
        graph_rank=graph_rank,
        embedding_source_metadata={},
        graph_source_metadata=graph_source_metadata or {},
        source_hashes={},
    )


def _both_priors_bundle(graph_source_metadata=None) -> PriorGuidanceBundle:
    U_k, _, _ = build_embedding_subspace(_low_rank_embedding(), energy_threshold=0.90)
    C = _synthetic_graph()
    return _bundle(U_k=U_k, C=C, graph_source_metadata=graph_source_metadata)


def _tiny_cfg(**overrides):
    base = dict(
        in_channels=8,
        model_variant="empirical",
        d_u=8,
        encoder_hidden=(8,),
        decoder_hidden=(8,),
        head_hidden=(8,),
        is_noise_active=False,
    )
    base.update(overrides)
    return load_config(None, **base)


# --------------------------------------------------------------------------- #
# prior_mode_for
# --------------------------------------------------------------------------- #


def test_prior_mode_for_matches_every_guidance_arm():
    expected = {
        "empirical": "none",
        "u_real": "embedding",
        "u_random": "embedding",
        "graph_real": "graph",
        "graph_permuted": "graph",
        "graph_degree_matched": "graph",
        "both_real": "both",
        "u_real_graph_degree_matched": "both",
        "u_random_graph_real": "both",
        "both_random": "both",
    }
    assert set(expected) == set(GUIDANCE_ARMS)
    for name, (embedding_variant, graph_variant) in GUIDANCE_ARMS.items():
        assert prior_mode_for(embedding_variant, graph_variant) == expected[name], name


def test_prior_mode_for_raises_on_unknown_variant():
    with pytest.raises(ValueError):
        prior_mode_for("bogus", "none")
    with pytest.raises(ValueError):
        prior_mode_for("none", "bogus")


# --------------------------------------------------------------------------- #
# build_arm_guidance
# --------------------------------------------------------------------------- #


def test_build_arm_guidance_mode_and_ranks_for_every_arm():
    bundle = _both_priors_bundle()
    for name, (embedding_variant, graph_variant) in GUIDANCE_ARMS.items():
        guidance = build_arm_guidance(bundle, embedding_variant, graph_variant, gamma_prior=0.3, control_seed=1)
        assert guidance.mode == prior_mode_for(embedding_variant, graph_variant), name
        expected_embedding_rank = bundle.embedding_rank if embedding_variant != "none" else 0
        expected_graph_rank = bundle.graph_rank if graph_variant != "none" else 0
        assert guidance.embedding_rank == expected_embedding_rank, name
        assert guidance.graph_rank == expected_graph_rank, name


def test_random_embedding_and_degree_matched_graph_differ_from_real():
    bundle = _both_priors_bundle()
    real_u = build_arm_guidance(bundle, "real", "none", gamma_prior=0.3, control_seed=1)
    random_u = build_arm_guidance(bundle, "random", "none", gamma_prior=0.3, control_seed=1)
    assert not torch.equal(real_u.q_u, random_u.q_u)

    real_graph = build_arm_guidance(bundle, "none", "real", gamma_prior=0.3, control_seed=1)
    matched_graph = build_arm_guidance(bundle, "none", "degree_matched", gamma_prior=0.3, control_seed=1)
    assert not torch.equal(real_graph.q_c, matched_graph.q_c)


def test_control_seed_is_reproducible_and_varies():
    bundle = _both_priors_bundle()
    a = build_arm_guidance(bundle, "random", "none", gamma_prior=0.3, control_seed=5)
    b = build_arm_guidance(bundle, "random", "none", gamma_prior=0.3, control_seed=5)
    c = build_arm_guidance(bundle, "random", "none", gamma_prior=0.3, control_seed=6)
    assert torch.equal(a.q_u, b.q_u)
    assert not torch.equal(a.q_u, c.q_u)


# --------------------------------------------------------------------------- #
# arm_parameter_counts
# --------------------------------------------------------------------------- #


def test_arm_parameter_counts_equal_across_every_arm():
    bundle = _both_priors_bundle()
    cfg = _tiny_cfg()
    counts = arm_parameter_counts(cfg, bundle)
    assert set(counts) == set(GUIDANCE_ARMS)
    assert len(set(counts.values())) == 1


# --------------------------------------------------------------------------- #
# sign_test_p
# --------------------------------------------------------------------------- #


def test_sign_test_p_extremes_and_tie():
    assert sign_test_p(10, 10) == pytest.approx(0.001953125, abs=1e-6)
    assert sign_test_p(0, 10) == pytest.approx(0.001953125, abs=1e-6)
    assert sign_test_p(5, 10) == 1.0
    assert sign_test_p(0, 0) == 1.0


def test_sign_test_p_rejects_out_of_range_wins():
    with pytest.raises(ValueError):
        sign_test_p(11, 10)


# --------------------------------------------------------------------------- #
# run_guidance_sweep produces only numeric run-dict values
# --------------------------------------------------------------------------- #


def test_run_guidance_sweep_run_dicts_are_numeric_only(monkeypatch):
    bundle = _both_priors_bundle()
    cfg = _tiny_cfg()

    def fake_run_realization(
        realization,
        arm_cfg,
        prior_path=None,
        val_fraction=0.3,
        seed=0,
        verbose=False,
        control="none",
        control_seed=None,
        expert_bundle=None,
        guidance=None,
    ):
        return {
            "realization": float(realization),
            "out_pehe": 1.0 + 0.01 * realization,
            "out_eps_ate": 0.1,
            "val_factual_objective_normalized": 0.2,
            "pool_smd_reduction": 0.3,
            "train_alignment_active": 0.5,
            "train_guidance_loss": 0.05,
            "train_x_guided_delta_ratio": 0.1,
        }

    monkeypatch.setattr(guidance_sweep, "_load_bundle", lambda base, expert_dir: bundle)
    monkeypatch.setattr(guidance_sweep.experiments, "run_realization", fake_run_realization)

    arms = {"empirical": GUIDANCE_ARMS["empirical"], "u_real": GUIDANCE_ARMS["u_real"]}
    rows, summaries = run_guidance_sweep(
        realizations=(1, 2), arms=arms, cfg=cfg, gamma_prior=0.3, lambda_prior=0.1
    )

    assert set(rows) == set(arms)
    for label, runs in rows.items():
        assert len(runs) == 2
        for run in runs:
            for key, value in run.items():
                assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                    f"{label}.{key} = {value!r} is not numeric"
                )
    assert summaries["empirical"]["out_pehe"]["n_runs"] == 2


# --------------------------------------------------------------------------- #
# format_guidance_sweep provenance note
# --------------------------------------------------------------------------- #


def _fake_rows_and_summaries():
    rows = {
        "empirical": [{"realization": 1.0, "out_pehe": 1.0}, {"realization": 2.0, "out_pehe": 1.1}],
        "u_real": [{"realization": 1.0, "out_pehe": 0.9}, {"realization": 2.0, "out_pehe": 1.2}],
    }
    summaries = {name: aggregate(runs) for name, runs in rows.items()}
    return rows, summaries


def test_format_guidance_sweep_notes_authored_document_when_flag_absent():
    rows, summaries = _fake_rows_and_summaries()
    bundle = _both_priors_bundle(graph_source_metadata={})
    text = format_guidance_sweep(
        rows, summaries, DEFAULT_GAMMA_PRIOR, DEFAULT_LAMBDA_PRIOR,
        param_counts={"empirical": 10, "u_real": 10}, bundle=bundle,
    )
    assert "NOTE the expert document was authored" in text


def test_format_guidance_sweep_notes_authored_document_when_flag_false():
    rows, summaries = _fake_rows_and_summaries()
    bundle = _both_priors_bundle(graph_source_metadata={"response_is_generated": False})
    text = format_guidance_sweep(
        rows, summaries, DEFAULT_GAMMA_PRIOR, DEFAULT_LAMBDA_PRIOR,
        param_counts={"empirical": 10, "u_real": 10}, bundle=bundle,
    )
    assert "NOTE the expert document was authored" in text


def test_format_guidance_sweep_omits_note_when_flag_true():
    rows, summaries = _fake_rows_and_summaries()
    bundle = _both_priors_bundle(graph_source_metadata={"response_is_generated": True})
    text = format_guidance_sweep(
        rows, summaries, DEFAULT_GAMMA_PRIOR, DEFAULT_LAMBDA_PRIOR,
        param_counts={"empirical": 10, "u_real": 10}, bundle=bundle,
    )
    assert "NOTE the expert document was authored" not in text


def test_format_guidance_sweep_never_declares_a_winner():
    rows, summaries = _fake_rows_and_summaries()
    text = format_guidance_sweep(rows, summaries, DEFAULT_GAMMA_PRIOR, DEFAULT_LAMBDA_PRIOR)
    for banned in ("confirms", "proves", "better overall", "demonstrates", "evidence that"):
        assert banned not in text.lower()


# --------------------------------------------------------------------------- #
# end to end (real artifacts, real training, skipped without them)
# --------------------------------------------------------------------------- #

_IHDP_EXPERT_DIR = Path("artifacts/ihdp/expert_prior_2")


@pytest.mark.skipif(
    not (has_replication_set() and _IHDP_EXPERT_DIR.exists()),
    reason="IHDP replication set or the real expert-prior artifact is not present",
)
def test_end_to_end_guidance_sweep_smoke():
    cfg = load_config(
        str(experiments.DEFAULT_CONFIG),
        epochs=3,
        d_u=8,
        encoder_hidden=(8,),
        decoder_hidden=(8,),
        head_hidden=(8,),
    )
    arms = {name: GUIDANCE_ARMS[name] for name in ("empirical", "u_real", "graph_real")}

    rows, summaries = run_guidance_sweep(
        realizations=(1,),
        arms=arms,
        cfg=cfg,
        expert_dir=_IHDP_EXPERT_DIR,
        val_fraction=0.3,
    )

    assert set(rows) == set(arms)
    bundle = guidance_sweep._load_bundle(cfg, _IHDP_EXPERT_DIR)
    param_counts = arm_parameter_counts(cfg, bundle, arms, DEFAULT_GAMMA_PRIOR)
    assert len(set(param_counts.values())) == 1

    for name in arms:
        assert len(rows[name]) == 1
        run = rows[name][0]
        assert "out_pehe" in run
        assert "gamma_prior" in run

    text = format_guidance_sweep(
        rows, summaries, DEFAULT_GAMMA_PRIOR, DEFAULT_LAMBDA_PRIOR, param_counts, bundle
    )
    assert text
