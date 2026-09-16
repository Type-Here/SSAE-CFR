"""Tests for the evaluation harness.

Training a real model is covered elsewhere, so these focus on what the harness itself
decides: that a binary outcome's effect is taken on the probability scale rather than
from the raw head logits, that the metric set adapts to what the dataset can support,
and that aggregation across runs handles the nan a non-identified metric returns.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ssae_cfr.config import load_config
from ssae_cfr.data.base import Dataset
from ssae_cfr.evaluate import (
    LOADERS,
    OUTCOME_IS_BENEFIT,
    _forward_eval,
    _potential_outcomes,
    _sensitivity,
    aggregate,
    factual_objective,
    final_training_diagnostics,
    format_summary,
    score_split,
)
from ssae_cfr.models import SSAECFR
from ssae_cfr.train import admission_diagnostics


def _dataset(outcome_type: str = "continuous", n: int = 120, m: int = 6, oracle: bool = True) -> Dataset:
    rng = np.random.default_rng(0)
    t = (rng.random(n) < 0.4).astype(np.int64)
    x = rng.standard_normal((n, m))
    if outcome_type == "binary":
        mu0 = np.full(n, 0.3)
        mu1 = np.full(n, 0.5)
        yf = (rng.random(n) < np.where(t == 1, mu1, mu0)).astype(np.float64)
    else:
        mu0 = x[:, 0]
        mu1 = mu0 + 1.0
        yf = np.where(t == 1, mu1, mu0)
    return Dataset(
        name="synthetic",
        x=x,
        t=t,
        yf=yf,
        feature_names=[f"x{j}" for j in range(m)],
        mu0=mu0 if oracle else None,
        mu1=mu1 if oracle else None,
        outcome_type=outcome_type,
        standardized=True,
    )


def _model(m: int, outcome_type: str = "continuous") -> SSAECFR:
    cfg = load_config(None, k_latent=4, encoder_hidden=(8,), decoder_hidden=(8,),
                      head_hidden=(4,), gating_hidden=(4,))
    P_U = torch.eye(m, dtype=torch.float32)
    return SSAECFR(m=m, P_U=P_U, cfg=cfg, outcome_type=outcome_type)


# -- scale handling --------------------------------------------------------


def test_binary_outcomes_are_squashed_to_probabilities():
    """The heads emit logits; a risk difference has to be taken after the sigmoid."""
    out = {"y0_hat": np.array([0.0, 2.0]), "y1_hat": np.array([0.0, -2.0])}
    y0, y1 = _potential_outcomes(out, "binary")
    assert y0 == pytest.approx([0.5, 1.0 / (1.0 + np.exp(-2.0))])
    assert np.all((y0 >= 0.0) & (y0 <= 1.0)) and np.all((y1 >= 0.0) & (y1 <= 1.0))


def test_continuous_outcomes_pass_through_untouched():
    out = {"y0_hat": np.array([1.5, -3.0]), "y1_hat": np.array([2.5, -1.0])}
    y0, y1 = _potential_outcomes(out, "continuous")
    assert y0 == pytest.approx([1.5, -3.0])
    assert y1 == pytest.approx([2.5, -1.0])


def test_continuous_outcomes_are_put_back_on_the_callers_scale():
    """A head trained against a standardized outcome emits nobody's units until undone."""
    out = {"y0_hat": np.array([1.5, -3.0]), "y1_hat": np.array([2.5, -1.0])}
    y0, y1 = _potential_outcomes(out, "continuous", 3.5, 2.0)
    assert y0 == pytest.approx([6.5, -2.5])
    assert y1 == pytest.approx([8.5, 1.5])


def test_normalized_factual_objective_divides_by_the_outcome_variance():
    """The selection criterion has to be comparable across differing outcome scales."""
    ds = _dataset()
    model = _model(ds.x.shape[1])
    raw = factual_objective(model, ds)
    assert factual_objective(model, ds, normalized=True) == pytest.approx(raw)

    scaled = SSAECFR(m=ds.x.shape[1], P_U=torch.eye(ds.x.shape[1], dtype=torch.float32),
                     cfg=load_config(None, k_latent=4, encoder_hidden=(8,), decoder_hidden=(8,),
                                     head_hidden=(4,), gating_hidden=(4,)),
                     outcome_type="continuous", y_loc=0.0, y_scale=4.0)
    scaled.load_state_dict(model.state_dict() | {
        "y_loc": torch.tensor(0.0), "y_scale": torch.tensor(4.0)
    })
    assert factual_objective(scaled, ds, normalized=True) == pytest.approx(
        factual_objective(scaled, ds) / 16.0
    )


def test_normalization_is_a_no_op_for_a_binary_outcome():
    """A cross-entropy is already scale-free; there is nothing to divide by."""
    ds = _dataset(outcome_type="binary")
    model = _model(ds.x.shape[1], outcome_type="binary")
    assert factual_objective(model, ds, normalized=True) == pytest.approx(
        factual_objective(model, ds)
    )


def test_the_outcome_scale_is_fit_on_the_training_split_only():
    """Like the covariate standardizer: a property of the data the model was shown.

    Reading it off the test split would leak the held-out outcome's location and spread
    into the predictions scored against it.
    """
    from ssae_cfr.evaluate import fit_and_score

    train, test = _dataset(n=100), _dataset(n=40)
    # move the test outcome far away; the fitted scale must not follow it
    test = Dataset(
        name=test.name, x=test.x, t=test.t, yf=test.yf * 10.0 + 50.0,
        feature_names=test.feature_names, mu0=test.mu0, mu1=test.mu1,
        outcome_type=test.outcome_type, standardized=True,
    )
    cfg = load_config(None, k_latent=4, encoder_hidden=(8,), decoder_hidden=(8,),
                      head_hidden=(4,), gating_hidden=(4,), epochs=2)
    model, _ = fit_and_score(train, test, cfg)
    loc, scale = model.outcome_affine
    assert loc == pytest.approx(float(np.mean(train.yf)))
    assert scale == pytest.approx(float(np.std(train.yf)))


# -- sensitivity -----------------------------------------------------------


def test_binary_sensitivity_uses_the_ratio_of_predicted_risks():
    n = 300
    y0 = np.full(n, 0.2)
    y1 = np.full(n, 0.4)
    result = _sensitivity(y0, y1, np.zeros(n), "binary", seed=0)
    assert result["risk_ratio"] == pytest.approx(2.0)
    assert result["e_value"] == pytest.approx(2.0 + np.sqrt(2.0))


def test_a_null_effect_gives_the_smallest_possible_e_value():
    n = 200
    y = np.full(n, 0.3)
    result = _sensitivity(y, y.copy(), np.zeros(n), "binary", seed=0)
    assert result["risk_ratio"] == pytest.approx(1.0)
    assert result["e_value"] == pytest.approx(1.0)
    assert result["e_value_ci"] == pytest.approx(1.0)


def test_continuous_sensitivity_goes_through_the_approximation():
    rng = np.random.default_rng(0)
    n = 400
    yf = rng.standard_normal(n)  # sd about 1, so d is about the raw effect
    y0 = np.zeros(n)
    y1 = np.full(n, 0.5)
    result = _sensitivity(y0, y1, yf, "continuous", seed=0)
    assert result["risk_ratio"] == pytest.approx(np.exp(0.91 * 0.5 / yf.std()), rel=1e-6)
    assert result["e_value"] > 1.0


def test_sensitivity_is_nan_for_a_constant_continuous_outcome():
    n = 20
    result = _sensitivity(np.zeros(n), np.ones(n), np.full(n, 3.0), "continuous", seed=0)
    assert np.isnan(result["e_value"])


# -- the metric set adapts to the dataset ----------------------------------


def test_oracle_metrics_appear_only_with_an_oracle():
    with_oracle = score_split(_model(6), _dataset(oracle=True), benefit=True)
    without = score_split(_model(6), _dataset(oracle=False), benefit=True)
    assert "pehe" in with_oracle and "eps_ate" in with_oracle
    assert "pehe" not in without and "eps_ate" not in without
    assert "smd_reduction" in without  # balance never needs an oracle


def test_continuous_outcomes_report_policy_value_not_risk():
    """1 - value is only meaningful for an outcome bounded in [0, 1]."""
    scores = score_split(_model(6), _dataset("continuous"), benefit=True)
    assert "policy_value" in scores
    assert "policy_value_treat_all" in scores and "policy_value_treat_none" in scores
    assert not any(key.startswith("policy_risk") for key in scores)


def test_binary_outcomes_report_the_policy_risk_table():
    scores = score_split(_model(6), _dataset("binary"), benefit=False)
    assert {"policy_risk_policy", "policy_risk_treat_all", "policy_risk_treat_none"} <= set(scores)
    assert "policy_value" not in scores


def test_scoring_an_empty_split_returns_nothing():
    ds = _dataset().subset(np.array([], dtype=int))
    assert score_split(_model(6), ds, benefit=True) == {}


def test_scoring_leaves_the_model_in_its_original_mode():
    model = _model(6)
    model.train()
    score_split(model, _dataset(), benefit=True)
    assert model.training


# -- aggregation and reporting ---------------------------------------------


def test_aggregate_reports_mean_and_std_per_metric():
    runs = [{"in_pehe": 1.0}, {"in_pehe": 3.0}]
    summary = aggregate(runs)
    assert summary["in_pehe"]["mean"] == pytest.approx(2.0)
    assert summary["in_pehe"]["std"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
    assert summary["in_pehe"]["n_runs"] == 2


def test_aggregate_drops_nan_and_records_how_many_runs_counted():
    runs = [{"a": 1.0}, {"a": float("nan")}, {"a": 3.0}]
    summary = aggregate(runs)
    assert summary["a"]["mean"] == pytest.approx(2.0)
    assert summary["a"]["n_runs"] == 2


def test_aggregate_survives_an_all_nan_metric():
    summary = aggregate([{"a": float("nan")}, {"a": float("nan")}])
    assert np.isnan(summary["a"]["mean"])
    assert summary["a"]["n_runs"] == 0


def test_single_run_std_is_zero_not_nan():
    summary = aggregate([{"a": 5.0}])
    assert summary["a"]["std"] == 0.0


def test_format_summary_pairs_the_two_splits():
    summary = aggregate([{"in_pehe": 1.0, "out_pehe": 2.0, "treated_fraction_train": 0.2}])
    text = format_summary("ihdp", summary, [0])
    assert "in-sample" in text and "out-of-sample" in text
    assert "pehe" in text and "treated_fraction_train" in text


def test_every_loader_declares_an_outcome_direction():
    """A missing entry would silently default to 'harm' and invert the policy ranking."""
    assert set(OUTCOME_IS_BENEFIT) == set(LOADERS)


# -- the selection criterion -----------------------------------------------


def test_factual_objective_is_the_mse_on_the_observed_outcome():
    ds = _dataset("continuous")
    model = _model(6)
    out = _forward_eval(model, ds)
    y0, y1 = _potential_outcomes(out, "continuous")
    t = ds.t.astype(np.float64)
    expected = np.mean((t * y1 + (1.0 - t) * y0 - ds.yf) ** 2)
    assert factual_objective(model, ds) == pytest.approx(expected)


def test_factual_objective_is_cross_entropy_for_a_binary_outcome():
    """Not MSE on logits: the criterion has to match the loss the model was fit with."""
    ds = _dataset("binary")
    value = factual_objective(_model(6, "binary"), ds)
    assert value > 0.0 and np.isfinite(value)


def test_factual_objective_of_an_empty_split_is_nan():
    ds = _dataset().subset(np.array([], dtype=int))
    assert np.isnan(factual_objective(_model(6), ds))


def test_factual_objective_needs_no_oracle():
    """The point of this criterion: it is computable where PEHE is not."""
    ds = _dataset(oracle=False)
    assert np.isfinite(factual_objective(_model(6), ds))


# -- training diagnostics carried into the scores ----------------------------

def test_final_training_diagnostics_reads_the_last_scored_epoch():
    history = [
        {"share_L_fact": 0.9, "share_L_rec": 0.1, "b_mean": 0.5, "epoch": 0},
        {"share_L_fact": 0.4, "share_L_rec": 0.6, "b_mean": 0.2, "epoch": 1},
    ]
    out = final_training_diagnostics(history)
    assert out["train_share_L_rec"] == 0.6
    assert out["train_b_mean"] == 0.2


def test_final_training_diagnostics_skips_the_early_stopping_record():
    """`fit` appends a bookkeeping entry after the loop; it carries no diagnostics."""
    history = [
        {"share_L_fact": 0.4, "b_mean": 0.2, "epoch": 1},
        {"early_stopped_to_epoch": 1, "best_val_objective": 0.3},
    ]
    assert final_training_diagnostics(history)["train_share_L_fact"] == 0.4


def test_final_training_diagnostics_of_an_empty_history_is_empty():
    assert final_training_diagnostics([]) == {}


# -- admission-gate diagnostics ----------------------------------------------

def _rank_reduced_model(m: int, k: int) -> SSAECFR:
    """A model whose P_U keeps only k directions, so the residual is non-trivial."""
    cfg = load_config(None, k_latent=4, encoder_hidden=(8,), decoder_hidden=(8,),
                      head_hidden=(4,), gating_hidden=(4,))
    torch.manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(m, k))
    return SSAECFR(m=m, P_U=(U @ U.T).float(), cfg=cfg)


def test_score_split_reports_the_admission_gate():
    ds = _dataset(m=6)
    scores = score_split(_rank_reduced_model(6, 3), ds, benefit=True)
    for key in ("b_mean", "b_std", "b_patient_std", "b_within_cov_share",
                "residual_admitted"):
        assert key in scores and np.isfinite(scores[key])
    assert 0.0 <= scores["b_mean"] <= 1.0
    assert 0.0 <= scores["residual_admitted"] <= 1.0


def test_admission_is_reported_per_covariate_as_flat_keys():
    """Flat scalars, so `aggregate` averages them across runs with no special case."""
    ds = _dataset(m=6)
    scores = score_split(_rank_reduced_model(6, 3), ds, benefit=True)
    per_cov = [scores[f"b_cov_{j:02d}"] for j in range(6)]
    assert len(per_cov) == 6 and all(0.0 <= v <= 1.0 for v in per_cov)
    assert f"b_cov_{6:02d}" not in scores, "one key per covariate, no more"
    assert np.mean(per_cov) == pytest.approx(scores["b_mean"], abs=1e-6)


def test_residual_admitted_is_zero_when_there_is_no_residual():
    """P_U = I leaves nothing to admit; the ratio must not divide by zero."""
    ds = _dataset(m=6)
    scores = score_split(_model(6), ds, benefit=True)  # _model uses P_U = I
    assert scores["residual_admitted"] == 0.0


def test_fixed_gate_modes_show_up_in_the_diagnostics():
    """b_mode is the nested ablation, so the diagnostic must read it back exactly."""
    ds = _dataset(m=6)
    for mode, expected in (("one", 1.0), ("zero", 0.0)):
        cfg = load_config(None, k_latent=4, encoder_hidden=(8,), decoder_hidden=(8,),
                          head_hidden=(4,), gating_hidden=(4,), b_mode=mode)
        torch.manual_seed(0)
        U, _ = torch.linalg.qr(torch.randn(6, 3))
        model = SSAECFR(m=6, P_U=(U @ U.T).float(), cfg=cfg)
        scores = score_split(model, ds, benefit=True)
        assert scores["b_mean"] == pytest.approx(expected)
        assert scores["b_patient_std"] == pytest.approx(0.0)
        assert scores["b_within_cov_share"] == pytest.approx(0.0)
        assert scores["residual_admitted"] == pytest.approx(expected)


def test_within_cov_share_separates_a_covariate_mask_from_a_per_patient_gate():
    """The collapse tell. A gate that is one fixed mask for everyone must read ~0.

    This is the diagnostic `b_patient_std` cannot provide: averaging b over the m
    covariates before taking the spread cancels independent per-patient deviations, so a
    genuinely per-patient gate still reads a near-zero per-patient sd.
    """
    x_res = torch.randn(200, 25)

    mask = torch.rand(1, 25).expand(200, 25)  # same decision for every patient
    assert admission_diagnostics(mask, x_res)["b_within_cov_share"] == pytest.approx(0.0)

    per_patient = torch.rand(200, 25)  # no per-covariate structure at all
    assert admission_diagnostics(per_patient, x_res)["b_within_cov_share"] > 0.95


def test_patient_std_understates_a_per_patient_gate_by_root_m():
    """Why the tell had to change: the two diagnostics disagree by construction."""
    torch.manual_seed(0)
    b = torch.rand(500, 25)
    diag = admission_diagnostics(b, torch.randn(500, 25))
    # Independent across covariates, so the per-patient mean concentrates as 1/sqrt(m).
    assert diag["b_patient_std"] == pytest.approx(float(b.std()) / 5.0, rel=0.2)
    assert diag["b_within_cov_share"] > 0.95
