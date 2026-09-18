"""Hard invariants of the SSAE-CFR v3 architecture.

The empirical host always sees the full standardized input; the U (structural) and
W (semantic) corrections are strictly additive, exactly zero at init, removable by
setting their reliability to zero, and never touched by the reconstruction loss.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from ssae_v3.hparams import MODEL_VARIANTS, load_config
from ssae_v3.model_v3 import SSAECFRv3
from ssae_v3.prior_build.projector import build_prior_bundle
from ssae_v3.prior_modules.w_adapter import ValueSemanticTokenizer, WSemanticAdapter
from ssae_v3.training.experiments import _arm_config

M = 6
R_W = 3


def _cfg(**overrides):
    base = dict(
        in_channels=M,
        d_u=8,
        encoder_hidden=(8,),
        decoder_hidden=(8,),
        head_hidden=(8,),
        u_adapter_hidden=(8,),
        d_token=8,
        phi_hidden=(8,),
        d_s=8,
        rho_hidden=(8,),
        r_W_rank=R_W,
    )
    base.update(overrides)
    return load_config(None, **base)


def _dummy_prior(m: int = M, r_w: int = R_W, seed: int = 0):
    """A structurally-valid but semantically-meaningless (U_k, q_tilde) pair."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(m, r_w, generator=gen)
    q, _ = torch.linalg.qr(a)
    u_k = q
    q_tilde = torch.randn(m, r_w, generator=gen)
    return u_k, q_tilde


def _paired_models():
    """(empirical, u_w_adapter-with-r=0) models sharing every weight the empirical path has."""
    cfg_emp = _cfg(model_variant="empirical")
    cfg_uw = _cfg(model_variant="u_w_adapter", r_U=0.0, r_W=0.0)
    u_k, q_tilde = _dummy_prior()

    torch.manual_seed(0)
    model_emp = SSAECFRv3(cfg_emp)
    torch.manual_seed(1)
    model_uw = SSAECFRv3(cfg_uw, U_k=u_k, q_tilde=q_tilde)

    model_uw.load_state_dict(model_emp.state_dict(), strict=False)
    # FixedReliability's constants are buffers and travel with load_state_dict; the
    # copy above may already have zeroed them, but the intent (r_U = r_W = 0) is set
    # explicitly so the test does not depend on that coincidence.
    model_uw.reliability.r_U_const.fill_(0.0)
    model_uw.reliability.r_W_const.fill_(0.0)

    model_emp.eval()
    model_uw.eval()
    return model_emp, model_uw


def test_prior_off_reproduces_empirical_predictions():
    """model_variant='u_w_adapter' with r_U=r_W=0 predicts bit-identically to 'empirical'."""
    model_emp, model_uw = _paired_models()
    x = torch.randn(10, M)
    t = (torch.rand(10) > 0.5).float()

    out_emp = model_emp(x, t, omega=0.0)
    out_uw = model_uw(x, t, omega=0.0)

    for key in ("y0_hat", "y1_hat", "x_hat", "u"):
        assert torch.equal(out_emp[key], out_uw[key])


def test_zero_reliability_blocks_all_adapter_gradient():
    """With r_U=r_W=0, backprop of the total loss leaves every adapter grad None or zero."""
    cfg = _cfg(model_variant="u_w_adapter", r_U=0.0, r_W=0.0)
    u_k, q_tilde = _dummy_prior()
    torch.manual_seed(3)
    model = SSAECFRv3(cfg, U_k=u_k, q_tilde=q_tilde)

    x = torch.randn(12, M)
    t = (torch.rand(12) > 0.5).float()
    yf = torch.randn(12)

    out = model(x, t, omega=0.0)
    terms = model.loss_terms(out, x, t, yf)
    loss, _ = model.total_loss(terms, cfg)
    loss.backward()

    adapter_params = list(model.u_adapter.parameters()) + list(model.w_adapter.parameters())
    assert adapter_params
    for p in adapter_params:
        assert p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad))


@pytest.mark.parametrize("variant", MODEL_VARIANTS)
def test_full_x_reaches_encoder_unfiltered(variant):
    """No masking or filtering happens before the empirical encoder, for every variant."""
    cfg = _cfg(model_variant=variant)
    u_k, q_tilde = _dummy_prior()
    kwargs = {}
    if cfg.use_u_adapter:
        kwargs["U_k"] = u_k
    if cfg.use_w_adapter:
        kwargs["q_tilde"] = q_tilde
    model = SSAECFRv3(cfg, **kwargs)

    seen = {}

    def hook(module, args, output):
        seen["x"] = args[0]

    model.empirical.encoder.register_forward_hook(hook)

    x = torch.randn(9, M)
    t = (torch.rand(9) > 0.5).float()
    model(x, t, omega=0.0)

    assert torch.equal(seen["x"], x)


def test_decoder_never_sees_a_fused_code():
    """x_hat is bit-identical whether r_U/r_W are 0 or 1; only u_out changes."""
    cfg = _cfg(model_variant="u_w_adapter", r_U=0.0, r_W=0.0)
    u_k, q_tilde = _dummy_prior()
    torch.manual_seed(5)
    model = SSAECFRv3(cfg, U_k=u_k, q_tilde=q_tilde)

    # make the corrections genuinely nonzero once reliability is switched on
    nn.init.normal_(model.u_adapter.net[-1].weight, std=1.0)
    nn.init.normal_(model.u_adapter.net[-1].bias, std=1.0)
    nn.init.normal_(model.w_adapter.a_w_head[-1].weight, std=1.0)
    nn.init.normal_(model.w_adapter.a_w_head[-1].bias, std=1.0)
    model.eval()

    x = torch.randn(8, M)
    t = (torch.rand(8) > 0.5).float()

    out0 = model(x, t, omega=0.0)
    model.reliability.r_U_const.fill_(1.0)
    model.reliability.r_W_const.fill_(1.0)
    out1 = model(x, t, omega=0.0)

    assert torch.equal(out0["x_hat"], out1["x_hat"])
    assert torch.equal(out0["u"], out1["u"])
    assert not torch.equal(out0["u_out"], out1["u_out"])


def test_reconstruction_gradient_never_reaches_adapters():
    """Backprop of L_rec alone leaves every adapter parameter's grad None or zero."""
    cfg = _cfg(model_variant="u_w_adapter", r_U=1.0, r_W=1.0)
    u_k, q_tilde = _dummy_prior()
    torch.manual_seed(6)
    model = SSAECFRv3(cfg, U_k=u_k, q_tilde=q_tilde)
    nn.init.normal_(model.u_adapter.net[-1].weight, std=1.0)
    nn.init.normal_(model.w_adapter.a_w_head[-1].weight, std=1.0)

    x = torch.randn(8, M)
    t = (torch.rand(8) > 0.5).float()
    out = model(x, t, omega=0.0)
    rec_loss = torch.nn.functional.mse_loss(out["x_hat"], x)
    rec_loss.backward()

    for p in list(model.u_adapter.parameters()) + list(model.w_adapter.parameters()):
        assert p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad))


def test_tokenizer_uses_value_times_semantics_interaction():
    """The tokenizer's phi input contains x_ij * q~_j exactly, not a linear rewrite."""
    m, r_w = 5, 3
    q_tilde = torch.randn(m, r_w)
    tok = ValueSemanticTokenizer(q_tilde, d_token=6, hidden=(6,))

    captured = {}

    def hook(module, args, output):
        captured["token_in"] = args[0]

    tok.phi.register_forward_hook(hook)

    n = 4
    x = torch.randn(n, m)
    tok(x)

    expected_interaction = (x.unsqueeze(-1) * q_tilde.unsqueeze(0)).reshape(n * m, r_w)
    got = captured["token_in"]
    assert got.shape == (n * m, 1 + 2 * r_w)
    assert torch.equal(got[:, 1 + r_w:], expected_interaction)


def test_w_branch_output_is_not_affine_in_x():
    """A linear rewrite of the W branch is rejected: a_W is not affine in x."""
    m, r_w = 5, 3
    q_tilde = torch.randn(m, r_w)
    torch.manual_seed(11)
    adapter = WSemanticAdapter(q_tilde, d_u=6, d_token=6, phi_hidden=(6,), d_s=6, rho_hidden=(6,))
    nn.init.normal_(adapter.a_w_head[-1].weight, std=1.0)
    nn.init.normal_(adapter.a_w_head[-1].bias, std=1.0)

    x1 = torch.randn(3, m) * 3.0
    x2 = torch.randn(3, m) * 3.0

    a1 = adapter(x1)
    a2 = adapter(x2)
    a_sum = adapter(x1 + x2)
    a_double = adapter(2.0 * x1)

    assert not torch.allclose(a_sum, a1 + a2, atol=1e-5)
    assert not torch.allclose(a_double, 2.0 * a1, atol=1e-5)


def test_zero_reliability_removes_augmentation_even_with_live_adapters():
    """r_U=r_W=0 reproduces the empirical predictions even when the adapters are non-trivial."""
    model_emp, model_uw = _paired_models()
    nn.init.normal_(model_uw.u_adapter.net[-1].weight, std=1.0)
    nn.init.normal_(model_uw.u_adapter.net[-1].bias, std=1.0)
    nn.init.normal_(model_uw.w_adapter.a_w_head[-1].weight, std=1.0)
    nn.init.normal_(model_uw.w_adapter.a_w_head[-1].bias, std=1.0)
    model_uw.eval()

    x = torch.randn(11, M)
    t = (torch.rand(11) > 0.5).float()
    out_emp = model_emp(x, t, omega=0.0)
    out_uw = model_uw(x, t, omega=0.0)

    assert not torch.equal(out_uw["c"], torch.zeros_like(out_uw["c"]))
    assert torch.equal(out_emp["u"], out_uw["u"])
    assert torch.equal(out_emp["x_hat"], out_uw["x_hat"])
    assert torch.equal(out_emp["y0_hat"], out_uw["y0_hat"])
    assert torch.equal(out_emp["y1_hat"], out_uw["y1_hat"])


def test_predictions_derive_only_from_u_out():
    """y0_hat/y1_hat equal heads(u_out) exactly; there is no alternate causal path."""
    cfg = _cfg(model_variant="u_w_adapter", r_U=1.0, r_W=1.0)
    u_k, q_tilde = _dummy_prior()
    torch.manual_seed(9)
    model = SSAECFRv3(cfg, U_k=u_k, q_tilde=q_tilde)
    model.eval()

    x = torch.randn(7, M)
    t = (torch.rand(7) > 0.5).float()
    out = model(x, t, omega=0.0)
    y0_direct, y1_direct = model.heads(out["u_out"])

    assert torch.equal(out["y0_hat"], y0_direct)
    assert torch.equal(out["y1_hat"], y1_direct)


@pytest.mark.parametrize("batchnorm", [False, True])
@pytest.mark.parametrize("training_mode", [True, False])
def test_corrections_exactly_zero_at_init(training_mode, batchnorm):
    """c and a_W are exactly 0.0 at init, in every train/eval x batchnorm combination."""
    cfg = _cfg(model_variant="u_w_adapter", r_U=1.0, r_W=1.0, batchnorm=batchnorm)
    u_k, q_tilde = _dummy_prior()
    model = SSAECFRv3(cfg, U_k=u_k, q_tilde=q_tilde)
    model.train(training_mode)

    x = torch.randn(6, M)  # n > 1 so BatchNorm1d in train mode does not error
    t = (torch.rand(6) > 0.5).float()
    out = model(x, t, omega=0.0)

    assert out["c"].abs().max().item() == 0.0
    assert out["a_W"].abs().max().item() == 0.0


def test_prior_math_modules_do_not_import_dataset_loaders():
    """projector.py and descriptions.py never import a dataset loader carrying outcomes."""
    import ssae_v3.prior_build.descriptions as descriptions_mod
    import ssae_v3.prior_build.projector as projector_mod

    for mod in (projector_mod, descriptions_mod):
        source = inspect.getsource(mod)
        assert "ssae_v3.data" not in source
        assert "..data" not in source


def test_build_prior_bundle_needs_only_v_and_feature_names():
    """build_prior_bundle takes V and feature names; no dataset/treatment/outcome object."""
    sig = inspect.signature(build_prior_bundle)
    for forbidden in ("dataset", "t", "yf", "mu0", "mu1"):
        assert forbidden not in sig.parameters

    rng = np.random.default_rng(0)
    v = rng.standard_normal((M, 10))
    bundle = build_prior_bundle(v, feature_names=[f"x{i}" for i in range(M)])
    assert bundle.P_U.shape == (M, M)


def test_split_and_standardize_feeds_identical_x_to_every_variant():
    """The same split/standardize call feeds bit-identical, unfiltered x to every variant."""
    from ssae_v3.data.base import Dataset
    from ssae_v3.utils.split import split_and_standardize

    rng = np.random.default_rng(0)
    n, m = 60, M
    x = rng.standard_normal((n, m))
    t = (rng.random(n) < 0.5).astype(np.int64)
    yf = rng.standard_normal(n)
    ds = Dataset(name="synthetic", x=x, t=t, yf=yf, feature_names=[f"x{i}" for i in range(m)])

    splits_a = split_and_standardize(ds, test_size=0.3, seed=0)
    splits_b = split_and_standardize(ds, test_size=0.3, seed=0)

    assert np.array_equal(splits_a.train.x, splits_b.train.x)
    assert np.array_equal(splits_a.test.x, splits_b.test.x)
    assert np.array_equal(splits_a.standardizer.mean_, splits_b.standardizer.mean_)
    assert np.array_equal(splits_a.standardizer.scale_, splits_b.standardizer.scale_)

    u_k, q_tilde = _dummy_prior(m=m)
    x_tensor = torch.as_tensor(splits_a.train.x, dtype=torch.float32)
    t_tensor = torch.as_tensor(splits_a.train.t, dtype=torch.float32)
    for variant in MODEL_VARIANTS:
        cfg = _cfg(model_variant=variant, in_channels=m)
        kwargs = {}
        if cfg.use_u_adapter:
            kwargs["U_k"] = u_k
        if cfg.use_w_adapter:
            kwargs["q_tilde"] = q_tilde
        model = SSAECFRv3(cfg, **kwargs)
        model.eval()

        seen = {}

        def _capture(mod, args, out, seen=seen):
            # a plain assignment (not an expression): returning a non-None value
            # from a forward hook replaces the module's actual output, which would
            # corrupt the very thing this test is trying to observe.
            seen["x"] = args[0]

        model.empirical.encoder.register_forward_hook(_capture)
        model(x_tensor, t_tensor, omega=0.0)
        assert torch.equal(seen["x"], x_tensor)


def test_arm_config_restores_reliability_from_an_empirical_base():
    """Deriving any ladder arm from an empirical-variant base gives nonzero reliability."""
    base = load_config(None, in_channels=M, model_variant="empirical")
    assert base.r_U == 0.0 and base.r_W == 0.0

    for variant in ("u_adapter", "w_adapter", "u_w_adapter"):
        arm_cfg = _arm_config(base, variant)
        if arm_cfg.use_u_adapter:
            assert arm_cfg.r_U > 0.0
        if arm_cfg.use_w_adapter:
            assert arm_cfg.r_W > 0.0

