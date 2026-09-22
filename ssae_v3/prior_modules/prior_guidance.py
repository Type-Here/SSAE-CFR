"""PriorGuidance: fixed covariate-space projectors that nudge the empirical encoder's
input and pull its first layer toward the prior subspace, without ever masking or
filtering the raw covariates.

    x_guided = x_std + gamma_prior * mean over active s of (x_std @ p_s)
    W_eff    = (I + gamma_prior * mean over active s of p_s) @ first_layer_weight.T
    L_guide  = || (I - p_active) @ W_eff ||_F^2 / (|| W_eff ||_F^2 + eps)

`p_s` and `q_s` are supplied from elsewhere (an embedding-subspace prior and an
expert-graph prior); this module only implements the transform and the loss. Both
projectors and both bases are stored as non-trainable buffers, so `PriorGuidance` never
owns a trainable parameter.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from ..hparams import PRIOR_MODES


def first_linear_weight(module: nn.Module) -> Tensor:
    """Return the weight of the first `nn.Linear` found walking `module.modules()`.

    Raises if a module encountered before that first linear owns parameters of its
    own (e.g. a normalization layer), since that would invalidate the assumption that
    the located linear is the encoder's true first affine map. Raises if no linear is
    found at all.
    """
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            return sub.weight
        if any(True for _ in sub.parameters(recurse=False)):
            raise ValueError(
                f"{type(sub).__name__} owns parameters before any nn.Linear was "
                "found; cannot locate the first linear layer unambiguously"
            )
    raise ValueError("module contains no nn.Linear")


def _check_square(p: Tensor, name: str) -> None:
    if p.dim() != 2 or p.shape[0] != p.shape[1]:
        raise ValueError(f"{name} must be square (m, m); got {tuple(p.shape)}")


def _check_pair(p: Tensor, q: Tensor, name_p: str, name_q: str) -> None:
    if q.dim() != 2 or q.shape[0] != p.shape[0]:
        raise ValueError(
            f"{name_q} must have shape (m, r) matching {name_p}'s m={p.shape[0]}; "
            f"got {tuple(q.shape)}"
        )


def _clone_buffer(t: Optional[Tensor]) -> Optional[Tensor]:
    """Detach, cast to float32 and clone so the buffer never aliases the caller's tensor."""
    if t is None:
        return None
    return t.detach().to(dtype=torch.float32).clone()


def _mean_row_norm(t: Tensor) -> float:
    return float(t.norm(dim=1).mean())


def _union_basis(q_u: Tensor, q_c: Tensor) -> Tensor:
    """An orthonormal basis for the column space of [q_u | q_c], via a float64 SVD.

    Singular vectors are kept while their singular value exceeds
    max(shape) * eps(float64) * s_max, then the result is cast back to float32.
    """
    cat = torch.cat([q_u, q_c], dim=1).to(torch.float64)
    u, s, _ = torch.linalg.svd(cat, full_matrices=False)
    if s.numel() == 0:
        return torch.zeros((cat.shape[0], 0), dtype=torch.float32)
    tol = max(cat.shape) * torch.finfo(torch.float64).eps * s[0]
    rank = int((s > tol).sum())
    return u[:, :rank].to(dtype=torch.float32)


class PriorGuidance(nn.Module):
    """Fixed additive guidance in covariate space, plus a subspace-alignment loss.

    Two independent prior sources may be active: an embedding-subspace prior
    (`p_u`, `q_u`) and an expert-graph prior (`p_c`, `q_c`). `mode` selects which are
    used; a mode never receives a prior it was not built with, and a prior supplied
    but unused by `mode` is rejected rather than silently ignored. All four prior
    tensors, plus the ones this module derives from them (`T`, `q_active`,
    `p_active`), are non-trainable buffers: `PriorGuidance` holds zero trainable
    parameters in every mode.
    """

    def __init__(
        self,
        p_u: Optional[Tensor] = None,
        q_u: Optional[Tensor] = None,
        p_c: Optional[Tensor] = None,
        q_c: Optional[Tensor] = None,
        mode: str = "none",
        gamma_prior: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if mode not in PRIOR_MODES:
            raise ValueError(f"mode must be one of {PRIOR_MODES}; got {mode!r}")

        gamma_prior = float(gamma_prior)
        if gamma_prior < 0:
            raise ValueError(f"gamma_prior must be >= 0; got {gamma_prior}")

        needs_embedding = mode in ("embedding", "both")
        needs_graph = mode in ("graph", "both")

        if needs_embedding and (p_u is None or q_u is None):
            raise ValueError(f"mode={mode!r} requires both p_u and q_u")
        if not needs_embedding and (p_u is not None or q_u is not None):
            raise ValueError(
                f"p_u/q_u were supplied but mode={mode!r} does not use the embedding prior"
            )
        if needs_graph and (p_c is None or q_c is None):
            raise ValueError(f"mode={mode!r} requires both p_c and q_c")
        if not needs_graph and (p_c is not None or q_c is not None):
            raise ValueError(
                f"p_c/q_c were supplied but mode={mode!r} does not use the graph prior"
            )

        if p_u is not None:
            _check_square(p_u, "p_u")
            _check_pair(p_u, q_u, "p_u", "q_u")
        if p_c is not None:
            _check_square(p_c, "p_c")
            _check_pair(p_c, q_c, "p_c", "q_c")
        if p_u is not None and p_c is not None and p_u.shape[0] != p_c.shape[0]:
            raise ValueError(
                f"p_u and p_c must share the same m; got {p_u.shape[0]} and {p_c.shape[0]}"
            )

        self.register_buffer("p_u", _clone_buffer(p_u))
        self.register_buffer("q_u", _clone_buffer(q_u))
        self.register_buffer("p_c", _clone_buffer(p_c))
        self.register_buffer("q_c", _clone_buffer(q_c))

        self._mode = mode
        self._gamma_prior = gamma_prior
        self._eps = float(eps)

        self._embedding_rank = self.q_u.shape[1] if self.q_u is not None else 0
        self._graph_rank = self.q_c.shape[1] if self.q_c is not None else 0

        if mode == "none":
            self._n_active = 0
            self._active_rank = 0
            return

        active_p = [p for p in (self.p_u, self.p_c) if p is not None]
        self._n_active = len(active_p)
        m = active_p[0].shape[0]
        p_mean = torch.stack(active_p, dim=0).mean(dim=0)
        T = torch.eye(m, dtype=torch.float32) + gamma_prior * p_mean
        self.register_buffer("T", T)

        if mode == "embedding":
            q_active = self.q_u.clone()
        elif mode == "graph":
            q_active = self.q_c.clone()
        else:
            q_active = _union_basis(self.q_u, self.q_c)

        self.register_buffer("q_active", q_active)
        self.register_buffer("p_active", q_active @ q_active.t())
        self._active_rank = q_active.shape[1]

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def gamma_prior(self) -> float:
        return self._gamma_prior

    @property
    def n_active(self) -> int:
        return self._n_active

    @property
    def embedding_rank(self) -> int:
        return self._embedding_rank

    @property
    def graph_rank(self) -> int:
        return self._graph_rank

    @property
    def active_rank(self) -> int:
        return self._active_rank

    def transform(self, x_std: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Additively nudge x_std toward the active prior subspace(s).

        Returns x_std itself (same object, unmodified) when mode is "none". The raw
        input always keeps coefficient 1; the active views are averaged before
        gamma_prior is applied, so a two-prior configuration is not perturbed twice
        as hard as a one-prior one.
        """
        if self._mode == "none":
            with torch.no_grad():
                x_std_norm = _mean_row_norm(x_std)
            diagnostics = {
                "x_std_norm": x_std_norm,
                "x_prior_norm": 0.0,
                "x_guided_delta_ratio": 0.0,
            }
            return x_std, diagnostics

        views = []
        if self.p_u is not None:
            p_u = self.p_u.to(device=x_std.device, dtype=x_std.dtype)
            views.append(x_std @ p_u)
        if self.p_c is not None:
            p_c = self.p_c.to(device=x_std.device, dtype=x_std.dtype)
            views.append(x_std @ p_c)
        x_prior = torch.stack(views, dim=0).mean(dim=0)
        x_guided = x_std + self._gamma_prior * x_prior

        with torch.no_grad():
            x_std_norm = _mean_row_norm(x_std)
            x_prior_norm = _mean_row_norm(x_prior)
            delta_norm = _mean_row_norm(x_guided - x_std)
            x_guided_delta_ratio = delta_norm / (x_std_norm + self._eps)

        diagnostics = {
            "x_std_norm": x_std_norm,
            "x_prior_norm": x_prior_norm,
            "x_guided_delta_ratio": x_guided_delta_ratio,
        }
        return x_guided, diagnostics

    def _alignment_against(self, p: Optional[Tensor], W_eff: Tensor, eye_m: Tensor) -> float:
        """1 - normalized off-subspace energy of W_eff against a single projector p."""
        if p is None:
            return float("nan")
        p = p.to(device=W_eff.device, dtype=W_eff.dtype)
        w_out = (eye_m - p) @ W_eff
        denom = (W_eff**2).sum() + self._eps
        return float(1.0 - (w_out**2).sum() / denom)

    def guidance_loss(self, first_layer_weight: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Scale-normalized off-active-subspace energy of the effective first layer.

        Uses W_eff = T @ first_layer_weight.T, the mapping that actually acts on the
        original covariates once the input transform is folded in, never the raw
        weight. Zero and gradient-free when mode is "none".
        """
        if self._mode == "none":
            loss = torch.zeros(
                (), device=first_layer_weight.device, dtype=first_layer_weight.dtype
            )
            with torch.no_grad():
                W1 = first_layer_weight.t()
                first_layer_effective_norm = float(torch.linalg.norm(W1))
            diagnostics = {
                "guidance_loss": 0.0,
                "alignment_active": float("nan"),
                "alignment_U": float("nan"),
                "alignment_C": float("nan"),
                "first_layer_effective_norm": first_layer_effective_norm,
            }
            return loss, diagnostics

        device, dtype = first_layer_weight.device, first_layer_weight.dtype
        T = self.T.to(device=device, dtype=dtype)
        p_active = self.p_active.to(device=device, dtype=dtype)
        eye_m = torch.eye(T.shape[0], device=device, dtype=dtype)

        W1 = first_layer_weight.t()
        W_eff = T @ W1
        W_out = (eye_m - p_active) @ W_eff
        denom = (W_eff**2).sum() + self._eps
        loss = (W_out**2).sum() / denom

        with torch.no_grad():
            guidance_loss_val = float(loss)
            alignment_active = 1.0 - guidance_loss_val
            first_layer_effective_norm = float(torch.linalg.norm(W_eff))
            alignment_U = self._alignment_against(self.p_u, W_eff, eye_m)
            alignment_C = self._alignment_against(self.p_c, W_eff, eye_m)

        diagnostics = {
            "guidance_loss": guidance_loss_val,
            "alignment_active": alignment_active,
            "alignment_U": alignment_U,
            "alignment_C": alignment_C,
            "first_layer_effective_norm": first_layer_effective_norm,
        }
        return loss, diagnostics
