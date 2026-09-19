"""Negative-control transformations of the prior.

A capacity-matched control answer whether a measured gain comes
from the LLM's semantic knowledge, or merely from the extra parameters/degrees
of freedom the U/W branches add. Each function below returns
a same-shape, same-capacity replacement for `U_k` or `q_tilde` with the semantic
content destroyed in a specific, named way. All are deterministic given a seed and
use a private `torch.Generator`, so they never disturb the caller's global RNG state.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

CONTROLS = (
    "none",
    "random_basis",
    "shuffled_semantics",
    "random_semantics",
    "onehot_semantics",
    "nonsemantic",
)


def random_basis(m: int, rank: int, seed: int) -> Tensor:
    """A random orthonormal basis of a rank-dimensional subspace of R^m, as (m, rank).

    The Q factor of a QR decomposition of a random (m, rank) matrix: orthonormal
    columns spanning a uniformly random subspace - the same algebraic shape as U_k,
    with no semantic content. The adapter reads U_k^T x, so this feeds it rank
    coordinates of a subspace chosen without reference to any covariate's meaning.
    """
    if not (1 <= rank <= m):
        raise ValueError(f"rank must be in [1, m={m}]; got {rank}")
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(m, rank, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    return q.to(dtype=torch.float32)


def shuffled_semantics(q_tilde: Tensor, seed: int) -> Tensor:
    """The same rows of `q_tilde`, permuted, so feature j gets feature perm[j]'s meaning.

    Capacity and the value distribution of q_tilde are untouched; only the
    feature-to-meaning assignment is destroyed. Redraws if the permutation happens
    to be the identity (a no-op control would silently pass as a real one).
    """
    m = q_tilde.shape[0]
    gen = torch.Generator().manual_seed(seed)
    identity = torch.arange(m)
    perm = torch.randperm(m, generator=gen)
    while m > 1 and torch.equal(perm, identity):
        perm = torch.randperm(m, generator=gen)
    return q_tilde[perm].clone()


def random_semantics(q_tilde: Tensor, seed: int) -> Tensor:
    """A random matrix shaped like `q_tilde`, rescaled to the same root-mean-square.

    Matches dimensionality and magnitude, so a difference from the real semantics
    cannot be attributed to either.
    """
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(q_tilde.shape, generator=gen, dtype=torch.float64)
    real_rms = torch.sqrt(torch.mean(q_tilde.to(torch.float64) ** 2))
    noise_rms = torch.sqrt(torch.mean(noise ** 2))
    if float(noise_rms) == 0.0:
        return noise.to(dtype=q_tilde.dtype)
    return (noise * (real_rms / noise_rms)).to(dtype=q_tilde.dtype)


def onehot_semantics(q_tilde: Tensor, seed: int = 0) -> Tensor:
    """A scaled identity (m, m): feature j's "meaning" is the fact that it is feature j.

    The floor of the semantic ladder. The tokenizer still gets a per-feature vector, and
    the value-semantics interaction x_ij * q~_j still exists, but the vector carries no
    relation between features at all - two covariates that mean almost the same thing are
    as far apart as any other pair. Anything the real prior earns over this arm is earned
    by the geometry of meaning, not by the adapter merely knowing which feature it reads.

    Scaled so the whole matrix has the same root-mean-square as `q_tilde`, which is the
    magnitude the shared phi sees. Row norms then differ by sqrt(m / r_W) - a one-hot
    concentrates its mass in one coordinate where the real q~ spreads it over r_W - and
    that is intrinsic: m distinct one-hots do not fit in R^{r_W} when r_W < m. The
    dimensionality therefore does NOT match the other controls, and the phi input width
    (1 + 2 * r_W) changes with it; report the parameter counts beside this arm.

    Deterministic. `seed` is accepted only so the dispatch table can call every semantic
    control the same way.
    """
    m = q_tilde.shape[0]
    rms = torch.sqrt(torch.mean(q_tilde.to(torch.float64) ** 2))
    eye = torch.eye(m, dtype=torch.float64)
    # RMS of I_m is sqrt(1/m), so this factor lands the result exactly on `rms`.
    return (eye * (rms * float(m) ** 0.5)).to(dtype=q_tilde.dtype)


# control -> (replace U_k?, how to replace q_tilde with which seed offset). None means
# "leave that branch alone". "nonsemantic" is the capacity-matched control: both
# branches replaced at once, the offset keeping the two draws independent while each
# stays reproducible from `seed`.
_DISPATCH = {
    "random_basis": (True, None),
    "shuffled_semantics": (False, (shuffled_semantics, 0)),
    "random_semantics": (False, (random_semantics, 0)),
    "onehot_semantics": (False, (onehot_semantics, 0)),
    "nonsemantic": (True, (random_semantics, 1)),
}


def apply_control(
    U_k: Optional[Tensor],
    q_tilde: Optional[Tensor],
    control: str,
    seed: int,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Dispatch one named control over whichever of U_k/q_tilde is not None.

    A branch that was not supplied (its input is None, e.g. a U-only run has no
    q_tilde) is left as None rather than fabricated.
    """
    if control not in CONTROLS:
        raise ValueError(f"unknown control {control!r}; choose from {CONTROLS}")
    if control == "none":
        return U_k, q_tilde

    replace_u, semantics = _DISPATCH[control]
    u_out = U_k
    if replace_u and U_k is not None:
        u_out = random_basis(U_k.shape[0], U_k.shape[1], seed)
    q_out = q_tilde
    if semantics is not None and q_tilde is not None:
        transform, offset = semantics
        q_out = transform(q_tilde, seed + offset)
    return u_out, q_out
