"""Negative-control transformations of the prior.

A capacity-matched control answer whether a measured gain comes
from the LLM's semantic knowledge, or merely from the extra parameters/degrees
of freedom the U/W branches add. Each function below returns
a same-shape, same-capacity replacement for `P_U` or `q_tilde` with the semantic
content destroyed in a specific, named way. All are deterministic given a seed and
use a private `torch.Generator`, so they never disturb the caller's global RNG state.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

CONTROLS = ("none", "random_projector", "shuffled_semantics", "random_semantics", "nonsemantic")


def random_projector(m: int, rank: int, seed: int) -> Tensor:
    """A random rank-matched orthogonal projector in R^{m x m}.

    Built from the Q factor of a QR decomposition of a random (m, rank) matrix, so
    Q has orthonormal columns and P = Q @ Q.T is genuinely symmetric, idempotent,
    with trace == rank - the same algebraic shape as P_U, with no semantic content.
    """
    if not (1 <= rank <= m):
        raise ValueError(f"rank must be in [1, m={m}]; got {rank}")
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(m, rank, generator=gen, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    p = q @ q.T
    return p.to(dtype=torch.float32)


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


def apply_control(
    P_U: Optional[Tensor],
    q_tilde: Optional[Tensor],
    control: str,
    seed: int,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Dispatch one named control over whichever of P_U/q_tilde is not None.

    `"nonsemantic"` is the capacity-matched control: a random projector in place of
    P_U AND random semantics in place of q_tilde, drawn from different seeds so
    they are independent but each still reproducible given `seed`. A branch that
    was not supplied (its input is None, e.g. a U-only run has no q_tilde) is left
    as None rather than fabricated.
    """
    if control not in CONTROLS:
        raise ValueError(f"unknown control {control!r}; choose from {CONTROLS}")
    if control == "none":
        return P_U, q_tilde

    p_out, q_out = P_U, q_tilde
    if control in ("random_projector", "nonsemantic") and P_U is not None:
        rank = int(round(float(torch.trace(P_U))))
        p_out = random_projector(P_U.shape[0], rank, seed)
    if control in ("shuffled_semantics", "nonsemantic") and q_tilde is not None:
        if control == "shuffled_semantics":
            q_out = shuffled_semantics(q_tilde, seed)
        else:
            q_out = random_semantics(q_tilde, seed + 1)
    if control == "random_semantics" and q_tilde is not None:
        q_out = random_semantics(q_tilde, seed)
    return p_out, q_out
