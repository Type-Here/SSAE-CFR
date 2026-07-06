"""Prior projector P_U = U_k U_k^T from the SVD of the covariate embeddings V.

The semantic prior enters the model as a fixed, analytic orthogonal projector that
lives in covariate space R^m (one axis per covariate). Given the LLM embedding matrix
`V` (m x d_LLM, one row per covariate description), we take its left singular vectors:

    U, S, Wt = svd(V)          # U is m x m, columns are left singular vectors
    U_k = U[:, :k_svd]         # the top-k directions in covariate space
    P_U = U_k @ U_k.T          # m x m, symmetric and idempotent

`P_U x` keeps the part of a patient vector that lies in the span of the k dominant
"semantic" directions (the clinically explainable subspace); `(I - P_U) x` is the
residual the model is later pushed to shrink. Because `U_k` has orthonormal columns,
`P_U` needs no inverse and its eigenvalues are exactly {0, 1} (1 on the retained
subspace, 0 elsewhere).

Two things are deliberately kept apart:
  - `k_svd` is the rank of `P_U` (how many semantic directions we retain). `P_U` is
    always m x m regardless of `k_svd`.
  - `k_latent` (elsewhere) is the encoder bottleneck. The two are independent knobs.

`P_U` is data-independent: it depends only on the covariate descriptions, not on the
patients, so it is computed once and cached per dataset.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def svd_energy(V: np.ndarray) -> np.ndarray:
    """Cumulative explained-variance curve of the SVD spectrum of `V`.

    Returns an array `energy` of length `min(m, d_LLM)` where `energy[k-1]` is the
    fraction of squared singular-value mass captured by the top-k singular vectors.
    Used both by `choose_k_svd` and for logging the spectrum in a run.
    """
    V = np.asarray(V, dtype=np.float64)
    s = np.linalg.svd(V, compute_uv=False)
    total = np.sum(s ** 2)
    if total == 0.0:
        raise ValueError("V has zero singular-value mass (all-zero embeddings?)")
    return np.cumsum(s ** 2) / total


def choose_k_svd(
    V: np.ndarray,
    energy_threshold: float = 0.90,
    protected: Optional[Sequence[int]] = None,
    retention_floor: float = 0.5,
    k_min: int = 2,
    k_max: Optional[int] = None,
) -> int:
    """Pick the rank `k_svd` of `P_U` from the SVD spectrum of `V`.

    One interpretable knob, `energy_threshold`: choose the smallest k whose top-k
    singular vectors capture at least that fraction of the spectral energy. A safety
    net then bumps k upward until every `protected` covariate (indices of clinically
    critical variables that must survive the projection) has per-covariate retention
    `diag(P_U)[j] >= retention_floor`. The retention of covariate j under the top-k
    subspace is exactly `sum_i U[j, i]^2` for i < k, i.e. the diagonal of `P_U`.

    `k_max` defaults to `m - 1` so `P_U` can never become the identity (a pass-through
    filter that would make the prior vacuous). Erring toward larger k is safer: a
    too-small k suppresses weak clinical signals, which the alignment loss would then
    wrongly penalize.
    """
    V = np.asarray(V, dtype=np.float64)
    m = V.shape[0]
    hard_max = m - 1
    if k_max is None:
        k_max = hard_max
    k_max = max(k_min, min(k_max, hard_max))

    U, s, _ = np.linalg.svd(V, full_matrices=True)
    total = np.sum(s ** 2)
    if total == 0.0:
        raise ValueError("V has zero singular-value mass (all-zero embeddings?)")
    energy = np.cumsum(s ** 2) / total

    # smallest k with cumulative energy >= threshold (searchsorted gives the index)
    k = int(np.searchsorted(energy, energy_threshold) + 1)
    k = max(k_min, min(k, k_max))

    if protected:
        while k < k_max:
            U_k = U[:, :k]
            retention = np.einsum("ij,ij->i", U_k, U_k)  # diag(P_U), in [0, 1]
            if all(retention[j] >= retention_floor for j in protected):
                break
            k += 1

    return k


def build_projector(V: np.ndarray, k_svd: int) -> np.ndarray:
    """Build the m x m orthogonal projector `P_U = U_k U_k^T` from `V` and a fixed k.

    `V` is the embedding matrix (m x d_LLM); `k_svd` the number of leading left
    singular vectors to retain. The result is symmetric, idempotent, has rank `k_svd`
    and eigenvalues in {0, 1}.
    """
    V = np.asarray(V, dtype=np.float64)
    m = V.shape[0]
    if not (1 <= k_svd <= m):
        raise ValueError(f"k_svd must be in [1, m={m}]; got {k_svd}")
    U, _, _ = np.linalg.svd(V, full_matrices=True)
    U_k = U[:, :k_svd]
    return U_k @ U_k.T


def retention(P_U: np.ndarray) -> np.ndarray:
    """Per-covariate retention under `P_U`: the diagonal, each entry in [0, 1].

    `retention[j]` is the fraction of covariate j's unit vector kept by the
    projection; a value near 1 means covariate j is well inside the semantic
    subspace, near 0 means it is pushed almost entirely into the residual.
    """
    return np.diag(np.asarray(P_U, dtype=np.float64)).copy()