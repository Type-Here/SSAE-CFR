"""Prior bundle: SVD of the covariate embeddings V into P_U, W_r and Q.

The semantic prior enters the model as fixed, analytic objects derived from one SVD
of the LLM embedding matrix V (m x d_LLM, one row per covariate description):

    V_c = U S W^T                 # V_c is V (optionally centered)
    U_k = U[:, :k_U]               # top-k_U left singular vectors: covariate space
    P_U = U_k @ U_k.T               # m x m orthogonal projector, symmetric idempotent
    W_r = W[:, :r_W_rank]           # top-r_W right singular vectors: semantic space
    Q   = V_c @ W_r                 # (m, r_W) semantic coordinates, q_j = W_r.T @ v_j

`P_U x` keeps the part of a patient vector that lies in the span of the k dominant
covariate-space directions; `W_r`/`Q` describe what each covariate means, for the
value-semantic tokenizer. `k_U` (rank of P_U) and `r_W_rank` (rank of W_r) are
independent knobs; the first default is `r_W_rank = k_U`, not a claim they must match.

Both `P_U` and the prior bundle are data-independent: they depend only on the
covariate descriptions, not on the patients, so they are computed once per dataset.

Centering
---------
`center=True` (the supported default) subtracts the mean embedding before the SVD.
LM embedding spaces are strongly anisotropic: every string lands in a narrow cone, so
V is close to `1 mu^T` plus small deviations, and uncentered the leading singular
direction is that shared offset (its left singular vector is approximately the
all-ones vector in R^m) rather than a semantic direction. Measured on the real IHDP V:
uncentered k at 90% energy = 2, mean off-diagonal cosine +0.885; centered k at 90% =
12, cosine -0.039. Centering does not create structure - a flat centered spectrum
means V is genuinely empty of between-covariate structure.

Consequence: after centering, `1^T V_c = 0`, so the all-ones direction lies in the
null space of P_U by construction, hence `P_U @ ones == 0` exactly. A uniform shift
across standardized covariates is therefore never explained by the prior.

Note: `P_U` retention (its diagonal) is not a clinical-relevance ranking - it keeps
variance-dominant directions of the gloss cloud, and the complement of P_U is not
noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def _prepare(V: np.ndarray, center: bool) -> np.ndarray:
    """`V` as float64, with the mean embedding removed when `center` is set."""
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(f"V must be 2-D (m, d_LLM); got shape {V.shape}")
    return V - V.mean(axis=0, keepdims=True) if center else V


def svd_energy(V: np.ndarray, center: bool = False) -> np.ndarray:
    """Cumulative explained-variance curve of the SVD spectrum of `V`.

    Returns an array of length `min(m, d_LLM)` where entry k-1 is the fraction of
    squared singular-value mass captured by the top-k singular vectors.
    """
    s = np.linalg.svd(_prepare(V, center), compute_uv=False)
    total = np.sum(s ** 2)
    if total == 0.0:
        raise ValueError("V has zero singular-value mass (all-zero embeddings?)")
    return np.cumsum(s ** 2) / total


def cosine_similarities(V: np.ndarray, center: bool = False) -> np.ndarray:
    """The m x m matrix of cosine similarities between covariate embeddings.

    A mean off-diagonal similarity near 1 means every covariate embedded to nearly
    the same place, and the prior will be degenerate whatever rank is chosen.
    """
    V = _prepare(V, center)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = V / norms
    return unit @ unit.T


def choose_k_svd(
    V: np.ndarray,
    energy_threshold: float = 0.90,
    protected: Optional[Sequence[int]] = None,
    retention_floor: float = 0.5,
    k_min: int = 2,
    k_max: Optional[int] = None,
    center: bool = False,
) -> int:
    """Pick the rank `k_U` of `P_U` from the SVD spectrum of `V`.

    Chooses the smallest k whose top-k singular vectors capture at least
    `energy_threshold` of the spectral energy, then bumps k upward until every
    `protected` covariate index has per-covariate retention (`diag(P_U)[j]`, i.e.
    `sum_i U[j, i]^2` for i < k) at least `retention_floor`.

    `k_max` defaults to `m - 1` so `P_U` can never become the identity (a
    pass-through that would make the prior vacuous). Erring toward larger k is
    safer: a too-small k suppresses weak semantic signal.
    """
    V = _prepare(V, center)
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


def build_projector(V: np.ndarray, k_svd: int, center: bool = False) -> np.ndarray:
    """Build the m x m orthogonal projector `P_U = U_k U_k^T` from `V` and a fixed k.

    Result is symmetric, idempotent, rank `k_svd`, eigenvalues in {0, 1}, whether or
    not `V` was centered - centering changes which subspace is retained, never the
    fact that `P_U` is an orthogonal projector.
    """
    V = _prepare(V, center)
    m = V.shape[0]
    if not (1 <= k_svd <= m):
        raise ValueError(f"k_svd must be in [1, m={m}]; got {k_svd}")
    U, _, _ = np.linalg.svd(V, full_matrices=True)
    U_k = U[:, :k_svd]
    return U_k @ U_k.T


def retention(P_U: np.ndarray) -> np.ndarray:
    """Per-covariate retention under `P_U`: the diagonal, each entry in [0, 1].

    Near 1 means covariate j is well inside the semantic subspace; near 0 means it
    is pushed almost entirely into the residual.
    """
    return np.diag(np.asarray(P_U, dtype=np.float64)).copy()


def _canonicalize_signs(U: np.ndarray, Wt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fix the SVD's sign ambiguity: flip each (U, W) column pair together.

    For column i, if the largest-magnitude entry of U[:, i] is negative, flip the
    sign of both U[:, i] and Wt[i, :], leaving U @ diag(s) @ Wt unchanged. Applied
    once, right after the SVD, so every array derived downstream is deterministic.
    """
    U = U.copy()
    Wt = Wt.copy()
    for i in range(U.shape[1]):
        j = int(np.argmax(np.abs(U[:, i])))
        if U[j, i] < 0:
            U[:, i] = -U[:, i]
            Wt[i, :] = -Wt[i, :]
    return U, Wt


@dataclass
class PriorBundle:
    """The full semantic prior for one dataset: covariate and semantic spaces.

    U_k/P_U act in covariate space (m axes); W_r/Q/Q_tilde describe what each
    covariate means in the LLM's semantic space. `meta` carries provenance
    (dataset, model_name, V_sha256, created, energy_at_k, ...) not needed for the
    math but needed to trust and reproduce the bundle.
    """

    U_k: np.ndarray
    singular_values: np.ndarray
    W_r: np.ndarray
    P_U: np.ndarray
    Q: np.ndarray
    Q_tilde: np.ndarray
    s_Q: float
    k_U: int
    r_W_rank: int
    feature_names: Sequence[str]
    centered: bool
    meta: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        return self.P_U.shape[0]

    @property
    def d_LLM(self) -> int:
        return self.W_r.shape[0]


def build_prior_bundle(
    V: np.ndarray,
    k_U: Optional[int] = None,
    r_W_rank: Optional[int] = None,
    center: bool = True,
    feature_names: Optional[Sequence[str]] = None,
    eps: float = 1e-8,
    **choose_k_kwargs: Any,
) -> PriorBundle:
    """Build the full prior bundle from one SVD of `V`.

    `k_U=None` picks it with `choose_k_svd` (energy_threshold/protected/
    retention_floor/k_min/k_max forwarded via `choose_k_kwargs`); `r_W_rank=None`
    defaults to `k_U`. Both ranks are independent - equal by default only, not by
    requirement. `q_j` is `W_r.T @ v_j`, i.e. `Q = V_c @ W_r`, which already equals
    `U_r * s_r`; it must not be multiplied by the singular values again.
    """
    V64 = np.asarray(V, dtype=np.float64)
    if V64.ndim != 2:
        raise ValueError(f"V must be 2-D (m, d_LLM); got shape {V64.shape}")
    m, d_LLM = V64.shape

    if k_U is None:
        k_U = choose_k_svd(V64, center=center, **choose_k_kwargs)
    if r_W_rank is None:
        r_W_rank = k_U
    if not (1 <= k_U <= m):
        raise ValueError(f"k_U must be in [1, m={m}]; got {k_U}")
    if not (1 <= r_W_rank <= min(m, d_LLM)):
        raise ValueError(f"r_W_rank must be in [1, min(m, d_LLM)={min(m, d_LLM)}]; got {r_W_rank}")

    V_c = _prepare(V64, center)
    # one decomposition; everything below is derived from it, not recomputed
    U, s, Wt = np.linalg.svd(V_c, full_matrices=False)
    U, Wt = _canonicalize_signs(U, Wt)
    W = Wt.T

    U_k = U[:, :k_U]
    P_U = U_k @ U_k.T
    W_r = W[:, :r_W_rank]
    Q = V_c @ W_r

    s_Q = float(np.sqrt(np.mean(Q ** 2)) + eps)
    Q_tilde = Q / s_Q

    total = np.sum(s ** 2)
    if total == 0.0:
        raise ValueError("V has zero singular-value mass (all-zero embeddings?)")
    energy_at_k = float(np.cumsum(s ** 2)[k_U - 1] / total)

    # left empty rather than invented: a fabricated order could silently pass the
    # loader's covariate-order check against a dataset it was never built for
    names = list(feature_names) if feature_names is not None else []
    return PriorBundle(
        U_k=U_k,
        singular_values=s,
        W_r=W_r,
        P_U=P_U,
        Q=Q,
        Q_tilde=Q_tilde,
        s_Q=s_Q,
        k_U=k_U,
        r_W_rank=r_W_rank,
        feature_names=names,
        centered=center,
        meta={"m": m, "d_LLM": d_LLM, "energy_at_k": energy_at_k},
    )


def save_bundle(bundle: PriorBundle, path: PathLike) -> Path:
    """Persist a PriorBundle: arrays to `path` (.npz), human-readable meta to `<path>.json`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        U_k=bundle.U_k,
        singular_values=bundle.singular_values,
        W_r=bundle.W_r,
        P_U=bundle.P_U,
        Q=bundle.Q,
        Q_tilde=bundle.Q_tilde,
        s_Q=np.float64(bundle.s_Q),
        k_U=np.int64(bundle.k_U),
        r_W_rank=np.int64(bundle.r_W_rank),
        centered=np.bool_(bundle.centered),
        feature_names=np.array(list(bundle.feature_names)),
    )
    sidecar_meta = dict(bundle.meta)
    sidecar_meta.update({
        "feature_names": list(bundle.feature_names),
        "m": bundle.m,
        "d_LLM": bundle.d_LLM,
        "k_U": bundle.k_U,
        "r_W_rank": bundle.r_W_rank,
        "s_Q": bundle.s_Q,
        "centered": bundle.centered,
    })
    path.with_suffix(".json").write_text(json.dumps(sidecar_meta, indent=2), encoding="utf-8")
    return path


def load_bundle(path: PathLike) -> PriorBundle:
    """Load a PriorBundle written by `save_bundle`. Raises if the file is absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no prior bundle at {path}; run the build first")

    with np.load(path) as data:
        U_k = data["U_k"]
        singular_values = data["singular_values"]
        W_r = data["W_r"]
        P_U = data["P_U"]
        Q = data["Q"]
        Q_tilde = data["Q_tilde"]
        s_Q = float(data["s_Q"])
        k_U = int(data["k_U"])
        r_W_rank = int(data["r_W_rank"])
        centered = bool(data["centered"])
        feature_names: List[str] = [str(x) for x in data["feature_names"]]

    sidecar = path.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}

    return PriorBundle(
        U_k=U_k,
        singular_values=singular_values,
        W_r=W_r,
        P_U=P_U,
        Q=Q,
        Q_tilde=Q_tilde,
        s_Q=s_Q,
        k_U=k_U,
        r_W_rank=r_W_rank,
        feature_names=feature_names,
        centered=centered,
        meta=meta,
    )
