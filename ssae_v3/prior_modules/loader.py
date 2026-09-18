"""Online-side reader for the semantic prior bundle.

Resolves a dataset name (or an explicit path) to its `prior_bundle.npz`, checks the
bundle's covariate order against the dataset's own `feature_names`, and returns the
prior objects the model needs as torch tensors. No nn.Module, no config coupling -
wiring these tensors into the U/W adapters happens where the model is assembled.

The U branch consumes the basis `U_k`, not the projector `P_U = U_k U_k^T`: the
adapter reads the k_U coordinates `U_k^T x` rather than their embedding back into
R^m. Same subspace, same information, k_U inputs instead of m. `P_U` stays in the
bundle for the offline retention diagnostics, which are about covariate space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from ..data.roles import repo_root
from ..prior_build.projector import PriorBundle, load_bundle

PathLike = Union[str, Path]


@dataclass
class PriorTensors:
    """The prior objects the model consumes, as float32 torch tensors."""

    U_k: Tensor           # (m, k_U)
    q_tilde: Tensor        # (m, r_W_rank)
    k_U: int
    r_W_rank: int
    s_Q: float
    centered: bool
    feature_names: Sequence[str]
    meta: dict


def bundle_path(dataset: str) -> Path:
    """Default location of a dataset's prior bundle: artifacts/<dataset>/prior_bundle.npz."""
    return repo_root() / "artifacts" / dataset / "prior_bundle.npz"


def _check_feature_order(bundle_names: Sequence[str], dataset_names: Sequence[str]) -> None:
    """Raise a clear error naming the first mismatch between the bundle and the dataset.

    Row j of the embedding matrix lines up with covariate j only because the order
    is identical; a silent reordering would corrupt the prior with no shape error
    to catch it.
    """
    bundle_names = list(bundle_names)
    dataset_names = list(dataset_names)
    if len(bundle_names) != len(dataset_names):
        raise ValueError(
            f"prior bundle has {len(bundle_names)} covariates, dataset has "
            f"{len(dataset_names)}"
        )
    for j, (b, d) in enumerate(zip(bundle_names, dataset_names)):
        if b != d:
            raise ValueError(
                f"prior bundle covariate order does not match the dataset at index "
                f"{j}: bundle has {b!r}, dataset has {d!r}"
            )


def load_prior_tensors(
    feature_names: Sequence[str],
    dataset: Optional[str] = None,
    path: Optional[PathLike] = None,
) -> PriorTensors:
    """Load a prior bundle and check it against `feature_names`, as torch tensors.

    Either `dataset` (resolved to the default artifact location) or an explicit
    `path` must be given.
    """
    if path is None:
        if dataset is None:
            raise ValueError("load_prior_tensors needs either dataset or path")
        path = bundle_path(dataset)

    bundle: PriorBundle = load_bundle(path)
    _check_feature_order(bundle.feature_names, feature_names)

    return PriorTensors(
        U_k=torch.as_tensor(bundle.U_k, dtype=torch.float32),
        q_tilde=torch.as_tensor(bundle.Q_tilde, dtype=torch.float32),
        k_U=bundle.k_U,
        r_W_rank=bundle.r_W_rank,
        s_Q=bundle.s_Q,
        centered=bundle.centered,
        feature_names=list(bundle.feature_names),
        meta=dict(bundle.meta),
    )
