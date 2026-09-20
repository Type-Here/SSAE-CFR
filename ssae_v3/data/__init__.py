"""Dataset interface and adapters.

`base` defines the `Dataset` container every adapter produces. Adapters (`ihdp`,
`actg175`, `mimic`) read per-dataset column roles from the repo-root `config.py`
(via `roles`) and emit a `Dataset`.

The ACTG175 and MIMIC adapters are optional: their modules are not present in every
checkout, and a checkout without them must still be able to train on IHDP. Their
entry points are therefore bound to None when the module is missing rather than
raising at import time, which would take down every module that builds a dataset
registry. A registry drops the None entries, so an absent adapter shows up as an
unknown dataset at the point of use and never as an ImportError somewhere else.
"""

from importlib import import_module
from typing import Callable, Dict, Optional

from .base import Dataset
from .ihdp import N_REALIZATIONS, has_replication_set, load_ihdp, load_ihdp_realization


def _optional(module: str, *names: str) -> Dict[str, Optional[Callable[[], Dataset]]]:
    """Bind `names` from an adapter module, or to None when it is not in this checkout.

    Only a missing module is tolerated. An adapter that exists and raises while
    importing is a real error and propagates.
    """
    try:
        adapter = import_module(f".{module}", __name__)
    except ModuleNotFoundError as exc:
        if exc.name not in (f"{__name__}.{module}", module):
            raise
        return {name: None for name in names}
    return {name: getattr(adapter, name) for name in names}


_actg175 = _optional("actg175", "load_actg175_rct", "load_actg175_pseudo_obs")
load_actg175_rct = _actg175["load_actg175_rct"]
load_actg175_pseudo_obs = _actg175["load_actg175_pseudo_obs"]

_mimic = _optional("mimic", "load_diur_v1", "load_sepsis_v2")
load_diur_v1 = _mimic["load_diur_v1"]
load_sepsis_v2 = _mimic["load_sepsis_v2"]


def available(loaders: Dict[str, Optional[Callable[[], Dataset]]]) -> Dict[str, Callable[[], Dataset]]:
    """Drop the datasets whose adapter this checkout does not have."""
    return {name: fn for name, fn in loaders.items() if fn is not None}


__all__ = [
    "Dataset",
    "N_REALIZATIONS",
    "available",
    "has_replication_set",
    "load_ihdp",
    "load_ihdp_realization",
    "load_actg175_rct",
    "load_actg175_pseudo_obs",
    "load_diur_v1",
    "load_sepsis_v2",
]
