"""Dataset interface and adapters.

`base` defines the `Dataset` container every adapter produces. Adapters (`ihdp`,
`actg175`, `mimic`) read per-dataset column roles from the repo-root `config.py`
(via `roles`) and emit a `Dataset`.
"""

from .base import Dataset
from .ihdp import N_REALIZATIONS, has_replication_set, load_ihdp, load_ihdp_realization
from .actg175 import load_actg175_rct, load_actg175_pseudo_obs
from .mimic import load_diur_v1, load_sepsis_v2

__all__ = [
    "Dataset",
    "N_REALIZATIONS",
    "has_replication_set",
    "load_ihdp",
    "load_ihdp_realization",
    "load_actg175_rct",
    "load_actg175_pseudo_obs",
    "load_diur_v1",
    "load_sepsis_v2",
]