"""Online prior modules: the U/W adapters, the reliability interface, and negative
controls for telling a semantic gain apart from a capacity-matched one."""

from .controls import CONTROLS, apply_control, random_basis, random_semantics, shuffled_semantics
from .loader import PriorTensors, bundle_path, load_prior_tensors
from .reliability import FixedReliability
from .u_adapter import UStructuralAdapter
from .w_adapter import ValueSemanticTokenizer, WSemanticAdapter

__all__ = [
    "PriorTensors",
    "bundle_path",
    "load_prior_tensors",
    "FixedReliability",
    "UStructuralAdapter",
    "ValueSemanticTokenizer",
    "WSemanticAdapter",
    "CONTROLS",
    "apply_control",
    "random_basis",
    "random_semantics",
    "shuffled_semantics",
]
