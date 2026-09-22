"""Online prior modules: the fixed prior-guidance transform, the U/W adapters, the
expert-graph adapter, the reliability interface, and negative controls for telling a
semantic gain apart from a capacity-matched one."""

from .controls import CONTROLS, apply_control, random_basis, random_semantics, shuffled_semantics
from .expert_adapter import ExpertPriorAdapter
from .expert_bundle import ExpertBundleError, ExpertPriorBundle, load_expert_bundle
from .expert_graph_controls import (
    EXPERT_GRAPH_CONTROLS,
    apply_graph_control,
    permuted_graph,
    random_matched_graph,
)
from .guidance_bundle import (
    GuidanceBundleError,
    PriorGuidanceBundle,
    build_embedding_subspace,
    build_graph_subspace,
    expert_dir_for,
    load_guidance_bundle,
    orthonormal_basis,
)
from .guidance_controls import (
    EMBEDDING_VARIANTS,
    GRAPH_VARIANTS,
    apply_guidance_controls,
    make_degree_matched_graph_control,
    make_permuted_graph_control,
    make_random_embedding_control,
)
from .prior_guidance import PriorGuidance, first_linear_weight
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
    "ExpertPriorAdapter",
    "ExpertPriorBundle",
    "ExpertBundleError",
    "load_expert_bundle",
    "EXPERT_GRAPH_CONTROLS",
    "apply_graph_control",
    "permuted_graph",
    "random_matched_graph",
    "PriorGuidance",
    "first_linear_weight",
    "PriorGuidanceBundle",
    "GuidanceBundleError",
    "load_guidance_bundle",
    "expert_dir_for",
    "orthonormal_basis",
    "build_embedding_subspace",
    "build_graph_subspace",
    "EMBEDDING_VARIANTS",
    "GRAPH_VARIANTS",
    "apply_guidance_controls",
    "make_random_embedding_control",
    "make_permuted_graph_control",
    "make_degree_matched_graph_control",
]
