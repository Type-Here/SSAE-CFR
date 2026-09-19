"""ExpertPriorAdapter: a correction built from the expert's feature-to-concept graph.

    x_ij on active edges -> [x_ij, onehot(r_jc)] -> phi_edge -> mean per concept
    -> concat -> rho -> zero-initialized head -> a_expert

Patient values come from the dataset; the expert supplies only which features are
evidence about which concept, and what kind of evidence each is. phi_edge never sees a
feature id, a concept id, an embedding, a similarity, treatment or outcome, so it
cannot learn a separate function per covariate. Its parameter count depends on neither
m nor the edge count.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..core.modules import build_mlp


class ExpertPriorAdapter(nn.Module):
    """Maps x_std to a correction a_expert in R^{d_u} through the expert graph.

    The graph is stored as non-trainable buffers: it moves with `.to(device)` and is
    saved in `state_dict`, but is never optimized. The final head is zero-initialized
    (weight and bias), so a_expert is exactly zero at init whatever the input.
    """

    def __init__(
        self,
        relation_mask: Tensor,
        relation_type_index: Tensor,
        n_relation_types: int,
        d_u: int,
        d_edge: int = 16,
        rho_hidden: Sequence[int] = (32,),
        activation: str = "elu",
        phi_hidden: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if relation_mask.shape != relation_type_index.shape:
            raise ValueError(
                f"relation_mask {tuple(relation_mask.shape)} and relation_type_index "
                f"{tuple(relation_type_index.shape)} must have the same shape"
            )
        mask = relation_mask.detach().to(dtype=torch.bool).clone()
        index = relation_type_index.detach().to(dtype=torch.long).clone()
        m, K = mask.shape
        if not bool(mask.any(dim=0).all()):
            raise ValueError("every concept must have at least one active edge")

        feature_idx, concept_idx = mask.nonzero(as_tuple=True)
        # concept-major with ascending feature inside each concept, so the edge order
        # is fixed by the graph rather than by traversal order
        order = torch.argsort(concept_idx * m + feature_idx)
        feature_idx, concept_idx = feature_idx[order], concept_idx[order]
        relation_type = index[feature_idx, concept_idx]
        if int(relation_type.min()) < 0 or int(relation_type.max()) >= n_relation_types:
            raise ValueError(
                f"relation type index out of range for n_relation_types={n_relation_types}"
            )

        self.m = m
        self.n_concepts = K
        self.d_edge = d_edge
        self.n_relation_types = int(n_relation_types)

        self.register_buffer("relation_mask", mask)
        self.register_buffer("relation_type_index", index)
        self.register_buffer("edge_feature_idx", feature_idx)
        self.register_buffer("edge_concept_idx", concept_idx)
        self.register_buffer("edge_relation_type", relation_type)
        self.register_buffer(
            "edge_onehot",
            torch.nn.functional.one_hot(relation_type, self.n_relation_types).to(torch.float32),
        )
        self.register_buffer("edge_count", mask.sum(dim=0).to(torch.long))

        hidden = tuple(phi_hidden) if phi_hidden is not None else (d_edge,)
        self.phi_edge = build_mlp(1 + self.n_relation_types, hidden, d_edge, activation)
        rho_hidden = tuple(rho_hidden)
        d_z = rho_hidden[-1] if rho_hidden else K * d_edge
        self.rho = build_mlp(K * d_edge, rho_hidden, d_z, activation)
        self.head = nn.Linear(d_z, d_u)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def n_active_edges(self) -> int:
        return int(self.edge_feature_idx.numel())

    def concept_representations(self, x_std: Tensor) -> Tensor:
        """Patient-specific concept states h_ic, (n, K, d_edge)."""
        n = x_std.shape[0]
        if x_std.shape[1] != self.m:
            raise ValueError(f"expected {self.m} covariates; got {x_std.shape[1]}")
        E = self.n_active_edges

        value = x_std[:, self.edge_feature_idx].unsqueeze(-1)          # (n, E, 1)
        onehot = self.edge_onehot.to(dtype=x_std.dtype)
        onehot = onehot.unsqueeze(0).expand(n, -1, -1)                  # (n, E, R)
        edge_in = torch.cat([value, onehot], dim=-1)                    # (n, E, 1+R)
        # one shared phi over all edges: flatten (patient, edge) into one batch axis
        edges = self.phi_edge(edge_in.reshape(n * E, -1)).reshape(n, E, self.d_edge)

        sums = edges.new_zeros(n, self.n_concepts, self.d_edge)
        sums.index_add_(1, self.edge_concept_idx, edges)
        # edge_count >= 1 is enforced when the graph is built, so no epsilon is needed
        return sums / self.edge_count.to(dtype=edges.dtype).view(1, -1, 1)

    def forward(self, x_std: Tensor) -> Tuple[Tensor, Dict[str, object]]:
        """x_std (n, m) -> a_expert (n, d_u) and diagnostics."""
        h = self.concept_representations(x_std)                         # (n, K, d_edge)
        z = self.rho(h.reshape(x_std.shape[0], -1))                     # (n, d_z)
        a_expert = self.head(z)                                         # (n, d_u)

        diagnostics = {
            "a_expert_norm": a_expert.detach().norm(dim=1).mean(),
            "concept_repr_norms": h.detach().norm(dim=2).mean(dim=0),
            "n_active_edges": self.n_active_edges,
            "edges_per_concept": self.edge_count.detach().clone(),
        }
        return a_expert, diagnostics
