"""Loss terms and their weighted sum.

    L = L_fact + alpha_mmd*L_mmd + beta_l1*L_sparse + lambda_rec*L_rec

Modules: `factual` (MSE / BCE-with-logits), `mmd` (RBF, median heuristic), `sparse`
(L1 on a code), `total` (the four-term assembly plus `weighted_breakdown`).
"""

from .factual import factual_loss
from .mmd import mmd_rbf
from .sparse import sparse_loss
from .total import total_loss, weighted_breakdown

__all__ = ["factual_loss", "mmd_rbf", "sparse_loss", "total_loss", "weighted_breakdown"]
