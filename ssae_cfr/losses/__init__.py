"""Loss terms and their weighted sum.

    L = L_fact + alpha_mmd*L_mmd + beta_l1*L_sparse + lambda_rec*L_rec + gamma(epoch)*L_pref

Modules: `factual` (MSE/BCE), `mmd` (RBF, median heuristic), `preference` (mean_j b_j on
the admission gate), `total` (assembly + warm-up).
"""

from .factual import factual_loss
from .mmd import mmd_rbf
from .preference import preference_loss
from .total import total_loss

__all__ = ["factual_loss", "mmd_rbf", "preference_loss", "total_loss"]