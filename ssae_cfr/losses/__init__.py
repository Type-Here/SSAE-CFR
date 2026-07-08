"""Loss terms and their weighted sum.

    L = L_fact + alpha_mmd*L_mmd + beta_l1*L_sparse + lambda_rec*L_rec + gamma(epoch)*L_align

Modules: `factual` (MSE/BCE), `mmd` (RBF, median heuristic), `align` (mean ||mu_res||^2),
`total` (assembly + warm-up).
"""

from .align import align_loss
from .factual import factual_loss
from .mmd import mmd_rbf
from .total import total_loss

__all__ = ["factual_loss", "mmd_rbf", "align_loss", "total_loss"]