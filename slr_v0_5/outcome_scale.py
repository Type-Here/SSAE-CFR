"""The outcome-scale conversion, shared by the model and the evaluation runner.

A continuous head is trained against a standardized outcome, so predictions need the
inverse affine map back to the caller's units. A binary head emits a logit, so it is
squashed to a probability instead - only then is y1 - y0 a risk difference rather
than a log-odds ratio. One definition, torch or numpy, so the two callers cannot drift.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from torch import Tensor


def to_outcome_scale(
    y: Union[Tensor, np.ndarray],
    outcome_type: str,
    loc: float = 0.0,
    scale: float = 1.0,
) -> Union[Tensor, np.ndarray]:
    """Map a head output to the scale the outcome actually lives on.

    Binary: sigmoid(y), and loc/scale must be 0/1 (rejected otherwise, not ignored).
    Continuous: y * scale + loc, the inverse of the standardization applied on input.
    """
    if outcome_type == "binary":
        if loc != 0.0 or scale != 1.0:
            raise ValueError("a binary outcome is never standardized; got loc/scale != 0/1")
        if isinstance(y, Tensor):
            return torch.sigmoid(y)
        return 1.0 / (1.0 + np.exp(-np.asarray(y, dtype=np.float64)))
    if loc == 0.0 and scale == 1.0:
        return y
    return y * scale + loc
