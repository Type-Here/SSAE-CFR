"""Training schedules.

Alignment warm-up: gamma ramps linearly from 0 to `gamma_align` over the first
`gamma_warmup_epochs` epochs, then holds. Early on, gamma is small, so the model is free
to explore the residual code z_res and fit the outcome; only as training settles does the
alignment pressure toward the prior subspace grow to full strength. This avoids letting a
possibly-wrong prior dictate the representation from epoch zero (a self-fulfilling
prophecy). With `gamma_warmup_epochs <= 0` the full weight applies immediately.
"""

from __future__ import annotations


def gamma_warmup(epoch: int, gamma_align: float, gamma_warmup_epochs: int) -> float:
    """Linear 0 -> gamma_align ramp over `gamma_warmup_epochs` (0-based epoch)."""
    if gamma_warmup_epochs <= 0:
        return gamma_align
    frac = min(1.0, epoch / gamma_warmup_epochs)
    return gamma_align * frac