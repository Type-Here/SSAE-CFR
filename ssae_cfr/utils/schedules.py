"""Training schedules.

Preference warm-up: the weight on `L_pref` ramps linearly from 0 to `gamma_pref` over the
first `gamma_warmup` epochs, then holds. Early on the weight is small, so the admission
gate `b` is free and is driven only by the reconstruction, which pushes it toward 1 - i.e.
the model starts by looking at all of x. Only as training settles does the preference for
the documented subspace grow to full strength. This avoids letting a possibly-wrong prior
dictate the representation from epoch zero (a self-fulfilling prophecy). With
`warmup_epochs <= 0` the full weight applies immediately.

The ramp is generic, which is why it is named for its shape and not for the term it
weights: it previously carried `gamma_align`, and the alignment term it belonged to is
gone.
"""

from __future__ import annotations


def linear_warmup(epoch: int, target: float, warmup_epochs: int) -> float:
    """Linear 0 -> `target` ramp over `warmup_epochs` (0-based epoch)."""
    if warmup_epochs <= 0:
        return target
    frac = min(1.0, epoch / warmup_epochs)
    return target * frac