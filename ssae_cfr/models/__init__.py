"""Model components: SSAE encoder/decoder, PGAG decomposition + admission gate, heads.

Assembled by `ssae_cfr.models.ssae_cfr` into the full v1 / Variant A model, which runs
one encoder pass over `x_mod = P_U x + b(x) * (I - P_U) x` and reads the decoder, the two
TARNet heads and the MMD off the single resulting code. Modules subclass
`torch.nn.Module`.
"""

from .heads import OutcomeHeads
from .pgag import PGAG, PGAGOutput
from .ssae import Decoder, Encoder, build_mlp
from .ssae_cfr import SSAECFR

__all__ = [
    "Encoder",
    "Decoder",
    "build_mlp",
    "OutcomeHeads",
    "PGAG",
    "PGAGOutput",
    "SSAECFR",
]