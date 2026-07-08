"""Model components: SSAE encoder/decoder, PGAG decomposition+gating, TARNet heads.

Assembled by `ssae_cfr.models.ssae_cfr` into the full v1 / Variant A model. Modules
subclass `torch.nn.Module`. Exports grow as each component is implemented (Milestones
5-7); the encoder/decoder land first.
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