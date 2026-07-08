"""The full SSAE-CFR model (v1, Variant A).

One forward pass runs three encodings through the single shared encoder:

    x        --> encode --> mu_enc, z_enc --> decode --> x_hat      (L_rec, L_sparse)
    P_U x    --> encode --> z_prior                                 (gating)
    (I-P_U)x --> encode --> mu_res, z_res                           (L_align, gating)

then gates the prior/residual codes into z_mod, and reads the two outcome heads off it:

    z_mod --> h0, h1 --> y0_hat, y1_hat --> yf_hat                  (L_fact)
    z_mod | t         --> MMD                                       (L_mmd)

`forward` returns every raw piece (predictions, reconstruction, codes, gate, diagnostics)
without collapsing anything into a loss, so the caller stays in control. `loss_terms`
turns those pieces into the five scalar terms `total_loss` expects. The plain
reconstruction pass encodes the *whole* x (not a decomposed half): it is what keeps the
code informative and gives the L1 sparsity something to compress, given there is no KL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..losses import align_loss, factual_loss, mmd_rbf
from .heads import OutcomeHeads
from .pgag import PGAG
from .ssae import Decoder, Encoder

if TYPE_CHECKING:
    from ..config import TrainConfig


class SSAECFR(nn.Module):
    """Encoder + decoder + PGAG (shared encoder) + two heads, over a fixed P_U."""

    def __init__(self, m: int, P_U: Tensor, cfg: "TrainConfig") -> None:
        super().__init__()
        self.m = m
        self.k_latent = cfg.k_latent
        self.l1_target = cfg.l1_target

        self.encoder = Encoder(m, cfg.encoder_hidden, cfg.k_latent, cfg.activation, cfg.batchnorm)
        self.decoder = Decoder(cfg.k_latent, cfg.decoder_hidden, m, cfg.activation, cfg.batchnorm)
        self.pgag = PGAG(self.encoder, P_U, cfg.gating_hidden, cfg.activation, cfg.batchnorm)
        self.heads = OutcomeHeads(cfg.k_latent, cfg.head_hidden, cfg.activation, cfg.batchnorm)

    def forward(self, x: Tensor, t: Tensor, omega: float = 0.0) -> Dict[str, Tensor]:
        # reconstruction pass over the whole x (shared encoder)
        mu_enc, z_enc = self.encoder.encode(x, omega)
        x_hat = self.decoder(z_enc)

        # prior/residual decomposition + gating
        pg = self.pgag(x, omega)
        y0_hat, y1_hat = self.heads(pg.z_mod)
        tf = t.to(y1_hat.dtype)
        yf_hat = tf * y1_hat + (1.0 - tf) * y0_hat

        return {
            "y0_hat": y0_hat,
            "y1_hat": y1_hat,
            "yf_hat": yf_hat,
            "x_hat": x_hat,
            "mu_enc": mu_enc,
            "z_enc": z_enc,
            "z_mod": pg.z_mod,
            "lam": pg.lam,
            "mu_res": pg.mu_res,
            "z_prior": pg.z_prior,
            "z_res": pg.z_res,
        }

    def loss_terms(
        self,
        out: Dict[str, Tensor],
        x: Tensor,
        t: Tensor,
        yf: Tensor,
        outcome_type: str = "continuous",
    ) -> Dict[str, Tensor]:
        """Assemble the five scalar loss terms from a forward output."""
        code = out["z_enc"] if self.l1_target == "z" else out["mu_enc"]
        return {
            "L_fact": factual_loss(out["y0_hat"], out["y1_hat"], t, yf, outcome_type),
            "L_mmd": mmd_rbf(out["z_mod"], t),
            "L_sparse": code.abs().sum(dim=-1).mean(),
            "L_rec": F.mse_loss(out["x_hat"], x),
            "L_align": align_loss(out["mu_res"]),
        }

    @torch.no_grad()
    def predict_tau(self, x: Tensor) -> Tensor:
        """Deterministic CATE estimate tau_hat(x) = y1_hat - y0_hat (eval, no noise)."""
        was_training = self.training
        self.eval()
        out = self.forward(x, torch.zeros(x.shape[0], device=x.device), omega=0.0)
        if was_training:
            self.train()
        return out["y1_hat"] - out["y0_hat"]