from torch import nn

from ssae_v3.core.empirical import Empirical
from hparams import DefaultConfig

class SSAECFRv3(nn.Module):
    """
    SSAE-CFR model main class, connecting each module.
        From Empirical: u
        From U_adapter: c
        From W_adapter: a_W
        Input: x -> x_std

        x_std -> u + c -> u_shared (MMD) - +a_W -> u_out -> to heads h0 and h1
    """

    def __init__(self):
        super().__init__()
        self.empirical = Empirical(DefaultConfig)

    def forward(self, x):
        # TODO
        return self.empirical(x)