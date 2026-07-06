"""SSAE-CFR training hyperparameters (model config, not dataset column roles).

- Dataset *column roles* live in the repo-root "config.py";
- model/training *hyperparameters* live here.

See :mod:`ssae_cfr.config.hparams`.
"""

from .hparams import TrainConfig, load_config

__all__ = ["TrainConfig", "load_config"]
