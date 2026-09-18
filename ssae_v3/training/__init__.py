"""Training and evaluation harness: the fit loop, split scoring, and the IHDP protocol.

`train` fits one model, `evaluate` scores fitted models and aggregates runs, and
`experiments` runs the IHDP realization protocol and the paired variant ladder.

Nothing is re-exported here on purpose: `train` and `experiments` are both run as
`python -m ssae_v3.training.<module>`, and importing them from this package's __init__
would load each twice under runpy. Import from the submodule directly.
"""
