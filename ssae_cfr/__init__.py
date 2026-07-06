"""SSAE-CFR: Sparse Stochastic Autoencoder for Counterfactual Regression.

CATE estimation with a stochastic denoising sparse autoencoder + an LLM-derived
semantic prior (analytic projector ``P_U``), with orthogonal decomposition in
feature space (Variant A).

See ``internal_docs/Implementation_plan.md`` for the full spec and ``CLAUDE.md``
for living context. This is v1 / Variant A.
"""

__version__ = "0.0.1"