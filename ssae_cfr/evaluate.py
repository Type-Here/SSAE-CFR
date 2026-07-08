"""Evaluation entry point - STUB.

Dispatch on dataset:
* IHDP → PEHE, eps_ATE (mean ± std over realizations, in/out-of-sample).
* ACTG175 RCT → ATE vs RCT reference; CATE stability.
* ACTG175 pseudo-obs → PEHE-against-zero, SMD reduction, E-value.
* MIMIC (diur_v1, sepsis_v2) → SMD reduction, policy risk, E-value, head variance.

Uses :mod:`ssae_cfr.utils.metrics`.
"""

from __future__ import annotations


def main() -> None:
    """Evaluate a trained run and emit the dataset-appropriate metrics. STUB."""
    raise NotImplementedError("Evaluation harness not yet implemented")


if __name__ == "__main__":
    main()