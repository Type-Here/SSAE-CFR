"""Baselines for comparison.

Build order: TARNet/CFRNet first (deterministic, no prior/noise - pipeline sanity
check), then the BCAUSS adapter (primary comparison; isolates the gain from the
`U_k` prior).
"""