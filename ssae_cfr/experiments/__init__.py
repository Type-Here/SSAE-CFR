"""Benchmark protocols - one module per benchmark whose comparison rules are fixed.

`evaluate.py` answers "how does the model do on this dataset, under a split we chose".
That is the right question for the observational subsets, where nobody else has published
a number to match. It is the wrong question for a benchmark whose published results were
produced under a specific protocol: there the split, the number of replications and the
aggregation are part of the benchmark, not free parameters. Those protocols live here.
"""