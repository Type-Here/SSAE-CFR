# SSAE-CFR

**Sparse Stochastic Autoencoders for Counterfactual Regression with Semantic Priors.**

Estimating conditional average treatment effects (CATE) from observational tabular data,
with two departures from the usual counterfactual-regression recipe:

1. **A stochastic autoencoder in place of a deterministic encoder.**
   Balanced-representation methods (CFR, TARNet) penalize the distance between the treated
   and control representations. When the two groups are well separated, that penalty has
   almost nothing to push against - the distributions barely overlap, so the gradient is
   weak and the representation collapses to whatever the outcome heads want. Injecting
   noise into the encoder restores overlap, and therefore restores the balancing signal.
   The noise scale is not learned: it is a function of the measured covariate imbalance
   (standardized mean difference on the raw covariates, computed per batch and detached),
   so the model cannot make the penalty easy by shrinking its own noise.

2. **A semantic prior derived from a language model, injected as a fixed projector.**
   Each covariate is described in a sentence, that sentence is embedded by a frozen
   domain LLM, and the resulting matrix `V` is reduced by SVD into an orthogonal
   projector `P_U = U_k U_k^T`. Every patient vector `x` splits, in feature space, into
   `x_prior = P_U x` (the part lying in the span of what is clinically documented) and a
   residual `x_res = x - x_prior`. An **admission gate** then decides, per covariate and
   per patient, how much of that residual to let in:

       x_mod = x_prior + b(x) * x_res,    b(x) in [0, 1]^m

   and a bounded penalty `L_pref = mean_j b_j` prices the admission. The effect is to make
   the model exhaust known clinical explanations before it leans on undocumented
   correlations - a guard against balance that rests on spurious structure - while `b`
   stays directly readable: it says which covariates needed the part the prior does not
   name. The form is additive rather than a blend, so `b = 1` recovers `x` exactly: the
   model without a prior is a setting of this model, not a different one.

Good distributional balance is not the same as valid causal identification. The prior is
there for the second problem, the stochasticity for the first.

## Status

The v1 model (feature-space decomposition, Gaussian latent noise, no KL term) is complete
and trains. The evaluation harness runs on all five dataset adapters, and the real IHDP
prior is built (BioMistral-7B, `k_svd = 12` of 25). What is **not** done:

- **The admission gate is new and unmeasured.** It replaced an earlier design that blended
  the two encoded branches convexly in latent space; that form could not represent the
  full covariate vector at all, so the "no prior" case was unreachable. The replacement is
  argued, tested and not yet benchmarked - no result in this repo was produced with it.
- **The prior's contribution to tau is still unmeasured.** The prior carries real semantic
  structure (covariate groups are recovered from the gloss text alone, with no data), but
  whether the decomposition improves the estimate is exactly what `b = 1` vs learned `b`
  is meant to answer, and that comparison has not been run.
- A selection criterion that targets tau. The only oracle-free criterion available is the
  held-out factual objective, which is a reliable filter and an unreliable ranker, so no
  hyperparameter here is honestly tuned.
- Baselines (TARNet/CFRNet, BCAUSS) are not implemented; IHDP runs print published
  reference numbers for orientation instead. Those are orientation only, not a
  like-for-like comparison.
- Priors for the non-IHDP datasets. Without a cached `P_U` those runs fall back to a
  *placeholder* random `V` - geometrically valid, semantically meaningless, and never for
  reported results. See "Building the prior" below.

## Install

```bash
conda env create -f environment.yml
conda activate ssae-cfr
```

Torch is pinned to the CPU build: the model is small and the datasets are small. Only the
one-off LLM embedding step wants a GPU, and that step is deliberately separable.

## Data

Raw data is not distributed with the repo. Place it under `data/raw/<DATASET>/`:

| Adapter key | Path | Notes |
|---|---|---|
| `ihdp` | `data/raw/IHDP/ihdp.csv` | single realization, 747 rows |
| `ihdp` (benchmark) | `data/raw/IHDP/ihdp_npci_1-100.{train,test}.npz` | the 100 realizations |
| `aids_v1` | `data/raw/AIDS_V1/` | ACTG175, randomized |
| `aids_v1_biased` | `data/raw/AIDS_V1_BIASED/` | ACTG175 pseudo-observational (sharp null) |
| `diur_v1`, `sepsis_v2` | `data/raw/DIUR_V1/`, `data/raw/SEPSIS_V2/` | MIMIC subsets |

The IHDP replication files are the standard ones from <https://www.fredjo.com/>:

```bash
curl -O https://www.fredjo.com/files/ihdp_npci_1-100.train.npz
curl -O https://www.fredjo.com/files/ihdp_npci_1-100.test.npz
mv ihdp_npci_1-100.*.npz data/raw/IHDP/
```

Column roles for the ACTG175 and MIMIC subsets (which column is the treatment, which the
outcome, which to drop) come from `config.py` at the repo root. Model hyperparameters live
separately, in `ssae_v3/hparams.py` and `ssae_v3/config/`, so "what the data is" and
"how the model is trained" never get conflated.

## Running

Train one model, print the training curve:

```bash
python -m ssae_v3.training.train --config ssae_v3/config/ihdp.yaml
```

Evaluate on any dataset, over several seeds, with the metric set that dataset can support
(PEHE and eps_ATE where an oracle exists; balance, policy value and E-values everywhere):

```bash
python -m ssae_v3.training.evaluate --dataset ihdp --seeds 0 1 2 --val-size 0.2
python -m ssae_v3.training.evaluate --dataset diur_v1 --seeds 0 1 2 --json-out out/diur.json
```

Run the **IHDP benchmark protocol** - 100 realizations, the train/test partition the
benchmark files ship, a 63/27 split inside the training portion, averaged the way
published tables average:

```bash
python -m ssae_v3.training.experiments --realizations 100 --json-out out/ihdp_bench.json
```

This is the entry point to use for anything comparative. `evaluate --dataset ihdp` varies
the seed and the split of a *single* realization, which measures our own variance rather
than the benchmark's, and its numbers do not belong in a table next to published ones.

## Building the prior

The prior bundle is built offline and cached, because it is fixed: it depends on the
covariate set and the embedding model, not on anything learned. One SVD of the centered
embedding matrix yields the covariate-space projector `P_U` and the semantic coordinates
`q~` that the two adapters consume.

An existing `artifacts/<dataset>/` built before the bundle existed is rebuilt without a
GPU in seconds:

```bash
python -m ssae_v3.prior_build.build build --dataset ihdp --reuse-V
```

1. **Emit a gloss template** (locally; needs the dataset present, to read column order):

   ```bash
   python -m ssae_v3.prior_build.build emit --dataset diur_v1
   ```

   This writes `ssae_v3/prior_build/glosses/<dataset>.yaml`, one line per covariate.

2. **Write the glosses.** This is the human step and the one that determines whether the
   prior means anything. Each gloss is the phrase describing that covariate to the LLM.
   A gloss file may also set its own prompt template under the reserved key `_template`:
   the default asks for a physiological and prognostic reading, which suits the ICU and
   trial datasets but not IHDP, whose covariates are obstetric and socioeconomic.
   Row order must match the dataset's covariate order and must not be changed - it is
   what guarantees row `j` of `V` describes covariate `j`.

3. **Dry-run** without a GPU or a model, to check the glosses parse and the rank selection
   behaves:

   ```bash
   python -m ssae_v3.prior_build.build build --dataset ihdp --placeholder
   rm -rf artifacts/ihdp artifacts/manifest.json   # a placeholder prior is not a prior
   ```

4. **Build for real**, on a machine with a GPU (needs only the committed gloss YAML, not
   the raw data - no dataset ever has to leave your machine):

   ```bash
   pip install "transformers>=4.40" accelerate sentencepiece
   python -m ssae_v3.prior_build.build build --dataset ihdp --model BioMistral/BioMistral-7B
   ```

   This caches `artifacts/<dataset>/V.npz` and `prior_bundle.npz` and records the recipe -
   model, precision, prompt template, gloss file, chosen rank, energy captured, covariate
   order and a hash of `V` - in the versioned `artifacts/manifest.json`. The `.npz` files
   themselves are gitignored; move them by `rsync`.

   `--dtype` defaults to float16 on a GPU and float32 on CPU. A 7B model at float32 needs
   about 28 GB of weights and will be killed during loading on any modest host, so leave
   it on `auto` unless you know the machine can take it. Nothing here hard-codes a model
   size or embedding width: `--model` accepts any HuggingFace id and `d_LLM` is read from
   the model, so a small sentence encoder is a legitimate choice if a 7B one is awkward
   to run.

5. **Check it before trusting it**, which costs nothing - it runs on the cached `V`, with
   no GPU, no model and no dataset:

   ```bash
   python -m ssae_v3.prior_build.build diagnose --dataset ihdp
   ```

   It prints the SVD spectrum and the mean pairwise cosine, then each covariate's nearest
   neighbors. Read the neighbors: covariates that mean similar things must sit together.
   A `k_U` of 1 or 2 is a failure signal rather than an efficient prior - it means the
   embeddings are nearly collinear, which is why `V` is centered before the SVD by default
   (`--no-center` to disable, not recommended).

   `V` is the expensive artifact; the bundle is one SVD of an `m x d_LLM` matrix. So rank
   selection is free once `V` is cached - `--reuse-V` rebuilds it without re-embedding:

   ```bash
   python -m ssae_v3.prior_build.build build --dataset ihdp --reuse-V --energy 0.95
   ```

6. **Train against it.** The bundle is found automatically under `artifacts/<dataset>/`;
   `--prior` overrides the path:

   ```bash
   python -m ssae_v3.training.experiments --variants empirical,u_w_adapter --controls nonsemantic
   ```

   The covariate order stored with the bundle is checked against the dataset's, and a
   mismatch is an error rather than a silently wrong projection. There is no placeholder
   fallback: a variant that needs a prior and finds no bundle refuses to run.

## Negative controls

A variant beating the baseline does not by itself show that the LLM's knowledge helped -
the adapters also add parameters. `--controls` runs capacity-matched arms beside the real
one, replacing the projector with a rank-matched random one and the semantic coordinates
with shuffled or random ones of the same shape and scale. If the control matches the real
arm, the gain is capacity, not semantics.

## Layout

```
ssae_v3/
  core/            encoder, decoder, noise injector, outcome heads, outcome scaling
  prior_build/     glosses -> LLM embeddings V -> the cached prior bundle (offline)
  prior_modules/   U and W adapters, reliability, bundle loader, negative controls
  losses/          factual, mmd, sparse, and the weighted total
  data/            dataset adapters -> a single `Dataset` container
  utils/           standardization, splitting, SMD, metrics
  training/        fit one model, score splits, run the IHDP protocol and the ladder
  model_v3.py      the composed model: empirical host + the two additive corrections
  hparams.py       model and training hyperparameters
  config/          per-dataset YAML overrides
config.py          per-dataset column roles (data, not hyperparameters)
tests/             pytest suite
```

Two `data` directories exist on purpose: `data/raw/` holds the datasets, `ssae_v3/data/`
holds the adapter code that reads them.

## Tests

```bash
python -m pytest
```

Tests that need a dataset on disk skip when it is absent, so the suite runs on a fresh
checkout.

## Notation

| | |
|---|---|
| `m` | number of covariates |
| `k_svd` | rank of the projector `P_U` (`P_U` is always `m x m`) |
| `k_latent` | encoder bottleneck width |
| `omega` | noise scale, `tanh(alpha_smd * mean SMD)`, detached |
| `b` | the admission gate in `[0, 1]^m`: how much of each covariate's residual is let in |
| `x_mod` | `P_U x + b * (I - P_U) x`, the encoder's input |
| `z` | the representation: what the decoder, the heads and the MMD all read |

`k_svd` and `k_latent` are independent and are never tied to each other.

## License

GPL-3.0. See `LICENSE`.