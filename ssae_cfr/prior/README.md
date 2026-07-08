# Quick Start for Prior Module

## Last Implemented Scripts Reference
1. `prior/descriptions.py` - turns covariate names into the LLM prompts, via committed per-dataset gloss YAMLs (ssae_cfr/prior/glosses/). 
2. `prior/build.py` - the CLI that does the whole thing (glosses → prompts → V → P_U → cache + manifest), plus train.py --prior to consume the result.

Plus save_projector/load_projector and the reproducibility manifest.                                                                                                                                                             

## What to run, in order                                                                                                                                                                                                           
                                                                                                                                                                                                                                   
### Phase 1 -- locally (no GPU needed):
```sh
  python -m ssae_cfr.prior.build emit --dataset ihdp      # writes glosses/ihdp.yaml
  python -m ssae_cfr.prior.build emit --dataset diur_v1   # (already authored)
```
**Note**:
It is needed to edit `glosses/ihdp.yaml`: its columns are anonymized x1..x25, so the defaults are useless; 
A complete IHDP data dictionary is necessary.
`diur_v1` is filled as a worked example.

## Phase 2 -- GPU machine:
```sh
  conda activate ssae-cfr
  pip install "transformers>=4.40" accelerate sentencepiece   # + huggingface-cli login if gated
  python -m ssae_cfr.prior.build build --dataset ihdp --model BioMistral/BioMistral-7B
  python -m ssae_cfr.prior.build build --dataset diur_v1 --model BioMistral/BioMistral-7B \
         --protected creatinine,bun,vaso_any_baseline
```                                                                                                                                                                                 
The build reads only the committed glosses, not the raw datasets, so the data doesn't even need to be on the uni machine. 
Copy the `artifacts/` tree back.

### Phase 3 -- rain on the real prior:
```sh
  python -m ssae_cfr.train --prior artifacts/ihdp/P_U.npz
```  
One flag swaps placeholder → real; training refuses a covariate-order mismatch.  

## Already verified end-to-end
- The placeholder dry run (--placeholder) exercises the entire path with no LLM: on `diur_v1` it chose k=23 at 0.906 energy, all protected covariates cleared the retention floor, and wrote V/P_U/manifest. 
- The alignment guard: `load_prior_for` loads the `diur_v1` prior correctly and rejects it when misapplied to IHDP.
