# Paper Experiment Configurations

Index of the experiment configs behind *"From Configuration to Prediction"*.
For installation, credentials, data location, and the suggested run order, see
[`../README.md`](../README.md). Each experiment directory has its own README
with per-config details and expected results.

Every experiment is run the same way:

```bash
symfluence workflow run --config <config.yaml>
```

## Experiments

| Directory | Paper section | Figure | Configs | Approx. runtime |
|---|---|---|---|---|
| [`01_domain_definition/`](01_domain_definition/) | §2.1 | Figs 1–3, Table 1 | 14 | minutes each (Iceland: hours) |
| [`02_model_ensemble/`](02_model_ensemble/) | §4.2.1 | Fig 7 | 19 | 0.5–8 h per model |
| [`03_forcing_ensemble/`](03_forcing_ensemble/) | §4.1 | Fig 6 | 4 | 1–2 h each |
| [`04_calibration_ensemble/`](04_calibration_ensemble/) | §4.2.3 | Fig 9 | 130 | 10 min – 2 h each |
| [`05_benchmarking/`](05_benchmarking/) | §4.2.2 | Fig 8 | 1 | ~15 min |
| [`10_multivariate_evaluation/`](10_multivariate_evaluation/) | §4.2.4 | Fig 10 | 4 | 2–4 h each |
| [`11_data_pipeline/`](11_data_pipeline/) | §2.2 | Fig 4 | 3 | 0.5–2 h each |
| [`12_parallel_scaling/`](12_parallel_scaling/) | §5 | Fig 11 | 35 | laptop: minutes–hours; 20 configs need an HPC cluster |

Experiments 06–09 were cut from the final manuscript; the directory numbering
keeps the original gaps.

## Config anatomy

Every config is **standalone** — there is no base-config inheritance and no
override mechanism. What you see in one file is the complete experiment.

- **`system.workflow_steps`** — each config lists exactly the pipeline steps it
  needs, in order. `symfluence workflow run` executes only those steps:
  domain-definition configs stop after `discretize_domain`; the benchmarking
  config runs no model steps at all. Completed steps are tracked with stage
  markers, so configs that share a domain (e.g. all of experiment 02) reuse
  downloaded and preprocessed data automatically.
- **`system.data_dir: default` / `code_dir: default`** — inputs and results go
  to a `SYMFLUENCE_data/` directory created as a sibling of the cloned repo,
  overridable with the `SYMFLUENCE_DATA_DIR` environment variable. Configs are
  independent of the current working directory.
- **`system.random_seed`** — calibration configs pin the seed used for the
  paper (e.g. `random_seed: 42`), so optimization trajectories are
  deterministic on a given platform.

Results for a config land under `SYMFLUENCE_data/domain_<name>/` (see
`../README.md` §3 for the layout).
