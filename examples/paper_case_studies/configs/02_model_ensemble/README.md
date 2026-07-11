# Experiment 02 — Multi-Model Ensemble (§4.2.1, Fig 7)

Nineteen hydrological models are run on the identical experiment: Bow River at
Banff (lumped domain, WSC station 05BB001), ERA5 forcing 2002–2009, DDS
calibration with 1,000 iterations against KGE (`random_seed: 42`), calibration
2004–2007, evaluation 2008–2009, spin-up 2002–2003. Seventeen models form the
pooled ensemble of Fig 7; NGEN and VIC are run but **excluded from the pooled
ensemble** — they calibrate poorly under the lumped constraint, which is the
published result, not a bug.

## Configs (`models/`, one per model)

| Config | Model |
|---|---|
| `config_crhm.yaml` | CRHM |
| `config_fuse.yaml` | FUSE |
| `config_gr4j.yaml` | GR4J |
| `config_gsflow.yaml` | GSFLOW |
| `config_hbv.yaml` | HBV |
| `config_hechms.yaml` | HEC-HMS |
| `config_hype.yaml` | HYPE |
| `config_lstm.yaml` | LSTM (via FLASH) |
| `config_mesh.yaml` | MESH |
| `config_mhm.yaml` | mHM |
| `config_prms.yaml` | PRMS |
| `config_rhessys.yaml` | RHESSys |
| `config_sacsma.yaml` | SAC-SMA |
| `config_summa.yaml` | SUMMA |
| `config_swat.yaml` | SWAT |
| `config_topmodel.yaml` | TOPMODEL |
| `config_xinanjiang.yaml` | Xinanjiang (+ Snow17 snow model) |
| `config_ngen.yaml` | NGEN — excluded member; optional (`symfluence binary install ngen`) |
| `config_vic.yaml` | VIC — excluded member; optional (`symfluence binary install vic`) |

All 19 configs share the domain `Bow_at_Banff_lumped_era5` and run the full
pipeline (domain definition → observations → ERA5 acquisition → preprocessing
→ `run_model` → `calibrate_model`).

## Run

```bash
CFG=examples/paper_case_studies/configs/02_model_ensemble

# One model:
symfluence workflow run --config $CFG/models/config_summa.yaml

# All 19:
for f in $CFG/models/config_*.yaml; do
    symfluence workflow run --config "$f"
done
```

The first config downloads and preprocesses ERA5 for the shared domain
(~1–2 h); the remaining 18 skip straight to model-specific work via stage
markers. Compiled models must be installed first (`symfluence binary install
...` — see top-level README §1.2).

## Outputs

Under `SYMFLUENCE_data/domain_Bow_at_Banff_lumped_era5/`:

- `simulations/run_1/` — model output (all configs use `experiment_id: run_1`)
- `optimization/` — DDS iteration history and best parameters per model
- `evaluation/` — calibration/evaluation metrics

## Verify (Fig 7 reference values, evaluation KGE)

| Model | KGE | Model | KGE |
|---|---|---|---|
| LSTM | 0.90 | FUSE | 0.84 |
| SUMMA | 0.87 | TOPMODEL | 0.84 |
| PRMS | 0.85 | GR4J | 0.83 |

Pooled ensemble mean KGE 0.90, median 0.88. Runs are seeded, so ±0.01–0.02
deviations across platforms are expected.

## Runtime

0.5–8 h per model (1,000 model evaluations; JAX models are fastest, SUMMA/
MESH/SWAT slowest), plus the one-time ERA5 preprocessing.
