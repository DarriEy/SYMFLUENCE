# Paper 3 — Raw Configuration Files (configs_orig/)

This directory contains the original SYMFLUENCE configuration files used to produce
the results in **"From Configuration to Prediction: Multi-Model, Multi-Basin
Experiments with SYMFLUENCE"**.

Each subdirectory corresponds to one experiment or infrastructure demonstration in
the paper. Directories retain their original numbering; experiments that were removed
from the final manuscript (structural decision ensemble, sensitivity analysis,
large-sample and large-domain Iceland calibration) have been removed from this
archive as well, which is why the numbering has gaps.

## Directory Map

| Dir | Experiment | Paper Section | Configs | Description |
|-----|-----------|---------------|---------|-------------|
| `01_domain_definition/` | Domain definition | Section 2.1 | 4 | Point, lumped, semi-distributed, regional domains |
| `02_model_ensemble/` | Model ensemble | Section 4.2.1 | 28 | Multi-model intercomparison (Figure 7) |
| `03_forcing_ensemble/` | Forcing ensemble | Section 4.1 | 5 | ERA5, RDRS, AORC, CONUS404 at Paradise (Figure 6) |
| `04_calibration_ensemble/` | Calibration comparison | Section 4.2.3 | 130 | 17 algorithms × 8 models, fixed seed (Figure 9) |
| `05_benchmarking/` | Benchmarking | Section 4.2.2 | 1 | Reference-predictor comparison via HydroBM (Figure 8) |
| `10_multivariate_evaluation/` | Multivariate calibration | Section 4.2.4 | 8 | GRACE TWS + streamflow calibration (Figure 10) |
| `11_data_pipeline/` | Data pipeline | Section 2.2 | 4 | Automated data acquisition demos |
| `12_parallel_scaling/` | Parallel scaling | Section 5 | 93 | TauDEM, calibration, actors scaling (Figure 11) |

## Experiment Details

### 01 — Domain Definition (4 configs)
- `config_paradise_summa_optimization.yaml` — Point-scale (Paradise SNOTEL, WA)
- `config_Bow_lumped_era5.yaml` — Lumped catchment (Bow at Banff, AB)
- `config_Bow_lumped_elev_sd_routing_era5.yaml` — Semi-distributed with elevation bands
- `config_iceland_tutorial.yaml` — Regional distributed (Iceland)

### 02 — Model Ensemble (28 configs)
One config per hydrological model for the Bow at Banff domain. Models included:
SUMMA, FUSE, jFUSE, HBV, HYPE, RHESSys, CRHM, PRMS, CLM, VIC, MESH, mHM,
SWAT, GR4J, SAC-SMA, XAJ, XAJ+Snow17, HEC-HMS, TOPMODEL, SUMMA+MODFLOW,
GSFLOW, CLM+ParFlow, ParFlow, WRF-Hydro, WFLOW, WATFLOOD, LSTM, NGEN.

The paper reports the 19-member subset that could be reproduced end-to-end from a
clean install (Figure 7b); the remaining configs are provided as starting points.

**Excluded:** MIKE SHE (proprietary), duplicate MESH elevation band variants.

### 03 — Forcing Ensemble (5 configs)
Paradise SNOTEL with SUMMA under the four forcing products compared in the paper
(ERA5, RDRS, AORC, CONUS404) plus the shared base configuration.

### 04 — Calibration Comparison (130 configs)
Organized by model subdirectory (`hbv/`, `hechms/`, `topmodel/`, `xinanjiang/`,
`sacsma/`, `fuse/`, `summa/`, `hype/`).

Each model has one config per applicable calibration algorithm (DDS, SCE-UA, PSO,
DE, CMA-ES, GA, SA, Nelder-Mead, Basin-Hopping, Bayesian Optimization, GLUE,
L-BFGS, Adam, ABC, NSGA-II, MOEA/D, DREAM), all with the fixed random seed (42)
reported in the paper — 130 valid model–algorithm combinations in total.

### 05 — Benchmarking (1 config)
Evaluates the ensemble against 12 reference predictors computed from observed
streamflow and precipitation (HydroBM).

### 10 — Multivariate Calibration (8 configs)
- `bow_grace_tws/` — Streamflow + GRACE TWS joint calibration (6 configs)
- `iceland_scf_trend/`, `paradise_sca_sm/` — additional multivariate demonstrations
  not reported in the paper

### 11 — Data Pipeline (4 configs)
Automated data acquisition for Paradise, Bow, and Iceland domains.

### 12 — Parallel Scaling (93 configs)
- `exp1_taudem/` — TauDEM watershed delineation scaling (46 configs)
- `exp2_calibration/` — DDS calibration parallelism (36 configs)
- `exp3_distributed/` — SUMMA actors distributed execution (7 configs)
- `base/` — Base configs for scaling experiments (4 configs)

## Notes

- These are the **original** configs with absolute paths from the development machine.
  For portable versions with relative paths, see `../configs_nested/`.
- All configs follow the SYMFLUENCE YAML schema (6 sections: Global, Geospatial,
  Model Agnostic, Model Specific, Evaluation, Optimization).
- Settings with value `default` use SYMFLUENCE's built-in defaults.
