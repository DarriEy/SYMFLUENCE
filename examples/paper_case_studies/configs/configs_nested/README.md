# Paper 3 — Minimal Portable Configurations (configs_nested/)

Portable, documented SYMFLUENCE configuration files for reproducing the experiments
in **"From Configuration to Prediction"**. These configs use relative paths and
only include non-default settings.

## Prerequisites

```bash
pip install symfluence
```

SYMFLUENCE requires Python 3.9+ and installs all model backends automatically.
Some models (SUMMA, mizuRoute, WRF-Hydro) require pre-compiled executables — see
the [SYMFLUENCE documentation](https://symfluence.readthedocs.io) for installation guides.

## Data Requirements

Each domain requires specific input data. SYMFLUENCE can automatically download
most datasets when `DATA_ACCESS: cloud` is set.

| Domain | Location | Data Sources | Approx. Size |
|--------|----------|-------------|-------------|
| Paradise Creek | WA, USA (46.78°N, 121.75°W) | ERA5, SNOTEL, AORC, NEX-GDDP | ~2 GB |
| Bow at Banff | AB, Canada (51.17°N, 115.57°W) | ERA5, RDRS, WSC streamflow, GRACE | ~5 GB |
| Iceland (LamaH-Ice) | Iceland-wide | CARRA, IMO streamflow | ~20 GB |

### Manual data setup

If not using cloud access, create a `data/` directory alongside each config:

```
data/
├── domain_[DOMAIN_NAME]/
│   ├── shapefiles/           # Catchment, river, pour point shapefiles
│   ├── forcing/              # Meteorological forcing data
│   ├── observations/         # Streamflow, SWE, GRACE observations
│   └── parameters/           # Soil, land cover, DEM data
```

## How to Run

Each experiment can be run with:

```bash
symfluence run <config.yaml>
```

For configs using base config inheritance (prefixed with `_base_`), specify both:

```bash
symfluence run --base _base_bow_lumped.yaml models/config_summa.yaml
```

### Quick start — run a single model

```bash
cd configs_nested/02_model_ensemble/
symfluence run --base _base_bow_lumped.yaml models/config_summa.yaml
```

### Run an entire ensemble

```bash
cd configs_nested/02_model_ensemble/
for config in models/*.yaml; do
    symfluence run --base _base_bow_lumped.yaml "$config"
done
```

## Directory Structure

```
configs_nested/
├── 01_domain_definition/          Section 2.1: domain scale configs
├── 02_model_ensemble/             Section 4.2.1: multi-model ensemble
│   ├── _base_bow_lumped.yaml      Shared base config
│   └── models/                    Per-model overrides (27 files)
├── 03_forcing_ensemble/           Section 4.1: 4 forcing products
│   ├── _base_paradise_summa.yaml  Shared base config
│   └── forcings/                  Per-forcing overrides (4 files)
├── 04_calibration_ensemble/       Section 4.2.3: 17 algorithms × 8 models
│   ├── _base_bow_calibration.yaml Shared base config
│   └── algorithms/                Per-model/algorithm configs
├── 05_benchmarking/               Section 4.2.2: benchmark comparison
├── 10_multivariate_evaluation/    Section 4.2.4: GRACE TWS calibration
├── 11_data_pipeline/              Section 2.2: Data processing
└── 12_parallel_scaling/           Section 5: Parallel execution
```

Directories retain their original numbering; experiments removed from the final
manuscript (06–09) have been removed from this archive, hence the gaps.

## Config Inheritance

Experiments 2, 3, and 4 use a **base config + override** pattern:

- `_base_*.yaml` contains shared settings (domain, time period, forcing, etc.)
- Model/forcing-specific configs override only the settings that differ
- This reduces duplication and makes the experimental design transparent

## Expected Outputs

| Experiment | Output | Runtime (approx.) |
|-----------|--------|-------------------|
| 01 Domain | Delineated shapefiles, forcing, model setup | 10-30 min |
| 02 Models | 27 configured model runs + evaluation metrics | 2-8 hrs per model |
| 03 Forcing | 4 calibrated SUMMA runs under different forcings | 1-2 hrs per forcing |
| 04 Calibration | 130 calibration trajectories (fixed seed) | 30 min - 2 hrs each |
| 05 Benchmark | Benchmark statistics table | 5 min |
| 10 Multivariate | 3 calibration strategies + evaluation | 2-4 hrs each |
| 11 Pipeline | Data downloads + preprocessing | 30 min - 2 hrs |
| 12 Scaling | Timing benchmarks at various core counts | Varies |

## Differences from Raw Configs

Compared to `../configs_orig/` (raw configs), these minimal configs:

1. **Replace absolute paths** with `./data/` and `./`
2. **Remove redundant defaults** — only non-default settings included
3. **Add documentation** — inline comments explain each setting
4. **Use config inheritance** — base configs reduce duplication
5. **Include generation scripts** — for large-sample experiment automation

## Paper Reference

Eythorsson, D., et al. (2026). From Configuration to Prediction: Multi-Model,
Multi-Basin Experiments with SYMFLUENCE. Water Resources Research, submitted.
