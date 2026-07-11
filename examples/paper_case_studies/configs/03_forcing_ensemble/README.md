# Experiment 03 — Forcing Ensemble (§4.1, Fig 6)

SUMMA is calibrated at the Paradise SNOTEL station (Mt. Rainier, WA) four
times, identically except for the meteorological forcing product. Calibration
targets daily snow water equivalent (SWE, RMSE objective) with DDS and 1,000
iterations; calibration water years 2016–2018 (2015-10-01 – 2018-09-30),
evaluation WY2019–2020 (2018-10-01 – 2020-09-30). The experiment isolates how
much forcing choice alone changes both performance and the calibrated
parameters (Fig 6).

## Configs (`forcings/`)

| Config | Forcing product |
|---|---|
| `config_era5.yaml` | ERA5 global reanalysis (ARCO cloud mirror, public) |
| `config_rdrs.yaml` | RDRS (ECCC regional reanalysis) |
| `config_aorc.yaml` | AORC (NOAA analysis of record) |
| `config_conus404.yaml` | CONUS404 (4 km WRF reanalysis) |

All four share the point domain `paradise_snotel_wa` and run the full
pipeline through `calibrate_model`; each acquires its own forcing product
(all four products are public; ERA5 is read from the ARCO cloud mirror).

## Run

```bash
CFG=examples/paper_case_studies/configs/03_forcing_ensemble

# One product:
symfluence workflow run --config $CFG/forcings/config_aorc.yaml

# All four:
for f in $CFG/forcings/config_*.yaml; do
    symfluence workflow run --config "$f"
done
```

## Outputs

Under `SYMFLUENCE_data/domain_paradise_snotel_wa/`:

- `simulations/forcing_ensemble_<product>/` — SUMMA output per forcing
- `optimization/` — DDS iteration history and best parameters per experiment
- `evaluation/` — SWE metrics against SNOTEL observations

## Verify (Fig 6 reference values)

- Evaluation KGE (daily SWE) spans **0.60 (RDRS)** to **0.86 (AORC)**.
- Calibrated frozen-precipitation multiplier spans **0.76 (CONUS404)** to
  **3.89 (RDRS)** — the parameter absorbs each product's snowfall bias
  (Fig 6c).

## Runtime

1–2 h per config (forcing download + 1,000 SUMMA point runs). SUMMA must be
installed (`symfluence binary install summa sundials`).
