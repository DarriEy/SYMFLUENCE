# Reproducing "From Configuration to Prediction" (Paper 3)

This directory contains everything needed to reproduce the experiments in

> Eythorsson, D., et al. (2026). *From Configuration to Prediction: Multi-Model,
> Multi-Basin Experiments with SYMFLUENCE.* Water Resources Research, submitted.

Every experiment is a plain SYMFLUENCE configuration file. Every config is
**standalone** and **self-scoping**: it declares exactly the workflow steps it
needs, so reproducing any result is one command:

```bash
symfluence workflow run --config <config.yaml>
```

No script editing, no manual data downloads, no base-config assembly.

---

## 1. One-time setup

### 1.1 System prerequisites

- **Python 3.11 or 3.12**
- **GDAL** (library + headers; used for domain delineation):
  `brew install gdal` (macOS) / `apt install gdal-bin libgdal-dev` (Debian/Ubuntu)
- **Build toolchain** for the compiled models: C/C++/Fortran compilers and CMake
  (`brew install gcc cmake` / `apt install build-essential gfortran cmake`)
- **R ≥ 4** with the `airGR` package — only for the GR4J ensemble member
  (`Rscript -e 'install.packages("airGR")'`)

### 1.2 Install SYMFLUENCE (pinned version)

```bash
git clone https://github.com/symfluence-org/SYMFLUENCE.git
cd SYMFLUENCE
git checkout develop   # paper-pinned commit: see PINNED_VERSION.txt in this directory

python3 -m venv venv
source venv/bin/activate
pip install -e ".[jax]"                    # framework + JAX + the 5 JAX-native paper models
pip install "GDAL==$(gdal-config --version)"   # Python bindings matching your system GDAL
```

Running the MESH ensemble member (experiment 02 only) additionally needs two
packages that are not on PyPI, in this order:

```bash
pip install git+https://github.com/kasra-keshavarz/hydrant.git
pip install git+https://github.com/CH-Earth/meshflow.git@main
```

**Checkpoint** — both commands must succeed:

```bash
symfluence workflow list-steps
python -c "from osgeo import gdal; import jax; print('GDAL + JAX OK')"
```

### 1.3 Install model executables

The external (compiled) models are installed by SYMFLUENCE itself. Install only
what the experiments you plan to run require (see the matrix in §2), or
everything at once:

```bash
# Everything used by the paper (≈30–60 min of compilation):
symfluence binary install taudem sundials summa mizuroute fuse hype mesh crhm \
    gsflow mhm prms rhessys swat vic ngen

# Verify:
symfluence binary doctor
```

Executables land in `SYMFLUENCE_data/installs/<tool>/bin/` and are found
automatically — no config editing needed.

### 1.4 Credentials for data providers

| Provider | Needed for | Setup |
|---|---|---|
| Copernicus CDS | ERA5 (all Bow/Paradise experiments), CARRA (Iceland) | Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu), put the API key in `~/.cdsapirc` |
| NASA Earthdata | GRACE TWS, MODIS, SMAP (experiment 10) | Register at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov), add credentials to `~/.netrc` |
| AORC / RDRS / CONUS404 / SNOTEL / WSC | forcing + observations | Public — no credentials |

### 1.5 Where data goes

All inputs, model runs, and results are written to a single data directory,
`SYMFLUENCE_data/`, created **as a sibling of the cloned repo**. To put it
somewhere else:

```bash
export SYMFLUENCE_DATA_DIR=/path/with/space   # ~40 GB for the full suite
```

Everything below works from any working directory — paths in the configs are
location-independent.

---

## 2. The experiments

| Directory | Paper section | Figures | Configs | What it runs | Approx. time |
|---|---|---|---|---|---|
| `configs/01_domain_definition` | §2.1 | Figs 1–3, Table 1 | 14 | domain definition + discretization only | minutes each (Iceland: ~1 h) |
| `configs/02_model_ensemble` | §4.2.1 | Fig 7 | 19 | full pipeline + DDS calibration (1,000 iter) per model | 0.5–8 h per model |
| `configs/03_forcing_ensemble` | §4.1 | Fig 6 | 4 | SUMMA @ Paradise SNOTEL, one forcing product each | 1–2 h each |
| `configs/04_calibration_ensemble` | §4.2.3 | Fig 9 | 130 | 8 models × 17 algorithms (fixed seed) | 10 min – 2 h each |
| `configs/05_benchmarking` | §4.2.2 | Fig 8 | 1 | naive-predictor benchmarks (no model runs) | ~15 min |
| `configs/10_multivariate_evaluation` | §4.2.4 | Fig 10 | 4 | GRACE-TWS-constrained SUMMA calibration | 2–4 h each |
| `configs/11_data_pipeline` | §2.2 | Fig 4 | 3 | data acquisition → model-ready store only | 0.5–2 h each |
| `configs/12_parallel_scaling` | §5 | Fig 11 | 35 | calibration scaling; 14 laptop configs, 20 HPC configs | varies |

Each subdirectory has its own README with per-experiment details and expected
results.

---

## 3. Running the experiments

Run any single experiment:

```bash
symfluence workflow run --config examples/paper_case_studies/configs/01_domain_definition/config_bow_lumped.yaml
```

Because configs within an experiment share a domain, **preprocessing runs once
and is reused**: the first config of experiment 02 downloads and preprocesses
ERA5 for Bow at Banff (~1–2 h); the remaining 18 skip straight to
model-specific work. Completed steps are tracked with stage markers and never
repeat unless the config changed.

### Suggested order

```bash
CFG=examples/paper_case_studies/configs

# 1. Domain definition — fast, verifies your geospatial stack end to end
for f in $CFG/01_domain_definition/config_bow_*.yaml; do
    symfluence workflow run --config "$f"
done

# 2. Model ensemble (Fig 7) — one command per model
for f in $CFG/02_model_ensemble/models/config_*.yaml; do
    symfluence workflow run --config "$f"
done

# 3. Benchmarking (Fig 8) — reuses experiment 02's domain, no model runs
symfluence workflow run --config $CFG/05_benchmarking/config_bow_benchmark.yaml

# 4. Forcing ensemble (Fig 6)
for f in $CFG/03_forcing_ensemble/forcings/config_*.yaml; do
    symfluence workflow run --config "$f"
done

# 5. Calibration algorithm comparison (Fig 9) — 130 runs, the long haul
for f in $CFG/04_calibration_ensemble/*/config_bow_*.yaml; do
    symfluence workflow run --config "$f"
done

# 6. Multivariate calibration (Fig 10)
for f in $CFG/10_multivariate_evaluation/config_*.yaml; do
    symfluence workflow run --config "$f"
done
```

### Finding results

Per experiment `<domain>` and `<experiment_id>` (both set in each config):

```
SYMFLUENCE_data/domain_<domain>/
├── simulations/<experiment_id>/        # model output
├── optimization/                       # calibration iterations + best params
├── evaluation/                         # metrics, benchmark_scores.csv
└── _workLog_<domain>/                  # per-run logs and provenance
```

Quick sanity checks: `symfluence workflow status --config <config.yaml>` shows
step completion; calibration progress is in
`optimization/<experiment_id>_parallel_iteration_results.csv`.

---

## 4. What to expect (verification checkpoints)

| Experiment | Published reference value |
|---|---|
| 02 ensemble | Evaluation KGE: LSTM 0.90, SUMMA 0.87, PRMS 0.85, FUSE 0.84 … (Fig 7b); NGEN and VIC calibrate poorly by design (excluded from pooled ensemble) |
| 03 forcing | Evaluation KGE: AORC 0.86 … RDRS 0.60; RDRS frozen-precip multiplier ≈ 3.9 (Fig 6c) |
| 04 calibration | Top tier DE/CMA-ES/DDS/ADAM mean calibration KGE 0.867–0.872; ADAM mean evaluation KGE 0.769 vs DDS 0.705 on JAX models (Fig 9) |
| 05 benchmark | Daily-median benchmark evaluation KGE ≈ 0.80 (Fig 8) |
| 10 multivariate | Q-only: KGE 0.88 / TWS r 0.75; joint: KGE 0.87 / TWS r 0.84 (Fig 10) |

Small numerical deviations (±0.01–0.02 KGE) across platforms are expected;
calibration runs are seeded (`random_seed: 42`) so trajectories are
deterministic on a given platform.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `unknown step '<name>'` at startup | Typo in a hand-edited `workflow_steps` list — run `symfluence workflow list-steps` for valid names |
| CDS download hangs or 403s | `~/.cdsapirc` missing/stale, or dataset licence not yet accepted on the CDS website (one-time click per dataset) |
| `executable not found` for a model | `symfluence binary install <tool>`, then `symfluence binary doctor` |
| `GDAL Python bindings (osgeo) are required` | `pip install "GDAL==$(gdal-config --version)"` after installing system GDAL (§1.1) |
| Warning: h5py and netCDF4 bundle different libhdf5 | Known pip-wheel quirk; SYMFLUENCE disables the conflicting fallback automatically — safe to ignore |
| A step reruns that you expected to skip | The config section governing that step changed — stage markers hash the config; this is intentional |
| Want a completely fresh start | Delete `SYMFLUENCE_data/domain_<name>/` (or the whole data dir) and rerun |
| Diagnosing a failed run | `symfluence workflow diagnose --config <config.yaml>`, logs in `_workLog_<domain>/` |

Issues: https://github.com/symfluence-org/SYMFLUENCE/issues

---

## 6. Provenance

`experiment_logs/` contains the curated run logs, frozen resolved configs, and
step-level summaries for the runs behind the paper's figures, plus manifests
recording git commit, package versions, and platform for each run
(`experiment_logs/COVERAGE.md` maps every log to its experiment).
