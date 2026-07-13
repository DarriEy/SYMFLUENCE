# Reproducing the Paper 3 experiments

Configuration files for the experiments in *From Configuration to Prediction:
Multi-Model, Multi-Basin Experiments with SYMFLUENCE* (Paper 3 of the
SYMFLUENCE series). Every config is standalone and declares the workflow steps
it needs; reproducing any experiment is one command:

```bash
symfluence workflow run --config <config.yaml>
```

## 1. Setup

### 1.1 Install everything

```bash
git clone -b develop https://github.com/symfluence-org/SYMFLUENCE.git
cd SYMFLUENCE
./scripts/symfluence-bootstrap --paper-repro
```

That one command creates the Python environment, installs SYMFLUENCE and all
its models (including the JAX-native ones), sets up GDAL/R/NetCDF, and compiles
exactly the 13 model binaries these experiments use — RHESSys with the paper's
subsurface-GW patch. Expect 30–60 minutes, mostly compilation.

(`--paper-repro` differs from a plain `--install` only in which binaries get
built: 13 instead of all 26.)

Prerequisites: Python 3.11/3.12, a C/C++/Fortran toolchain and CMake
(`brew install gcc cmake` / `apt install build-essential gfortran cmake`), and
— for the GR4J member only — R ≥ 4 with `airGR`
(`Rscript -e 'install.packages("airGR")'`).

Running the MESH member additionally needs two packages that are not on PyPI
(the extra pins keep meshflow from pulling pandas 3):

```bash
pip install git+https://github.com/kasra-keshavarz/hydrant.git
pip install git+https://github.com/CH-Earth/meshflow.git@main "pandas>=2.0,<3" "pint-pandas<0.8"
```

**Check** — all three must succeed:

```bash
symfluence workflow list-steps
symfluence binary doctor
python -c "from osgeo import gdal; import jax; import jsacsma"
```

### 1.2 Credentials

| Provider | Needed for | Setup |
|---|---|---|
| NASA Earthdata | experiment 06 only, and only for the JPL GRACE mascons the paper used | credentials in `~/.netrc` ([urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov)); to run credential-free, set `evaluation.grace.product` to the public CSR solution |
| ERA5 (ARCO), AORC, RDRS, CONUS404, SNOTEL, WSC | forcing + observations | public, no credentials |

### 1.3 Data location

Everything is written to `SYMFLUENCE_data/`, created as a sibling of the cloned
repo; override with `export SYMFLUENCE_DATA_DIR=/path`. Budget several tens of
GB for the full suite. Configs are location-independent (run from any
directory).

## 2. Experiments

| Directory | Paper section | Figure | Configs |
|---|---|---|---|
| `configs/01_domain_definition` | §2.1 | Figs 1–3, Table 1 | 14 |
| `configs/02_model_ensemble` | §4.2.1 | Fig 7 | 17 |
| `configs/03_forcing_ensemble` | §4.1 | Fig 6 | 4 |
| `configs/04_calibration_ensemble` | §4.2.3 | Fig 9 | 130 |
| `configs/05_benchmarking` | §4.2.2 | Fig 8 | 1 |
| `configs/06_multivariate_evaluation` | §4.2.4 | Fig 10 | 4 |
| `configs/07_data_pipeline` | §2.2 | Fig 4 | 1 |
| `configs/08_parallel_scaling` | §5 | Fig 11 | 14 (laptop row; cluster row = README recipe) |

Each directory has a README with its config list and the reference values from
the manuscript.

## 3. Running

```bash
CFG=examples/paper_case_studies/configs

# single experiment
symfluence workflow run --config $CFG/05_benchmarking/config_bow_benchmark.yaml

# a whole set
for f in $CFG/02_model_ensemble/models/config_*.yaml; do
    symfluence workflow run --config "$f"
done
```

Run configs within one experiment sequentially — they share a domain, and
completed steps (downloads, preprocessing) are reused via stage markers, so
only the first config pays the acquisition cost. Different experiments can run
concurrently, except 02/05/12 which share the Bow lumped ERA5 domain.

Rough costs (Apple M3 Pro): domain definitions minutes each once attribute
data is cached (first config per region pays the downloads); one calibration
config 10 min–2 h depending on model; the full 130-run experiment 04 is a
multi-day serial job.

Results land in `SYMFLUENCE_data/domain_<name>/` — `simulations/`,
`optimization/` (per-iteration CSV + best parameters), `evaluation/`, and
run logs + provenance manifests in `_workLog_<name>/`.
`symfluence workflow status --config <file>` shows step completion;
`symfluence workflow diagnose --config <file>` helps on failures.

## 4. Troubleshooting

| Symptom | Fix |
|---|---|
| `GDAL Python bindings (osgeo) are required` | rerun `./scripts/symfluence-bootstrap --paper-repro` (it provisions GDAL), or install system GDAL + `pip install "GDAL==$(gdal-config --version)"` |
| `executable not found` | `symfluence binary install --paper-repro --force`, then `symfluence binary doctor` |
| RHESSys calibration stuck near KGE 0.15 | binary built without the SYMFLUENCE patch — `symfluence binary install rhessys --patched --force` |
| calibration log shows `Best: 9999.0` | the objective never saw valid observations — check the obs step in the log |
| warning: h5py/netCDF4 bundle different libhdf5 | known pip-wheel quirk, mitigated automatically, ignore |
| fresh start for one experiment | delete `SYMFLUENCE_data/domain_<name>/` and rerun |

Issues: https://github.com/symfluence-org/SYMFLUENCE/issues

## 5. Provenance

`experiment_logs/` holds the curated logs, frozen resolved configs, and
run manifests (git commit, package versions, platform) of the runs behind the
paper's figures; `experiment_logs/COVERAGE.md` maps each log to its
experiment. The shipped configs are kept aligned to those frozen records.
