# Reproducing the Paper 3 experiments

Configuration files for the experiments in *From Configuration to Prediction:
Multi-Model, Multi-Basin Experiments with SYMFLUENCE* (Paper 3 of the
SYMFLUENCE series). Every config is standalone and declares the workflow steps
it needs; reproducing any experiment is one command:

```bash
symfluence workflow run --config <config.yaml>
```

## 1. Setup

### 1.1 System prerequisites

- Python 3.11 or 3.12
- GDAL library + headers: `brew install gdal` / `apt install gdal-bin libgdal-dev`
- C/C++/Fortran compilers and CMake for the compiled models:
  `brew install gcc cmake` / `apt install build-essential gfortran cmake`
- R ≥ 4 with `airGR` (GR4J member only): `Rscript -e 'install.packages("airGR")'`

### 1.2 Install SYMFLUENCE

```bash
git clone -b develop https://github.com/symfluence-org/SYMFLUENCE.git
cd SYMFLUENCE

python3 -m venv venv && source venv/bin/activate
pip install -e ".[jax]"
pip install "GDAL==$(gdal-config --version)"
```

MESH member only (not on PyPI; the pins prevent a pandas-3 conflict):

```bash
pip install git+https://github.com/kasra-keshavarz/hydrant.git
pip install git+https://github.com/CH-Earth/meshflow.git@main "pandas>=2.0,<3" "pint-pandas<0.8"
```

Check: `symfluence workflow list-steps` and
`python -c "from osgeo import gdal; import jax"` must both succeed.

### 1.3 Model executables

```bash
symfluence binary install taudem sundials summa mizuroute fuse hype mesh crhm \
    gsflow mhm prms rhessys swat
symfluence binary doctor
```

Compiles into `SYMFLUENCE_data/installs/` (~4 GB); found automatically.

### 1.4 Credentials

| Provider | Needed for | Setup |
|---|---|---|
| NASA Earthdata | experiment 10 only, and only for the JPL GRACE mascons the paper used | credentials in `~/.netrc` ([urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov)); to run credential-free, set `evaluation.grace.product` to the public CSR solution |
| ERA5 (ARCO), AORC, RDRS, CONUS404, SNOTEL, WSC | forcing + observations | public, no credentials |

### 1.5 Data location

Everything is written to `SYMFLUENCE_data/`, created as a sibling of the cloned
repo; override with `export SYMFLUENCE_DATA_DIR=/path`. Budget several tens of
GB for the full suite. Configs are location-independent (run from any
directory).

## 2. Experiments

| Directory | Paper section | Figure | Configs |
|---|---|---|---|
| `configs/01_domain_definition` | §2.1 | Figs 1–3, Table 1 | 14 |
| `configs/02_model_ensemble` | §4.2.1 | Fig 7 | 19 |
| `configs/03_forcing_ensemble` | §4.1 | Fig 6 | 4 |
| `configs/04_calibration_ensemble` | §4.2.3 | Fig 9 | 130 |
| `configs/05_benchmarking` | §4.2.2 | Fig 8 | 1 |
| `configs/10_multivariate_evaluation` | §4.2.4 | Fig 10 | 4 |
| `configs/11_data_pipeline` | §2.2 | Fig 4 | 2 |
| `configs/12_parallel_scaling` | §5 | Fig 11 | 35 (14 laptop, 20 need a cluster) |

Each directory has a README with its config list and the reference values from
the manuscript. Directories 06–09 were experiments cut from the final
manuscript.

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
| `GDAL Python bindings (osgeo) are required` | install system GDAL, then `pip install "GDAL==$(gdal-config --version)"` |
| `executable not found` | `symfluence binary install <tool>`, then `symfluence binary doctor` |
| calibration log shows `Best: 9999.0` | the objective never saw valid observations — check the obs step in the log |
| warning: h5py/netCDF4 bundle different libhdf5 | known pip-wheel quirk, mitigated automatically, ignore |
| fresh start for one experiment | delete `SYMFLUENCE_data/domain_<name>/` and rerun |

Issues: https://github.com/symfluence-org/SYMFLUENCE/issues

## 5. Provenance

`experiment_logs/` holds the curated logs, frozen resolved configs, and
run manifests (git commit, package versions, platform) of the runs behind the
paper's figures; `experiment_logs/COVERAGE.md` maps each log to its
experiment. The shipped configs are kept aligned to those frozen records.
