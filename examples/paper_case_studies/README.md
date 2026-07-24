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
git clone -b <paper-release-tag> https://github.com/symfluence-org/SYMFLUENCE.git
cd SYMFLUENCE
./scripts/symfluence-bootstrap --paper-repro
```

Replace ``<paper-release-tag>`` with the version named in the paper's software
citation. Using that tag, rather than the moving ``develop`` branch, fixes the
code and configurations to the archived paper release.

That one command creates the Python environment, installs SYMFLUENCE and all
its models (including the JAX-native ones), sets up GDAL/R/NetCDF, and compiles
exactly the 13 model binaries these experiments use — RHESSys with the paper's
subsurface-GW patch. Expect 30–60 minutes, mostly compilation.

(`--paper-repro` differs from a plain `--install` in three ways: it builds 13
binaries instead of all 26, it installs the `[jax,ml]` extras the paper
ensemble needs, and it treats missing system build dependencies as fatal rather
than as a warning — a reproducibility bundle is all-or-nothing.)

Prerequisites: Python 3.11/3.12 plus the compilers and geospatial/netCDF
development libraries the 13 binaries build against. A C/C++/Fortran toolchain
alone is *not* enough — SUMMA, FUSE, mizuRoute, MESH and mHM need
netCDF-Fortran, RHESSys needs GDAL/PROJ/GEOS, TauDEM needs MPI, and building
the GDAL Python bindings needs the Python development headers:

```bash
# Debian/Ubuntu
sudo apt-get install -y build-essential gfortran cmake python3-dev \
    libnetcdf-dev libnetcdff-dev libgdal-dev libproj-dev proj-bin \
    libgeos-dev libopenmpi-dev openmpi-bin libopenblas-dev libudunits2-dev

# macOS
brew install gcc cmake netcdf netcdf-fortran gdal proj geos open-mpi openblas udunits
```

`--paper-repro` checks for all of these before it clones or compiles anything
and stops with the exact missing list, so a missing library costs seconds
rather than an hour of build time.

The GR4J member additionally needs R ≥ 4 with `airGR`. Install it somewhere on
R's default library path so `library(airGR)` resolves without extra setup:

```bash
Rscript -e 'install.packages("airGR", repos="https://cloud.r-project.org")'
Rscript -e 'library(airGR)'   # must succeed
```

MESH support (`hydrant` + `meshflow`, neither on PyPI) and the JAX/LSTM model
stacks are installed by `--paper-repro` automatically; no manual pip steps are
needed. Note that the `hydrant` name on PyPI belongs to an unrelated Kinesis
Firehose tool, so if you install meshflow by hand, install
`git+https://github.com/kasra-keshavarz/hydrant.git` first and constrain
`"pandas>=2.0,<3" "pint-pandas<0.8"` — pint-pandas 0.8 requires pandas 3 and
will otherwise break SYMFLUENCE's own `pandas<3` pin.

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
concurrently, except 02 and 05, which share the Bow lumped ERA5 domain.

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
| warning: h5py/netCDF4 bundle different libhdf5 | pip-wheel quirk: each wheel vendors its own libhdf5. SYMFLUENCE's own I/O is guarded, but third-party libraries that open netCDF files directly can still crash (see next row) — install both from one source: `pip uninstall h5py netCDF4 -y && conda install h5py netcdf4`, or rerun `./scripts/symfluence-bootstrap --paper-repro` |
| segfault / bus error in `xr.open_dataset` during forcing remapping (EASYMORE `__weighted_average`, common on macOS) | the libhdf5 conflict above hitting EASYMORE, which opens files outside SYMFLUENCE's guard — apply the same fix, and confirm the startup warning is gone (`--debug` re-shows it) |
| fresh start for one experiment | delete `SYMFLUENCE_data/domain_<name>/` and rerun |

Issues: https://github.com/symfluence-org/SYMFLUENCE/issues

## 5. Provenance

The repository tracks the provenance inventory and retrieval instructions in
[`provenance/`](provenance/). The larger bundle of curated logs, frozen
resolved configs, run manifests, and reference metrics is distributed as a
versioned GitHub Release asset and archived with the paper's Zenodo record.
This keeps generated logs out of Git while retaining a checksummed,
version-specific research artifact.

Maintainers create the archive with
`scripts/create_paper_provenance_bundle.sh`; see the provenance README for the
expected layout and release procedure.

## 6. Figure reproducibility scope

The experiment configurations reproduce the numerical workflows behind the
paper. Figure production has three explicitly different support levels:

- **Fully automated from experiment outputs:** Figures 1–3, 6–8, and 11 have
  staged plotting code. Figure 8 is exercised by the paper-release acceptance
  test using deterministic fixture data.
- **Automated with archived or separately staged inputs:** Figure 4 needs the
  pipeline-analysis JSON; Figure 9's staged code covers the HBV-by-algorithm
  slice but not the complete multi-model panel; Figure 10's domain panel needs
  the satellite subsets identified by `P3_MULTIVAR_ASSETS_DIR`.
- **Conceptual/manual:** Figure 5 is a framework schematic and has no
  data-driven plotting program.

The exact script, input, and current readiness of every figure are listed in
[`plotting/README.md`](plotting/README.md). A figure is not claimed as fully
reproducible unless all of its inputs and generation code are publicly
available.
