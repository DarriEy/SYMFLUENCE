# SYMFLUENCE Tutorial 04c — Logan River from the Command Line (SUMMA)

This workshop replicates the 04a Logan River notebook **without writing a single
line of Python**. Everything — configuration, watershed delineation, cloud data
acquisition, SUMMA execution, evaluation, and calibration — runs through the
`symfluence` CLI, which makes the workflow scriptable, reproducible, and ready
for HPC batch jobs.

A runnable version of every command below lives in
[`04c_logan_river_cli_workshop.sh`](04c_logan_river_cli_workshop.sh):

```bash
./04c_logan_river_cli_workshop.sh                    # setup → SUMMA run → results
RUN_CALIBRATION=1 ./04c_logan_river_cli_workshop.sh  # ...plus DDS calibration
```

**The domain:** Logan River above Logan, UT (USGS gauge 10109000) — a ~550 km²
snow-dominated mountain watershed in the Bear River Range. We model it as a
single lumped GRU with SUMMA, forced by RDRS reanalysis, over 2018–2021
(1 spin-up year, 1 calibration year, 1 evaluation year). The uncalibrated
pipeline — from empty directory to simulated hydrograph — takes about
15 minutes on a laptop, most of it cloud data acquisition; the SUMMA run
itself is seconds.

---

## Step 0 — Check your toolbox

```bash
symfluence doctor                    # system diagnostics: python env, binaries, paths
symfluence binary install summa      # build/install SUMMA if doctor says it's missing
symfluence binary summa --version    # pass-through: run the managed SUMMA binary directly
```

`symfluence binary <tool> [args...]` passes arguments straight to any managed
binary (SUMMA, FUSE, TauDEM tools, mizuRoute), so you never need to hunt for
install paths. Note that SUMMA writes a `*.log` file named after its arguments
into the current directory when invoked this way.

## Step 1 — Configuration

The notebook builds a `SymfluenceConfig` in Python. On the command line,
`project init` generates the same thing as a YAML file:

```bash
symfluence project init \
    --domain Logan_River_at_Logan \
    --model SUMMA \
    --forcing RDRS \
    --start-date 2018-01-01 \
    --end-date 2021-12-31 \
    --definition-method lumped \
    --discretization GRUs \
    --minimal
```

This writes `config_Logan_River_at_Logan.yaml` (≈60 lines) into the current
directory. `--minimal` keeps only the settings without workable defaults; drop
it for a fully-documented config, and see
`src/symfluence/resources/config_templates/` for the annotated reference
templates.

Now localize it for Logan River. Set the experiment ID and align the start
time with the hourly RDRS record:

```yaml
EXPERIMENT_ID: cli_workshop_1
EXPERIMENT_TIME_START: 2018-01-01 01:00
```

and append the watershed, data-source, and calibration block:

```yaml
# ── Logan River at Logan, UT (USGS 10109000) ─────────────────────────────────
POUR_POINT_COORDS: 41.743098/-111.786432
BOUNDING_BOX_COORDS: 42.15/-111.90/41.70/-111.40
LUMPED_WATERSHED_METHOD: TauDEM
SPINUP_PERIOD: 2018-01-01, 2018-12-31
CALIBRATION_PERIOD: 2019-01-01, 2019-12-31
EVALUATION_PERIOD: 2020-01-01, 2020-12-31

# Data sources (all cloud-hosted — no manual downloads)
DATA_ACCESS: cloud
DEM_SOURCE: copdem90
FORCING_MEASUREMENT_HEIGHT: 2.0

# Streamflow observations
STATION_ID: '10109000'
STREAMFLOW_DATA_PROVIDER: USGS
DOWNLOAD_USGS_DATA: true

# Calibration (Step 5b)
OPTIMIZATION_METHODS:
- iteration
OPTIMIZATION_TARGET: streamflow
ITERATIVE_OPTIMIZATION_ALGORITHM: DDS
OPTIMIZATION_METRIC: KGE
CALIBRATION_TIMESTEP: hourly
NUMBER_OF_ITERATIONS: 50
```

Then check your work — the validator catches typos and inconsistent settings
before you burn any compute:

```bash
symfluence workflow validate --config config_Logan_River_at_Logan.yaml
symfluence config resolve  --config config_Logan_River_at_Logan.yaml   # optional: see every resolved setting
```

## Step 2 — Domain definition

The full pipeline is 16 steps; list them (with their short aliases) any time:

```bash
symfluence workflow list-steps
```

Build the project skeleton, fetch terrain/soil/land-cover attributes from the
cloud, delineate the watershed with TauDEM, and discretize it into a single
lumped GRU:

```bash
symfluence workflow steps \
    setup_project create_pour_point acquire_attributes \
    define_domain discretize_domain \
    --config config_Logan_River_at_Logan.yaml
```

(Equivalent, for the alias-inclined: `symfluence workflow steps setup pp attrs domain discretize ...`)

Artifacts land under `$SYMFLUENCE_DATA_DIR/domain_Logan_River_at_Logan/`:
the delineated basin in `shapefiles/river_basins/`, the HRUs in
`shapefiles/catchment/`, and the DEM and attribute rasters in `attributes/`.

## Step 3 — Data acquisition and preprocessing

Download USGS streamflow, pull four years of RDRS forcing from the cloud, and
run the model-agnostic preprocessing (basin averaging, elevation lapse rates):

```bash
symfluence workflow steps \
    process_observed_data acquire_forcings model_agnostic_preprocessing \
    --config config_Logan_River_at_Logan.yaml
```

This is the long stage — RDRS arrives as monthly files that are remapped onto
the basin. Add `--debug` for verbose logs if a download misbehaves.

## Step 4 — Run SUMMA

Write SUMMA's input files (file manager, forcing list, attributes, cold state,
trial parameters), execute the simulation, and extract streamflow into a
standardized results CSV:

```bash
symfluence workflow steps \
    model_specific_preprocessing run_model postprocess_results \
    --config config_Logan_River_at_Logan.yaml
```

## Step 5 — Results

```bash
symfluence workflow status --config config_Logan_River_at_Logan.yaml
```

Simulated streamflow is in
`$SYMFLUENCE_DATA_DIR/domain_Logan_River_at_Logan/results/cli_workshop_1_results.csv`,
with the processed USGS observations alongside in
`data/observations/streamflow/preprocessed/`. Raw SUMMA NetCDF output is under
`simulations/cli_workshop_1/SUMMA/`. For quick diagnostic plots, re-run any
step with `--visualise`, or use `symfluence workflow diagnose` on the
completed outputs.

## Step 5b — Calibration

DDS optimization of SUMMA's snow and soil parameters against hourly KGE
(the config above uses 50 iterations for a fast demo; the notebook's 200 gives
a better optimum — expect one SUMMA run per iteration):

```bash
symfluence workflow step calibrate_model --config config_Logan_River_at_Logan.yaml
```

Progress and the best parameter set land in
`domain_Logan_River_at_Logan/optimization/`.

## Step 6 — Benchmarking

Compare SUMMA against reference predictors (climatology, persistence, …):

```bash
symfluence workflow step run_benchmarking --config config_Logan_River_at_Logan.yaml
```

---

## The one-liner

Once a config exists, the entire pipeline above is just:

```bash
symfluence workflow run --config config_Logan_River_at_Logan.yaml
```

Useful companions:

```bash
symfluence workflow run --dry-run ...          # show the execution plan without running
symfluence workflow resume --config ...        # pick up after an interruption
symfluence workflow step run_model --force-rerun ...   # redo one step
symfluence job submit --config ...             # same workflow as a SLURM job on HPC
```

## Switching models

The model-agnostic steps (2–3) don't change — swap the model at init time and
SYMFLUENCE generates the right model-specific configuration:

```bash
symfluence project init --domain Logan_River_at_Logan --model FUSE --forcing RDRS ...
```

`symfluence list models` shows everything available (SUMMA, FUSE, GR, HYPE,
MESH, NGEN, LSTM, ...).

## Troubleshooting

- **`workflow validate` fails** — the error lists unrecognized keys with
  did-you-mean suggestions; fix the spelling or remove the key.
- **A step fails midway** — logs are in
  `domain_Logan_River_at_Logan/_workLog_Logan_River_at_Logan/`; re-run the
  single step with `--debug`.
- **Missing binaries** — `symfluence binary doctor` diagnoses,
  `symfluence binary install summa taudem` fixes.
- **No network / workshop fallback** — unpack a pre-built
  `domain_Logan_River_at_Logan.zip` into `$SYMFLUENCE_DATA_DIR` and skip
  straight to Step 4.
