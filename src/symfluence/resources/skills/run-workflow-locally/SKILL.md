---
name: run-workflow-locally
description: >-
  Run the SYMFLUENCE workflow locally — full pipeline or single steps — via the
  `symfluence workflow` CLI (run/step/steps/resume/status/validate/clean). Covers
  the 16-step order + aliases, the config-hash stage-marker system (why a step is
  skipped or auto-re-runs), authoring/validating a config, and the domain_{NAME} layout.
when_to_use:
  - Setting up or running a SYMFLUENCE experiment (end to end or part of it)
  - '"Why did my step get skipped / re-run?", or resuming a partial run'
  - Authoring or validating a config, or understanding the pipeline order
---

# Running the SYMFLUENCE Workflow Locally

SYMFLUENCE runs one config-driven pipeline of 16 named steps. Each step is
tracked by a stage marker that stores a hash of the config sections it depends
on, so a step is skipped when already done **and** its config is unchanged. This
is the operator's guide. Code paths relative to `src/symfluence/`; run artifacts
under `SYMFLUENCE_DATA_DIR/domain_{NAME}/`.

## 1. The CLI surface (verified)

Entry point: `symfluence` (console script → `symfluence.main_cli:main`).
Command group `symfluence workflow <action>` (`cli/commands/workflow_commands.py`,
parser in `cli/argument_parser.py`). Actions:

```bash
symfluence workflow run        [--config C] [--force-rerun] [--continue-on-error] [--dry-run] [--debug]
symfluence workflow step  STEP [--config C] [--force-rerun]
symfluence workflow steps STEP [STEP ...] [--config C] [--force-rerun]
symfluence workflow resume STEP [--config C] [--force-rerun]
symfluence workflow status     [--config C]
symfluence workflow list-steps
symfluence workflow validate   [--config C]
symfluence workflow clean      [--level {intermediate,outputs,all}] [--dry-run] [--config C]
symfluence workflow diagnose   [--step STEP] [--config C]
```

- **run** — full pipeline in order; skips steps that are done & config-current
  (unless `--force-rerun`).
- **step STEP** — run one step; always executes it (no skip check).
- **steps S1 S2 …** — run the named steps, in the given order; always execute.
- **resume STEP** — run from STEP through the end of the pipeline.
- **status** — show which steps are complete for this config's domain.
- **list-steps** — print all steps + aliases (no config needed).
- **validate** — check the config file is valid.
- **clean** — remove intermediate / output files (`--dry-run` to preview).
- **diagnose** — diagnostic plots over existing outputs.

**Important flag facts:**
- The force flag is **`--force-rerun`** (not `--force`). It clears markers and
  re-executes. Config equivalent: `FORCE_RUN_ALL_STEPS: true`.
- **There is no `--start-from` / `--stop-at`.** To run "from a step onward" use
  `resume STEP`; to run a specific subset use `steps S1 S2 …`.
- `--config` default is `./config.yaml`.
- `--dry-run` (on `run`) prints the plan without executing — use it first.
- `--continue-on-error` (on `run`) sets `STOP_ON_ERROR=False` so a failing step
  doesn't halt the pipeline. Other shared flags: `--debug`, `--visualise`,
  `--diagnostic`, `--profile [--profile-output PATH] [--profile-stacks]`.

## 2. The 16 canonical steps (verbatim, in order)

Source: `workflow_steps.py` `WORKFLOW_STEP_ITEMS`. Run `symfluence workflow
list-steps` for the live list + aliases.

```
 1. setup_project                  Initialize project directory structure and shapefiles
 2. create_pour_point              Create pour point shapefile from coordinates
 3. acquire_attributes             Download/process geospatial attributes (soil, land class, DEM)
 4. define_domain                  Define hydrological domain boundaries and river basins
 5. discretize_domain              Discretize domain into HRUs / modeling units
 6. process_observed_data          Process observational data (streamflow, etc.)
 7. acquire_forcings               Acquire meteorological forcing data
 8. model_agnostic_preprocessing   Model-agnostic preprocessing of forcing + attributes
 9. build_model_ready_store        Build model-ready forcing/attributes/observations store
10. model_specific_preprocessing   Setup model-specific input files and configuration
11. run_model                      Execute the hydrological model simulation
12. postprocess_results            Postprocess and finalize model results
13. calibrate_model                Run model calibration / parameter optimization
14. run_benchmarking               Benchmark against observations
15. run_decision_analysis          Decision analysis for model comparison
16. run_sensitivity_analysis       Sensitivity analysis on model parameters
```

Common **aliases** accepted anywhere a STEP is expected (resolved by
`resolve_workflow_step_name`): `setup`→setup_project, `pp`/`pour_point`→create_pour_point,
`attrs`/`attributes`→acquire_attributes, `domain`→define_domain,
`discretize`→discretize_domain, `obs`→process_observed_data,
`forcings`→acquire_forcings, `map`/`agnostic_prep`→model_agnostic_preprocessing,
`store`→build_model_ready_store, `msp`/`specific_prep`→model_specific_preprocessing,
`model`→run_model, `post`/`postprocess`→postprocess_results,
`cal`/`calibrate`→calibrate_model, `bench`→run_benchmarking,
`sa`/`sensitivity`→run_sensitivity_analysis.

A "baseline run, no calibration" is steps 1–12; calibration/analysis is 13–16.

## 3. Stage markers — completion tracking with config hashing

Each finished step writes a marker (`core/stage_marker.py`):
```
SYMFLUENCE_DATA_DIR/domain_{DOMAIN_NAME}/.symfluence/stage_markers/{step_name}.json
```
(`_marker_dir = project_dir/".symfluence"/"stage_markers"`,
`_marker_path = _marker_dir/f"{step_name}.json"`). Marker filenames use the same
step names as the CLI/list-steps (e.g. `run_model.json`,
`model_specific_preprocessing.json`).

A marker stores: `stage`, `completed_utc`, **`config_hash`** (SHA-256 of the
config sections that step depends on), `symfluence_version`, `git_commit`.

**Skip decision** (orchestrator): a step runs when
`force_run OR not output_exists OR not marker_current`, where `marker_current`
means the marker exists AND its hash matches the current config. So:
- Step done + config unchanged → **skipped**.
- **Edit a relevant config section → that step (and dependents) auto-re-run** —
  no need to delete markers. `STAGE_CONFIG_SECTIONS` maps each step to its
  sections, e.g. `acquire_forcings: [forcing, domain]`,
  `run_models: [model, domain]`, `calibrate_model: [optimization, model,
  evaluation, domain]`. (The `STAGE_CONFIG_SECTIONS` key for the model-run step is
  `run_models`, plural — the CLI step name is `run_model`.)
- `--force-rerun` / `FORCE_RUN_ALL_STEPS` ignores markers entirely (clears them).

**Re-run a single step manually** (e.g. output looks wrong but config didn't
change): force just that step, or delete its marker:
```bash
symfluence workflow step run_model --config config.yaml --force-rerun
# or
rm "$SYMFLUENCE_DATA_DIR/domain_Bow_at_Banff/.symfluence/stage_markers/run_model.json"
symfluence workflow run --config config.yaml
```
**Start fresh:** `rm -rf .../domain_{NAME}/.symfluence/stage_markers/` (on-disk
artifacts remain; some steps also detect existing outputs). Inspect progress with
`symfluence workflow status`.

## 4. Minimal config to run

Templates in `resources/config_templates/`:
- `config_quickstart_minimal.yaml` (+ `_nested`) — the 10 required keys; start here.
- `config_template.yaml` — standard, documented.
- `config_template_comprehensive.yaml` (+ `_nested`) — all options.
- dataset presets: `camelsspat_template.yaml`, `fluxnet_template.yaml`,
  `norswe_template.yaml`.

The 10 **required** keys (verbatim from the minimal template):
```yaml
SYMFLUENCE_DATA_DIR: "/path/to/data"     # where domains live (§5)
SYMFLUENCE_CODE_DIR: "/path/to/code"     # the code checkout
DOMAIN_NAME: "MyBasin"                   # names the domain_{NAME} dir
EXPERIMENT_ID: "run_001"                 # names the simulations subdir
EXPERIMENT_TIME_START: "2010-01-01 00:00"
EXPERIMENT_TIME_END:   "2020-12-31 23:00"
DOMAIN_DEFINITION_METHOD: "lumped"       # lumped | point | semidistributed | distributed
SUB_GRID_DISCRETIZATION: "elevation"     # lumped | elevation | aspect | soilclass | landclass | grus ...
FORCING_DATASET: "ERA5"                  # registered data handler key
HYDROLOGICAL_MODEL: "SUMMA"              # registered model key
```
Plus model-specific keys the chosen model needs (e.g. SUMMA: `SUMMA_INSTALL_PATH`,
`SUMMA_EXE`, `SETTINGS_SUMMA_PATH` — all may be `"default"`). Optional but common:
`POUR_POINT_COORDS`, `BOUNDING_BOX_COORDS`, `CALIBRATION_PERIOD`,
`EVALUATION_PERIOD`, `ROUTING_MODEL`, `FORCE_RUN_ALL_STEPS`, optimization keys.

**Authoring a config:** copy a template (`config_quickstart_minimal.yaml` to
start, `config_template.yaml` for the documented standard), edit the keys above,
then `symfluence workflow validate --config my_config.yaml` before a long run.
The config is the public contract — keys are validated against the Pydantic
schema; unknown keys warn (strict mode errors). Legacy `CONFLUENCE_*` spellings
are auto-aliased (`core/config/legacy_aliases.py`).

Notes:
- `FORCING_DATASET` / `HYDROLOGICAL_MODEL` must be **registered keys** — run
  `symfluence list forcings` / `symfluence list models` to see the available names
  (see the explore-platform skill), and add-data-handler / add-model-handler to add
  new ones.
- `SYMFLUENCE_DATA_DIR`/`CODE_DIR` may be `"default"` (resolved relative to the
  project at runtime); set explicit absolute paths to avoid surprises.
- Env vars override file values (`SYMFLUENCE_*`, e.g.
  `export SYMFLUENCE_DOMAIN_NAME=test`).

## 5. Output directory layout

A run populates `SYMFLUENCE_DATA_DIR/domain_{DOMAIN_NAME}/`:
```
.symfluence/stage_markers/    step completion markers (JSON, §3)
shapefiles/                   pour point, catchment, river network, HRU/GRU
data/attributes/              DEM, soil, landcover, climate
data/forcing/                 acquired + processed meteorological forcing
data/observations/            streamflow & other obs
data/model_ready/             CF-1.8 model-ready store
settings/{MODEL}/             generated model config files
simulations/{EXPERIMENT_ID}/  model run outputs (per experiment)
optimization/{MODEL}/         calibration results
results/{EXPERIMENT_ID}_results.csv   standardized streamflow output
cache/                        cached intermediates
```
(Modern layout prefers `data/{subdir}`; legacy `{subdir}` is still accepted.)

## 6. Common recipes

```bash
# See the plan without doing anything
symfluence workflow run --config config.yaml --dry-run

# Full pipeline
symfluence workflow run --config config.yaml

# Baseline run only (through model output), no calibration
symfluence workflow resume setup_project --config config.yaml   # then Ctrl-C after run_model, OR:
symfluence workflow steps setup_project create_pour_point acquire_attributes define_domain \
    discretize_domain process_observed_data acquire_forcings model_agnostic_preprocessing \
    build_model_ready_store model_specific_preprocessing run_model --config config.yaml

# Re-run from preprocessing onward (after changing forcing/model setup)
symfluence workflow resume model_specific_preprocessing --config config.yaml --force-rerun

# Run a specific subset
symfluence workflow steps acquire_forcings run_model --config config.yaml

# Force one step to re-run
symfluence workflow step run_model --config config.yaml --force-rerun

# Progress / what's left
symfluence workflow status --config config.yaml

# Validate config before a long run
symfluence workflow validate --config config.yaml
```
Note: editing a config section makes its dependent steps re-run automatically on
the next `run` (§3) — often you don't need `--force-rerun` at all.

If acquisition needs an interactive login (CDS / Earthdata), do that first; see
the credential helpers in the add-data-handler skill.

## 7. Troubleshooting

- **"My step keeps getting skipped."** It's done and its config section is
  unchanged. Use `step … --force-rerun`, or delete its marker (§3). Check
  `workflow status`.
- **"A step re-runs unexpectedly."** You changed a config key in that step's
  `STAGE_CONFIG_SECTIONS` (§3) → hash changed → re-run. Expected behavior.
- **"Unknown action `list`."** It's `list-steps`. And the force flag is
  `--force-rerun`, not `--force`. There's no `--start-from`/`--stop-at` (use
  `resume` / `steps`).
- **"Step name not recognized."** Use a name or alias from `workflow list-steps`
  (§2) — e.g. `acquire_forcings` (plural), `model_specific_preprocessing`.
- **"Model / forcing not found."** Config value must be a registered key — see
  add-model-handler / add-data-handler.
- **Inspect, don't guess:** `--dry-run` shows exactly what will run vs skip.

## 8. Key file reference

| Concern | File |
|---------|------|
| Workflow CLI actions / handlers | `cli/commands/workflow_commands.py` |
| Workflow arg parser | `cli/argument_parser.py` |
| Canonical steps + aliases + resolver | `workflow_steps.py` (`WORKFLOW_STEP_ITEMS`, `WORKFLOW_STEP_ALIASES`, `resolve_workflow_step_name`) |
| Top-level entry class | `project/system.py` (`SYMFLUENCE`: `run_workflow`, `run_individual_steps`; re-exported from `symfluence/__init__.py`) |
| Orchestrator (step loop, skip logic) | `project/workflow_orchestrator.py` (`define_workflow_steps`, `run_workflow`) |
| Per-step manager creation | `project/manager_factory.py` (`LazyManagerDict`) |
| Stage markers + config hashing | `core/stage_marker.py` (`write_marker`, `is_stage_current`, `clear_markers`, `STAGE_CONFIG_SECTIONS`) |
| `FORCE_RUN_ALL_STEPS` config field | `core/config/models/system.py` |
| Config templates | `resources/config_templates/{config_quickstart_minimal,config_template_comprehensive}.yaml` |
| Legacy key aliases (CONFLUENCE_*) | `core/config/legacy_aliases.py` |
| CLI entry | `symfluence.main_cli:main` (console script `symfluence`) |
```
