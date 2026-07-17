# Changelog

All notable changes to SYMFLUENCE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **The SYMFLUENCE agent interface** (`symfluence agent`): `agent launch` now
  primes the host coding-agent CLI (Claude Code, Codex, Gemini, ...) as the
  SYMFLUENCE agent instead of opening it bare. Four provider-agnostic layers,
  each wired through whatever mechanism the CLI declares in the launcher
  registry: (1) the packaged skills (unchanged); (2) an identity block with
  live project context (detected configs, domain directories) and operating
  house rules, delivered via the CLI's system-prompt flag or the `AGENTS.md`
  preamble; (3) a dependency-free MCP server (`symfluence agent mcp`) exposing
  `list_capabilities`, `validate_config`, `workflow_status`, and
  `run_workflow_step`; (4) packaged specialist subagents
  (`calibration-debugger`, `platform-scout`).
- New `agent` verbs: `agent list` (registered CLIs and which one launch picks),
  `agent skills` (packaged skills), `agent doctor` (full setup diagnosis), and
  `agent mcp` (serve the MCP server on stdio); `agent launch` gains `--cli`
  and `--no-skills`.
- **Agent Command Center**: `symfluence agent launch` now opens a dedicated
  Agent screen in the SYMFLUENCE TUI — runtime selector, mission context,
  capabilities, and preflight checks — and hands off to the chosen agent CLI
  only after the TUI exits (typed `AgentHandoff` contract, ready for a future
  embedded agent). `--direct` (or a one-shot prompt, no TTY, or a missing TUI
  extra) restores the immediate handoff. The screen is also mode `7` inside
  `symfluence tui launch`.

---

## [0.9.2] - 2026-07-07

Archival release accompanying the SYMFLUENCE paper series
("Out-standing in Every Field", "The Registry as Social Contract",
"From Configuration to Prediction"; Eythorsson et al., 2026, Water
Resources Research, submitted). First release archived on Zenodo via the
GitHub integration.

### Added
- **Citation metadata**: `CITATION.cff` (GitHub "Cite this repository") and
  `.zenodo.json` (Zenodo deposit metadata) with the full author list and
  ORCIDs (#269).
- Guard test validating every shipped paper calibration config against the
  strict config schema.

### Changed
- **Paper supplementary configs trimmed to the final manuscript** (#268):
  removed the four cut experiments (decision ensemble, sensitivity analysis,
  large-sample and large-domain Iceland calibration), reduced the forcing
  ensemble to the four reported products (ERA5, RDRS, AORC, CONUS404), and
  kept only the 130 fixed-seed (42) calibration combinations the paper
  reports; all READMEs updated to the manuscript's section and figure
  numbering.

### Fixed
- Shipped calibration configs for the five JAX-native models (HBV, HEC-HMS,
  SAC-SMA, TOPMODEL, Xinanjiang) failed strict validation due to stale keys
  from older plugin schemas (`initial_params`; HBV also `smoothing`,
  `smoothing_factor`) and could not be loaded; the stale keys are removed
  (85 configs).
- Paper-config seed guard repointed after the #268 trim removed the config
  it loaded (#270).

---

## [0.9.1] - 2026-06-14

> **Note**: Versions 0.8.0–0.9.0 were tagged and published without CHANGELOG
> entries. This section documents the changes since the v0.9.0 release; the
> 0.8.x/0.9.0 history is not reconstructed here.

### Added
- **Canonical forcing contract (CFIF)**: a single canonical forcing reader keyed
  to CF standard names (CFIF) that model preprocessors consume through one path,
  with automatic resampling of the source to each model's cadence. Adopted by
  SWAT, Wflow, IGNACIO, LISFLOOD, CWatM, PCR-GLOBWB, RHESSys, CLM-ParFlow, and mHM.
- **Model-ready data store for attributes and observations**: new attributes
  reader plus observation/streamflow loaders that source from the model-ready
  store. SUMMA soil/land class and HYPE GeoData (soil/land/elevation) now read
  from the store; `StreamflowMetrics` and the remaining calibration loaders are
  routed through it.
- **Windows binary builds** for many models — WATFLOOD (bundled Waterloo CHARM
  prebuilt), ParFlow, CLM-ParFlow, PRMS, VIC, PIHM, mHM, WRF-Hydro, and glacier —
  via MinGW/MSYS2 POSIX shims and CMake/NetCDF wiring.
- **WATFLOOD**: full Unix and Windows source builds; CHARM runs on the datastore;
  parallel-calibration plumbing.
- **GSFLOW**: datastore-driven inputs and a distributed-groundwater option.
- **mHM**: CARRA forcing support and distributed-morphology (MPR) mode.
- WATFLOOD and GSFLOW standalone postprocessors registered.

### Changed
- **Agent interface simplified to a launcher.** `symfluence agent launch` now hands
  off to an installed coding-agent CLI (Claude Code, Codex, Gemini, ...), primed with
  the SYMFLUENCE skills, instead of running a bespoke in-house LLM agent. It detects
  the CLI on `PATH` and uses whichever API key is set (`ANTHROPIC_API_KEY` /
  `OPENAI_API_KEY` / `GEMINI_API_KEY`); override with `SYMFLUENCE_AGENT_CLI`, skip
  skill materialization with `SYMFLUENCE_NO_SKILLS`. The skills now ship inside the
  package. *(net ~ −7,000 LOC)*
- **Config schema canonicalization (toward 1.0)**: promoted data-handler,
  HYPE/NGEN/FUSE/GR/mizuRoute/path, multi-gauge, optimizer/evaluator, and
  parameter-bounds flat keys to typed Pydantic fields; `RECOGNIZED_FLAT_KEYS` is
  now closed; the `state` and `data_assimilation` sections are wired into the
  flat-key transform; IGNACIO/GNN/LSTM schemas reconciled with the registered
  ones. Strict config-key validation is enforced and dogfooded on shipped configs.
- Documentation: root `ARCHITECTURE.md` at-a-glance, an ADR practice with the
  initial ADR set, and a threat model (`docs/security.md`).
- Evaluation: unified catchment-area resolution between trial and final-eval
  metrics (resolves the FUSE final-eval KGE divergence).
- Performance: RTI review item 20 hot-spots — rasterio hoist, Julian-date
  vectorization, coastal-partition rewrite — plus a CLI performance gate.
- Core layering restored: no `core` → upper-layer imports (RTI item 19).

### Fixed
- **Data acquisition**: MERIT-Basins region table and dead-host download; Daymet
  gridded OPeNDAP route (descending y, DAP2); HRRR LCC-grid windowing; CERRA
  longwave variable name; NEX-GDDP-CMIP6 synthetic-airpres survival and hardening;
  DEM merge pixel-grid snap; MSWEP remote path and default version; SMHI
  skip-if-raw-exists guard; EM-Earth S3 handler (403 visibility, credentials,
  month window); WSC GeoMet paging corruption and NLDAS-2 v2.0 variable names;
  USGS NWIS gauge-local → UTC timestamps; GRACE atomic `.part` downloads.
- **HYPE**: clone over `https://` with retry, replacing the deprecated
  `git://` SourceForge transport that intermittently broke nightly and release
  builds.
- **CLM** build: NetCDF library symlinks for PIO (Debian multiarch), git
  `safe.directory` re-assert after HOME change, `mpi-serial` pin, and Perl
  `XML::LibXML` install.
- Hardcoded `*3600` hourly-conversion bugs in the Wflow and IGNACIO forcing paths.
- **Windows builds**: WATFLOOD DLL staging, PIHM `timegm` guard, ParFlow
  `mkdir`/path compatibility, PRMS NetCDF path quoting and `gnu17`, VIC version
  macros, mHM NetCDF detection, and WRF-Hydro checkout.
- **Glacier**: earthaccess auth for NSIDC, RGI region-name transform, and
  coldstart fixes.
- **PRMS**: resilience-handler tracebacks (`exc_info=True`) and HRU parameters
  driven from the datastore.
- SWAT preprocessor registered via `core.registries.R` to avoid a circular import.

### Removed
- **Deprecated `register*` decorator registry API** — registration is now
  R-facade only. *(breaking)*
- **Deprecated `*Postprocessor` spelling aliases** — use `*PostProcessor`.
  *(breaking; external plugins must update)*
- **HYDROGEOSPHERE** model stub. *(breaking)*
- The **CalMIP** example stub.
- **DPE** (differentiable PE) and other dead/external config keys; shipped
  configs curated accordingly.
- **The bespoke in-house LLM agent** (`symfluence.agent` internals) and the
  optional `[llm]`/`openai` dependency — replaced by the launcher above.
  `symfluence agent start`/`run` are deprecated aliases for `launch`. *(breaking:
  the agent now requires an external coding-agent CLI)*

### CI / Release
- `release-binaries`: build MODFLOW 6 from source on Linux, build ngen on
  Linux/macOS, skip CLM on Windows (ESMF/CIME on MinGW unsupported), close
  presence-check holes and surface failed optional builds, and don't clobber the
  latest release on a tagless `workflow_dispatch`.
- Strict config-key validation dogfooded on shipped configs (kept out of the
  lint job, which has no `symfluence` install).
- Require the `PostProcessor`-spelling plugin releases.

---

## [0.7.1] - 2026-03-01

### Fixed
- **HPC file-locking workaround**: Added tempfile fallback (attempt 4) to ERA5
  NetCDF writer for parallel filesystems (Lustre/GPFS/BeeGFS) where HDF5
  `fcntl()` locking fails even with `HDF5_USE_FILE_LOCKING=FALSE`. Writes to
  `$TMPDIR` first, then moves the file to the target path.
- **mizuRoute Makefile build**: Collapse multi-line `LIBNETCDF` before appending
  RPATH flags to avoid breaking backslash continuation (caused "missing separator"
  errors on CI).
- **ngen linker (expat)**: Detect system UDUNITS2 via `pkg-config`/`ldconfig`
  (not just explicit env var) so `-lexpat` is added for XML symbol resolution.
- **Wflow calibration**: Fix snow parameters, routing, and unit conversion issues.
  Add Oudin PET, log-transform bounds, and sub-daily resampling support.
- **FUSE preprocessing**: Make streamflow observations optional.
- **MPI launcher**: Add fallback when preferred launcher fails; prefer `mpirun`
  over `srun` for launcher detection.

### Added
- SPDX license headers on all source files.
- Coverage tracking in cross-platform CI workflow.
- Unit tests for NGEN, JFUSE, HYPE model runners and configs.
- Unit tests for core modules (file_utils, validation, path_resolver).
- Unit tests for `_safe_to_netcdf` fallback chain.
- WATFLOOD: Expand calibration to 16 parameters with tests.
- Bow River preset and default `DEM_SOURCE` to Copernicus.
- Config-hash stage invalidation for workflow orchestrator.

---

## [0.7.0] - 2026-02-10

> **Note**: This release jumps from the last tagged release v0.5.2 directly to v0.7.0.
> Versions 0.5.3–0.6.0 were development milestones on the `develop` branch and were
> never tagged or published to `main`. Their changes are included in this release and
> documented below in the [0.6.0] and [0.5.x] sections for historical reference.

> **Breaking Change**: This release refactors the CLI to a subcommand architecture.
> All existing CLI commands will need to be updated.

### Added
- **New CLI Commands**
  - `symfluence gui launch` - Panel-based web GUI for workflow management
  - `symfluence data download|list|info` - Standalone dataset acquisition without a full project
  - `--shapefile` option for `symfluence data download` to derive bounding box from a shapefile

- **Documentation Improvements**
  - New CFuse model guide (experimental differentiable PyTorch-based FUSE)
  - New JFuse model guide (experimental JAX-based FUSE with gradient support)
  - New WM-Fire model guide (wildfire spread simulation with RHESSys)
  - Comprehensive CLI reference documentation with all commands and options
  - Expanded configuration parameter reference
  - Fixed Python API examples in calibration documentation
  - New testing guide for contributors with pytest patterns and CI integration
  - Added agent-assisted contribution workflow documentation

- **Agent Improvements**
  - Enhanced `show_staged_changes` with diff statistics and file summary
  - Improved `generate_pr_description` with auto-detection of modified files
  - Better PR descriptions with context-specific sections and testing checklists
  - Added documentation for fuzzy matching threshold parameter

- **HBV Module Refactoring**
  - New `losses.py` module with differentiable NSE/KGE loss functions and gradient utilities
  - New `parameters.py` module with parameter bounds, defaults, and scaling utilities
  - Lazy imports in `__init__.py` for optional JAX dependency
  - Reduced `model.py` by ~650 lines through modularization

- **HBV Parameter Regionalization**
  - Neural transfer functions for spatially-varying parameter estimation
  - Support for catchment attribute-based parameter prediction

- **HBV Calibration Improvements**
  - Hydrograph signature metrics for multi-objective calibration
  - Enhanced optimizer with adaptive learning rates

- New modular command structure in `src/symfluence/cli/`:
  - `argument_parser.py` - Main parser with subcommand structure
  - `validators.py` - Validation utilities
  - `commands/` directory with category-specific handlers

### Changed
- **Complete CLI Refactor**
  - Replaced flat flag-based interface with modern two-level subcommand architecture
  - New structure: `symfluence <category> <action>` instead of `symfluence --flag`
  - 9 command categories: workflow, project, binary, config, job, example, agent, gui, data
  - Eliminated complex mode detection logic
  - Archived old `cli_argument_manager.py` for reference

- **New CLI Structure**
  ```bash
  # Workflow commands
  symfluence workflow run [--config CONFIG]
  symfluence workflow step STEP_NAME
  symfluence workflow list-steps
  symfluence workflow status

  # Project commands
  symfluence project init [PRESET]
  symfluence project pour-point LAT/LON --domain-name NAME --definition METHOD
  symfluence project list-presets

  # Binary/tool commands
  symfluence binary install [TOOL...]
  symfluence binary validate
  symfluence binary doctor

  # Configuration commands
  symfluence config validate
  symfluence config validate-env
  symfluence config list-templates

  # Job commands
  symfluence job submit [WORKFLOW_CMD] [SLURM_OPTIONS]

  # Example commands
  symfluence example launch EXAMPLE_ID
  symfluence example list

  # Agent commands
  symfluence agent start
  symfluence agent run PROMPT

  # GUI commands
  symfluence gui launch

  # Data commands
  symfluence data download DATASET --bbox W S E N
  symfluence data list
  symfluence data info DATASET
  ```

- **HBV Sub-Daily Parameter Scaling**
  - Implemented exact exponential scaling for recession coefficients (K0, K1, K2)
  - Formula: `k_sub = 1 - (1 - k_daily)^(dt/24)` replaces linear approximation
  - Eliminates ~5-13% error in recession behavior at sub-daily timesteps
  - Flux rate parameters (CFMAX, PERC) continue to use linear scaling
  - Added `FLUX_RATE_PARAMS` and `RECESSION_PARAMS` constants

- **MESH Preprocessing Consolidation**
  - Moved configuration defaults to dedicated `config_defaults.py`
  - Streamlined `config_generator.py` and `meshflow_manager.py`
  - Expanded `parameter_fixer.py` with robust parameter handling
  - Updated NALCMS to CLASS land cover mapping (wetland, snow/ice corrections)
  - Added unit conversion utilities for forcing data
  - Improved logging (replaced debug prints with proper logger calls)

- **FLUXCOM ET Acquisition Rework**
  - Replaced stub ICOS downloader with fully functional `_download_from_icos()` using ICOS Carbon Portal metadata API
  - Smart variable matching (exact name, then substring, then fallback)
  - Automatic unit conversion (W/m2, mm/hr, mm/day)
  - Nearest-neighbor fallback for basins smaller than the grid cell

- **MODIS ET Acquisition Improvements**
  - Spatial subsetting via sinusoidal projection for large tiles with small domains
  - Improved special-value masking (all MODIS QC codes, not just fill value)
  - SDS name matching now tries exact match before substring search

- **Reporting Module Cleanup**
  - Simplified plotter implementations
  - Improved shapefile handling

- **Example Notebooks — Typed Config Migration**
  - Migrated notebooks 02a–03b from legacy `yaml.safe_load` + flat dict pattern to `SymfluenceConfig.from_minimal()` typed config API
  - All 9 example notebooks (01a–04b) now use the same typed, validated, frozen config pattern
  - Removed MAF-specific language from all example notebooks; acquisition comments now reference `DATA_ACCESS: 'cloud'` or `'maf'` config setting
  - Fixed pre-existing bugs: 02a `NameError` on config access, 03b `cf.managers` wrong variable name, 03b config overwrite bug, 03b outdated CLI syntax

### Fixed
- MESH lumped mode routing: switched from `run_def` to `noroute` mode to correctly preserve lower-zone baseflow (`wf_lzs`); `run_def` in MESH 1.5.6 bypasses `STGGW` storage, causing zero baseflow
- Traceback handling in MESH postprocessor
- Added missing `pydantic` runtime dependency to `pyproject.toml`
- Fixed invalid escape sequences in `__init__.py` warning filters for Python 3.12+ compatibility
- CI pipeline now correctly fails on unit/integration test regressions (removed `|| true` from pytest invocations)

### Documentation
- Added comprehensive sub-daily simulation section to HBV model guide
- Documented parameter scaling approaches (exact vs linear)
- Added validation methodology for sub-daily implementations

### Migration Guide

| Old Command (v0.6.x) | New Command (v0.7.0) |
|----------------------|----------------------|
| `symfluence --calibrate_model` | `symfluence workflow step calibrate_model` |
| `symfluence --setup_project --create_pour_point` | `symfluence workflow steps setup_project create_pour_point` |
| `symfluence --get_executables summa` | `symfluence binary install summa` |
| `symfluence --validate_binaries` | `symfluence binary validate` |
| `symfluence --doctor` | `symfluence binary doctor` |
| `symfluence --init fuse-provo --scaffold` | `symfluence project init fuse-provo --scaffold` |
| `symfluence --list_presets` | `symfluence project list-presets` |
| `symfluence --workflow_status` | `symfluence workflow status` |
| `symfluence --list_steps` | `symfluence workflow list-steps` |
| `symfluence --pour_point 51/-115 --domain_def delineate --domain_name Test` | `symfluence project pour-point 51/-115 --domain-name Test --definition delineate` |
| `symfluence --agent` | `symfluence agent start` |
| `symfluence --example_notebook 1a` | `symfluence example launch 1a` |

**Benefits of new CLI:**
- Clearer command organization and discoverability
- Better help messages (`symfluence workflow --help` shows workflow-specific options)
- Easier to extend with new commands
- Industry-standard pattern (like git, docker, kubectl)

---

## [0.6.0] - 2025-12-29

### Added
- **Calibration Observation Data Utilities**
  - `download_smhi_discharge.py` - Download discharge data from Swedish Meteorological Institute
  - `prepare_streamflow_for_calibration.py` - Convert discharge CSV to calibration format
  - `setup_calibration.py` - Automated calibration setup with parameter bounds
  - Calibration demo tests for Elliðaár (Iceland) and Fyris (Sweden) basins

- **CARRA/CERRA Data Processing Improvements**
  - Fixed CARRA longitude normalization in spatial subsetting
  - `FORCING_TIME_STEP_SIZE` configuration support (10800s for CERRA)
  - `FORCING_SHAPE_ID_NAME` configuration support with default 'ID'

### Changed
- CLI orchestrator integration completed with full workflow execution support
- Version bump to 0.6.0 across all version references

### Removed
- **Deprecated CONFLUENCE backward compatibility**
  - Removed `CONFLUENCE.py` wrapper file
  - Removed `./confluence` shell script
  - Removed `CONFLUENCE_DATA_DIR` and `CONFLUENCE_CODE_DIR` configuration support
  - All documentation now uses SYMFLUENCE exclusively

### Fixed
- CARRA spatial subsetting for small basin extents
- EASYMORE remapping failures for CARRA datasets

---

## [0.5.11] - 2025-12-15

### Changed
- Enhanced ngen outlet detection
- Cleaned up technical debt

### Fixed
- Mypy type errors in config property inheritance
- Completed typed config migration for base classes

### Improved
- Centralized evaluation metric logic
- Improved linting and added ruff tests to pyproject.toml

---

## [0.5.3] - 2025-11-15

### Added
- Support for cloud acquisition of:
  - Copernicus DEM
  - MODIS land cover
  - Global USDA-NRCS soil texture class map
  - Forcing datasets: ERA5, NEX-GDDP, CONUS404, AORC
- Agnostic pipeline for cloud-based ERA5 matched workflows
- Full cloud-integrated workflow tested with ERA5
- Made MPI worker log generation optional
- Initial t-route support (in progress)

---

## [0.5.2] - 2025-11-12

### Major: Formal Initial Release with End-to-End CI Validation

**This release marks the first fully reproducible SYMFLUENCE workflow with continuous integration.**

### Added
- **End-to-End CI Pipeline (Example Notebook 2a Equivalent)**
  Integrated a comprehensive GitHub Actions workflow that builds, validates, and runs SYMFLUENCE automatically on every commit to `main`.
  - Compiles all hydrologic model dependencies (TauDEM, mizuRoute, FUSE, NGEN).
  - Validates MPI, NetCDF, GDAL, and HDF5 environments.
  - Executes key steps (`setup_project`, `create_pour_point`, `define_domain`, `discretize_domain`, `model_agnostic_preprocessing`, `run_model`, `calibrate_model`, `run_benchmarking`).
  - Confirms reproducible outputs under `SYMFLUENCE_DATA_DIR/domain_Bow_at_Banff`.
  - Runs both wrapper (`./symfluence`) and direct Python entrypoints equivalently.

### Changed
- Updated `external_tools_config.py` to include automatic path resolution for TauDEM binaries (e.g., `moveoutletstostrms → moveoutletstostreams`).
- Expanded logging and run summaries for CI visibility.
- Protected `main` branch to require successful CI validation before merge.

### Notes
This release formalizes SYMFLUENCE's **reproducibility framework**, guaranteeing that all supported workflows can be rebuilt and validated automatically on clean systems.

---

## [0.5.0] - 2025-01-09

### Major: CONFLUENCE → SYMFLUENCE Rebranding

**This is the rebranding release.** The project is now SYMFLUENCE (SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexii for Computational Exploration).

### Added
- Complete rebranding to SYMFLUENCE
- New domain: [symfluence.org](https://symfluence.org)
- PyPI package: `pip install symfluence`
- Backward compatibility for all CONFLUENCE names (with deprecation warnings)

### Changed
- Main script: `CONFLUENCE.py` → `symfluence.py`
- Shell command: `./confluence` → `./symfluence`
- Config parameters: `CONFLUENCE_*` → `SYMFLUENCE_*`
- Repository: `DarriEy/CONFLUENCE` → `symfluence-org/SYMFLUENCE`

### Deprecated
- All CONFLUENCE naming (removed in v0.6.0)

---

## Links

- **PyPI**: [pypi.org/project/symfluence](https://pypi.org/project/symfluence)
- **Documentation**: [symfluence.readthedocs.io](https://symfluence.readthedocs.io)
- **GitHub**: [github.com/symfluence-org/SYMFLUENCE](https://github.com/symfluence-org/SYMFLUENCE)
- **Issues**: [github.com/symfluence-org/SYMFLUENCE/issues](https://github.com/symfluence-org/SYMFLUENCE/issues)
