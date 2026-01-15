# Changelog

All notable changes to SYMFLUENCE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- See [v0.7.0] for upcoming CLI changes

---

## [0.7.0] - Upcoming

> **Breaking Change**: This release refactors the CLI to a subcommand architecture.
> All existing CLI commands will need to be updated.

### Changed
- **Complete CLI Refactor**
  - Replaced flat flag-based interface with modern two-level subcommand architecture
  - New structure: `symfluence <category> <action>` instead of `symfluence --flag`
  - 7 command categories: workflow, project, binary, config, job, example, agent
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
  ```

### Added
- New modular command structure in `src/symfluence/cli/`:
  - `argument_parser.py` - Main parser with subcommand structure
  - `validators.py` - Validation utilities
  - `commands/` directory with category-specific handlers

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

### Added
- **End-to-End CI Pipeline**
  - Comprehensive GitHub Actions workflow for automated testing
  - Compiles all hydrologic model dependencies (TauDEM, mizuRoute, FUSE, NGEN)
  - Validates MPI, NetCDF, GDAL, and HDF5 environments
  - Executes key workflow steps with reproducible outputs
  - Protected `main` branch requiring successful CI validation

### Changed
- Updated `external_tools_config.py` with automatic path resolution for TauDEM binaries
- Expanded logging and run summaries for CI visibility

---

## [0.5.0] - 2025-01-09

### Added
- Complete rebranding to SYMFLUENCE
- New domain: [symfluence.org](https://symfluence.org)
- PyPI package: `pip install symfluence`
- Backward compatibility for all CONFLUENCE names (with deprecation warnings)

### Changed
- Main script: `CONFLUENCE.py` → `symfluence.py`
- Shell command: `./confluence` → `./symfluence`
- Config parameters: `CONFLUENCE_*` → `SYMFLUENCE_*`
- Repository: `DarriEy/CONFLUENCE` → `DarriEy/SYMFLUENCE`

### Deprecated
- All CONFLUENCE naming (removed in v0.6.0)

---

## Links

- **PyPI**: [pypi.org/project/symfluence](https://pypi.org/project/symfluence)
- **Documentation**: [symfluence.readthedocs.io](https://symfluence.readthedocs.io)
- **GitHub**: [github.com/DarriEy/SYMFLUENCE](https://github.com/DarriEy/SYMFLUENCE)
- **Issues**: [github.com/DarriEy/SYMFLUENCE/issues](https://github.com/DarriEy/SYMFLUENCE/issues)
