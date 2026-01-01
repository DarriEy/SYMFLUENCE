# Changelog

All notable changes to SYMFLUENCE are documented here.

---
## [0.6.0] - 2025-12-29

### Major: Calibration Infrastructure & CLI Completion

**This release completes the CLI orchestrator integration and adds comprehensive calibration infrastructure including observation data acquisition utilities.**

### Added
- **Calibration Observation Data Utilities**
  - `utils/data/observation/download_smhi_discharge.py` - Download discharge data from Swedish Meteorological Institute
  - `utils/data/observation/prepare_streamflow_for_calibration.py` - Convert discharge CSV to calibration format
  - `utils/calibration/setup_calibration.py` - Automated calibration setup with parameter bounds and observational data
  - Calibration demo tests for Elliðaár (Iceland) and Fyris (Sweden) basins

- **CARRA/CERRA Data Processing Improvements**
  - Fixed CARRA longitude normalization in spatial subsetting
  - Improved spatial subsetting for small basins
  - Added `FORCING_TIME_STEP_SIZE` configuration support (10800s for CERRA)
  - Added `FORCING_SHAPE_ID_NAME` configuration support with default 'ID'

### Changed
- **CLI Orchestrator Integration Completed**
  - `src/symfluence/cli.py` now properly instantiates and executes the workflow orchestrator
  - Full integration with SYMFLUENCE class for both full workflow and individual step execution
  - Removed placeholder TODO and completed implementation

- **Version Bump to 0.6.0**
  - Updated all version references across codebase
  - `symfluence_version.py`: 0.6.0
  - `pyproject.toml`: 0.6.0
  - `__init__.py`: 0.6.0 fallback

### Removed
- **Deprecated CONFLUENCE Backward Compatibility**
  - Removed `CONFLUENCE.py` wrapper file
  - Removed `./confluence` shell script
  - Removed `CONFLUENCE_DATA_DIR` and `CONFLUENCE_CODE_DIR` configuration support
  - Removed all backward compatibility warnings and migration notes from documentation
  - Updated all documentation references to use SYMFLUENCE exclusively

### Fixed
- CARRA spatial subsetting for small basin extents
- EASYMORE remapping failures for CARRA datasets

### Notes
- Calibration utilities are complete but require manual execution before running calibration
- Future releases may integrate observation data acquisition into the main workflow
- All CONFLUENCE references have been removed; users must update configurations to use `SYMFLUENCE_DATA_DIR`

---
## [0.5.3] - DEVELOP: 2025-11-15

### Added
- Support for cloud acquisition of 
  - Copernicus DEM
  - MODIS land cover
  - Global USDA-NRCS soil texture class map
  - Forcing datasets ERA5, NEX-GDDP, CONUS404, AORC
  - Agnostic pipeline for cloud based era5 matched
  - Full cloud integraed workflow tested with ERA5

- made mpi_worker log generation optional
- Working on t-route support

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
This release formalizes SYMFLUENCE’s **reproducibility framework**, guaranteeing that all supported workflows can be rebuilt and validated automatically on clean systems.

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
- Repository: github.com/DarriEy/CONFLUENCE → github.com/DarriEy/SYMFLUENCE

### Deprecated
- All CONFLUENCE naming (will be removed in v1.0.0)
- Legacy names still work but show warnings

### Migration
```bash
# Update command
./confluence --install  # old (still works)
./symfluence --install  # new

# Update imports
from CONFLUENCE import CONFLUENCE  # old (still works)
from symfluence import SYMFLUENCE  # new

# Update config
CONFLUENCE_DATA_DIR → SYMFLUENCE_DATA_DIR
```

---

## Links
- PyPI: [pypi.org/project/symfluence](https://pypi.org/project/symfluence)
- Docs: [symfluence.readthedocs.io](https://symfluence.readthedocs.io)
- GitHub: [github.com/DarriEy/SYMFLUENCE](https://github.com/DarriEy/SYMFLUENCE)
