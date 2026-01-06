# SYMFLUENCE Configuration Quickstart Guide

**Quick Reference for Getting Started with SYMFLUENCE**

## 📋 Overview

SYMFLUENCE requires only **10 essential configuration options** to run. Everything else has sensible defaults.

Two quickstart templates are available:
- **Flat Style** (Legacy): `config_quickstart_minimal.yaml`
- **Nested Style** (Modern): `config_quickstart_minimal_nested.yaml`

Both contain the same required settings, just formatted differently.

---

## 🎯 10 Required Settings

These are the ONLY settings you MUST configure:

| # | Setting | Section | Type | Purpose |
|---|---------|---------|------|---------|
| 1 | `SYMFLUENCE_DATA_DIR` | System | Path | Root directory for all data |
| 2 | `SYMFLUENCE_CODE_DIR` | System | Path | Root directory for code/executables |
| 3 | `DOMAIN_NAME` | Domain | String | Watershed/basin identifier |
| 4 | `EXPERIMENT_ID` | Domain | String | Experiment/run identifier |
| 5 | `EXPERIMENT_TIME_START` | Domain | Datetime | Simulation start (YYYY-MM-DD HH:MM) |
| 6 | `EXPERIMENT_TIME_END` | Domain | Datetime | Simulation end (YYYY-MM-DD HH:MM) |
| 7 | `DOMAIN_DEFINITION_METHOD` | Domain | Enum | How to define domain (lumped/discretized/etc.) |
| 8 | `DOMAIN_DISCRETIZATION` | Domain | String | How to divide domain (elevation/landclass/etc.) |
| 9 | `FORCING_DATASET` | Forcing | String | Meteorological data source (ERA5/RDRS/etc.) |
| 10 | `HYDROLOGICAL_MODEL` | Model | String | Model(s) to run (SUMMA/FUSE/GR/etc.) |

---

## 🔄 Format Comparison

### Flat Style (Legacy)
```yaml
# Old style - single-level keys with UPPERCASE_UNDERSCORE names
SYMPHLUENCE_DATA_DIR: "/path/to/data"
DOMAIN_NAME: "MyBasin"
FORCING_DATASET: "ERA5"
HYDROLOGICAL_MODEL: "SUMMA"
```

**Pros:**
- Simple, flat structure
- Easy to search/grep
- Compatible with legacy code

**Cons:**
- Many keys with similar prefixes
- Can be hard to organize mentally
- No logical grouping

### Nested Style (Modern)
```yaml
# New style - hierarchical organization by section
system:
  data_dir: "/path/to/data"
domain:
  name: "MyBasin"
forcing:
  dataset: "ERA5"
model:
  hydrological_model: "SUMMA"
```

**Pros:**
- Clear logical organization
- Easier to understand relationships
- Future-proof design
- Better IDE/editor support

**Cons:**
- More indentation needed
- Slightly more typing
- Requires understanding hierarchy

---

## 🚀 Quick Start Instructions

### Step 1: Choose Your Format

**Option A: Flat Style (If Using Existing Code)**
```bash
cp src/symfluence/resources/config_templates/config_quickstart_minimal.yaml \
    my_config.yaml
```

**Option B: Nested Style (Recommended for New Projects)**
```bash
cp src/symfluence/resources/config_templates/config_quickstart_minimal_nested.yaml \
    my_config.yaml
```

### Step 2: Edit Required Fields

Open `my_config.yaml` and fill in ALL 10 required fields:

**Flat Style Example:**
```yaml
SYMFLUENCE_DATA_DIR: "/home/user/symphluence_data"
SYMPHLUENCE_CODE_DIR: "/home/user/symphluence"
DOMAIN_NAME: "MyTestBasin"
EXPERIMENT_ID: "test_run_001"
EXPERIMENT_TIME_START: "2020-01-01 00:00"
EXPERIMENT_TIME_END: "2020-12-31 23:00"
DOMAIN_DEFINITION_METHOD: "lumped"
DOMAIN_DISCRETIZATION: "elevation"
FORCING_DATASET: "ERA5"
HYDROLOGICAL_MODEL: "SUMMA"
```

**Nested Style Example:**
```yaml
system:
  data_dir: "/home/user/symphluence_data"
  code_dir: "/home/user/symphluence"

domain:
  name: "MyTestBasin"
  experiment_id: "test_run_001"
  time_start: "2020-01-01 00:00"
  time_end: "2020-12-31 23:00"
  definition_method: "lumped"
  discretization: "elevation"

forcing:
  dataset: "ERA5"

model:
  hydrological_model: "SUMMA"
```

### Step 3: Run SYMPHLUENCE

```bash
python run_symphluence.py my_config.yaml
```

That's it! SYMPHLUENCE will use defaults for all other 354 options.

---

## 📚 What Each Required Setting Does

### System Configuration

**`SYMPHLUENCE_DATA_DIR`**
- Root directory for all data files
- Should contain subdirectories: forcing/, observations/, outputs/
- Example: `/mnt/data/symphluence` or `/Users/username/symphluence_data`

**`SYMPHLUENCE_CODE_DIR`**
- Root directory for SYMPHLUENCE code and model executables
- Example: `/opt/symphluence` or `/Users/username/symphluence`

### Domain Configuration

**`DOMAIN_NAME`**
- Unique identifier for your watershed/basin
- Used in output filenames and logging
- Example: `Clear_Creek`, `Colorado_River`, `MyBasin_001`

**`EXPERIMENT_ID`**
- Unique identifier for this specific run/experiment
- Allows multiple runs on the same domain
- Example: `baseline_001`, `sensitivity_test_precip`, `calibration_v2`

**`EXPERIMENT_TIME_START`** and **`EXPERIMENT_TIME_END`**
- Define the simulation time period
- Format: `YYYY-MM-DD HH:MM`
- Should include spin-up years if doing model spin-up
- Example: Start `2008-01-01 00:00`, End `2020-12-31 23:00`

**`DOMAIN_DEFINITION_METHOD`**

Choose how to define your modeling domain:

| Method | Use Case | Example |
|--------|----------|---------|
| `lumped` | Single catchment, minimal discretization | Small test basin |
| `discretized` | Pre-defined smaller units within domain | HRU-based model |
| `distributed` | Grid-based spatial resolution | Regional model, 1km grid |
| `subset` | Subset of larger predefined domain | Tributary of major river |
| `point` | Single point/pixel location | FLUXNET site |
| `delineate` | Auto-delineate from DEM at pour point | Custom watershed boundary |

**`DOMAIN_DISCRETIZATION`**

Choose how to subdivide your domain for spatial variability:

| Method | Purpose | Example |
|--------|---------|---------|
| `elevation` | Account for elevation-based gradients | Mountainous terrain |
| `landclass` | Account for land cover differences | Mixed forest/grassland |
| `soilclass` | Account for soil type variations | Heterogeneous soils |
| `aspect` | Account for slope aspect | High mountains |
| `radiation` | Account for radiation differences | Complex topography |
| `grus` | Use pre-defined geographic units | Pre-created GRU files |
| `elevation,landclass` | Combine multiple criteria | Complex domains |

### Forcing Data

**`FORCING_DATASET`**

Choose meteorological forcing data source:

| Dataset | Coverage | Resolution | Details |
|---------|----------|-----------|---------|
| `ERA5` | Global | ~25 km | ECMWF reanalysis, hourly, most commonly used |
| `RDRS` | Canada | ~10 km | Canadian Regional DA System, hourly |
| `NLDAS` | CONUS | 0.125° | NOAA reanalysis, hourly, US only |
| `CONUS404` | CONUS | 4 km | Dynamically downscaled, hourly, US only |
| `custom` | Your data | Variable | User-provided forcing files |

### Hydrological Model

**`HYDROLOGICAL_MODEL`**

Choose one or more hydrological models:

| Model | Type | Use Case | Complexity |
|-------|------|----------|-----------|
| `SUMMA` | Process-based | Distributed, detailed physics | High |
| `FUSE` | Process-based | Flexible structure, modular | Medium |
| `GR` | Conceptual | Lumped, simple, minimal data | Low |
| `HYPE` | Process-based | Semi-distributed, Nordic focus | Medium |
| `NGEN` | Modular | Research community model | High |
| `MESH` | Process-based | Distributed, Canadian focus | High |
| `LSTM` | Data-driven | Fast neural network emulator | Medium |

For multiple models: `"SUMMA, FUSE"` or `["SUMMA", "FUSE"]`

---

## 🔧 Optional Enhancements

Once you have the basic 10 settings working, you can add:

### Most Recommended Optional Settings

```yaml
# (Flat style shown; add to nested sections as needed)

# Calibration/evaluation periods
CALIBRATION_PERIOD: "2010-01-01, 2015-12-31"
EVALUATION_PERIOD: "2016-01-01, 2020-12-31"

# System configuration
MPI_PROCESSES: 4                    # Enable parallel processing
DEBUG_MODE: false                   # Disable debugging

# Spatial location (if using point or delineate)
POUR_POINT_COORDS: "51.1722/-115.5717"

# Routing (recommended for spatial models)
ROUTING_MODEL: "mizuRoute"

# Model-specific paths (if using non-default locations)
SUMMA_INSTALL_PATH: "/opt/summa"
SETTINGS_SUMMA_PATH: "/opt/summa/settings"

# Observation data (for validation)
DOWNLOAD_USGS_DATA: true
STATION_ID: "09010500"
```

### Common Model-Specific Settings

**For SUMMA:**
```yaml
SUMMA_EXE: "summa_sundials.exe"
SUMMA_INSTALL_PATH: "/opt/summa"
SETTINGS_SUMMA_PATH: "/data/summa_settings"
SETTINGS_SUMMA_CONNECT_HRUS: true
```

**For FUSE:**
```yaml
FUSE_EXE: "fuse.exe"
FUSE_INSTALL_PATH: "/opt/fuse"
SETTINGS_FUSE_PATH: "/data/fuse_settings"
FUSE_SPATIAL_MODE: "lumped"
```

**For GR:**
```yaml
GR_EXE: "GR.r"
GR_INSTALL_PATH: "/opt/gr_models"
SETTINGS_GR_PATH: "/data/gr_settings"
```

---

## 📖 Documentation References

| Document | Use Case |
|----------|----------|
| **config_quickstart_minimal.yaml** | Get started quickly (flat style) |
| **config_quickstart_minimal_nested.yaml** | Get started quickly (nested style) |
| **config_template_comprehensive.yaml** | All 364 configuration options |
| **CONFIG_AUDIT_REPORT.md** | Detailed documentation and explanations |

---

## ✅ Verification Checklist

Before running SYMPHLUENCE, verify:

- [ ] `SYMPHLUENCE_DATA_DIR` exists and is writable
- [ ] `SYMPHLUENCE_CODE_DIR` exists and contains executables
- [ ] `DOMAIN_NAME` is set (no spaces recommended)
- [ ] `EXPERIMENT_ID` is unique for this run
- [ ] `EXPERIMENT_TIME_START` < `EXPERIMENT_TIME_END`
- [ ] Date format is correct: `YYYY-MM-DD HH:MM`
- [ ] `DOMAIN_DEFINITION_METHOD` is valid (see options above)
- [ ] `DOMAIN_DISCRETIZATION` is valid (see options above)
- [ ] `FORCING_DATASET` is available for your region
- [ ] `HYDROLOGICAL_MODEL` executables are installed

---

## 🆘 Common Issues

### "Path not found"
- Check that `SYMPHLUENCE_DATA_DIR` and `SYMPHLUENCE_CODE_DIR` exist
- Use absolute paths (start with `/` or `~/`)
- Check for typos in directory names

### "Invalid time format"
- Use format: `YYYY-MM-DD HH:MM`
- Example: `2020-01-15 09:30` (not `2020-1-15 9:30`)
- Start time must be before end time

### "Model executable not found"
- Verify model is installed in `SYMPHLUENCE_CODE_DIR`
- Check spelling of `HYDROLOGICAL_MODEL` (case-sensitive)
- For SUMMA: make sure `SUMMA_EXE` points to correct executable

### "Forcing data not found"
- Verify dataset is available for your region
- Check internet connection (ERA5 needs CDS download)
- Ensure `FORCING_DATASET` is spelled correctly

---

## 📝 Example Configurations

### Minimal SUMMA Setup
```yaml
# Flat style
SYMPHLUENCE_DATA_DIR: "/data/symphluence"
SYMPHLUENCE_CODE_DIR: "/opt/symphluence"
DOMAIN_NAME: "Test"
EXPERIMENT_ID: "v1"
EXPERIMENT_TIME_START: "2020-01-01 00:00"
EXPERIMENT_TIME_END: "2020-12-31 23:00"
DOMAIN_DEFINITION_METHOD: "lumped"
DOMAIN_DISCRETIZATION: "elevation"
FORCING_DATASET: "ERA5"
HYDROLOGICAL_MODEL: "SUMMA"
```

### Multi-Model Ensemble
```yaml
HYDROLOGICAL_MODEL: "SUMMA, FUSE, GR"
ROUTING_MODEL: "mizuRoute"
```

### Calibration Setup
```yaml
CALIBRATION_PERIOD: "2010-01-01, 2015-12-31"
EVALUATION_PERIOD: "2016-01-01, 2020-12-31"
OPTIMIZATION_METHODS: "PSO"
ITERATIVE_OPTIMIZATION_ALGORITHM: "PSO"
NUMBER_OF_ITERATIONS: 1000
POPULATION_SIZE: 50
```

---

## 🎓 Learning Path

1. **Start here:** Use `config_quickstart_minimal.yaml` or `config_quickstart_minimal_nested.yaml`
2. **Need more options?** See `config_template_comprehensive.yaml`
3. **Want details?** Read `CONFIG_AUDIT_REPORT.md`
4. **Explore code?** Check `src/symphluence/core/config/models/`

---

**Questions?** Refer to the full configuration audit report: `CONFIG_AUDIT_REPORT.md`
