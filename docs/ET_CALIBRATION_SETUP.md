# Remote Sensing ET Dataset Integration for SUMMA Calibration

This guide explains how to wire up remote sensing Evapotranspiration (ET) datasets for SUMMA model calibration using the SYMFLUENCE workflow.

## Overview

SYMFLUENCE supports three major remote sensing ET products for model calibration:

1. **GLEAM v3** (Global Land Evaporation Amsterdam Model)
   - Global coverage, daily data
   - ~0.25° resolution
   - Based on microwave remote sensing
   - Best for: Global applications

2. **MODIS ET** (Moderate Resolution Imaging Spectroradiometer)
   - 8-day composites
   - 500m resolution
   - Derived from MODIS land data products
   - Best for: High-resolution regional studies

3. **FLUXCOM ET** (Flux Community)
   - Daily estimates
   - ~0.05° resolution
   - Machine learning based (RF, neural networks)
   - Best for: Site-specific validations

## Workflow Steps

The complete ET-based calibration workflow consists of five steps:

### 1. **process_observations**
Acquires and processes remote sensing ET data:
```bash
python -m symfluence process_observations --config path/to/config.yaml
```

This step:
- Downloads ET data according to `ADDITIONAL_OBSERVATIONS` configuration
- Subsets data to your domain bounding box
- Converts ET to daily mm/day format if needed
- Outputs preprocessed CSV files to `observations/et/preprocessed/`

### 2. **run_model**
Executes SUMMA simulations with current parameter sets:
```bash
python -m symfluence run_model --config path/to/config.yaml
```

This step:
- Initializes SUMMA with parameters from `PARAMS_TO_CALIBRATE`
- Runs simulations for the calibration period
- Outputs ET and other fluxes matching observation frequency

### 3. **calibrate_model**
Optimizes model parameters using observed ET data:
```bash
python -m symfluence calibrate_model --config path/to/config.yaml
```

This step:
- Uses the optimization algorithm (ASYNC-DDS recommended)
- Compares simulated vs. observed ET using KGE metric
- Iteratively improves ET simulations
- Saves best parameters and performance metrics

## Configuration

### Minimal ET Configuration

To enable ET calibration in your YAML config:

```yaml
# ============================================
# Step 1: Enable ET Data Acquisition
# ============================================

# Add ET to additional observations
ADDITIONAL_OBSERVATIONS: GLEAM_ET
# OR: MODIS_ET, FLUXCOM_ET, or multiple: [GRACE, GLEAM_ET]

# Choose one of the following based on your dataset:

# --- GLEAM ET Configuration ---
GLEAM_ET_DOWNLOAD_URL: https://zenodo.org/api/records/7099512/files/E_GLEAM_v3.tar.gz/content
GLEAM_ET_PATH: /path/to/data/observations/et/gleam
ET_UNIT_CONVERSION: 1.0

# --- MODIS ET Configuration ---
# MODIS_ET_DIR: /path/to/data/observations/et/modis
# ET_UNIT_CONVERSION: 1.0

# --- FLUXCOM ET Configuration ---
# FLUXCOM_ET_PATH: /path/to/data/observations/et/fluxcom
# FLUXCOM_ET_DOWNLOAD_URL: https://your-download-url/fluxcom_bundle.tar.gz

# ============================================
# Step 2: Set Optimization Target to ET
# ============================================

OPTIMIZATION_TARGET: et
OPTIMIZATION_METRIC: KGE
EVALUATION_DATA: ['et']

# ET-specific settings (optional, defaults shown)
ET_OBS_SOURCE: gleam  # or fluxcom, modis_et
ET_TEMPORAL_AGGREGATION: daily_mean
ET_USE_QUALITY_CONTROL: true
ET_MAX_QUALITY_FLAG: 2

# Path to preprocessed ET observations (optional - auto-detected if not provided)
ET_OBS_PATH: /path/to/data/observations/et/preprocessed/domain_gleam_et_processed.csv

# ============================================
# Step 3: Select ET-Related Parameters
# ============================================

# ET-sensitive parameters (vegetation, soil, snow)
PARAMS_TO_CALIBRATE: 
  - k_soil           # Soil permeability
  - theta_sat        # Soil porosity
  - critSoilTranspire # Wilting point
  - Fcapil          # Capillary fringe height
  - z0Snow          # Snow surface roughness
  - albedoMax       # Maximum albedo
  - albedoMinWinter # Winter albedo
  - tempCritRain    # Rain-snow threshold

# Optional: Basin-level parameters
BASIN_PARAMS_TO_CALIBRATE:
  - routingGammaScale

# ============================================
# Step 4: Set Optimization Algorithm
# ============================================

OPTIMIZATION_METHODS: ['iteration']
ITERATIVE_OPTIMIZATION_ALGORITHM: ASYNC-DDS
OPTIMIZATION_METRIC: KGE

# Algorithm settings
DDS_R: 0.2
ASYNC_DDS_POOL_SIZE: 10
ASYNC_DDS_BATCH_SIZE: 10
MAX_STAGNATION_BATCHES: 10
NUMBER_OF_ITERATIONS: 500
```

### Full Example Config

See `test_ellioaar_grace_tws_optimization.yaml` for a complete example with GLEAM ET integrated.

## Parameter Guidance for ET Calibration

### ET-Sensitive Parameters

Parameters that most affect simulated ET:

**High Sensitivity:**
- `theta_sat`: Soil porosity (affects soil water availability)
- `critSoilTranspire`: Critical soil water for transpiration (wilting point)
- `Fcapil`: Capillary rise height (affects root zone moisture)
- `k_soil`: Hydraulic conductivity (affects drainage and moisture retention)

**Medium Sensitivity:**
- `z0Snow`: Snow surface roughness (affects snow sublimation)
- `albedoMax`: Maximum albedo (affects energy balance and melt)
- `LAI`: Leaf area index (affects canopy transpiration)

**Lower Sensitivity (but important):**
- `tempCritRain`: Rain-snow threshold (affects precipitation partitioning)
- `aquiferBaseflowRate`: Baseflow rate (indirect effect via water table)

### Recommended Parameter Bounds

```yaml
PARAMETER_BOUNDS:
  theta_sat: [0.40, 0.70]        # Common soil porosity range
  critSoilTranspire: [0.10, 0.40] # Between residual and saturation
  Fcapil: [0.05, 0.50]           # Capillary rise height (m)
  k_soil: [0.001, 0.1]           # Saturated hydraulic conductivity (m/day)
  albedoMax: [0.2, 0.5]          # Fresh snow albedo
  albedoMinWinter: [0.15, 0.4]   # Old snow albedo
  z0Snow: [0.0001, 0.01]         # Snow surface roughness (m)
```

## Data Acquisition Details

### GLEAM ET

**Pros:**
- Global coverage at moderate resolution
- Free and easy access via Zenodo
- Well-documented quality metrics
- Consistent methodology

**Cons:**
- Lower spatial resolution (~25 km)
- Requires downloading ~500 MB per year

**Download:**
```yaml
GLEAM_ET_DOWNLOAD_URL: https://zenodo.org/api/records/7099512/files/E_GLEAM_v3.tar.gz/content
GLEAM_ET_PATH: /path/to/data/observations/et/gleam
```

SYMFLUENCE automatically:
1. Downloads the tar.gz file
2. Extracts it to `GLEAM_ET_PATH`
3. Subsets to your domain bounding box
4. Converts to daily mm/day format
5. Saves to `observations/et/preprocessed/{domain}_gleam_et_processed.csv`

### MODIS ET

**Pros:**
- High spatial resolution (500 m)
- 8-day temporal frequency
- Consistent long-term record

**Cons:**
- Requires manual preprocessing into CSV format
- More files to manage

**Expected CSV Format:**
```csv
date,mean_et_mm
2015-01-01,2.5
2015-01-09,2.3
...
```

SYMFLUENCE automatically:
1. Reads CSV files from `MODIS_ET_DIR`
2. Converts 8-day cumulative to daily mean (divide by 8)
3. Saves to `observations/et/preprocessed/{domain}_modis_et_processed.csv`

### FLUXCOM ET

**Pros:**
- Daily temporal resolution
- Machine learning-based, combines multiple inputs
- High accuracy where validated

**Cons:**
- Requires explicit download from FLUXCOM website
- May have gaps in some regions

**Setup:**
```yaml
ADDITIONAL_OBSERVATIONS: FLUXCOM_ET
FLUXCOM_ET_PATH: /path/to/data/observations/et/fluxcom
FLUXCOM_ET_DOWNLOAD_URL: https://zenodo-link/fluxcom_bundle.tar.gz  # If available
```

## Multi-Objective Calibration (Advanced)

To calibrate against both TWS (GRACE) and ET simultaneously:

```yaml
# Acquire both datasets
ADDITIONAL_OBSERVATIONS: [GRACE, GLEAM_ET]

# Enable multi-objective optimization
OPTIMIZATION_METHODS: ['iteration']
OPTIMIZATION_TARGET: multivariate
EVALUATION_DATA: ['tws', 'et']

# Multi-objective weights
MULTIVARIATE_WEIGHTS:
  tws: 0.5    # 50% weight on GRACE TWS
  et: 0.5     # 50% weight on ET

# Or use Pareto-based optimization (e.g., NSGA2)
OPTIMIZATION_METHODS: ['nsga2']
NSGA2_PRIMARY_TARGET: et
NSGA2_SECONDARY_TARGET: tws
```

## Running the Workflow

### Complete ET Calibration (process_observations → run_model → calibrate_model)

```bash
# Full command with all three steps
python -m symfluence workflow steps \
  process_observations \
  run_model \
  calibrate_model \
  --config /path/to/config.yaml
```

### Individual Steps

```bash
# Step 1: Acquire and process ET observations
python -m symfluence workflow steps \
  process_observations \
  --config /path/to/config.yaml

# Step 2: Run SUMMA simulations
python -m symfluence workflow steps \
  run_model \
  --config /path/to/config.yaml

# Step 3: Calibrate against ET
python -m symfluence workflow steps \
  calibrate_model \
  --config /path/to/config.yaml
```

### Continuing from Checkpoint

If interrupted, resume calibration:
```bash
python -m symfluence workflow steps \
  calibrate_model \
  --config /path/to/config.yaml
```

SYMFLUENCE will detect existing iterations and continue from the last checkpoint.

## Output Files

After successful completion, you'll find:

```
project_directory/
├── observations/et/
│   ├── gleam/              # Downloaded raw GLEAM data
│   └── preprocessed/       # Processed daily ET CSV
├── settings/SUMMA/         # Model configuration
├── simulations/            # Model output NetCDF files
└── optimization/
    ├── iteration_0001/     # Calibration iterations
    ├── iteration_0002/
    └── ...
    └── results/
        ├── best_parameters.nc
        ├── optimization_metrics.csv
        └── performance_plots/
```

## Troubleshooting

### Issue: ET observations file not found

**Check:**
1. `ADDITIONAL_OBSERVATIONS` includes ET dataset name
2. `GLEAM_ET_DOWNLOAD_URL`, `MODIS_ET_DIR`, or `FLUXCOM_ET_PATH` is configured
3. Run `process_observations` step first

**Solution:**
```bash
python -m symfluence workflow steps process_observations --config config.yaml
```

### Issue: ET values unreasonable (very high or low)

**Cause:** Unit conversion mismatch

**Solution:**
```yaml
ET_UNIT_CONVERSION: 1.0  # Check your dataset's native units
# GLEAM: already in mm/day (1.0)
# Some datasets may need: 1.0, 0.001, 86.4, etc.
```

### Issue: Poor ET calibration performance

**Check:**
1. Calibration period has good ET data coverage (no gaps > 30 days)
2. ET-sensitive parameters in `PARAMS_TO_CALIBRATE`
3. Parameter bounds are physically reasonable
4. Sufficient iterations (try 500-1000)

**Solution:** Increase iterations or adjust parameter bounds:
```yaml
NUMBER_OF_ITERATIONS: 1000
ASYNC_DDS_POOL_SIZE: 20
```

### Issue: Memory issues with large ET datasets

**Solution:** Filter time period or spatial extent:
```yaml
EXPERIMENT_TIME_START: '2016-01-01 00:00'  # Shorter period
EXPERIMENT_TIME_END: '2017-12-31 23:00'
BOUNDING_BOX_COORDS: 65.0/-20.0/64.0/-22.0  # Smaller domain
```

## Advanced: Custom ET Datasets

To integrate custom ET observations:

1. **Create a CSV file** with columns: `date`, `et_mm`
2. **Place in:** `observations/et/preprocessed/{domain}_custom_et.csv`
3. **Configure:**
   ```yaml
   ET_OBS_PATH: /path/to/observations/et/preprocessed/{domain}_custom_et.csv
   OPTIMIZATION_TARGET: et
   ```

## References

- **GLEAM:** Miralles et al. (2011) - https://www.gleam.eu/
- **MODIS ET:** Running et al. (2017) - https://lpdaac.usgs.gov/products/mod16a2v006/
- **FLUXCOM:** Jung et al. (2019) - https://www.fluxcom.org/
- **SUMMA Documentation:** https://summa.readthedocs.io/

## Reporting Issues

For issues with ET calibration:

1. Check `DEBUG` logs in `LOG_LEVEL: DEBUG`
2. Verify ET data files in project directory
3. Report with:
   - Config file (with sensitive paths anonymized)
   - Log output
   - Error message
