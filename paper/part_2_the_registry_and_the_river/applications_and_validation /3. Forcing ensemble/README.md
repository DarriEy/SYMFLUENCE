# 4.3 Forcing Ensemble Study

Comparative analysis of atmospheric forcing dataset impact on snow water equivalent (SWE) simulation using the Paradise SNOTEL station as a test case.

## Study Overview

This experiment evaluates how different atmospheric forcing datasets affect hydrological model performance, specifically for snow accumulation and melt processes. The study uses four commonly available forcing datasets covering the continental United States.

### Forcing Datasets

| Dataset | Resolution | Coverage | Period | Source |
|---------|------------|----------|--------|--------|
| **ERA5** | ~31 km | Global | 1940-present | ECMWF Reanalysis |
| **AORC** | ~1 km | CONUS | 1979-present | NOAA NWM |
| **HRRR** | ~3 km | CONUS | 2014-present | NOAA NCEP |
| **CONUS404** | ~4 km | CONUS | 1979-present | NCAR/USGS WRF |

### Experimental Objectives

1. **Forcing Sensitivity**: Quantify SWE simulation sensitivity to forcing dataset choice
2. **Resolution Impact**: Assess whether higher resolution forcings improve SWE prediction
3. **Uncertainty Characterization**: Establish forcing-related uncertainty bounds for snow modeling
4. **Practical Guidance**: Provide recommendations for forcing dataset selection in snow-dominated basins

## Study Structure

```
4_3_forcing_ensemble/
├── configs/                          # Configuration files for each experiment
│   ├── config_paradise_era5.yaml     # ERA5 forcing
│   ├── config_paradise_aorc.yaml     # AORC forcing
│   ├── config_paradise_hrrr.yaml     # HRRR forcing
│   └── config_paradise_conus404.yaml # CONUS404 forcing
├── scripts/                          # Execution and analysis scripts
│   ├── generate_configs.py           # Generate configuration files
│   ├── run_study.py                  # Main study execution script
│   └── analyze_results.py            # Results analysis and visualization
├── results/                          # Study outputs (created during execution)
│   ├── calibration_results/          # Calibration outputs for each forcing
│   ├── plots/                        # Diagnostic plots and comparisons
│   └── forcing_comparison_report.html# Interactive summary report
└── README.md                         # This file
```

## Prerequisites

1. **SYMFLUENCE Installation**
   ```bash
   pip install -e ".[summa]"
   ```

2. **Domain Setup**
   - Domain data must exist at: `$SYMFLUENCE_DATA_DIR/domain_paradise_snotel_wa`
   - SNOTEL observations must be downloaded

3. **Cloud Data Access**
   - ERA5: Access via Google Cloud (no authentication required)
   - AORC: Access via AWS S3 (anonymous)
   - HRRR: Access via AWS S3 (anonymous)
   - CONUS404: Access via HyTEST catalog (public)

## Quick Start

### Step 1: Generate Configuration Files
```bash
cd scripts
python generate_configs.py
```

This creates all forcing-specific config files in the `configs/` directory.

### Step 2: Run the Study

**Run all forcing datasets:**
```bash
python run_study.py --forcing all
```

**Run specific forcing:**
```bash
# Run ERA5 only
python run_study.py --forcing era5

# Run AORC only
python run_study.py --forcing aorc

# Run HRRR only
python run_study.py --forcing hrrr

# Run CONUS404 only
python run_study.py --forcing conus404
```

**Dry run (see what will be executed):**
```bash
python run_study.py --forcing all --dry-run
```

### Step 3: Analyze Results
```bash
python analyze_results.py --output-dir ../results
```

## Manual Execution

You can run experiments manually using the SYMFLUENCE CLI:

### Example: ERA5 Forcing Experiment
```bash
# 1. Data acquisition
symfluence workflow step acquire_forcing \
  --config configs/config_paradise_era5.yaml

# 2. Model preprocessing
symfluence workflow step model_specific_preprocessing \
  --config configs/config_paradise_era5.yaml

# 3. Calibration
symfluence workflow step calibrate_model \
  --config configs/config_paradise_era5.yaml

# 4. Evaluation
symfluence workflow step run_benchmarking \
  --config configs/config_paradise_era5.yaml
```

## Study Configurations

### Common Settings (all experiments)
- **Domain**: Paradise SNOTEL Station, Washington
- **Coordinates**: 46.78°N, -121.75°W
- **Experiment Period**: 2015-01-01 to 2020-12-31
- **Calibration Period**: 2015-10-01 to 2018-09-30 (3 water years)
- **Evaluation Period**: 2018-10-01 to 2020-09-30 (2 water years)
- **Spinup Period**: 2015-01-01 to 2015-09-30
- **Hydrological Model**: SUMMA
- **Calibration Target**: Snow Water Equivalent (SWE)
- **Optimization Method**: DDS (10 iterations for testing)
- **Calibration Metric**: RMSE

### Variable Settings by Experiment

| Experiment | Forcing Dataset | Native Resolution | Temporal Resolution |
|------------|-----------------|-------------------|---------------------|
| Part A | ERA5 | ~31 km | Hourly |
| Part B | AORC | ~1 km | Hourly |
| Part C | HRRR | ~3 km | Hourly |
| Part D | CONUS404 | ~4 km | Hourly |

### Calibrated Parameters (Snow-focused)
```yaml
PARAMS_TO_CALIBRATE: >
  tempCritRain,tempRangeTimestep,frozenPrecipMultip,
  albedoMax,albedoMinWinter,albedoDecayRate,
  constSnowDen,mw_exp,k_snow,z0Snow
BASIN_PARAMS_TO_CALIBRATE: routingGammaScale
```

## Expected Results

### 1. Forcing Characteristics
- **Temperature**: HRRR and CONUS404 expected to show more spatial detail
- **Precipitation**: Higher resolution may better capture orographic effects
- **Radiation**: Differences in cloud parameterization between products

### 2. SWE Performance
- **ERA5**: Baseline global reanalysis performance
- **AORC/HRRR/CONUS404**: Expected improvement in complex terrain

### 3. Computational Considerations
- **Data Volume**: AORC > HRRR > CONUS404 > ERA5 (per time step)
- **Download Time**: Varies by domain size and network speed
- **Processing**: Similar across datasets after acquisition

## Interpreting Results

### Key Metrics to Compare

1. **SWE Simulation Performance**
   - RMSE (Root Mean Square Error)
   - KGE (Kling-Gupta Efficiency)
   - Bias (mean error)
   - Peak SWE timing and magnitude

2. **Forcing Data Characteristics**
   - Temperature bias at SNOTEL elevation
   - Precipitation totals vs. SNOTEL gauge
   - Snow/rain partitioning sensitivity

3. **Calibration Behavior**
   - Parameter convergence
   - Equifinality assessment
   - Transferability implications

### Diagnostic Plots

The analysis script generates:
- **SWE Time Series**: Observed vs simulated for each forcing
- **Forcing Comparison**: Temperature/precipitation differences
- **Performance Metrics**: Bar charts of RMSE, KGE by forcing
- **Parameter Distributions**: Calibrated parameter comparison
- **Uncertainty Bounds**: Ensemble spread across forcings

## Troubleshooting

### Common Issues

1. **"Forcing data missing"**
   - Ensure internet connectivity for cloud data access
   - Check that the requested time period is within dataset coverage
   - HRRR data starts 2014-07-30 (not available before)

2. **"No grid points in bbox"**
   - Verify bounding box coordinates are within CONUS
   - Check coordinate order: lon_min, lat_min, lon_max, lat_max

3. **"Calibration fails to converge"**
   - Try increasing NUMBER_OF_ITERATIONS
   - Check forcing data quality (missing values, discontinuities)
   - Verify SNOTEL observations are available for calibration period

4. **"Memory error during forcing download"**
   - Reduce domain size or time period
   - Process years individually and concatenate
   - Use a machine with more RAM

### Debugging

Enable debug mode:
```bash
symfluence workflow step acquire_forcing \
  --config configs/config_paradise_era5.yaml \
  --debug
```

Check logs:
```bash
tail -f $SYMFLUENCE_DATA_DIR/domain_paradise_snotel_wa/_workLog_paradise_snotel_wa/*.log
```

## References

### Forcing Datasets
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999-2049.
- NOAA. (2022). Analysis of Record for Calibration (AORC) v1.1. *NOAA National Water Model*.
- Benjamin, S.G., et al. (2016). A North American hourly assimilation and model forecast cycle: The Rapid Refresh. *Monthly Weather Review*, 144(4), 1669-1694.
- Rasmussen, R., et al. (2023). CONUS404: The NCAR-USGS 4-km long-term regional hydroclimate reanalysis. *Bulletin of the American Meteorological Society*.

### Snow Modeling
- Clark, M.P., et al. (2015). A unified approach for process-based hydrologic modeling. *Water Resources Research*, 51(4), 2498-2514.

## Contact

For questions or issues:
- GitHub: https://github.com/DarriEy/SYMFLUENCE/issues
- Documentation: https://symfluence.readthedocs.io/

## License

This study is part of SYMFLUENCE and follows the same license.
