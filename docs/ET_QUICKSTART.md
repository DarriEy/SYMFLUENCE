# Quick Start: ET Calibration with SUMMA

This is a 5-minute setup guide for ET-based SUMMA calibration.

## 1. Prepare Your Config File

Copy this into your `config.yaml`:

```yaml
# Domain and time period
EXPERIMENT_ID: my_et_calibration
DOMAIN_NAME: my_domain
EXPERIMENT_TIME_START: '2015-01-01 00:00'
EXPERIMENT_TIME_END: '2019-12-31 23:00'
CALIBRATION_PERIOD: '2016-01-01, 2017-12-31'

# ET observations (pick ONE)
ADDITIONAL_OBSERVATIONS: GLEAM_ET      # <-- Change this line

# Point to your data
BOUNDING_BOX_COORDS: lat_n/lon_w/lat_s/lon_e  # north/west/south/east
SYMPHLUENCE_DATA_DIR: /path/to/data

# ET calibration target
OPTIMIZATION_TARGET: et                # <-- This tells SYMFLUENCE to use ET
OPTIMIZATION_METRIC: KGE
EVALUATION_DATA: ['et']

# Parameters affecting ET
PARAMS_TO_CALIBRATE: theta_sat,critSoilTranspire,Fcapil,k_soil,z0Snow,albedoMax

# Optimization settings
OPTIMIZATION_METHODS: ['iteration']
ITERATIVE_OPTIMIZATION_ALGORITHM: ASYNC-DDS
NUMBER_OF_ITERATIONS: 500
DDS_R: 0.2
```

## 2. Choose Your ET Dataset

| Dataset | Code | Setup |
|---------|------|-------|
| **GLEAM** (easiest) | `GLEAM_ET` | Auto-downloads from Zenodo |
| **MODIS** (high-res) | `MODIS_ET` | Point to CSV directory |
| **FLUXCOM** (daily) | `FLUXCOM_ET` | Upload or download manually |

### Option A: GLEAM (Recommended for starting)

Just add this to your config:
```yaml
ADDITIONAL_OBSERVATIONS: GLEAM_ET
GLEAM_ET_DOWNLOAD_URL: https://zenodo.org/api/records/7099512/files/E_GLEAM_v3.tar.gz/content
```

### Option B: MODIS

Add CSV files to a directory with format:
```
my_modis_data/
├── ET8D_Basin_2015.csv
├── ET8D_Basin_2016.csv
└── ...
```

Config:
```yaml
ADDITIONAL_OBSERVATIONS: MODIS_ET
MODIS_ET_DIR: /path/to/my_modis_data
```

### Option C: FLUXCOM

Add CSV files, then:
```yaml
ADDITIONAL_OBSERVATIONS: FLUXCOM_ET
FLUXCOM_ET_PATH: /path/to/fluxcom_data
```

## 3. Run the Workflow

```bash
# Step 1: Download & process ET data
python -m symfluence workflow steps process_observations --config config.yaml

# Step 2: Run SUMMA simulations
python -m symfluence workflow steps run_model --config config.yaml

# Step 3: Calibrate against ET
python -m symfluence workflow steps calibrate_model --config config.yaml
```

Or all at once:
```bash
python -m symfluence workflow steps \
  process_observations run_model calibrate_model \
  --config config.yaml
```

## 4. Check Results

Your calibrated parameters are in:
```
SYMPHLUENCE_DATA_DIR/domain_my_domain/optimization/results/
├── best_parameters.nc
├── optimization_metrics.csv
└── performance_plots/
```

## Common Issues

**ET file not found?**
→ Run `process_observations` step first

**Poor calibration?**
→ Increase `NUMBER_OF_ITERATIONS` to 1000

**Out of memory?**
→ Reduce time period or grid size

## Need Help?

See full docs: [ET_CALIBRATION_SETUP.md](ET_CALIBRATION_SETUP.md)

## What ET parameters do?

| Parameter | Effect on ET | Units |
|-----------|-------------|-------|
| `theta_sat` | ↑ saturation = ↑ available water | dimensionless |
| `critSoilTranspire` | ↑ wilting point = ↓ transpiration | dimensionless |
| `Fcapil` | ↑ capillary rise = ↑ root zone water | m |
| `k_soil` | ↑ conductivity = ↑ drainage = ↓ ET | m/day |
| `z0Snow` | ↑ roughness = ↑ sublimation | m |
| `albedoMax` | ↑ albedo = ↓ energy = ↓ melt/ET | dimensionless |

## Next Steps

1. ✅ Run ET calibration
2. 📊 Evaluate with `python -m symfluence workflow steps analysis`
3. 🔄 Try multi-objective (ET + TWS) calibration
4. 📈 Compare with different ET datasets

Enjoy!
