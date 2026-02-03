# Section 4.8: Large Sample Study

## Overview

This experiment applies FUSE (with mizuRoute routing) across six LamaH-Ice catchments
in Iceland to demonstrate SYMFLUENCE's ability to handle multi-catchment large-sample
hydrological studies. Each catchment is run independently with consistent configuration,
enabling cross-catchment comparison of model performance.

## Domain

- **Catchments:** LamaH-Ice domains 42, 45, 98, 100, 102, 103
- **Model:** FUSE + mizuRoute
- **Forcing:** ERA5 (hourly, 0.25 degree)
- **Observation source:** LamaH-Ice streamflow records
- **Spinup period:** 2005-01-01 to 2006-12-31
- **Calibration period:** 2007-01-01 to 2011-12-31
- **Evaluation period:** 2012-01-01 to 2014-12-31

## Optimization

- **Algorithm:** DDS (Dynamically Dimensioned Search)
- **Iterations:** 1000
- **Metric:** KGE (Kling-Gupta Efficiency)

## Quick Start

```bash
# 1. Generate per-catchment configs from template
python scripts/generate_configs.py

# 2. Run FUSE on all catchments
python scripts/run_large_sample.py

# 3. Run a single catchment
python scripts/run_large_sample.py --domain 45

# 4. Dry run (print commands only)
python scripts/run_large_sample.py --dry-run

# 5. Cross-catchment analysis
python scripts/analyze_large_sample.py
```

## Output Files

### Configs (in configs/)
- `config_lamahice_{ID}_FUSE.yaml` — Per-catchment FUSE configuration

### Analysis (in analysis/)
- `large_sample_metrics.csv` — KGE, NSE, PBIAS for all catchments

### Figures (in figures/)
- `fig_large_sample_metrics.{png,pdf}` — Bar chart of metrics across catchments
- `fig_large_sample_kge_dist.{png,pdf}` — KGE distribution plot

## Data

LamaH-Ice data is stored at:
```
/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/lamahice/domain_{ID}/
```

Each domain contains pre-processed shapefiles, forcing data, and streamflow observations.
