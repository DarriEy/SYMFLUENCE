# Section 4.9: Large Domain Study

## Overview

This experiment runs FUSE (with mizuRoute routing) over the full Iceland domain to
demonstrate SYMFLUENCE's capability for regional-scale distributed hydrological modelling.
The domain is delineated from a bounding box covering all of Iceland, with coastal
watershed delineation enabled to capture the island's drainage structure.

## Domain

- **Region:** Iceland (full island)
- **Bounding box:** 66.5°N / 25.0°W / 63.0°N / 13.0°W
- **Model:** FUSE (distributed) + mizuRoute
- **Forcing:** ERA5 (hourly, 0.25 degree)
- **Domain definition:** Delineated with coastal watersheds
- **Stream threshold:** 10,000 grid cells
- **DEM source:** Copernicus DEM
- **Lapse rate correction:** 6.5 °C km⁻¹
- **PET method:** Oudin
- **Spinup period:** 2008-01-01 to 2008-12-31
- **Calibration period:** 2009-01-01 to 2009-12-31
- **Evaluation period:** 2010-01-01 to 2010-12-31

## Optimization

- **Algorithm:** DDS (Dynamically Dimensioned Search)
- **Iterations:** 1000
- **Metric:** KGE (Kling-Gupta Efficiency)

## Quick Start

```bash
# 1. Run FUSE on full Iceland domain
python scripts/run_large_domain.py

# 2. Dry run (print command only)
python scripts/run_large_domain.py --dry-run

# 3. Run a specific workflow step
python scripts/run_large_domain.py --step delineation

# 4. Regional analysis
python scripts/analyze_large_domain.py
```

## Output Files

### Analysis (in analysis/)
- `fuse_annual_stats.csv` — Annual water balance statistics (P, Q, PET, T, runoff ratio)
- `per_catchment_kge.csv` — KGE performance metrics per validated catchment
- `iceland_gru_statistics.csv` — GRU-level spatial statistics (area, elevation)
- `gauge_segment_matches.csv` — Gauge to river segment matching (CORRECT methodology)
- `routed_obs_comparison_data.csv` — Daily routed discharge vs observed (CORRECT)

### Deprecated (bug in methodology)
- `lamahice_hru_matches.csv` — HRU to LamaH-Ice matching (used INTERIOR HRUs, not outlets)
- `hru_obs_comparison_data.csv` — Comparison using local HRU runoff (INCORRECT - see below)

**IMPORTANT BUG FIX**: The original `hru_obs_comparison_data.csv` compared local HRU runoff
to catchment outlet discharge. This is incorrect because:
1. HRUs were matched by centroid distance (interior, not outlet)
2. Local HRU runoff ≠ accumulated routed discharge at catchment outlet

The correct comparison (`routed_obs_comparison_data.csv`) uses mizuRoute routed discharge
at river segments matching gauge locations. Run `scripts/create_proper_comparison.py`
after the simulation completes.

### Figures (in figures/)
- `fig_large_domain_overview.{png,pdf}` — 3-panel domain overview (GRU mesh, area/elevation distributions)
- `fig_fuse_baseline_results.{png,pdf}` — 6-panel baseline results (time series, water balance)
- `fig_sim_obs_comparison.{png,pdf}` — 3-panel simulated vs observed comparison
- `fig_hru_obs_comparison.{png,pdf}` — HRU-level model evaluation
- `fig_runoff_ratio_comparison.{png,pdf}` — Runoff ratio analysis
- `fig_spatial_kge.{png,pdf}` — Spatial map of per-catchment KGE performance

## FUSE Configuration

The FUSE model is run in distributed mode with the following decision options:
- **Rainfall error:** Additive and multiplicative
- **Upper architecture:** Tension storage (single)
- **Lower architecture:** Tension + parallel
- **Surface runoff:** ARNO/VIC and PRMS variants
- **Percolation:** Fraction to saturation
- **Snow:** Temperature index
- **Routing:** Gamma distribution (internal) + mizuRoute (river network)
