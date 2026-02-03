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
- **Stream threshold:** 2000
- **DEM source:** FABDEM
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
- `iceland_reach_statistics.csv` — Per-reach flow statistics

### Figures (in figures/)
- `fig_iceland_spatial_flow.{png,pdf}` — Spatial map of mean simulated flow
- `fig_iceland_flow_distribution.{png,pdf}` — Distribution of outlet flows

## FUSE Configuration

The FUSE model is run in distributed mode with the following decision options:
- **Rainfall error:** Additive and multiplicative
- **Upper architecture:** Tension storage (single)
- **Lower architecture:** Tension + parallel
- **Surface runoff:** ARNO/VIC and PRMS variants
- **Percolation:** Fraction to saturation
- **Snow:** Temperature index
- **Routing:** Gamma distribution (internal) + mizuRoute (river network)
