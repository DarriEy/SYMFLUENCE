# Section 4.7 - Multi-Model Sensitivity Analysis

## Overview

This experiment showcases SYMFLUENCE's sensitivity analysis module by running
parameter sensitivity analysis across the five calibrated models from Section 4.2
(FUSE, GR4J, HBV, HYPE, SUMMA). The goal is to identify:

1. Which parameters are most influential within each model
2. Which hydrological processes are consistently sensitive across structurally
   different models
3. How well different sensitivity methods agree
4. Whether models agree on the importance of key processes (snow, soil storage,
   routing, baseflow, etc.)

## SYMFLUENCE Sensitivity Methods

SYMFLUENCE implements four complementary sensitivity analysis methods:

| Method | Library | Type | Description |
|--------|---------|------|-------------|
| VISCOUS | pyviscous | Total-order | Variance-based indices using copulas |
| Sobol | SALib | Variance-based | Quasi-random variance decomposition |
| RBD-FAST | SALib | Fourier | Random Balance Designs - FAST |
| Correlation | SciPy | Rank | Spearman rank correlation |

## Domain and Models

- **Domain**: Bow River at Banff (lumped, 2210 km²)
- **Forcing**: ERA5
- **Calibration**: DDS, 1000 iterations (from Section 4.2)
- **Models**: FUSE (13 params), GR4J (4 params), HBV (14 params), HYPE (10 params), SUMMA (11 params)

## Directory Structure

```
7. Sensitivity analysis/
├── config/           # Standalone configs with unique EXPERIMENT_IDs per model
├── scripts/
│   ├── run_sensitivity.py        # Run SYMFLUENCE sensitivity for each model
│   ├── analyze_sensitivity.py    # Cross-model sensitivity comparison
│   └── visualize_sensitivity.py  # Publication-quality figures
├── analysis/         # CSV outputs and reports
├── figures/          # Generated figures
├── logs/             # Execution logs
└── README.md
```

## Quick Start

```bash
# 1. Run sensitivity analysis for all models (requires Section 4.2 calibration data)
python scripts/run_sensitivity.py

# 2. Analyze and compare results across models
python scripts/analyze_sensitivity.py

# 3. Generate publication figures
python scripts/visualize_sensitivity.py --format pdf
```

## Dependencies

- SYMFLUENCE (with sensitivity module)
- SALib (Sobol, RBD-FAST)
- pyviscous (VISCOUS method)
- scipy (Spearman correlation)
- matplotlib, numpy, pandas

## Process Grouping

Parameters are grouped by hydrological process for cross-model comparison:

| Process | FUSE | GR4J | HBV | HYPE | SUMMA |
|---------|------|------|-----|------|-------|
| Snow | MBASE, MFMAX, MFMIN, PXTEMP | - | tt, cfmax, sfcf, cfr, cwh | ttmp, cmlt | albedoMax, albedoMinWinter, newSnowDenMin |
| Soil Storage | MAXWATR_1, MAXWATR_2, FRACTEN | X1 | fc, beta | - | Fcapil, k_soil, theta_sat, theta_res, f_impede |
| Baseflow | BASERTE, QB_POWR | - | k1, k2 | rrcs1, rrcs2 | - |
| Routing | TIMEDELAY | X3, X4 | maxbas | rivvel, damp | routingGammaShape, routingGammaScale |
| Evapotranspiration | - | - | lp | cevp, epotdist | critSoilWilting |
| Surface Runoff | RTFRAC1 | - | k0, uzl | - | - |
| Percolation | PERCRTE | - | perc | - | - |
| Groundwater | - | X2 | - | rcgrw | - |
