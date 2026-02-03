# Section 4.5: Benchmarking

## Overview

This experiment establishes performance baselines for the Bow River at Banff domain
using the HydroBM library (Knoben et al., 2020). Simple reference models (mean flow,
seasonal climatology, rainfall-runoff ratios, Schaefli & Gupta benchmarks) are evaluated
against observed streamflow. The resulting benchmark KGE scores are compared to the
multi-model ensemble from Section 4.2 to quantify the value added by calibrated
hydrological models.

## Domain

- **Catchment:** Bow River at Banff (lumped, 2,210 km2)
- **Forcing:** ERA5 (hourly, 0.25 degree)
- **Station:** WSC 05BB001
- **Calibration period:** 2004-01-01 to 2007-12-31
- **Evaluation period:** 2008-01-01 to 2009-12-31

## Benchmarks Computed

### Streamflow benchmarks
- `mean_flow`, `median_flow` (time-invariant)
- `annual_mean_flow`, `annual_median_flow` (interannual)
- `monthly_mean_flow`, `monthly_median_flow` (seasonal cycle)
- `daily_mean_flow`, `daily_median_flow` (daily climatology)

### Rainfall-runoff ratio benchmarks
- Long-term: `rainfall_runoff_ratio_to_{all,annual,monthly,daily,timestep}`
- Short-term: `monthly_rainfall_runoff_ratio_to_{monthly,daily,timestep}`

### Schaefli & Gupta (2007) benchmarks
- `scaled_precipitation_benchmark`
- `adjusted_precipitation_benchmark`
- `adjusted_smoothed_precipitation_benchmark`

## Quick Start

```bash
# 1. Run HydroBM benchmarks via SYMFLUENCE
python scripts/run_benchmarking.py

# 2. Analyze results and compare to Section 4.2
python scripts/analyze_benchmarks.py

# 3. Generate publication figures
python scripts/create_publication_figures.py
```

## Manual Execution

```bash
# Run benchmarking directly via SYMFLUENCE CLI
symfluence workflow step run_benchmarking --config config/config_Bow_benchmark_era5.yaml
```

## Output Files

### Benchmark results (in SYMFLUENCE_data/domain_Bow_at_Banff_lumped_era5/evaluation/)
- `benchmark_input_data.csv` - Preprocessed streamflow + forcing data
- `benchmark_flows.csv` - Time series of benchmark flows
- `benchmark_scores.csv` - Performance metrics (NSE, KGE, MSE, RMSE) per benchmark
- `benchmark_metadata.json` - Processing metadata

### Analysis (in analysis/)
- `benchmark_summary_*.csv` - All benchmark scores
- `benchmark_vs_models_*.csv` - Comparison with Section 4.2 models
- `benchmark_groups_*.csv` - Group-level statistics
- `benchmark_analysis_report_*.txt` - Narrative summary

### Figures (in figures/)
- `fig_benchmark_heatmap.{png,pdf}` - Heatmap of benchmark scores
- `fig_model_vs_benchmark.{png,pdf}` - Bar chart comparison
- `fig_benchmark_flows.{png,pdf}` - Flow time series envelopes

## Section 4.2 Reference Values

| Model | Type | Eval KGE |
|-------|------|----------|
| SUMMA | Process-based | 0.88 |
| FUSE | Process-based | 0.88 |
| GR4J | Conceptual | 0.79 |
| HBV | Conceptual | 0.70 |
| HYPE | Process-based | 0.81 |
| LSTM | Data-driven | 0.88 |
| **Ensemble Mean** | - | **0.94** |
| **Ensemble Median** | - | **0.92** |

## References

- Schaefli, B., & Gupta, H. V. (2007). Do Nash values have value? Hydrological Processes, 21(15), 2075-2080.
- Knoben, W. J. M., et al. (2020). A brief analysis of conceptual model structure uncertainty using multiple models and multiple catchments. HESS.
