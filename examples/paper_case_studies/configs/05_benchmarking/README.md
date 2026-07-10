# Experiment 05 — Streamflow Benchmarking (§4.2.2, Fig 8)

A single config runs SYMFLUENCE's HydroBM-based Benchmarker: 12 naive
reference predictors (mean/median flow, annual/monthly/daily climatologies,
precipitation-scaled variants, …) computed from observed streamflow and ERA5
precipitation for Bow River at Banff (WSC 05BB001). The benchmarks are
model-agnostic — **no hydrological model is run** — and set the performance
floor against which the experiment 02 ensemble is judged in Fig 8.

## Config

| Config | Description |
|---|---|
| `config_bow_benchmark.yaml` | Naive-predictor benchmarks; `workflow_steps` ends at `run_benchmarking`, no model steps |

The config uses the same domain as experiment 02
(`Bow_at_Banff_lumped_era5`), same periods (calibration 2004–2007, evaluation
2008–2009). If experiment 02 has already run, its downloaded/preprocessed
data is reused via stage markers; run standalone, the config performs its own
domain definition, observation processing, and ERA5 acquisition first.

## Run

```bash
symfluence workflow run --config \
    examples/paper_case_studies/configs/05_benchmarking/config_bow_benchmark.yaml
```

## Outputs

Under `SYMFLUENCE_data/domain_Bow_at_Banff_lumped_era5/`:

- `evaluation/benchmark_scores.csv` — one row per benchmark predictor, with
  calibration- and evaluation-period scores

## Verify (Fig 8 reference value)

The strongest naive predictor, the **daily-median climatology**, reaches an
evaluation KGE of **≈ 0.80** — the bar a calibrated model must clear to add
value at this snowmelt-dominated basin.

## Runtime

~15 min if experiment 02's domain already exists; ~1–2 h standalone (ERA5
download + preprocessing dominate). No model executables required.
