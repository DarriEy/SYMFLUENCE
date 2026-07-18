# Paper figure plotting scripts

Figure-producing scripts for *From Configuration to Prediction* (P3), staged
from the paper's per-experiment working folders and minimally adapted to run
against the reproduction outputs of `examples/paper_case_studies/configs/`.

## Conventions

- **Data root**: every script reads `SYMFLUENCE_DATA_DIR` (default: a
  `SYMFLUENCE_data` directory next to the repo root). No absolute paths.
- **Output**: figures are written to `plotting/output/` (created on demand;
  not tracked).
- **Domains / experiment ids** match the shipped configs
  (`Bow_at_Banff_lumped_era5` + `run_1`,
  `Bow_at_Banff_lumped_calibration_ensemble` + `cal_ensemble_<model>_<algo>`,
  `paradise_snotel_wa` + `forcing_ensemble_<forcing>`,
  `Bow_at_Banff_multivar` + `bow_exp*`, `Bow_at_Banff_lumped` + `pipeline_bow`,
  Iceland_regional / Iceland_coastal / Iceland_coastal_elev).
- Scripts degrade gracefully: missing inputs skip the affected figure/panel
  and the rest still render.

Run any script with the venv active, e.g.:

```bash
source venv/bin/activate
SYMFLUENCE_DATA_DIR=/path/to/SYMFLUENCE_data \
    python examples/paper_case_studies/plotting/01_domain_definition/plot_domain_final.py
```

## Figure map

Status meanings:

- **fully automated** — public experiment outputs are sufficient and the
  complete plotting program is staged here;
- **additional archived input required** — plotting code is staged, but a
  named intermediate or external asset must also be supplied;
- **partially automated** — the staged program covers only part of the
  published figure;
- **conceptual/manual** — the figure is not generated from experiment data.

| Paper figure | Script | Required experiment outputs | Status |
|---|---|---|---|
| Fig 1 (Paradise point domain) | `01_domain_definition/plot_domain_final.py` (`create_paradise_figure`) | 01 paradise domain (shapefiles + DEM) **plus** the ERA5 forcing-cell shapefile `domain_paradise_snotel_wa/shapefiles/forcing/forcing_ERA5.shp` from the 03 ERA5 config | needs-experiment-03 (ERA5 acquisition writes the forcing shapefile; smoke run skipped this figure) |
| Fig 2 (Iceland discretization) | `01_domain_definition/plot_domain_final.py` (`create_iceland_figure`) | 01 Iceland_regional / Iceland_coastal / Iceland_coastal_elev domains | ready — rendered in smoke run (`output/figure_4_1b_iceland.*`) |
| Fig 3 (Bow discretization 3×3) | `01_domain_definition/plot_domain_final.py` (`create_bow_figure`) | 01 Bow domains (lumped_era5, lumped_land_classes, semidistributed_era5/_elev/_elev_aspect, distributed) | ready — rendered in smoke run (`output/figure_4_1c_bow.*`) |
| Fig 4 (data pipeline / forcing transformation) | `07_data_pipeline/visualize_pipeline_paper.py` (`fig_paper_forcing`) | 07 `pipeline_bow` run on domain `Bow_at_Banff_lumped` (raw + basin-averaged + SUMMA-input forcing, HRU/ERA5 shapefiles). The architecture/observation figures additionally need a `pipeline_analysis_*.json` from the paper's unstaged `analyze_pipeline.py`. | **additional archived input required** — SUMMA-input/lapse panels come from experiment 07; panels d–f use placeholders without the archived analysis JSON. |
| Fig 5 (framework schematic) | — | — | **conceptual/manual** — drawn framework schematic; no data-driven plotting program exists. |
| Fig 6 (forcing ensemble @ Paradise) | `03_forcing_ensemble/create_publication_figures.py` (reads `results/*.csv` built by `03_forcing_ensemble/analyze_results.py`) | 03 `forcing_ensemble_{era5,aorc,conus404,rdrs}` calibrated runs on `paradise_snotel_wa` + SNOTEL SWE observations | needs-experiment-03 |
| Fig 7 (17-model ensemble hydrograph) | `02_model_ensemble/fig_ensemble_hydrograph.py` (loaders in `ensemble_analysis.py`) | 02 `run_1` DDS calibrations for the 17 models on `Bow_at_Banff_lumped_era5` (final_evaluation outputs + metric JSONs) | needs-experiment-02 (4/17 model runs present in the repro data at staging time) |
| Fig 8 (benchmarking) | `05_benchmarking/create_publication_figures.py` | 05 `benchmark` run (`domain_Bow_at_Banff_lumped_era5/evaluation/benchmark_{scores,flows,input_data}.csv`) | **fully automated** — rendered in the smoke run and exercised with deterministic fixture inputs by the paper-release acceptance test. |
| Fig 9 (calibration-algorithm ensemble) | `04_calibration_ensemble/create_publication_figures.py` + `create_consolidated_figures.py` (data prep: `analyze_results.py`) | 04 `cal_ensemble_hbv_<algo>` runs on `Bow_at_Banff_lumped_calibration_ensemble` | **partially automated** — staged scripts cover the HBV × algorithms slice; the published eight-model panel has no staged plotting program. |
| Fig 10 (multivariate / GRACE trade-off) | `06_multivariate_evaluation/fig2_tradeoff_v2.py` (primary); companions `fig3_validation_comprehensive.py`, `fig1_domain_v5.py` (+ shared `bow_banff_style.py`) | 06 `bow_exp1_streamflow_only`, `bow_exp2_tws_only`, `bow_exp3_joint`, `bow_exp4_moead_joint` on `Bow_at_Banff_multivar` + GRACE/streamflow observations; `fig1_domain_v5.py` additionally needs hand-staged satellite subsets (set `P3_MULTIVAR_ASSETS_DIR`) | **additional archived input required** for the domain panel; the trade-off plot is generated from experiment 06 outputs. |
| Fig 11 (parallel scaling) | `08_parallel_scaling/create_figures.py` (reads `analysis/*.json` built by `analyze_scaling.py` from timing CSVs) | 08 `calib_summa_{pp,mpi}_100evals_np*` runs; the timing CSVs (`results/timing_*_latest.csv`) come from the scaling harness described in `configs/08_parallel_scaling/README.md` | needs-experiment-08 |

## Provenance and adaptations

Source: the paper's Google Drive working folders
(`applications_and_validation /<N>. <Experiment>/scripts/`). Staged scripts
keep the original plotting logic; only the following was changed:

- hardcoded absolute data/figure paths replaced with the
  `SYMFLUENCE_DATA_DIR` convention and `plotting/output/`;
- domain names / experiment ids aligned to the shipped configs (e.g.
  02 run ids normalized to `run_1`, 04 ids to `cal_ensemble_hbv_<algo>`,
  06 exp3 to `nsga-ii_bow_exp3_joint`, 03 collapsed to the shared
  `paradise_snotel_wa` domain);
- observation/forcing lookups also accept the current `data/…` domain layout
  (originals used a flat layout);
- per-figure try/except in `01` and `07` mains so partial reproductions
  still render the available figures.

Experiments cut from the paper (source folders `6. Model decision ensemble`,
`7. Sensitivity analysis`, `8. Large sample`, `9. Large domain`,
`13. Attribute analysis`, `14. Differentiable coupling`) were inventoried but
intentionally not staged.
