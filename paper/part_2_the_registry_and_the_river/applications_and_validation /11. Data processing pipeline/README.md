# Section 4.12 — Data Processing Pipeline

Cross-scale evaluation of SYMFLUENCE's end-to-end data processing pipeline across
three canonical domains spanning the full range of scales in the paper.

## Domains

| Domain | Scale | HRUs | GRUs | Area (km²) | Paper sections |
|--------|-------|------|------|------------|----------------|
| Paradise SNOTEL | Point | 1 | 1 | 0.01 | 4.3, 4.10 |
| Bow at Banff | Watershed | 49 | 49 | 2,210 | 4.2, 4.4–4.7 |
| Iceland | Regional | 21,474 | 6,600 | 103,000 | 4.8, 4.9 |

## Experiment overview

Traces the 16-stage pipeline DAG through all three domains, quantifying:

- **Pipeline architecture**: 16 stages, 25 data-flow edges, 5 categories
- **Data transformation across scales**: array shape changes from raw grid to model-ready
- **Spatial remapping**: weight matrix structure (trivial → dense → highly sparse)
- **Variable standardisation**: ERA5 → CFIF mapping for 7 forcing variables
- **Observation coverage**: domain-specific obs availability and gap fractions
- **Scaling behaviour**: log-log relationships between GRU count, data volume, compression
- **Compression crossover**: point where basin-averaged exceeds raw forcing volume

## Directory structure

```
11. Data processing pipeline/
├── configs/
│   ├── config_Paradise_pipeline_era5.yaml   # Point-scale configuration
│   ├── config_Bow_pipeline_era5.yaml        # Watershed configuration (lumped)
│   ├── config_Bow_pipeline_distributed.yaml # Watershed configuration (49 GRUs)
│   └── config_Iceland_pipeline_era5.yaml    # Regional configuration
├── scripts/
│   ├── run_pipeline_experiment.py    # Execute pipeline stages & profile
│   ├── analyze_pipeline.py           # Post-hoc analysis (11 analyses, 3 domains)
│   ├── visualize_pipeline.py         # Publication-quality figures (5 figures)
│   ├── visualize_pipeline_data.py   # Data-centric figures (4 figures, actual data)
│   └── visualize_pipeline_paper.py  # Composite publication figures (3 figures)
├── analysis/          # JSON/CSV analysis outputs
├── figures/           # Generated figures
├── logs/              # Timestamped execution logs
├── section_4_12_draft.md   # Paper section draft (7 subsections)
└── README.md
```

## Running the experiment

```bash
# Analysis of existing pipeline outputs (3 domains)
python scripts/analyze_pipeline.py

# Generate publication figures (requires analysis output)
python scripts/visualize_pipeline.py --format png

# PDF output for paper
python scripts/visualize_pipeline.py --format pdf
```

## Analyses produced

| # | Analysis | Output |
|---|----------|--------|
| 1 | Data volume inventory (3 domains) | `pipeline_data_volumes_*.csv` |
| 2 | CF compliance check | JSON report |
| 3 | Remapping weight characterisation | JSON report |
| 4 | Observation coverage (3 domains) | JSON report |
| 5 | Variable standardisation audit | JSON report |
| 6 | Forcing I/O profiling | JSON report |
| 7 | Spatial compression ratios | `pipeline_compression_*.csv` |
| 8 | Stage dependency DAG | JSON (16 stages, 25 edges) |
| 9 | ERA5 → CFIF variable mapping | `pipeline_variable_mapping_*.csv` |
| 10 | Cross-scale summary | `pipeline_cross_scale_*.csv` |
| 11 | Data shape tracking | JSON (per-stage dimensions) |

## Figures produced

| Figure | File | Description |
|--------|------|-------------|
| 1 | `fig_layered_dag.png` | Layered DAG with swim-lanes per category, curved edges coloured by data product type, domain execution banner |
| 2 | `fig_data_flow_sankey.png` | Sankey/alluvial: data volumes through 4 pipeline columns with 3 colour-coded domain bands |
| 3 | `fig_multiscale_panel.png` | 3×4 panel: domain info, weight matrix structure, compression butterfly, storage footprint |
| 4 | `fig_observation_timeline.png` | Gantt-style timeline with 3 domain groups, experiment period overlays, section cross-references |
| 5 | `fig_scaling_law.png` | 2-panel log-log: GRU count vs data volume by category; GRU count vs compression ratio with break-even |

### Data-centric figures (`visualize_pipeline_data.py`)

| Figure | File | Description |
|--------|------|-------------|
| 6 | `fig_forcing_transformation.png` | 2×3 panel: temperature and precipitation through raw ERA5 grid → remapped HRUs → time series comparison |
| 7 | `fig_spatial_remapping.png` | 3-panel: ERA5 grid overlaid on HRU polygons, EASYMORE weight matrix heatmap, temperature mapped to HRUs |
| 8 | `fig_grace_remote_sensing.png` | GRACE TWS anomalies (3 Mascon solutions), annual cycle with ±1σ, data availability by solution |
| 9 | `fig_lapse_rate_effect.png` | Temperature vs elevation scatter, lowest/highest HRU time series, spatial correction magnitude map |

**Colour scheme**: Paradise = green, Bow = blue, Iceland = red (consistent across all figures).

### Composite publication figures (`visualize_pipeline_paper.py`)

Consolidates the 9 figures above into 3 publication-ready composites using `matplotlib.gridspec`.

| Figure | File | Description |
|--------|------|-------------|
| P1 | `fig_paper_architecture.png` | Pipeline architecture + scaling: layered DAG (top), log-log volume and compression panels (bottom). Panels (a)-(c). |
| P2 | `fig_paper_forcing.png` | 3×3 grid: spatial remapping geometry (row 1), variable transformation with lapse-rate (row 2), cross-scale weight matrices (row 3). Panels (a)-(i). |
| P3 | `fig_paper_observations.png` | Observation timeline (top), GRACE TWS time series + annual cycle (bottom). Panels (a)-(c). |

## Running the data-centric figures

```bash
# Generate data-centric figures showing actual forcing/obs data through pipeline
python scripts/visualize_pipeline_data.py --format png
```

## Running the composite publication figures

```bash
# Generate 3 composite figures (PNG, 300 DPI default)
python scripts/visualize_pipeline_paper.py --format png

# PDF output for paper submission
python scripts/visualize_pipeline_paper.py --format pdf --dpi 600
```

## Dependencies

- SYMFLUENCE (in PATH or via `sys.path`)
- numpy, pandas, matplotlib, xarray, geopandas
