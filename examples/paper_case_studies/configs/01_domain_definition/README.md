# Experiment 01 — Domain Definition Across Scales (§2.1, Figs 1–3, Table 1)

Fourteen configs define hydrological domains from point scale (Paradise
SNOTEL) through watershed scale (Bow River at Banff, ten discretization
variants) to regional scale (Iceland, three variants). Each config runs the
domain-definition stage only — `setup_project` → `create_pour_point` →
`acquire_attributes` → `define_domain` → `discretize_domain` — and stops
before any forcing download or model run. Together they produce the spatial
configurations of Figs 1–3 and the GRU/HRU/segment counts of Table 1.

## Configs

Counts below are the published values (also stated in each config header).

| Config | GRUs | HRUs | Segments | Notes |
|---|---|---|---|---|
| `config_paradise_point.yaml` | 1 | 1 | 0 | Point-scale, Paradise SNOTEL (WA) |
| `config_bow_lumped.yaml` | 1 | 1 | 0 | Single catchment GRU |
| `config_bow_lumped_elev_bands.yaml` | 1 | 12 | 0 | 200 m elevation bands |
| `config_bow_lumped_land_classes.yaml` | 1 | 9 | 0 | IGBP land-cover classes |
| `config_bow_lumped_elev_aspect.yaml` | 1 | 94 | 0 | Elevation × aspect classes |
| `config_bow_lumped_distributed_routing.yaml` | 1 | 1 | 49 | Lumped hydrology, distributed routing |
| `config_bow_lumped_elev_distributed_routing.yaml` | 1 | 12 | 49 | Elevation bands + distributed routing |
| `config_bow_semidistributed.yaml` | 49 | 49 | 49 | TauDEM sub-basin GRUs |
| `config_bow_semidistributed_elev.yaml` | 49 | 379 | 49 | Sub-basins + elevation bands |
| `config_bow_semidistributed_elev_aspect.yaml` | 49 | 2,596 | 49 | Sub-basins × elevation × aspect |
| `config_bow_distributed.yaml` | 2,335 | 2,335 | 2,335 | 1 km grid cells |
| `config_iceland_regional.yaml` | 1,894 | 1,894 | 1,895 | TauDEM river basins, Copernicus 90 m DEM (Fig 3a) |
| `config_iceland_coastal.yaml` | 2,284 | 2,284 | 1,895 | + coastal watersheds (Fig 3b) |
| `config_iceland_coastal_elev.yaml` | 2,284 | 7,185 | 1,895 | coastal + elevation-band HRUs (Fig 3c) |

(Counts measured from clean-environment runs of the pinned recipe,
Copernicus 90 m + threshold 5000; the manuscript's Table 1 / Fig 3
values are being updated to these.)

Key mechanics (documented in the config comments):

- Bow sub-basins are delineated with TauDEM from the Copernicus 90 m DEM;
  `stream_threshold: 3400` reproduces the published 49 GRUs.
- The Iceland domains are delineated with TauDEM from the Copernicus 90 m DEM
  (`stream_threshold: 5000`, the Iceland_National run recipe), with coastal
  watersheds added via
  `delineate_coastal_watersheds`.
- Hydrology and routing discretizations are independent: a lumped GRU can be
  paired with a 49-segment routing network (`delineation.routing:
  river_network`).

## Run

```bash
CFG=examples/paper_case_studies/configs/01_domain_definition

# One domain:
symfluence workflow run --config $CFG/config_bow_lumped.yaml

# All 14:
for f in $CFG/config_*.yaml; do
    symfluence workflow run --config "$f"
done
```

## Outputs

Each config writes its geometry to `SYMFLUENCE_data/domain_<name>/shapefiles/`
(catchment, river basins, river network, pour point) — one `domain_*`
directory per config (domain names differ per variant). No forcing or model
output is produced.

## Verify

Compare GRU/HRU/segment counts of the generated shapefiles against the table
above (= Table 1 in the paper). Feature counts, e.g.:

```bash
python -c "import geopandas as g; print(len(g.read_file('<...>/shapefiles/catchment/<file>.shp')))"
```

## Runtime

Minutes per Bow/Paradise config once attribute data is cached (the first config per region pays the downloads); the Iceland domains take hours (large DEM + national-scale TauDEM run)
(delineation + attribute acquisition). Requires no credentials
for attribute/DEM sources per the top-level README; no model executables.
