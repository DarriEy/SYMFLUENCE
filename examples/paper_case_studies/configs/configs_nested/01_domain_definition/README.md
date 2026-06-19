# Experiment 1: Domain Definition Across Scales (Section 2)

## Scientific Question
How does SYMFLUENCE represent hydrological domains across spatial scales, from
point-scale to regional distributed modeling, and what discretization options
are available within each scale?

## All 14 Spatial Configurations (Table 2, Figure 1)

### Paradise — Point-scale (Section 2.1)
| Config | GRUs | HRUs | Segments | Figure |
|--------|------|------|----------|--------|
| `config_paradise_point.yaml` | 1 | 1 | 0 | 1a |

### Bow at Banff — Watershed-scale (Section 2.2)

**Lumped (single GRU, no routing):**
| Config | GRUs | HRUs | Segments | Figure |
|--------|------|------|----------|--------|
| `config_bow_lumped.yaml` | 1 | 1 | 0 | 1d |
| `config_bow_lumped_elev_bands.yaml` | 1 | 12 | 0 | 1g |
| `config_bow_lumped_land_classes.yaml` | 1 | 9 | 0 | 1j |
| `config_bow_lumped_elev_aspect.yaml` | 1 | 94 | 0 | 1k |

**Hybrid (lumped GRU + distributed routing):**
| Config | GRUs | HRUs | Segments | Figure |
|--------|------|------|----------|--------|
| `config_bow_lumped_distributed_routing.yaml` | 1 | 1 | 49 | 1l |
| `config_bow_lumped_elev_distributed_routing.yaml` | 1 | 12 | 49 | 1i |

**Semi-distributed (sub-basin GRUs):**
| Config | GRUs | HRUs | Segments | Figure |
|--------|------|------|----------|--------|
| `config_bow_semidistributed.yaml` | 49 | 379 | 49 | 1e |
| `config_bow_semidistributed_elev.yaml` | 49 | 379 | 49 | 1h |
| `config_bow_semidistributed_elev_aspect.yaml` | 49 | 2,596 | 49 | 1k |

**Distributed (gridded):**
| Config | GRUs | HRUs | Segments | Figure |
|--------|------|------|----------|--------|
| `config_bow_distributed.yaml` | 2,335 | 2,335 | 2,335 | 1f |

### Iceland — Regional-scale (Section 2.3)
| Config | GRUs | HRUs | Segments | Figure |
|--------|------|------|----------|--------|
| `config_iceland_regional.yaml` | 6,606 | 6,606 | 6,606 | 1b |
| `config_iceland_coastal.yaml` | 7,618 | 7,618 | 6,606 | 1c |
| `config_iceland_coastal_elev.yaml` | 7,618 | 21,474 | 6,606 | — |

## Key Configuration Choices
- **Point scale**: `DOMAIN_DEFINITION_METHOD: point` — single grid cell, no routing
- **Lumped**: `definition_method: lumped` — TauDEM delineation to a single GRU at the outlet
- **Semi-distributed (Bow)**: `definition_method: semidistributed` — TauDEM sub-basin
  delineation from the DEM (`stream_threshold` controls sub-basin count). This is the
  default; it does **not** subset TDX. The 90 m DEM with `stream_threshold: 3400` gives
  the published 49 GRUs (the 30 m DEM / threshold 5000 give very different counts).
- **TDX subset (Iceland)**: `subset_from_geofabric: true` + `geofabric_type: TDX` — extracts
  pre-existing sub-basins from the TDX-Hydro geofabric. Required for the Iceland domains;
  without `subset_from_geofabric: true` the workflow delineates with TauDEM instead.
- **Distributed**: `definition_method: distributed` + `grid_cell_size: 1000` — 1 km grid cells
- **Discretization names** (exact tokens the discretizer accepts): `grus`, `elevation`,
  `landclass`, `aspect`, `soilclass`, or a comma list such as `elevation,aspect`.
- **Aspect**: `aspect_class_number: 8` (cardinal + intercardinal directions)
- **Coastal**: `delineate_coastal_watersheds: true` adds terminal ocean-draining basins
- **Hybrid routing**: `delineation.routing: river_network` (+ `stream_threshold`) delineates
  a routing network alongside a single lumped GRU (Figures 1i, 1l)

## Key Insight (Section 2.4)
Discretization choices for hydrology and routing are independent in SYMFLUENCE.
A lumped GRU can be paired with distributed routing (Figure 1l), and elevation-band
HRUs can be combined with semi-distributed routing (Figure 1i). This decoupling
allows independent control of vertical complexity (elevation, land cover, aspect)
and horizontal complexity (routing network density).

## Paper Figures/Tables
- Table 2: All 14 spatial configurations with GRU/HRU/segment counts
- Figure 1: Domain definition across scales (3x3 Bow grid + Paradise + Iceland)

## Data Requirements
- Paradise: ERA5 forcing, SNOTEL observations (station 679)
- Bow: ERA5 forcing, WSC streamflow (station 05BB001), Copernicus 90 m DEM (TauDEM delineation)
- Iceland: CARRA forcing, IMO streamflow, TDX-Hydro geofabric (subset)
