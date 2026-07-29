---
name: debug-delineation
description: >-
  Diagnose SYMFLUENCE domain delineation and discretization that produces a wrong
  geofabric — a basin that is too large or too small, an outlet sitting inside the
  polygon instead of on its boundary, more river segments than GRUs, missing coastal
  area, or a routing topology that no longer matches the shapefiles. Fault-tree for
  the TauDEM pipeline and the lumped / semidistributed / distributed / point paths.
when_to_use:
  - A delineated basin's area disagrees with the published gauge drainage area
  - The pour point/gauge is not on the edge of the delineated polygon
  - Segment and GRU counts disagree, or a reach has no catchment
  - Coastal or edge areas are missing, or basins extend offshore
  - Routed results look wrong after re-delineating a domain
---

# Debugging SYMFLUENCE Delineation

Delineation turns a DEM plus a pour point into the modelling units everything
downstream depends on: GRUs, the river network, and the HRUs discretized from
them. It runs in the `define_domain` step (`discretize_domain` follows). A wrong
geofabric is rarely loud — the workflow completes and the error shows up much
later as an area mismatch or a routing oddity. This skill is the fault-tree.
Paths relative to `src/symfluence/`.

## 1. The pipeline, end to end

```
define_domain
  → delineation.py picks a strategy from R.delineation_strategies
      DOMAIN_DEFINITION_METHOD = point | lumped | semidistributed | distributed
  → BaseGeofabricDelineator (geofabric/base/base_delineator.py)
      _condition_dem  → optional stream burning (processors/stream_burner.py)
      TauDEM chain    → processors/taudem_executor.py
                        pitremove → d8flowdir → aread8 → threshold
                        → moveoutletstostrm (snaps the outlet to a stream cell)
                        → streamnet / gagewatershed
      interim rasters land in  {project_dir}/taudem-interim-files/
  → per-strategy assembly of basins + river network  (delineators/*.py)
  → shapefiles/river_basins/{domain}_riverBasins_{suffix}.shp
    shapefiles/river_network/{domain}_riverNetwork_{suffix}.shp

discretize_domain
  → geospatial/discretization/  splits GRUs into HRUs by
    elevation / aspect / landclass / soilclass / radiation / combined
  → shapefiles/catchment/...
```

`{suffix}` encodes the definition method and its options (see
`_get_method_suffix` in `core/path_resolver.py`) — `lumped`,
`semidistributed`, `distributed_{cellsize}m`, `..._subset_{geofabric}`. If you
are looking at a file whose suffix doesn't match the config you think you ran,
that is the bug.

**The paths differ more than the names suggest.** This is the single most useful
thing to know when a bug appears in one mode but not another:

| mode | how the basin set is decided |
|------|------------------------------|
| `semidistributed` | TauDEM watershed raster → sub-basins, then a **graph traversal**: anchor on the primary outlet (`id == 0` in the pour-point file), build a directed graph from the network's `DSLINKNO` pointers, keep everything upstream. Interior gauges only break the network into more sub-basins; they never widen the domain. |
| `lumped` | TauDEM `gagewatershed` → **polygonize** → select the outlet's region → dissolve. Falls back, in order, to dissolving whole `streamnet` sub-watersheds, then to a valid-mask polygonization. |
| `distributed` | regular grid cells over the domain (`geospatial/geofabric/delineators/grid_delineator.py`), one routing reach per cell |
| `point` | a small buffer around the pour point; no network |

So a defect in the polygonize/select path shows up **only in lumped**, and a
defect in graph traversal shows up **only in semidistributed**. Before assuming a
shared bug, check whether the code you suspect is even on the other path.

## 2. Triage by symptom

### A. Basin area disagrees with the published gauge area
The most common real defect, and the easiest to confirm — compare against the
gauge's published drainage area before anything else.

1. **Is the outlet on the boundary?** Plot the basin and the pour point. The
   gauge should sit *on* the edge. If it sits inside, the basin extends
   downstream past the gauge — you are getting a whole sub-watershed that
   continues to the next confluence rather than the gauge's own catchment.
2. **Did a fallback fire?** In lumped mode the exact path can fail and degrade
   silently to a coarser one. Grep the run log for the fallback warnings and for
   `All lumped basin delineation strategies failed`. A fallback that "worked"
   still produces a different (usually larger) basin.
3. **Cross-check against `aread8`.** The contributing-area raster at the snapped
   outlet is the authority. `_warn_if_fallback_area_looks_inconsistent` in
   `geospatial/geofabric/delineators/lumped_delineator.py` does this
   automatically and logs when the polygon disagrees materially.
4. **Compare modes.** Delineate the same domain `lumped` and `semidistributed`.
   They should agree closely. A large disagreement localises the bug to whichever
   path is the outlier — see the table in §1.
5. **Check the snap distance.** See §D — the outlet may have snapped to the wrong
   stream.

### B. Segment and GRU counts disagree
The invariant is **one routable reach per river-basin GRU**.

- **More segments than GRUs:** TauDEM represents a confluence of three or more
  streams as two binary junctions joined by a **zero-length connector link**
  (start point == end point, `Length`/`Slope`/`StraightL` all 0, upstream
  contributing area equal to downstream). It is a topological placeholder, not a
  reach, and it has no catchment. `RiverGraphProcessor.drop_degenerate_reaches`
  in `geospatial/geofabric/processors/graph_processor.py` removes these during
  delineation and rewires the topology around them. If you are inspecting an
  older shapefile, expect one extra link per such confluence.
- **Reaches with no contributing GRU:** usually reaches lying outside the land
  mask. Ocean-masked builds remove offshore *basins* but their reaches can
  survive in the network. Check what fraction of a suspect reach's length falls
  inside the domain polygon.
- **More GRUs than reaches:** basins whose drainage never met `STREAM_THRESHOLD`,
  so no stream was defined for them. Usually tiny; confirm by area before
  worrying.

### C. Coastal / edge area missing or basins extend offshore
- Areas that drain directly to the sea are not part of any river basin. Set
  `DELINEATE_COASTAL_WATERSHEDS: true` to add coastal GRUs
  (`geospatial/geofabric/delineators/coastal_delineator.py`).
- **Count coastal units by the `is_coastal` flag, not by differencing two
  builds.** The coastal delineation can also absorb or dissolve river-basin GRUs,
  so `len(with_coastal) - len(without)` understates how many coastal units exist.
- Basins sprawling far offshore mean the sea was never masked: a flat
  zero-elevation ocean routes into thin radial "tentacle" basins. Controlled by
  `MASK_OCEAN_WATERSHEDS` / `SEA_LEVEL_THRESHOLD`; `GeometryProcessor.remove_spikes`
  cleans thin tentacles but will not remove bulk offshore area.

### D. The outlet snapped to the wrong place
`moveoutletstostrm` moves the pour point onto the nearest stream cell. The
snapped file (`taudem-interim-files/.../gauges.shp`) carries a `Dist_moved`
field.

- **`Dist_moved` is in grid cells, not metres.** A value of `4` on a ~90 m DEM is
  ~360 m, not 4 km. Misreading this unit sends you hunting for a nonexistent
  problem. Measure the true distance geometrically if it matters.
- A large snap means the pour point sits far from the stream network as the DEM
  sees it — usually coordinates in the wrong order/CRS, or a stream threshold so
  high that the local reach was never defined.
- A snap onto the *wrong* tributary changes the basin completely. Verify the
  snapped point is on the intended reach, not just near the original.

### E. Discretization: wrong HRU count, or HRUs with no attribute
- HRU counts come from `DOMAIN_DISCRETIZATION` (legacy alias of
  `SUB_GRID_DISCRETIZATION`) and its band settings, e.g. `ELEVATION_BAND_SIZE`,
  with `MIN_HRU_SIZE` / `MIN_GRU_SIZE` dropping slivers.
- **Nodata sentinels in attribute columns.** A sliver HRU can capture no raster
  cell centre and take a nodata value (e.g. `-9999`) in `elev_mean`. It will
  quietly become the minimum of any range you compute and the bottom of any colour
  scale. Always exclude or repair sentinels before summarising an attribute
  column; a fallback to the band's own mean is usually available.

### F. Routed results wrong after re-delineating
`topology.nc` is only rewritten by mizuRoute preprocessing. Re-delineating a
domain **without** re-running that step leaves the routing model using a topology
that describes the previous geofabric — a silent mismatch, since nothing compares
the two.

Check that the topology's `segId`/`hruId` counts match the current
`riverNetwork_*.shp` / `riverBasins_*.shp` feature counts. If they differ, re-run
mizuRoute preprocessing before trusting any routed output.

## 3. Fast diagnostic moves

- **Area against the gauge.** `gdf.to_crs(<equal-area or local UTM>).area.sum() / 1e6`
  versus the published drainage area. Two lines, catches most real defects.
- **Plot the basin with the pour point.** The gauge on the boundary is the
  single best visual invariant — both a too-large basin and a bad snap show up
  immediately.
- **Count the invariant:** one reach per river-basin GRU. Compare
  `len(river_network)` against `len(river_basins)` and, when a `gru_to_seg`
  column exists, use it rather than raw lengths — it excludes reaches that have
  no contributing GRU by construction.
- **Delineate the same domain two ways** (§A.4) and diff the areas.
- **Read the interim rasters.** `taudem-interim-files/` keeps every TauDEM stage;
  the failing step is usually visible there before it reaches a shapefile.
- **Check the suffix.** Confirm the file you are inspecting was produced by the
  config you think you ran.

## 4. Where each thing lives

| Concern | File |
|---------|------|
| Strategy dispatch, `define_domain` entry | `geospatial/delineation.py`, `geospatial/delineation_registry.py` |
| Shared TauDEM setup, DEM conditioning, interim dir | `geospatial/geofabric/base/base_delineator.py` |
| TauDEM command execution / allowed commands | `geospatial/geofabric/processors/taudem_executor.py` |
| Lumped basin + fallback chain | `geospatial/geofabric/delineators/lumped_delineator.py` |
| Semidistributed subset + graph traversal | `geospatial/geofabric/delineators/distributed_delineator.py` |
| Grid / point / coastal strategies | `geospatial/geofabric/delineators/{grid,point,coastal}_delineator.py` |
| Subsetting a pre-built hydrofabric (MERIT/TDX/NWS/HydroSHEDS) | `geospatial/geofabric/delineators/subsetter.py` |
| Raster → polygon conversion | `geospatial/geofabric/processors/gdal_processor.py` |
| Network topology, upstream tracing, degenerate reaches | `geospatial/geofabric/processors/graph_processor.py` |
| Geometry cleaning / despiking | `geospatial/geofabric/processors/geometry_processor.py` |
| Stream definition methods | `geospatial/geofabric/methods/{stream_threshold,curvature,slope_area,multi_scale}.py` |
| GRU → HRU discretization | `geospatial/discretization/`, `attributes/*.py` |
| Method suffix / path conventions | `core/path_resolver.py` |

## 5. Config keys that affect the geofabric

`DOMAIN_DEFINITION_METHOD` (`point`/`lumped`/`semidistributed`/`distributed`),
`DELINEATION_METHOD` (`stream_threshold`/`curvature`/`slope_area`/`multi_scale`),
`STREAM_THRESHOLD`, `DELINEATE_BY_POURPOINT`, `POUR_POINT_COORDS`,
`ROUTING_DELINEATION`, `DELINEATE_COASTAL_WATERSHEDS`,
`SUBSET_FROM_GEOFABRIC` + `GEOFABRIC_TYPE`, `MIN_GRU_SIZE`, `MIN_HRU_SIZE`,
`ELEVATION_BAND_SIZE`, and `SUB_GRID_DISCRETIZATION` (legacy alias
`DOMAIN_DISCRETIZATION`).

`STREAM_THRESHOLD` is the usual first knob: too high and small tributaries (and
their sub-basins) never appear; too low and the network fragments into far more
GRUs than intended.

Two ocean-masking overrides are read straight from the config dict and have no
schema field, so they do not appear in the shipped template and searching for
them there turns up nothing: `MASK_OCEAN_WATERSHEDS` (defaults to whatever
`DELINEATE_COASTAL_WATERSHEDS` is set to) and `SEA_LEVEL_THRESHOLD` (default
`0.0`). Both are consumed in
`geospatial/geofabric/delineators/distributed_delineator.py`.

## 6. TauDEM behaviours worth knowing

These are properties of the tool, not bugs in it, and each has caused real
confusion:

- **`gagewatershed` ids are 0-based.** Each watershed is labelled with the
  *gauge's* id from `moveoutletstostreams`, so a single-gauge run produces a
  watershed labelled `0` — never `1`. Code that selects `ID == 1` matches
  nothing.
- **`gagewatershed` regions are nested.** An upstream gauge's area is carved
  *out* of the downstream gauge's region rather than included in it. Dissolving
  every region is therefore correct only when the pour-point file holds a single
  gauge; with several, it fuses unrelated watersheds. Use the `-id` table
  (`id`/`iddown`, outlet has `iddown == -1`) to select the outlet's drainage.
- **Zero-length connector links** appear at 3-or-more-way confluences — see §B.
- **`Dist_moved` counts cells, not metres** — see §D.
- **The outlet sub-watershed extends past the gauge.** `streamnet` sub-watersheds
  end at confluences, not at your pour point, so anything that dissolves whole
  sub-watersheds will overshoot downstream.
