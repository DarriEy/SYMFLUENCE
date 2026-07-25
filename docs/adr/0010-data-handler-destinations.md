# ADR-0010: Destination mapping for the in-tree data handlers (community-service liftoff)

Date: 2026-07-25
Status: **Draft — destination decisions pending for the "Ambiguous" section**

## Context

The service decomposition anticipates lifting the in-tree data handlers into
the four community services (CFS forcing, CAS attributes, CSFS streamflow
observations, COS non-streamflow observations). The architecture is ready:
handlers are consumed registry-first behind the versioned
`AcquisitionBackend` protocol (contract 0.6.0), the framework imports and
runs with the handler subpackages physically absent (handlers-absent smoke),
and the layering guard forbids handler-module imports from outside `data/`.

What remains is the per-handler destination call. This table is the draft;
unambiguous rows ship as proposed, the Ambiguous section lists the decisions
that need a domain owner's sign-off.

## Proposed destinations

### CFS — community-forcing-service (meteorological forcing)

`aorc`, `cds_datasets` (CDS infra), `chirps`, `conus404`, `daymet`,
`em_earth`, `era5`, `era5_cds`, `era5_land`, `era5_processing`, `hrrr`,
`merra2`, `mswep`, `gpm`, `nex_gddp`, `nldas`, `rdrs`, `worldclim`.

Paired framework pieces that travel with (or are exercised by) CFS datasets:
`preprocessing/dataset_handlers/` dispatches on the contract's `SchemaId`
(sidecar-manifest tier), so it can either travel to CFS or stay
framework-side as schema-driven preprocessing — see Ambiguous.

### CAS — community-attribute-service (static basin attributes)

`aridity_index`, `bedrock_depth`, `can_height`, `dem`, `glacier`, `glclu`,
`glhymps`, `glwd`, `gssurgo`, `hydrolakes`, `hydrosheds`, `landcover`,
`merit_basins`, `merit_hydro`, `pelletier`, `polaris`,
`root_zone_storage`, `soilgrids`, `soilgrids_properties`, `tdx_hydro`,
`wokam`, `modis_lai`, `modis_ndvi` (as static/climatological attributes;
see Ambiguous for the MODIS split), plus
`preprocessing/attribute_processors/` (the external path is already proven
by climaclass via the `symfluence.attribute_processors` entry point).

### CSFS — community-streamflow-service (streamflow observations)

Observation handlers: `usgs`, `wsc`, `grdc`, `ana`, `dga`, `hubeau`,
`smhi`; acquisition-side `grdc`.

### COS — community-observation-service (non-streamflow observations)

`ascat_sm`, `canswe`, `cmc_snow`, `cnes_grgs_tws`, `esa_cci_sm`,
`fluxcom`/`fluxcom_et`, `ggmn`, `gldas_tws`, `gleam`/`gleam_et`,
`globsnow`, `grace`, `ims_snow`, `ismn`, `jrc_water`, `modis_et`,
`modis_lst`, `modis_sca`, `modis_snow`, `openet`, `sentinel1_sm`,
`sentinel2_snow`, `smap`, `smos_sm`, `snodas`, `snotel`, `soil_moisture`,
`ssebop`, `viirs_snow`.

### Stays in the framework

The contract and machinery: `backends/` (protocol, native adapter as long as
any in-tree handler remains, selection, errors), the acquisition/observation
registries and base classes, `model_ready/`, `cache/`, `preprocessing/`
core (resampling, remapping, alignment), shared infra bases
(`earthaccess_base`, `appeears_base` — see Ambiguous).

## Ambiguous — decisions needed

| Handler / piece | Options | Recommendation |
|---|---|---|
| `fluxnet` (acq + obs; towers feed both ET evaluation and forcing-adjacent use) | COS vs CFS | **COS** — it is consumed as observations (ET evaluator resolves `FLUXNET_ET`) |
| `camels`, `lamah_ice` (basin bundles: streamflow + attributes + forcing) | CSFS vs split per component | **CSFS** — primary consumer is streamflow calibration; document that their attribute/forcing components ride along |
| `nwm3_retrospective` (modeled streamflow used as pseudo-obs) | CSFS vs COS | **CSFS** — consumed on the streamflow path |
| `nws_hydrofabric` (geofabric/attributes for NGEN) | CAS vs stay (models-adjacent) | **CAS** |
| MODIS family split (`modis` base + et/lai/lst/ndvi/sca) | split CAS/COS (shared `modis`/`modis_utils` base duplicated or extracted to a shared lib) | **split as tabled above**, `modis` base travels to COS with CAS importing it cross-service — needs a call on the shared-base mechanics |
| Obs-side `chirps`, `daymet`, `era5_land`, `gpm`, `mswep` (forcing datasets consumed as precipitation/temperature observations) | COS vs CFS-with-obs-flavor | **COS** — the contract's flavour system already distinguishes forcing vs observation delivery, but the obs handlers are thin and their home should follow the consumer |
| `earthaccess_base`, `appeears_base`, `cds_datasets` (shared acquisition infra) | duplicate per service vs shared runtime lib in framework core-data vs a fifth micro-package | **keep in framework** (`data/acquisition/` base tier) — services import symfluence anyway |
| `preprocessing/dataset_handlers/` (per-dataset forcing preprocessing, SchemaId-dispatched) | travel to CFS vs stay schema-driven in framework | **stay initially** — the sidecar-manifest schema dispatch was built precisely so preprocessing needn't know which backend delivered the artifact; revisit if CFS gains datasets with novel schemas |
| Parity-gating endgame | keep native as reference until each family fully migrates vs flip `DATA_ACCESS` default to community per family | **migrate family-by-family**: a handler family leaves the tree only when its community counterpart holds a LIVE parity grade; `DATA_ACCESS` default flips per family at that moment |

## Consequences

- Each service absorbs its handlers plus their tests; the native backend
  shrinks per family until it adapts an empty registry (already proven safe).
- The framework keeps the contract, registries, bases, and schema-driven
  preprocessing — community services remain `symfluence`-dependent packages,
  so shared infra stays importable.
- Sequencing per the campaign plan: models split first, then handler
  families in parity-graded order (CSFS/COS observation handlers are the
  easiest first candidates; CFS forcing last, it has the deepest
  preprocessing pairing).
