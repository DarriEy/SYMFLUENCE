# ADR-0010: Destination mapping for the in-tree data handlers (community-service liftoff)

Date: 2026-07-25
Status: **Accepted** (destinations decided 2026-07-25)

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
`wokam`, `nws_hydrofabric`, and the **entire MODIS family** (the shared
`modis` base, `modis_utils`, and all products: `modis_et`, `modis_lai`,
`modis_lst`, `modis_ndvi`, `modis_sca`, `modis_snow` — one service owns
MODIS, acquisition and observation flavours alike), plus
`preprocessing/attribute_processors/` (the external path is already proven
by climaclass via the `symfluence.attribute_processors` entry point).

### CSFS — community-streamflow-service (streamflow observations)

Observation handlers: `usgs`, `wsc`, `grdc`, `ana`, `dga`, `hubeau`,
`smhi`; acquisition-side `grdc`.

### COS — community-observation-service (non-streamflow observations)

`ascat_sm`, `canswe`, `cmc_snow`, `cnes_grgs_tws`, `esa_cci_sm`,
`fluxcom`/`fluxcom_et`, `ggmn`, `gldas_tws`, `gleam`/`gleam_et`,
`globsnow`, `grace`, `ims_snow`, `ismn`, `jrc_water`, `openet`,
`sentinel1_sm`, `fluxnet` (observation flavour), the observation
flavours of `chirps`/`daymet`/`era5_land`/`gpm`/`mswep` (their
acquisition flavours live with CFS — dual-service datasets),
`sentinel2_snow`, `smap`, `smos_sm`, `snodas`, `snotel`, `soil_moisture`,
`ssebop`, `viirs_snow`.

### Stays in the framework

The contract and machinery: `backends/` (protocol, native adapter as long as
any in-tree handler remains, selection, errors), the acquisition/observation
registries and base classes, `model_ready/`, `cache/`, `preprocessing/`
core (resampling, remapping, alignment), shared infra bases
(`earthaccess_base`, `appeears_base` — see Ambiguous).

## Decisions (resolved 2026-07-25)

1. **fluxnet** → COS (consumed as observations; the ET evaluator resolves it
   registry-first).
2. **CAMELS / LamaH-ICE bundles** → CSFS whole; their attribute/forcing
   components ride along with the bundle.
3. **nwm3_retrospective** → CSFS (pseudo-observations on the streamflow path).
4. **MODIS family** → CAS, wholly: the shared base and every product travel
   together so no cross-service base dependency exists. Consumers keep
   resolving products registry-first (e.g. the ET evaluator's `modis_et`),
   served by CAS after liftoff.
5. **Dual-flavour datasets** (`chirps`, `daymet`, `era5_land`, `gpm`,
   `mswep`) → both services: acquisition (forcing) flavour with CFS,
   observation flavour with COS. The contract's flavour system already keeps
   the two deliveries distinct.
6. **Shared acquisition infrastructure** (`earthaccess_base`,
   `appeears_base`, `cds_datasets`) → moved upstream out of `handlers/` into
   `data/acquisition/` (implemented alongside this ADR; shims left at the
   old paths). The framework keeps it; services import it from symfluence.
7. **`preprocessing/dataset_handlers/`** → machinery (base, registry,
   schema dispatch) stays framework-side — it is already seam-shaped:
   per-dataset ``*_utils`` self-register via ``@R.dataset_handlers.add`` and
   the package loads them fail-safe, so each is individually extractable
   with its CFS dataset later.
8. **Parity-gating endgame** → family-by-family: a handler family leaves the
   tree only when its community counterpart holds a LIVE parity grade, and
   the ``DATA_ACCESS`` default flips per family at that moment.

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
