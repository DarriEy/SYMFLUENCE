# Section 3.3 — Draft text

> **Notes for authors:**
> - Figure number (currently "Figure 5") should be verified against final numbering.
> - Specific provider lists (USGS, WSC, SMHI, etc.) are drawn from the codebase; verify completeness against the release version.
> - The EASYMORE citation should reference Gharari et al. (2020) or the appropriate version used.
> - Pint library reference: https://pint.readthedocs.io/

---

## 3.3 Data Management

Hydrological modelling workflows require forcing fields, observational records, and static catchment attributes, each originating from different providers, formats, and spatiotemporal grids. SYMFLUENCE consolidates these heterogeneous inputs through a unified data management layer centred on the `DataManager` class, which dispatches work to three registered pipelines: forcing preprocessing, observation processing, and attribute processing (Figure 5). All three pipelines converge on a shared model-ready data store written in NetCDF-4 with CF-1.8 metadata conventions, indexed per hydrological response unit (HRU).

**Pipeline Registry.** Each pipeline is registered with the `DataManager` through a decorator-based registry (`@register_pipeline`), following the same extensibility pattern used elsewhere in the framework (Section 3.1). This design decouples the orchestrator from pipeline implementations: adding support for a new data source or processing chain requires registering a new pipeline class rather than modifying dispatch logic. At runtime the `DataManager` resolves the appropriate pipeline for the requested data type and delegates execution, maintaining a clear separation between orchestration and domain-specific processing.

**Data Acquisition.** Before any pipeline executes, raw data must be obtained. SYMFLUENCE supports three acquisition modes that share a common interface but differ in backend. *Cloud mode* retrieves gridded products directly from remote object stores and APIs (Zarr, S3, OpenDAP). *MAF mode* targets high-performance computing environments, invoking `gistool` and `datatool` under SLURM to stage large datasets from institutional archives. *User-supplied mode* bypasses acquisition entirely, reading pre-staged local files that conform to the expected directory layout. The acquisition mode is selected once in the configuration and applies uniformly across all pipelines, so users switching between laptop and cluster workflows need only change a single setting.

### 3.3.1 Forcing Preprocessing

The forcing pipeline transforms raw meteorological fields—typically on the native grid of a reanalysis or forecast product—into per-HRU time series suitable for model input (Figure 5a).

**Spatial Remapping.** Grid-to-catchment remapping is performed using pre-computed weight matrices following the EASYMORE approach. For each HRU, a sparse weight vector $\mathbf{w}^{\!\top}$ maps source grid cells to the target unit based on areal intersection, producing area-weighted averages that conserve the integrated field. Weight matrices are computed once during domain setup and reused across variables and time steps, amortising the geometric intersection cost.

**Temporal Processing.** Source fields are aggregated or disaggregated to the model time step. Sub-daily products may be accumulated to daily totals (e.g., hourly precipitation), while monthly climatologies can be temporally downscaled when higher-frequency inputs are unavailable. Temporal alignment ensures that all variables share a consistent calendar, handling leap-year conventions and time-zone offsets.

**Variable Standardisation.** Provider-specific variable names and units are mapped to a common internal vocabulary using the Pint unit library. A declarative mapping table defines the source name, target name, and unit conversion for each variable, eliminating ad-hoc conversion factors scattered through processing scripts.

**Elevation Correction.** For variables with strong elevation dependence—primarily temperature and precipitation—the pipeline applies lapse-rate corrections and precipitation scaling factors derived from the elevation difference between the source grid cell and the HRU centroid. This step is critical when coarse reanalysis grids span large elevation ranges within a single catchment.

### 3.3.2 Observation Processing

The observation pipeline ingests evaluation data from multiple providers and produces standardised, evaluation-ready time series (Figure 5b). Five observation families are currently supported: streamflow (USGS, WSC, SMHI), snow (SNOTEL, MODIS), soil moisture (SMAP, ESA CCI, ISMN), evapotranspiration (MODIS MOD16, FLUXNET), and terrestrial water storage (GRACE/GRACE-FO). Each source is handled by a dedicated retrieval-and-conversion module that manages API access, product-specific quality flags, and initial unit harmonisation.

**Backbone Processing.** After source-specific retrieval, all observation streams pass through a shared four-stage backbone. *Unit conversion* maps provider-native units to SI equivalents. *Quality control* applies flag-based filtering and range validation, removing suspect or missing-flagged records. *Gap handling* detects remaining data voids and either interpolates short gaps, masks long gaps, or leaves them for the model's evaluation logic to handle, depending on a configurable policy. *Temporal alignment* standardises time zones, calendar encoding, and timestamp conventions so that observations and model outputs are directly comparable without post-hoc shifting.

### 3.3.3 Attribute Processing

The attribute pipeline derives static catchment descriptors—soil properties, land-cover fractions, terrain metrics—and writes them as per-HRU parameter templates (Figure 5c).

**Acquisition and Mosaicking.** Global raster products (DEM tiles, soil maps, land-cover classifications) are downloaded and mosaicked into seamless layers covering the study domain. Terrain derivatives (slope, aspect, curvature) are computed from the mosaicked DEM at this stage, avoiding tile-boundary artifacts.

**Zonal Statistics.** Each raster layer is summarised per catchment polygon, computing the mean, mode, or area-weighted distribution as appropriate for the variable type (continuous vs. categorical). Because zonal computation over thousands of HRUs can be long-running, the pipeline implements a checkpoint-and-resume mechanism: completed HRU batches are persisted to disk so that interrupted runs restart from the last completed chunk rather than from the beginning.

**Output Formatting.** The resulting attribute table is written in the same NetCDF-4/CF-1.8 structure used by the other pipelines, ensuring that downstream model setup routines consume forcing, observations, and attributes through a single reader interface.

### 3.3.4 Model-Ready Data Store

All three pipelines deposit their outputs into a shared data store that serves as the sole entry point for model setup and evaluation (Figure 5, bottom). Storing forcing, observations, and attributes in a common NetCDF-4 structure with CF-1.8 conventions and per-HRU indexing eliminates format-translation glue code between data preparation and modelling stages. The store is self-describing: variable metadata, unit information, spatial reference, and provenance attributes are embedded in the files, enabling downstream tools to validate compatibility programmatically rather than relying on filename conventions or external documentation.
