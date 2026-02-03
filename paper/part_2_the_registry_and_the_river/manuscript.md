# The Registry and the River: Architectural Patterns for Community Hydrological Modeling

**Authors:** Darri Eythorsson, Nicolas Vasquez, Cyril Thébault, Frank Han, Kasra Keshavarz, Wouter Knoben, Dave Casson, Mohammed Ismail Ahmed, Ashley Van Beusekom, Hongli Liu, Befekadu Taddesse Woldegiorgis, Camille Gautier, Katherine Reece, Peter Wagener, Ignacio Aguirre, Paul Coderre, Neharika Bhattarai, Junwei Guo, Shadi Hatami, David Tarboton, James Halgrem, Jordan Reed, Steve Burian, Raymond Spiteri, Alain Pietroniro, and Martyn Clark

## Abstract

The companion paper (Eythorsson et al., 2025a) argued that predictive stagnation in computational hydrology reflects an infrastructure deficit rather than a scientific one, and proposed five architectural principles -- declarative specification, registry-based extensibility, end-to-end orchestration, automatic provenance, and separation of scientific from computational concerns -- as a path toward shared community infrastructure. This paper presents the concrete realization of those principles in SYMFLUENCE (SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexus for Computational Exploration), a four-tier layered architecture that integrates 25 hydrological models spanning Fortran, C, Python, R, and Julia; 41 data acquisition handlers; 31 observation processing pipelines; and 21 optimization algorithms within a single declarative, registry-governed, provenance-capturing workflow system. We describe each architectural tier in detail: the configuration system that replaces imperative scripting with type-safe, validatable YAML specifications; the registry pattern that enables decentralized model contribution without core code modification; the workflow orchestrator that manages the complete modeling lifecycle from data acquisition through calibration and evaluation; the data management layer that unifies heterogeneous forcing, observation, and attribute sources behind a common interface; the model integration architecture that standardizes interaction with models of radically different heritage and design; and the optimization framework that decouples algorithms from models. Throughout, we emphasize the governance implications of architectural choices: how interface definitions function as social contracts, how registries enable coordination without committees, and how layered separation of concerns converts the impossible generalist problem into manageable specialization. The architecture is offered not as a monolithic prescription but as a reference implementation of patterns the community can adopt, adapt, or independently re-implement to achieve interoperability across the growing ecosystem of hydrological tools.

---

## 1. Introduction

The first paper in this series (Eythorsson et al., 2025a) documented what we called the impossible generalist problem: the compound expectation that each computational hydrologist will independently master field observation, process understanding, mathematical foundations, legacy numerical codes, data science at scale, parameter estimation, optimization algorithms, differentiable modeling, machine learning, high-performance computing, and reproducibility infrastructure -- before addressing their actual scientific question. We argued that this expectation reflects not a talent deficit but an infrastructure deficit, and that the path forward lies in shared architectural principles rather than individual heroism. Crucially, we argued that until models can be deployed without implementation error confounding results, the community cannot assess whether process-based models are scientifically adequate or merely inadequately deployed -- that workflow infrastructure is a prerequisite to scientific inference, not an afterthought to it.

This paper describes how those principles translate into engineering practice. Where the companion paper asked *why* the community needs shared architecture, this paper asks *how* -- and answers with a specific, implemented system that demonstrates the feasibility of the proposed approach.

The system is SYMFLUENCE. It is not the only possible implementation of the principles we advocate, and we do not claim it should be the only one. But it is a complete one -- spanning the full modeling lifecycle from data acquisition through calibration and evaluation, integrating models written in five programming languages across three decades of software conventions, and operating on platforms from laptops to HPC clusters. Its architecture embodies choices that other frameworks could adopt independently: a configuration system that separates scientific intent from computational implementation, a registry pattern that enables extensibility without central gatekeeping, and a layered design that allows specialization without isolation.

The V4.0 paper from which this work descends framed the integration challenge through Alfred North Whitehead's concept of *concrescence*: "the growing together of distinct elements into a unified actual entity." We retain this framing deliberately. SYMFLUENCE does not merely couple models; it enacts concrescence by fusing the technical, procedural, and epistemic dimensions of modeling -- transforming fragmented workflows into coherent systems of inference. The architecture described below is the mechanism through which that concrescence is achieved: registries that grow together contributions from independent groups, configurations that unify scientific intent with computational execution, and orchestration that binds discrete workflow stages into a single reproducible process.

The architectural decisions described here were not made in a vacuum. They respond to specific technical barriers documented across the hydrological modeling literature: the glue-code fragmentation that Hutton et al. (2016) identified as the primary obstacle to reproducibility; the workflow automation that Knoben et al. (2022) and Keshavarz et al. (2024) demonstrated for individual pipeline stages; the model interoperability that the Basic Model Interface (Peckham et al., 2013; Hutton et al., 2020) and eWaterCycle (Hut et al., 2022) pursued through standardized execution interfaces; the multi-hypothesis flexibility that SUMMA (Clark et al., 2015a,b), FUSE (Clark et al., 2008), and Raven (Craig et al., 2020) achieved through configurable process representations; and the data management challenges that McCabe et al. (2017) and Bierkens et al. (2015) documented as the "data deluge" confronting Earth system science.

Each of these efforts solved part of the problem. SYMFLUENCE attempts to solve the integration problem: connecting these partial solutions into a coherent whole through deliberate architectural design. The following sections describe that design in sufficient detail to serve both as documentation and as a pattern language that other groups can apply to their own systems.

### 1.1 Design principles

Five design principles, articulated in the companion paper and grounded in software engineering practice, guided every architectural decision in SYMFLUENCE. We restate them here briefly, as each subsequent section demonstrates their concrete implementation.

**Declarative specification.** Experimental designs should be expressed as configurations, not as scripts. A researcher specifies *what* an experiment should do; the framework determines *how*. The configuration file becomes the primary artifact of record -- human-readable, machine-executable, version-controllable, and diffable.

**Registry-based extensibility.** Adding new components (models, data sources, algorithms) should not require modification to existing code. Components declare their capabilities and register themselves; the framework discovers them at runtime. The registry is a governance model: an architectural social contract that specifies what the community expects of a contribution and what it provides in return.

**End-to-end orchestration.** Partial automation that addresses individual workflow stages still recreates fragmentation at the interfaces between stages. Only a framework that manages the complete lifecycle -- from data acquisition through model execution, calibration, and evaluation -- can eliminate the integration burden entirely.

**Automatic provenance.** Reproducibility should be a byproduct of normal operation, not an additional burden. Every workflow execution should automatically record its computational context: software versions, resolved configuration, Git state, and platform details.

**Separation of concerns.** Scientific choices (which model, which forcing, which evaluation metric) should be expressible independently of computational choices (how many cores, which cluster, which file system). This separation is what ultimately converts the impossible generalist problem into manageable specialization.

### 1.2 Paper organization

Section 2 presents the system architecture overview: the four-tier layered design, the recurring architectural patterns, the workflow orchestration model, and the scale-invariance principle. Section 3 describes the configuration system in detail. Section 4 covers the data management layer: acquisition, preprocessing, and the model-ready data store. Section 5 presents the domain definition and spatial discretization system. Section 6 describes the model integration architecture: base classes, the component registry, and routing integration. Section 7 covers the calibration and optimization framework. Section 8 describes the analysis layer: metrics, multi-variable evaluation, visualization, and benchmarking. Section 9 discusses the user interface and deployment infrastructure. Section 10 reflects on the governance implications of these architectural choices.

---

## 2. System Architecture Overview

### 2.1 Four-tier layered architecture

SYMFLUENCE employs a four-tier layered architecture that separates user interfaces, workflow orchestration, domain-specific managers, and core infrastructure (Figure 1). Each tier communicates with adjacent layers through defined interfaces, enabling independent evolution while maintaining system cohesion. A single declarative YAML configuration file governs the entire stack, specifying the complete experimental design from domain definition through evaluation and eliminating the ad hoc scripting that the companion paper identified as a primary source of irreproducibility.

**Figure 1.** SYMFLUENCE system architecture diagram showing the four-tier layered design: User Interface Layer (Python API, CLI, AI Agent), Workflow Orchestration Layer (step sequencing, dependency management), Manager Layer (seven specialized facades), and Core Infrastructure Layer (configuration, path resolution, logging, profiling).

The layers are as follows:

**User Interface Layer.** The topmost layer provides multiple entry points for interaction. The Python API exposes the `SYMFLUENCE` class as the primary programmatic interface, enabling integration with Jupyter notebooks, custom scripts, and automated pipelines. The command-line interface offers equivalent functionality through a structured command hierarchy, supporting both interactive use and batch execution within job schedulers. An experimental AI-assisted interface enables natural-language specification of modeling tasks, building on the INDRA framework (Eythorsson & Clark, 2025). These interfaces share no execution logic; they serve solely to translate user intent into calls to the orchestration layer below.

**Workflow Orchestration Layer.** The orchestrator sequences the modeling lifecycle through discrete, dependency-aware steps. It maintains execution state, enforces preconditions between stages, and coordinates the domain managers that perform actual computation. By centralizing workflow logic in a dedicated layer, the architecture prevents the ad hoc control flow that characterizes script-based workflows, where execution order is implicit in code structure rather than explicit in system design.

**Manager Layer.** Seven specialized managers -- for project setup, data acquisition, domain definition, model execution, optimization, analysis, and reporting -- encapsulate subsystem complexity behind facade interfaces. Each implements the facade pattern, presenting simplified entry points to complex subsystems and shielding upper layers from implementation detail. Table 1 summarizes their responsibilities.

**Table 1.** Core Manager classes in SYMFLUENCE, their responsibilities, and the underlying complexity they encapsulate.

| Manager | Responsibility | Underlying Complexity |
|---|---|---|
| **ProjectManager** | Project initialization and structure | Directory creation, configuration snapshots, state tracking |
| **DataManager** | Data acquisition and processing | 41+ data source handlers, cloud APIs, caching, format conversion |
| **DomainManager** | Geospatial operations | Delineation algorithms, discretization strategies, geofabric integration |
| **ModelManager** | Model execution | 25 model implementations, preprocessing/postprocessing pipelines |
| **OptimizationManager** | Parameter calibration | 21 algorithms, parameter transformation, parallel evaluation |
| **AnalysisManager** | Performance evaluation | 100+ metrics, sensitivity analysis, benchmarking |
| **ReportingManager** | Visualization | Domain, optimization, and diagnostic plotters |

The facade pattern serves two purposes. First, it shields upper layers from implementation volatility: adding a new data source or model requires no changes to the orchestrator or interfaces. Second, it provides a natural extension point for community contributions -- new capabilities register with existing managers rather than requiring architectural modifications. Each manager inherits from a `BaseManager` class that provides shared infrastructure: configuration access, logging, path resolution, and error handling. Managers communicate through explicit method calls rather than shared state, maintaining clear boundaries between responsibilities.

**Core Infrastructure Layer.** The foundation layer provides cross-cutting services consumed by all components above. The configuration subsystem parses and validates YAML specifications, producing immutable configuration objects that flow upward through the architecture. The path resolver abstracts filesystem operations, translating logical resource identifiers into platform-appropriate paths. The logging infrastructure provides unified output formatting with configurable verbosity. Type validation, implemented through Pydantic models, catches configuration errors at parse time rather than during execution.

### 2.2 Recurring architectural patterns

Three patterns recur throughout the layered design.

**The registry pattern** enables runtime discovery of pluggable components -- models, data sources, metrics, and analysis methods -- by having them register themselves upon import. Components announce their presence through decorators; the framework enumerates available capabilities without hard-coded lists. This pattern is the architectural backbone of SYMFLUENCE's extensibility story, and we discuss its governance implications in detail in Section 10.

```python
@ComponentRegistry.register_preprocessor('SUMMA')
class SUMMAPreprocessor(BaseModelPreProcessor):
    """Register SUMMA preprocessor at import time."""
    ...

@ComponentRegistry.register_runner('SUMMA', method_name='run')
class SUMMARunner(BaseModelRunner):
    """Register SUMMA runner at import time."""
    ...
```

Adding a new model requires implementing the appropriate interfaces and applying the registration decorator. No modification to existing code is necessary. The orchestrator discovers the new model at runtime and makes it available alongside all existing models.

**Lazy initialization** defers resource allocation until first use. Managers are instantiated at startup but do not load data, compile models, or allocate memory until explicitly invoked. A `LazyManagerDict` pattern keeps startup time constant regardless of configuration complexity and enables the framework to validate configurations without incurring computational overhead.

**The mixin pattern** provides reusable behaviors across class hierarchies without deep inheritance chains. `ConfigurableMixin` composes six atomic mixins -- `LoggingMixin`, `ConfigMixin`, `ProjectContextMixin`, `FileUtilsMixin`, `ValidationMixin`, and `TimingMixin` -- granting consistent access to configuration, logging, file operations, and validation across all framework components.

### 2.3 Workflow orchestration

The `WorkflowOrchestrator` transforms declarative configurations into executable sequences, managing the complete modeling lifecycle from project initialization through analysis.

**Figure 2.** SYMFLUENCE workflow orchestration showing the six stage categories: project initialization, domain definition, model-agnostic preprocessing, model-specific operations, optimization, and analysis. Each stage produces well-defined artifacts that serve as preconditions for downstream stages.

**Pipeline structure.** The orchestrator defines fifteen stages grouped into six categories (Figure 2). Each stage produces a well-defined artifact -- shapefiles, NetCDF datasets, processed CSV files, or optimized parameter sets -- that serves as a precondition for downstream stages. Each stage corresponds to a method on the `WorkflowOrchestrator` class, which delegates to the appropriate manager, maintaining separation between orchestration logic (*what* to do and *when*) and domain logic (*how* to do it).

**Dependency management.** Stages encode explicit dependencies that the orchestrator enforces at runtime. Before executing any stage, the orchestrator verifies that all prerequisite stages have completed successfully -- for example, domain discretization cannot proceed until the basin shapefile produced by domain definition exists, and model execution requires that preprocessing has generated the necessary input files. Dependency violations produce informative errors identifying missing prerequisites rather than cryptic failures deep in execution. This explicit dependency tracking replaces the implicit ordering of procedural scripts, where execution sequence is encoded in line numbers rather than in declared relationships.

**Execution modes.** The orchestrator supports three execution modes. *Sequential execution* processes stages in dependency order, blocking until each completes. *Selective execution* allows users to specify individual stages or stage ranges; the orchestrator verifies that prerequisites have been satisfied before proceeding. *Forced re-execution* overrides completion tracking, re-running stages regardless of prior state -- essential when upstream data or configurations change.

**Completion tracking.** A persistent state records which stages have completed successfully. Upon completion, a marker file is written encoding the stage name, timestamp, configuration hash, and framework version. Configuration hashing enables automatic invalidation: if parameters relevant to a stage change between runs, the marker is considered stale and the stage re-executes, balancing efficiency against correctness.

**Error handling and recovery.** Stage execution is wrapped in error handling that captures failures, logs diagnostics, and records partial state. *Fail-fast mode* (default) halts on first error, preserving system state for debugging. *Continue-on-error mode* logs failures but proceeds to subsequent stages where dependencies permit. The orchestrator's resumption capability allows recovery from transient failures: after addressing the underlying issue, users re-invoke the workflow, and completed stages are skipped automatically.

### 2.4 Scale-invariance principle

A property of the architecture that deserves explicit naming is *scale-invariance*: the same 16-stage workflow DAG executes identically regardless of domain size, from a single-HRU point-scale flux tower experiment to a 21,474-HRU regional application spanning 103,000 km². Switching between scales requires changing only the domain coordinates, discretization parameters, and observation sources in the YAML configuration file. The framework automatically adapts data acquisition extents, remapping matrix dimensions, observation source selection, and output indexing based on the configuration.

This property is not merely convenient; it is architecturally consequential. Scale-invariance means that a workflow validated at the point scale is structurally identical to a workflow deployed at the regional scale -- the same stage dependencies, the same interface contracts between stages, and the same provenance capture. Bugs discovered at one scale are bugs at all scales, and fixes propagate uniformly. The CF-Intermediate Format (Section 4.2) is central to this property: the same ERA5-to-CFIF mapping produces identically structured outputs whether the target is 1 HRU or 21,474 HRUs, providing a well-defined interface between the data acquisition layer and the model execution layer that is invariant to scale.

The practical consequence is that workflow complexity is absorbed by the framework rather than imposed on the user. Total pipeline output may span three orders of magnitude -- from ~12 MB at the point scale to ~4 GB at the regional scale -- but the user's interaction with the system remains structurally unchanged.

---

## 3. Configuration System

The declarative specification principle requires a configuration system capable of expressing complete experimental designs while maintaining type safety, validation, and provenance. This section describes SYMFLUENCE's configuration architecture, which replaces ad hoc scripting with machine-readable specifications that serve simultaneously as execution instructions, documentation, and reproducibility artifacts.

### 3.1 Hierarchical YAML specification

SYMFLUENCE configurations employ a hierarchical YAML structure that organizes approximately 350 parameters into semantically coherent sections (Table 2). Rather than scattering experimental choices across multiple scripts and environment variables, this structure consolidates the complete experimental design into a single, version-controllable document.

**Table 2.** SYMFLUENCE configuration schema: top-level sections and their responsibilities.

| Section | Responsibility | Parameter Count |
|---|---|---|
| `system` | Runtime environment, parallelism, logging | ~11 |
| `domain` | Spatial extent, temporal bounds, discretization | ~38 |
| `data` | Geospatial data sources, observation retrieval | 10+ |
| `forcing` | Meteorological forcing dataset selection and processing | ~18 |
| `model` | Hydrological model selection and parameterization | 150+ |
| `optimization` | Calibration algorithm, objective functions, constraints | 50+ |
| `evaluation` | Observation data sources, performance metrics | 100+ |
| `paths` | File system locations, shapefile field mappings | ~37 |

This hierarchical organization mirrors the natural decomposition of hydrological experiments: a researcher specifying a new study modifies only the relevant sections, and a domain specification developed for one model can be combined with different model or optimization configurations without modification. A minimal working configuration requires only ten parameters; all others inherit validated, model-aware defaults:

```yaml
experiment:
  name: Bow_at_Banff_SUMMA
  start: "2004-01-01"
  end: "2010-12-31"
domain:
  pour_point: [51.17, -115.57]
forcing:
  dataset: ERA5
model:
  name: SUMMA
optimization:
  algorithm: DDS
  iterations: 1000
  objective: KGE
  calibration_period: ["2004-01-01", "2007-12-31"]
  evaluation_period: ["2008-01-01", "2010-12-31"]
evaluation:
  targets: [STREAMFLOW]
```

This configuration is both machine-executable and human-readable. It specifies the complete experimental design in 17 lines, and it is diffable: two experiments can be compared by comparing their configuration files, with differences in scientific choices immediately visible. A comprehensive configuration example spanning all sections appears in Appendix A.

### 3.2 Validation and type safety

Configuration errors represent a significant source of debugging effort in computational workflows. A misspelled parameter name, an invalid coordinate, or an incompatible combination of settings can produce cryptic failures hours into execution. SYMFLUENCE addresses this through comprehensive validation that catches errors at configuration load time rather than during execution.

**Pydantic foundation.** The configuration system builds on Pydantic, a data validation library that enforces type constraints through Python's type annotation system. Each configuration section is implemented as a Pydantic model with typed fields, enabling both static analysis and runtime validation. The `Literal` type constrains categorical parameters to valid options, preventing silent failures from typos (e.g., `'lumpd'` instead of `'lumped'`).

**Immutability.** All configuration models are frozen (`frozen=True`), preventing modification after construction. Immutability provides several benefits: configurations can be safely shared across threads and processes without synchronization concerns; the framework can cache derived values without invalidation logic; and accidental mutation -- a common source of subtle bugs -- becomes impossible. When changes are required, a new configuration object must be explicitly constructed, making the change visible and intentional.

**Cross-field validation.** Many configuration constraints span multiple fields. SYMFLUENCE implements six cross-field validators:

- *Temporal consistency*: experiment start precedes end; calibration periods fall within experiment bounds; evaluation periods are properly specified.
- *Coordinate validation*: pour point coordinates conform to the expected `'lat/lon'` format with valid ranges.
- *Model requirements*: model-specific parameters are present when their model is selected, consulting the model registry so new models can declare requirements without modifying core validation code.
- *Spatial mode consistency*: mismatches between domain definition method and model spatial mode produce warnings.
- *Optimization configuration*: algorithm existence, positive population sizes and iteration counts, and algorithm-specific parameter ranges.
- *Path validation*: required file paths exist and are accessible.

**Default value management.** Default values are declared directly in Pydantic field definitions, providing a single source of truth that is automatically documented and enforceable. The framework also maintains model-specific and forcing-specific default registries that provide appropriate values based on context -- ERA5 forcing defaults differ from RDRS defaults, and SUMMA model defaults differ from FUSE defaults.

**Error reporting.** When validation fails, the framework produces structured error reports that group issues by type, suggest corrections for likely typos using fuzzy matching, and indicate expected versus actual values. This transforms configuration debugging from searching through stack traces to reading structured diagnostics.

### 3.3 Configuration transformation and backward compatibility

The hierarchical configuration model represents a deliberate design choice, but hydrological modeling has a long history of flat key-value configurations. SYMFLUENCE bridges this through a bidirectional transformation system.

The `transform_flat_to_nested()` function converts flat dictionaries to nested `SymfluenceConfig` objects via a cached, thread-safe mapping (`FLAT_TO_NESTED_MAP`) auto-generated from the Pydantic model hierarchy. The inverse, `flatten_nested_config()`, converts back for compatibility with components that expect flat access. Deprecated keys are detected with warnings, guiding users toward the canonical form without breaking existing configurations.

Factory methods support multiple initialization pathways:

```python
config = SymfluenceConfig.from_file(path, overrides={}, use_env=True, validate=True)
config = SymfluenceConfig.from_preset('bow_at_banff_summa')
config = SymfluenceConfig.from_minimal({'DOMAIN_NAME': 'bow', ...})
```

### 3.4 Provenance capture

Reproducibility requires capturing not only the configuration but the complete computational context in which it was executed. SYMFLUENCE's provenance system automatically records this context at workflow initialization:

- **Python environment:** version, installed package versions with checksums, virtual environment path.
- **System information:** operating system, architecture, hostname, available memory and CPU cores.
- **Framework state:** SYMFLUENCE version, Git commit hash, branch name, presence of uncommitted changes.
- **Execution context:** working directory, environment variables (filtered for security), timestamp.

Each execution writes three artifacts to the project directory: the *resolved configuration* (full configuration after applying defaults, overrides, and validation -- exactly what the framework used), the *original configuration* (the user's input before resolution), and the *override record* (any programmatic or environment-based overrides applied). The resolved configuration uses canonical formatting (sorted keys, consistent indentation) that facilitates diff-based comparison between runs.

The `LoggingManager` provides hierarchical logging with separate console (concise, color-coded via the Rich library) and file (comprehensive, DEBUG-level) handlers. Upon workflow completion, execution summaries record timing per stage, resource utilization, completion status, and output inventories -- creating self-documenting result packages that support both reproducibility and performance analysis.

---

## 4. Data Management

Hydrological modeling workflows require forcing fields, observational records, and static catchment attributes, each originating from different providers, formats, and spatiotemporal grids. SYMFLUENCE consolidates these heterogeneous inputs through a unified data management layer centered on the `DataManager` class, which dispatches work to three registered pipelines: forcing preprocessing, observation processing, and attribute processing (Figure 3). All three pipelines converge on a shared model-ready data store written in NetCDF-4 with CF-1.8 metadata conventions, indexed per hydrological response unit (HRU).

**Figure 3.** SYMFLUENCE data management architecture showing the three processing pipelines (forcing, observation, attribute) converging on a shared model-ready data store.

### 4.1 Pipeline architecture and acquisition

Each pipeline is registered with the `DataManager` through a decorator-based registry (`@register_pipeline`), following the same extensibility pattern used throughout the framework. This design decouples the orchestrator from pipeline implementations: adding support for a new data source requires registering a new pipeline class rather than modifying dispatch logic.

SYMFLUENCE supports three acquisition modes that share a common interface but differ in backend:

- **Cloud mode** retrieves gridded products directly from remote object stores and APIs (Zarr, S3, OpenDAP).
- **MAF mode** targets HPC environments, invoking `gistool` and `datatool` under SLURM to stage large datasets from institutional archives.
- **User-supplied mode** bypasses acquisition entirely, reading pre-staged local files that conform to the expected directory layout.

The acquisition mode is selected once in the configuration and applies uniformly across all pipelines, so users switching between laptop and cluster workflows need only change a single setting.

### 4.2 Forcing preprocessing

The forcing pipeline transforms raw meteorological fields -- typically on the native grid of a reanalysis or forecast product -- into per-HRU time series suitable for model input.

**Spatial remapping.** Grid-to-catchment remapping is performed using pre-computed weight matrices following the EASYMORE approach (Gharari et al., 2020). For each HRU, a sparse weight vector maps source grid cells to the target unit based on areal intersection, producing area-weighted averages that conserve the integrated field. Weight matrices are computed once during domain setup and reused across variables and time steps, amortizing the geometric intersection cost.

**Temporal processing.** Source fields are aggregated or disaggregated to the model time step. Sub-daily products may be accumulated to daily totals; monthly climatologies can be temporally downscaled when higher-frequency inputs are unavailable. Temporal alignment ensures consistent calendar handling, including leap-year conventions and time-zone offsets.

**Variable standardization and the CF-Intermediate Format (CFIF).** Provider-specific variable names and units are mapped to a common internal vocabulary -- the CF-Intermediate Format (CFIF) -- using the Pint unit library. CFIF is a scale-invariant intermediate representation that decouples dataset-specific conventions from model-specific input requirements. Accumulated energy fluxes (J m⁻²) become instantaneous fluxes (W m⁻²), accumulated precipitation (m) becomes precipitation rate (kg m⁻² s⁻¹), dew-point temperature is converted to specific humidity, and wind components are combined into scalar wind speed. This standardization layer means that the current implementation supports 10+ forcing datasets and 25 hydrological models through approximately 20 adapter implementations rather than the ~250 that would be required without the intermediate format. A declarative mapping table defines the source name, target name, and unit conversion for each variable, eliminating ad hoc conversion factors scattered throughout processing scripts. CFIF is a named architectural contribution: it provides the well-defined interface between the data acquisition layer and the model execution layer that makes independent evolution of both layers possible.

**Elevation correction.** For variables with strong elevation dependence -- primarily temperature and precipitation -- the pipeline applies lapse-rate corrections and precipitation scaling factors derived from the elevation difference between the source grid cell and the HRU centroid.

Nine forcing datasets are currently supported: ERA5, ERA5-Land, Daymet, AORC, CONUS404, HRRR, CARRA, RDRS, and EM-Earth. Each is handled by a dedicated acquisition handler that manages API authentication, rate limits, caching, and retry logic with exponential backoff.

### 4.3 Observation processing

The observation pipeline ingests evaluation data from multiple providers and produces standardized, evaluation-ready time series. Thirty-one observation handlers span five families:

| Family | Sources | Count |
|---|---|---|
| Streamflow | USGS, WSC, GRDC | 3 |
| Snow | SNOTEL, SNODAS, VIIRS | 3 |
| Soil moisture | SMAP, ISMN, Sentinel-1, ESA-CCI | 4 |
| Evapotranspiration | MODIS, FluxNet, GLEAM, OPENet | 4 |
| Other | LAI, LST, GRACE, GGMN, etc. | 17 |

After source-specific retrieval, all observation streams pass through a shared four-stage backbone: *unit conversion* to SI equivalents, *quality control* via flag-based filtering and range validation, *gap handling* with configurable interpolation or masking policies, and *temporal alignment* to standardize time zones, calendars, and timestamps for direct comparability with model outputs.

### 4.4 Attribute processing

The attribute pipeline derives static catchment descriptors -- soil properties, land-cover fractions, terrain metrics -- and writes them as per-HRU parameter templates. Global raster products (DEM tiles, soil maps, land-cover classifications) are downloaded, mosaicked into seamless layers covering the study domain, and summarized per catchment polygon via zonal statistics. A checkpoint-and-resume mechanism enables interrupted runs to restart from the last completed HRU batch.

### 4.5 Model-ready data store

All three pipelines deposit their outputs into a shared data store that serves as the sole entry point for model setup and evaluation. Storing forcing, observations, and attributes in a common NetCDF-4 structure with CF-1.8 conventions and per-HRU indexing eliminates format-translation glue code between data preparation and modeling stages. The store is self-describing: variable metadata, unit information, spatial reference, and provenance attributes are embedded in the files, enabling downstream tools to validate compatibility programmatically rather than relying on filename conventions.

---

## 5. Domain Definition and Spatial Discretization

SYMFLUENCE separates spatial discretization into two distinct levels -- Grouped Response Units (GRUs) and Hydrological Response Units (HRUs) -- decoupling the lateral routing structure from within-catchment process heterogeneity.

### 5.1 Spatial hierarchy

**GRUs** define the routing topology. Each GRU corresponds to a subcatchment draining to a single river segment, and the complete set of GRUs tiles the catchment without overlap. The drainage network connecting GRUs encodes upstream-downstream relationships that govern lateral water transfer through routing models such as mizuRoute.

**HRUs** capture sub-grid heterogeneity within each GRU. An HRU is defined by one or more discretizing attributes -- elevation bands, aspect classes, land-cover types, soil classifications -- intersected with the parent GRU boundary. Because HRUs nest strictly within their parent GRU, the mapping from each HRU to its routing segment is unambiguous.

This hierarchical design carries two practical consequences. First, increasing HRU complexity within a GRU does not alter the routing network -- practitioners can refine vertical process representation independently of lateral connectivity. Second, all HRUs within a GRU share calibrated parameter values by default; HRUs differ only in their forcing adjustments and static attributes, so adding discretization layers does not increase the calibration parameter space.

### 5.2 Domain definition methods

The domain definition system employs a strategy pattern coordinated by the `DomainDelineator` class. Each method registers itself with the `DelineationRegistry`, enabling runtime selection based on configuration. Four methods span the spectrum from point-scale to fully distributed representations:

**Table 3.** Domain definition methods supported in SYMFLUENCE.

| Method | Spatial Units | Routing | Use Cases |
|---|---|---|---|
| Point | Single bounding box | None | Flux tower sites, lysimeter experiments |
| Lumped | Single watershed | Optional river network | Bucket models, calibration studies |
| Semi-distributed | Multiple subcatchments | River network | Process-based distributed modeling |
| Distributed | Regular grid cells | D8 flow directions | Land surface models |

A distinguishing capability is *lumped-to-distributed routing*: when configured with `routing: river_network`, the delineator internally performs full subcatchment delineation while presenting a lumped domain to the model. This enables area-weighted disaggregation of lumped model outputs to subcatchments for distributed routing through mizuRoute, combining computational efficiency of lumped simulation with spatial detail in flow routing.

For applications requiring consistency with established hydrographic databases, SYMFLUENCE supports extraction of domains from existing geofabrics through graph-based upstream tracing. The `GeofabricSubsetter` implements this for three major geofabrics: MERIT-Basins (global, ~90 m), TDX-Hydro (global, ~30 m), and NWS Hydrofabric (CONUS, variable resolution).

### 5.3 HRU discretization

The `DomainDiscretizer` supports six discretization attributes that can be applied individually or in combination:

**Table 4.** Discretization methods supported in SYMFLUENCE.

| Method | Attribute Source | Classification | Typical Classes |
|---|---|---|---|
| GRUs only | Delineation output | None (1:1 mapping) | 1 per subcatchment |
| Elevation | DEM | Bands by interval | 4--10 bands (200 m default) |
| Land cover | MODIS/NLCD | Discrete classes | 5--15 classes |
| Soil | SoilGrids | Discrete classes | 5--12 classes |
| Aspect | DEM-derived | Cardinal directions | 4--8 classes |
| Radiation | DEM-derived | Bands by interval | 4--8 classes |

For combined discretization, the algorithm creates HRUs for each unique combination of attributes present within a GRU. Combining 5 elevation bands with 4 land-cover classes could produce up to 20 HRUs per GRU. The intersection algorithm handles spatially disconnected regions as MultiPolygon geometries, includes topology repair operations, degenerate geometry filtering, and boundary clipping to ensure HRUs nest exactly within their parent GRU.

All delineation and discretization methods produce standardized outputs: unique identifiers (GRU_ID, LINKNO), topological references (downstream segment ID), and geometric properties (area, length, slope). This standardization ensures that downstream model preprocessors can consume any spatial configuration through a single reader interface.

---

## 6. Model Integration

The diversity of hydrological models -- spanning conceptual rainfall-runoff formulations, process-based land surface schemes, and data-driven approaches -- presents a significant integration challenge. Each model brings distinct input requirements, execution mechanisms, and output formats. SYMFLUENCE addresses this heterogeneity through a unified component interface that standardizes model interaction while preserving model-specific capabilities (Figure 4).

**Figure 4.** Model integration component architecture showing the four standardized interfaces (preprocessor, runner, postprocessor, result extractor), the `ComponentRegistry` for runtime discovery, and the shared mixin infrastructure.

### 6.1 Component architecture

Each integrated model provides four components that implement abstract base classes:

**PreProcessor** (`BaseModelPreProcessor`) transforms SYMFLUENCE's standardized data formats into model-specific inputs: forcing data conversion (reformatting meteorological variables, adjusting units, interpolating timesteps), spatial attribute generation (mapping HRU characteristics to model parameter files), configuration file creation (control files, namelist files, decision files), and initial condition setup.

**Runner** (`BaseModelRunner`) executes the model: binary invocation (subprocess management for compiled executables), environment configuration (working directories, library paths), execution monitoring (progress tracking, timeout handling), and error capture (exit codes, log parsing).

**PostProcessor** (`BaseModelPostProcessor`) extracts and standardizes model outputs: output parsing (NetCDF, CSV, binary formats), variable extraction, unit conversion (model-native to SI), and spatial aggregation (HRU-level to basin-level where needed). All postprocessors produce results in a common format enabling consistent evaluation.

**ResultExtractor** (`ModelResultExtractor`) provides flexible access to any output variable for diagnostic and research purposes beyond the standard evaluation targets.

### 6.2 The ComponentRegistry

The `ComponentRegistry` provides centralized discovery and instantiation of model components through decorator-based registration. Separate registries are maintained for each component type -- `preprocessors`, `runners`, `postprocessors`, and `result_extractors` -- enabling fine-grained capability declaration.

This pattern yields three architectural benefits:

1. **Loose coupling** isolates model implementations from orchestration logic. The `ModelManager` need not know implementation details of any specific model.
2. **Extensibility** enables new models to be added by implementing component interfaces and registering. No modifications to existing code are required.
3. **Graceful degradation** allows partial model support -- a model might provide only a runner (for manual preprocessing) or only an extractor (for analyzing external outputs).

### 6.3 Shared infrastructure

All four component base classes inherit from a set of mixins that provide cross-cutting capabilities: `ModelComponentMixin` initializes model-specific paths and configuration; `PathResolverMixin` resolves directory structures; `ShapefileAccessMixin` loads catchment geometry with 20+ property accessors for shapefile column names; and `ConfigMixin` provides typed configuration coercion. This mixin architecture ensures consistent behavior across all model implementations while avoiding code duplication.

### 6.4 Supported models

SYMFLUENCE integrates 25 models spanning the spectrum from parsimonious conceptual formulations to complex process-based representations and data-driven approaches:

**Table 5.** Hydrological models integrated in SYMFLUENCE.

| Model | Type | Spatial Modes | Routing |
|---|---|---|---|
| SUMMA | Process-based LSM | Point, Lumped, Semi-dist., Distributed | External (mizuRoute) |
| MESH | Process-based LSM | Semi-distributed, Distributed | Internal |
| VIC | Process-based LSM | Distributed | External |
| Noah-MP | Process-based LSM | Distributed | External |
| NextGen/NGen | Modular framework | Lumped, Semi-distributed | Internal (t-Route) |
| FUSE | Flexible conceptual | Lumped, Semi-distributed | External (mizuRoute) |
| Raven | Flexible framework | Lumped, Semi-distributed | Internal |
| GR4J/GR6J | Conceptual | Lumped, Semi-distributed | External (mizuRoute) |
| HYPE | Conceptual | Lumped, Semi-distributed | Internal |
| HBV | Conceptual | Lumped, Semi-distributed | External (mizuRoute) |
| RHESSys | Ecohydrological | Distributed | Internal |
| LSTM | Machine learning | Lumped | None |
| GNN | Machine learning | Semi-distributed | Internal (learned) |
| jFUSE / cFUSE | Julia / C variants | Lumped | External |
| MizuRoute | Routing | N/A | IRF, KWT, Diffusive Wave |

### 6.5 Routing integration

Hydrological models that simulate vertical water balance processes require coupling with routing models to translate hillslope runoff into channel discharge. SYMFLUENCE automates this coupling through dependency resolution and standardized interfaces.

The `RoutingDecider` determines when routing is required based on configuration and model characteristics. When the `ModelManager` resolves the execution workflow, it automatically inserts routing models after their source models, ensuring correct execution order. The coupling requires coordination across three dimensions:

- **Runoff variable mapping:** each model produces runoff under different variable names and units (SUMMA: `basRunoff` in mm/timestep; FUSE: `q_routed` in mm/day). Model-specific adapters handle the translation.
- **Spatial correspondence:** remapping files specify which HRUs contribute to which river segments, with area-weighted aggregation when boundaries differ.
- **Temporal alignment:** the preprocessor handles temporal aggregation or disaggregation as needed.

---

## 7. Calibration and Optimization

Parameter calibration remains essential for hydrological model application, whether to compensate for structural deficiencies, adapt models to local conditions, or quantify predictive uncertainty. SYMFLUENCE provides a comprehensive optimization framework in which algorithms, objective functions, calibration targets, and execution strategies are decoupled components that can be combined flexibly (Figure 5).

**Figure 5.** Calibration and optimization framework showing the decoupled components: algorithm library, objective registry, calibration target registry, and execution strategies.

### 7.1 Architecture

The `OptimizationManager` orchestrates calibration workflows, delegating to model-specific optimizers retrieved from the `OptimizerRegistry`. Each optimizer inherits from `BaseModelOptimizer`, which provides parameter management, parallel execution, results tracking, and final evaluation as reusable capabilities. Algorithms implement a common callback interface for solution evaluation, parameter denormalization, and progress logging, enabling them to remain model-agnostic while optimizers handle model-specific details.

Three component registries supply the building blocks for any calibration experiment:

**Algorithm Library.** Twenty-one algorithms spanning:

| Category | Algorithms |
|---|---|
| Local search | DDS (Tolson & Shoemaker, 2007), Nelder-Mead |
| Population-based | PSO, DE, SCE-UA (Duan et al., 1992), GA, CMA-ES |
| Gradient-based | Adam, L-BFGS |
| Multi-objective | NSGA-II, MOEA/D |
| Bayesian/MCMC | DREAM (Vrugt et al., 2009), GLUE (Beven & Binley, 1992), ABC |
| Other | Basin-Hopping, Simulated Annealing, Async-DDS, Bayesian Optimization |

**ObjectiveRegistry.** Metrics for quantifying simulation-observation agreement. For maximization metrics (KGE, NSE), the framework transforms to minimization via `cost = 1 - metric`, enabling consistent algorithm implementations.

**Calibration Target Registry.** Maps variable types (streamflow, SWE, ET, soil moisture, groundwater, total water storage) to model-specific evaluator implementations through a three-tier lookup: dynamic registry, model-specific overrides, and generic defaults. A `MultivariateTarget` combines multiple variables into a single scalar objective via configurable weighting, enabling multi-variable calibration without algorithm modification.

### 7.2 Parameter normalization

All algorithms operate in a normalized [0, 1] parameter space, enabling consistent search behavior regardless of parameter magnitudes or units. The `BaseParameterManager` implements bidirectional transformation between normalized and physical spaces. Parameter bounds derive from model-specific sources -- SUMMA parses `localParamInfo.txt` and `basinParamInfo.txt`; HBV maintains a central bounds registry; FUSE uses hardcoded ranges -- making algorithm implementations fully portable across registered models.

### 7.3 Calibration loop

The calibration workflow proceeds through an iterative cycle. After initialization (algorithm selection, iteration budget, population size, parallelization settings), the loop repeats until convergence or budget exhaustion:

1. The algorithm proposes candidate solutions in normalized space.
2. Parameters denormalize to physical values and update model files.
3. Models execute -- in parallel for population-based methods.
4. Objective functions evaluate simulations against observations.
5. The algorithm updates its internal state based on fitness.

Progress logs and checkpoints persist at each iteration.

Final evaluation applies the best parameters to a full-period simulation spanning both calibration and evaluation windows. Metrics computed separately for each period enable detection of overfitting, following the split-sample testing philosophy that Klemeš (1986) advocated.

### 7.4 Execution distribution

Parallel execution employs a strategy pattern with automatic runtime selection. Three strategies cascade automatically:

1. **MPI strategy** distributes tasks round-robin across ranks on distributed-memory HPC clusters (detected via environment variables).
2. **ProcessPool strategy** uses Python's `ProcessPoolExecutor` for shared-memory parallelism on multi-core workstations.
3. **Sequential strategy** provides a fallback when parallelism is unavailable.

Process isolation prevents file conflicts during parallel evaluation. Each candidate evaluation receives a dedicated directory structure, and configuration files are automatically updated with process-specific paths, ensuring concurrent model executions do not interfere.

---

## 8. Analysis Layer

The analysis layer provides the evaluation, visualization, and diagnostic infrastructure that connects model outputs to scientific interpretation. Where previous layers produce simulations, this layer answers *how good* those simulations are, *why* they succeed or fail, and *whether* they outperform simple reference predictors.

### 8.1 Performance metrics

SYMFLUENCE implements over 100 performance metrics through a centralized `METRIC_REGISTRY` organized into four categories:

**Efficiency metrics:** Nash-Sutcliffe Efficiency (NSE; Nash & Sutcliffe, 1970), log-transformed NSE for low-flow emphasis, Kling-Gupta Efficiency (KGE; Gupta et al., 2009) decomposing performance into correlation, variability ratio, and bias ratio, modified KGE' (Kling et al., 2012), and KGE_np using Spearman correlation for non-Gaussian robustness.

**Error metrics:** RMSE, MAE, and normalized variants (NRMSE, MARE) for cross-site comparison.

**Bias metrics:** absolute bias and percent bias (PBIAS).

**Correlation metrics:** Pearson r, R², and Spearman rank correlation.

The `METRIC_REGISTRY` stores function references alongside metadata -- optimal value, valid range, optimization direction, units, and literature references -- and supports multiple aliases (e.g., 'kge', 'KGE', 'kling_gupta'). This centralized design guarantees that any metric referenced by name resolves to a single, validated implementation.

### 8.2 Multi-variable evaluation

Robust model assessment requires evaluation against multiple hydrological variables, since models may reproduce streamflow accurately while misrepresenting internal states. SYMFLUENCE implements a registry-based evaluator system where variable-specific evaluators inherit from a common `ModelEvaluator` base class:

- **StreamflowEvaluator:** automatic output format detection (standalone, routing, or coupled), mass-flux-to-volumetric conversion, catchment area resolution through a priority cascade.
- **ETEvaluator:** supports MODIS MOD16A2, FLUXCOM, FluxNet, and GLEAM with source-specific unit conversions and quality filters.
- **SnowEvaluator:** assesses SWE (continuous) and SCA (fractional, from MODIS/Landsat), applying appropriate metric types for each.
- **SoilMoistureEvaluator:** integrates SMAP, ESA CCI, and ISMN with automatic depth matching and configurable tolerances.
- **GroundwaterEvaluator:** well-based evaluation with automatic datum alignment and GRACE-based evaluation summing simulated storage components.
- **TWSEvaluator:** vertically integrated simulated storage compared against GRACE satellite observations, with linear detrending for glacierized basins.

Evaluators register via decorators (`@EvaluationRegistry.register('STREAMFLOW')`), and new evaluators need only subclass `ModelEvaluator` and apply the registration decorator.

### 8.3 Visualization

SYMFLUENCE provides publication-ready visualization through a modular plotting architecture comprising specialized plotter classes and reusable panel components:

- **DomainPlotter:** catchment boundaries, discretization units, river networks, elevation distributions.
- **ForcingComparisonPlotter:** raw gridded forcing vs. HRU-remapped values with remapping statistics.
- **OptimizationPlotter:** convergence curves, parameter evolution traces, Pareto frontiers.
- **ModelComparisonPlotter:** time series, flow duration curves, scatter plots, metric tables, monthly boxplots, residual analysis.
- **BenchmarkPlotter:** performance heatmaps comparing calibrated skill against reference predictors.
- **WorkflowDiagnosticPlotter:** validates each workflow step before calibration.

Plotters compose from six reusable panel types implementing a common rendering interface: `TimeSeriesPanel`, `FDCPanel`, `ScatterPanel`, `MetricsTablePanel`, `MonthlyBoxplotPanel`, and `ResidualAnalysisPanel`. All plotters output PNG at configurable resolution (default 300 DPI) into a structured directory hierarchy.

### 8.4 Benchmarking and sensitivity analysis

The `Benchmarker` class compares calibrated model skill against a hierarchy of reference predictors: mean flow, median flow, monthly climatology, daily climatology, long-term and short-term rainfall-runoff ratios. Following Schaefli and Gupta (2007), benchmarking contextualizes model performance -- a calibrated model achieving KGE = 0.7 provides limited value if monthly climatology achieves KGE = 0.65, whereas the same score represents substantial skill if the best benchmark achieves only KGE = 0.3.

The `SensitivityAnalyzer` implements variance-based methods (total-order Sobol' indices) to identify influential parameters, enabling dimensionality reduction without sacrificing performance. The `BaseStructureEnsembleAnalyzer` enables systematic exploration of model structural choices, generating all combinations, executing runs, and identifying optimal configurations.

---

## 9. User Interfaces and Deployment

### 9.1 Access modalities

SYMFLUENCE provides three access modalities that share a common configuration and execution architecture:

**Python API.** The `SYMFLUENCE` class serves as the primary programmatic interface. Users instantiate with a configuration object and invoke methods corresponding to individual workflow steps or complete workflows.

**Command-line interface.** The CLI organizes commands into seven categories: `workflow` (step execution, status, validation, resumption), `project` (initialization from presets), `binary` (compilation and validation of external models), `config` (management and validation), `job` (SLURM submission), `agent` (AI-assisted interaction), and `docs` (documentation generation). The Rich library provides structured terminal output with progress indicators.

**AI-assisted workflows.** The `AgentManager` orchestrates AI-assisted workflows through a multi-provider interface (OpenAI, Groq, Ollama) with automatic fallback cascade. Over 50 registered tools span codebase analysis, file operations, workflow management, and GitHub integration.

### 9.2 Computational environment

**Dependency management.** Dependencies span three categories: Python dependencies (scientific stack, geospatial libraries, visualization), system dependencies (GDAL, PROJ, NetCDF-C), and external model binaries (compiled executables for SUMMA, mizuRoute, FUSE, etc.). Python dependencies are specified with version bounds in `pyproject.toml`; system dependencies are managed through multiple installation pathways; model binaries are provided pre-compiled for common platforms with compilation scripts for others.

**Platform abstraction.** `pathlib` handles path differences; symbolic link support is detected with fallbacks; resource detection queries available CPU cores, memory, and GPU devices, adapting default parallelism automatically.

**HPC integration.** The framework detects SLURM, PBS, and SGE environments, adjusts parallelism to match allocated resources, and writes intermediate files to configurable scratch locations to avoid shared-filesystem I/O bottlenecks.

### 9.3 Continuous integration

Every commit triggers automated testing across the supported platform matrix shown in Table 6.

**Table 6.** SYMFLUENCE continuous integration matrix. Every combination of operating system, Python version, and installation method is tested on each commit.

| Dimension | Variants Tested |
|---|---|
| **Operating System** | Ubuntu 22.04, macOS 13 (Apple Silicon), Windows Server 2022 |
| **Python Version** | 3.10, 3.11, 3.12 |
| **Installation Method** | conda, pip, bootstrap |

This matrix produces 27 environment combinations (3 × 3 × 3), each executing the full test suite. Tests span unit tests (individual component behavior), integration tests (manager interactions), and end-to-end tests (complete workflows on small domains). Over 99 test files with 70+ pytest markers enable selective execution by speed, data requirements, component, model, and dataset. Binary compilation is validated on each platform, ensuring that model executables build correctly from source.

Code quality is enforced through automated pre-commit hooks (Ruff linting, Bandit security scanning, MyPy type checking, file hygiene, and notebook output stripping). The contribution workflow follows a fork-and-branch model with `develop` as the integration branch and `main` reserved for stable releases, governed by semantic versioning. Release workflows compile external modeling tools from source for Linux x86_64 and macOS ARM64 targets, and Sphinx documentation builds and deploys automatically.

This testing matrix transforms portability from a documentation claim to a verified property. When tests pass, users can be confident that the framework will function on their platform; when tests fail, developers are alerted before changes reach users.

---

## 10. Architectural Governance: Registries as Social Contracts

The companion paper argued that shared architectural principles could transform hydrological modeling from a craft practice into a cumulative engineering discipline. This section reflects on the governance implications of the patterns described above -- how architectural choices shape community behavior independently of any committee, funding agency, or editorial board.

### 10.1 The interface as social contract

The most consequential architectural choice in SYMFLUENCE is not any specific algorithm or data handler but the definition of the interfaces between components. When we define what a `BaseModelPreProcessor` must provide -- `run_preprocessing()`, `_validate_required_config()`, `_validate_forcing_data()` -- we are not merely specifying a software contract. We are specifying a *social* contract: what the community expects of a model integration and what it provides in return.

A group that implements this interface gains access to the full ecosystem: 41 data handlers will prepare their inputs, 21 optimization algorithms will calibrate their parameters, 100+ metrics will evaluate their outputs, and publication-ready visualizations will present their results. The cost of this access is compliance with the interface -- implementing the four component classes and registering them. The cost is proportional to the interface, not to the complexity of the host system.

This is a fundamentally different governance model from the current practice, where integrating a model into a workflow requires understanding the workflow's internal structure. The registry pattern scales with community participation rather than with central maintainer bandwidth.

### 10.2 Decentralized contribution

The registry pattern enables a specific form of decentralized scientific contribution. A research group developing a new snow model need not understand, modify, or even possess the complete SYMFLUENCE source code. They implement the defined interfaces, apply the registration decorators, and package their model as a standalone module. When installed alongside the framework, their model automatically becomes available for calibration, evaluation, and comparison against all other registered models.

No central authority needs to approve, review, or integrate each contribution. The architecture itself enables coordination. The pattern is analogous to how package managers (pip, conda, npm) enable decentralized software contribution: the registry defines the contract, and compliance with the contract is both necessary and sufficient for participation.

### 10.3 Separation as specialization enabler

The layered architecture converts the impossible generalist problem into manageable specialization. A process hydrologist can specify a scientific experiment without understanding MPI. An HPC specialist can optimize execution without understanding snow physics. A data scientist can add a new satellite product without understanding model internals. A graduate student can calibrate a model without writing format-conversion scripts.

This is not a hypothetical benefit. It is the operational consequence of consistent interface definitions between layers. When the boundary between "scientific choice" and "computational implementation" is architecturally enforced rather than merely aspirational, specialization becomes productive rather than isolating.

### 10.4 Limitations and open questions

We do not claim that SYMFLUENCE has solved the community architecture problem. Several limitations deserve explicit acknowledgment:

**Interface stability.** The interfaces described here reflect the current understanding of what model integration requires. As new modeling paradigms emerge -- differentiable models that require gradient flow through the execution pipeline, for instance -- these interfaces may need to evolve. Managing interface evolution without breaking existing integrations is an unsolved governance challenge.

**Adoption barriers.** The value of a registry-based ecosystem grows with the number of registered components. This creates a bootstrapping problem: the ecosystem is most valuable when many models are registered, but registration effort is only justified if the ecosystem already provides value. SYMFLUENCE addresses this by providing substantial standalone value (data acquisition, preprocessing, evaluation) even with a single model, but the network effect that the companion paper envisions requires broader adoption.

**Community governance.** Architectural choices are governance choices, but they are not *sufficient* for governance. Questions of interface versioning, contribution standards, quality assurance, and long-term maintenance require social institutions in addition to software patterns. The architecture enables coordination; it does not guarantee it.

**Performance trade-offs.** Abstraction layers introduce overhead. The facade pattern, mixin composition, and registry lookups add indirection that a hand-crafted, model-specific script would not incur. In practice, this overhead is negligible compared to model execution time -- a SUMMA calibration spends 99.9% of its wall clock in Fortran, not in Python dispatch -- but the principle should be acknowledged.

---

## 11. Conclusion

This paper has described the architecture of SYMFLUENCE: a four-tier layered system that implements the five design principles advocated in the companion paper (Eythorsson et al., 2025a). The system integrates 25 hydrological models, 41 data acquisition handlers, 31 observation processing pipelines, and 21 optimization algorithms within a single declarative, registry-governed, provenance-capturing workflow.

The contribution is not the specific implementation but the demonstration that the proposed principles are feasible, practical, and sufficient to address the infrastructure barriers documented across the hydrological modeling literature. More fundamentally, the architecture establishes the infrastructure necessary to distinguish scientific inadequacy from implementation failure. When models are deployed through fragmented, irreproducible workflows, poor performance cannot definitively indicate either the limits of process understanding or the presence of technical barriers. The architecture described here removes this ambiguity: by ensuring that models are configured correctly, forced consistently, and evaluated uniformly, it enables the community to conduct fair tests of whether detailed process representation improves prediction. This is the epistemological stake of infrastructure work -- workflow architecture is not merely an engineering convenience but a prerequisite to scientific inference. Declarative specification works: a 17-line YAML file can encode a complete experiment. Registry-based extensibility works: new models integrate through decorators, not through pull requests to a monolithic codebase. End-to-end orchestration works: dependency-aware stage management eliminates the glue code that Hutton et al. (2016) identified as the primary obstacle to reproducibility. Automatic provenance works: every execution produces self-documenting result packages. And separation of concerns works: the same scientific experiment can execute on a laptop or an HPC cluster by changing a single configuration section.

Whether the community adopts SYMFLUENCE specifically or re-implements these patterns in other frameworks is less important than whether it adopts the patterns themselves. The registry and the river are both systems of flow -- one of data and control, the other of water. Both require channels, both require connections, and both work best when the infrastructure is shared. The architecture described here is one way to build those channels. We offer it as a starting point, not an endpoint -- as a demonstration that the foundation the companion paper called for can, in fact, be poured.

---

## References

Addor, N., & Melsen, L. A. (2019). Legacy, rather than adequacy, drives the selection of hydrological models. *Water Resources Research*, 55(1), 378--390. https://doi.org/10.1029/2018WR022958

Beven, K., & Binley, A. (1992). The future of distributed models: Model calibration and uncertainty prediction. *Hydrological Processes*, 6(3), 279--298. https://doi.org/10.1002/hyp.3360060305

Bierkens, M. F. P., et al. (2015). Hyper-resolution global hydrological modelling: What is next? *Hydrological Processes*, 29(2), 310--320. https://doi.org/10.1002/hyp.10391

Clark, M. P., Slater, A. G., Rupp, D. E., Woods, R. A., Vrugt, J. A., Gupta, H. V., Wagener, T., & Hay, L. E. (2008). Framework for Understanding Structural Errors (FUSE): A modular framework to diagnose differences between hydrological models. *Water Resources Research*, 44(12), W00B02. https://doi.org/10.1029/2007WR006735

Clark, M. P., Kavetski, D., & Fenicia, F. (2011). Pursuing the method of multiple working hypotheses for hydrological modeling. *Water Resources Research*, 47(9), W09301. https://doi.org/10.1029/2010WR009827

Clark, M. P., et al. (2015a). A unified approach for process-based hydrological modeling: 1. Modeling concept. *Water Resources Research*, 51(4), 2498--2514. https://doi.org/10.1002/2015WR017198

Clark, M. P., et al. (2015b). A unified approach for process-based hydrological modeling: 2. Model implementation and case studies. *Water Resources Research*, 51(4), 2515--2542. https://doi.org/10.1002/2015WR017200

Craig, J. R., et al. (2020). Flexible watershed simulation with the Raven hydrological modelling framework. *Environmental Modelling & Software*, 129, 104728. https://doi.org/10.1016/j.envsoft.2020.104728

Duan, Q., Sorooshian, S., & Gupta, V. K. (1992). Effective and efficient global optimization for conceptual rainfall-runoff models. *Water Resources Research*, 28(4), 1015--1031. https://doi.org/10.1029/91WR02985

Eythorsson, D., & Clark, M. P. (2025). Toward Automated Scientific Discovery in Hydrology: The Opportunities and Dangers of AI Augmented Research Frameworks. *Hydrological Processes*, 39(1), e70065. https://doi.org/10.1002/hyp.70065

Eythorsson, D., et al. (2025a). Outstanding in every field: The case for shared architectural vision in computational hydrology. *Geoscientific Model Development* [companion paper].

Eythorsson, D., et al. (2025b). From configuration to prediction: Multi-model, multi-basin experiments with SYMFLUENCE. *Geoscientific Model Development* [companion paper].

Fatichi, S., et al. (2016). An overview of current applications, challenges, and future trends in distributed process-based models in hydrology. *Journal of Hydrology*, 537, 45--60. https://doi.org/10.1016/j.jhydrol.2016.03.026

Fenicia, F., Kavetski, D., & Savenije, H. H. G. (2011). Elements of a flexible approach for conceptual hydrological modeling: 1. Motivation and theoretical development. *Water Resources Research*, 47(11), W11510. https://doi.org/10.1029/2010WR010174

Freeze, R. A., & Harlan, R. L. (1969). Blueprint for a physically-based, digitally-simulated hydrologic response model. *Journal of Hydrology*, 9(3), 237--258. https://doi.org/10.1016/0022-1694(69)90020-1

Gharari, S., Clark, M. P., Mizukami, N., Knoben, W. J. M., Wong, J. S., & Pietroniro, A. (2020). Flexible vector-based spatial configurations in land models. *Hydrology and Earth System Sciences*, 24(11), 5953--5971. https://doi.org/10.5194/hess-24-5953-2020

Gil, Y., et al. (2016). Toward the geoscience paper of the future. *Earth and Space Science*, 3(10), 388--415. https://doi.org/10.1002/2015EA000136

Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria. *Journal of Hydrology*, 377(1--2), 80--91. https://doi.org/10.1016/j.jhydrol.2009.08.003

Hrachowitz, M., & Clark, M. P. (2017). HESS Opinions: The complementary merits of competing modelling philosophies in hydrology. *Hydrology and Earth System Sciences*, 21(8), 3953--3973. https://doi.org/10.5194/hess-21-3953-2017

Hut, R., et al. (2022). The eWaterCycle platform for open and FAIR hydrological collaboration. *Geoscientific Model Development*, 15(13), 5371--5390. https://doi.org/10.5194/gmd-15-5371-2022

Hutton, C., et al. (2016). Most computational hydrology is not reproducible, so is it really science? *Water Resources Research*, 52(10), 7548--7555. https://doi.org/10.1002/2016WR019285

Hutton, E. W. H., Piper, M. D., & Tucker, G. E. (2020). The Basic Model Interface 2.0. *Journal of Open Source Software*, 5(51), 2317. https://doi.org/10.21105/joss.02317

Kavetski, D., & Fenicia, F. (2011). Elements of a flexible approach for conceptual hydrological modeling: 2. Application and experimental insights. *Water Resources Research*, 47(11), W11511. https://doi.org/10.1029/2011WR010748

Keshavarz, K., Knoben, W. J. M., & Clark, M. P. (2024). Community workflows for advanced reproducibility in hydrologic modeling. *Hydrological Processes*, 38(1), e15044. https://doi.org/10.1002/hyp.15044

Klemeš, V. (1986). Operational testing of hydrological simulation models. *Hydrological Sciences Journal*, 31(1), 13--24. https://doi.org/10.1080/02626668609491024

Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios. *Journal of Hydrology*, 424--425, 264--277. https://doi.org/10.1016/j.jhydrol.2012.01.011

Knoben, W. J. M., et al. (2019). Modular Assessment of Rainfall-Runoff Models Toolbox (MARRMoT) v1.2. *Geoscientific Model Development*, 12(6), 2463--2480. https://doi.org/10.5194/gmd-12-2463-2019

Knoben, W. J. M., et al. (2022). Community Workflows to Advance Reproducibility in Hydrologic Modeling (CWARHM). *Water Resources Research*, 58(11), e2022WR032702. https://doi.org/10.1029/2022WR032702

Kollet, S. J., et al. (2010). Proof of concept of regional scale hydrologic simulations at hydrologic resolution utilizing massively parallel computer resources. *Water Resources Research*, 46(4), W04201. https://doi.org/10.1029/2009WR008730

McCabe, M. F., et al. (2017). The future of Earth observation in hydrology. *Hydrology and Earth System Sciences*, 21(7), 3879--3914. https://doi.org/10.5194/hess-21-3879-2017

Maxwell, R. M., Condon, L. E., & Kollet, S. J. (2015). A high-resolution simulation of groundwater and surface water over most of the continental US with the integrated hydrologic model ParFlow v3. *Geoscientific Model Development*, 8(3), 923--937. https://doi.org/10.5194/gmd-8-923-2015

Mölder, F., et al. (2021). Sustainable data analysis with Snakemake. *F1000Research*, 10, 33. https://doi.org/10.12688/f1000research.29032.2

Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I -- A discussion of principles. *Journal of Hydrology*, 10(3), 282--290. https://doi.org/10.1016/0022-1694(70)90255-6

Nearing, G. S., et al. (2021). What role does hydrological science play in the age of machine learning? *Water Resources Research*, 57(3), e2020WR028091. https://doi.org/10.1029/2020WR028091

Paniconi, C., & Putti, M. (2015). Physically based modeling in catchment hydrology at 50. *Water Resources Research*, 51(9), 7090--7129. https://doi.org/10.1002/2015WR017780

Peckham, S. D., Hutton, E. W. H., & Norris, B. (2013). A component-based approach to integrated modeling in the geosciences: The design of CSDMS. *Computers & Geosciences*, 53, 3--12. https://doi.org/10.1016/j.cageo.2012.04.002

Schaefli, B., & Gupta, H. V. (2007). Do Nash values have value? *Hydrological Processes*, 21(15), 2075--2080. https://doi.org/10.1002/hyp.6825

Tolson, B. A., & Shoemaker, C. A. (2007). Dynamically dimensioned search algorithm for computationally efficient watershed model calibration. *Water Resources Research*, 43(1), W01413. https://doi.org/10.1029/2005WR004723

Vrugt, J. A., et al. (2009). Accelerating Markov chain Monte Carlo simulation by differential evolution with self-adaptive randomized subspace sampling. *International Journal of Nonlinear Sciences and Numerical Simulation*, 10(3), 273--290. https://doi.org/10.1515/IJNSNS.2009.10.3.273

Whitehead, A. N. (1929). *Process and Reality: An Essay in Cosmology*. Macmillan, New York.

Wood, E. F., et al. (2011). Hyperresolution global land surface modeling: Meeting a grand challenge for monitoring Earth's terrestrial water. *Water Resources Research*, 47(5), W05301. https://doi.org/10.1029/2010WR010090

---

## Appendix A: Comprehensive Configuration Example

```yaml
# ============================================================
# SYMFLUENCE Configuration: Bow River at Banff
# Complete example demonstrating all configuration sections
# ============================================================

# --- System settings ---
system:
  SYMFLUENCE_DATA_DIR: /data/symfluence
  SYMFLUENCE_CODE_DIR: /code/symfluence
  num_processes: 8
  log_level: INFO

# --- Domain definition ---
domain:
  DOMAIN_NAME: bow_at_banff
  EXPERIMENT_ID: summa_era5_dds
  POUR_POINT_COORDS: 51.17/-115.57
  DOMAIN_DEFINITION_METHOD: subset
  GEOFABRIC: Merit
  HYDROLOGICAL_MODEL_SPATIAL_MODE: semi_distributed
  DISCRETIZATION_METHOD: elevation
  ELEVATION_BAND_SIZE: 200
  EXPERIMENT_TIME_START: "2004-01-01"
  EXPERIMENT_TIME_END: "2010-12-31"

# --- Data sources ---
data:
  SOIL_CLASS_DATA_SOURCE: SOILGRIDS
  LAND_CLASS_DATA_SOURCE: MODIS
  DEM_SOURCE: NASADEM

# --- Meteorological forcing ---
forcing:
  FORCING_DATASET: ERA5
  FORCING_START_YEAR: 2004
  FORCING_END_YEAR: 2010

# --- Model configuration ---
model:
  HYDROLOGICAL_MODEL: SUMMA
  summa:
    SUMMA_INSTALL_PATH: /code/summa/bin
    decisions:
      snowLayers: CLM_2010
      snowDenNew: pahinger_2001
      soilCatTbl: ROSETTA

# --- Optimization ---
optimization:
  CALIBRATION_ALGORITHM: DDS
  CALIBRATION_PERIOD: "2004-01-01/2007-12-31"
  EVALUATION_PERIOD: "2008-01-01/2010-12-31"
  NUMBER_OF_ITERATIONS: 1000
  OPTIMIZATION_METRIC: KGE
  dds:
    perturbation_value: 0.2

# --- Evaluation ---
evaluation:
  STREAMFLOW_RAW_PATH: /data/obs/wsc/05BB001.csv
  EVALUATION_DATA_TYPES: [STREAMFLOW]

# --- Paths ---
paths:
  CATCHMENT_SHP_NAME: bow_catchment.shp
  RIVER_NETWORK_SHP_NAME: bow_rivers.shp
```

---

## Notes for co-authors

**Scope:** This paper is the technical companion to "Outstanding in Every Field" (Episode 1). Where Episode 1 argues *why* the community needs shared architecture, this paper describes *how* SYMFLUENCE implements it. The tone is technical rather than philosophical -- concrete descriptions of patterns, interfaces, and design decisions.

**Figures needed:**
1. System architecture diagram (four tiers) -- **Figure 1**
2. Workflow orchestration sequence -- **Figure 2**
3. Data management pipeline architecture -- **Figure 3**
4. Model integration component architecture -- **Figure 4**
5. Calibration/optimization framework -- **Figure 5**
6. Spatial discretization hierarchy (GRU/HRU) -- from V4.0 paper
7. Domain definition examples (Bow, Iceland) -- from V4.0 paper

**Relationship to V4.0 paper:** This manuscript ports and restructures Section 3 of the original V4.0 SYMFLUENCE paper. Content from V4.0 Sections 3.1--3.8 has been reorganized, expanded with architectural rationale, and framed as a pattern language rather than purely descriptive documentation. The governance discussion (Section 10) is new.

**Length:** Currently ~8,000 words of body text. For a GMD technical note or full paper, this is within range. The comprehensive configuration appendix adds ~500 words.

**Episode 3 dependency:** The applications and validation content from V4.0 Section 4 is reserved for Episode 3 ("From Configuration to Prediction"). This paper should not include results or performance numbers -- only architecture, design, and patterns.
