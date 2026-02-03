# 3.5 Model Integration

The diversity of hydrological models — spanning conceptual rainfall-runoff formulations, process-based land surface schemes, and data-driven approaches — presents a significant integration challenge. Each model brings distinct input requirements, execution mechanisms, and output formats. SYMFLUENCE addresses this heterogeneity through a unified component interface that standardises model interaction while preserving model-specific capabilities (Figure 7). This architecture enables rigorous intercomparison by ensuring that differences in simulation results reflect model formulation rather than preprocessing inconsistencies or configuration errors.

## 3.5.1 Component Architecture

The model integration layer employs a component-based architecture where each model implements four standardised interfaces: preprocessing, execution, postprocessing, and result extraction. A registry pattern enables runtime discovery of model implementations, supporting extensibility without modification to core framework code.

**Unified Component Interface.** Each integrated model provides four components that implement abstract base classes (Figure 7, middle row):

*PreProcessor* (`BaseModelPreProcessor`) transforms SYMFLUENCE's standardised data formats into model-specific inputs. Responsibilities include forcing data conversion (reformatting meteorological variables, adjusting units, interpolating timesteps), spatial attribute generation (mapping HRU characteristics to model parameter files), configuration file creation (control files, namelist files, decision files), and initial condition setup. The preprocessor receives an optional parameter dictionary enabling calibration workflows to generate model inputs for candidate parameter sets.

*Runner* (`BaseModelRunner`) executes the model with preprocessed inputs. Responsibilities include binary invocation (subprocess management for compiled executables), environment configuration (working directories, library paths), execution monitoring (progress tracking, timeout handling), and error capture (exit codes, log parsing). Runners register a method name (e.g., `run_summa`, `run_fuse`) that the orchestrator invokes.

*PostProcessor* (`BaseModelPostProcessor`) extracts and standardises model outputs. Responsibilities include output parsing (NetCDF, CSV, binary formats), variable extraction (streamflow, snow water equivalent, evapotranspiration), unit conversion (model-native units to SI), and spatial aggregation (HRU-level to basin-level where needed). All postprocessors produce results in a common CSV format enabling consistent evaluation.

*ResultExtractor* (`ModelResultExtractor`) provides flexible access to model outputs for analysis workflows. Responsibilities include file pattern matching (locating outputs across directory structures), variable-specific extraction (handling different output files for different variables), and metadata preservation (units, coordinates, time information). While the postprocessor targets streamflow for calibration and evaluation, the extractor enables retrieval of any output variable for diagnostic and research purposes.

**Registry Pattern.** The `ComponentRegistry` provides centralised discovery and instantiation of model components through decorator-based registration (Figure 7, top rows). Separate registries are maintained for each component type — `preprocessors`, `runners`, `postprocessors`, and `result_extractors` — enabling fine-grained model capability declaration. This pattern yields several architectural benefits:

- *Loose coupling* isolates model implementations from orchestration logic — the `ModelManager` need not know implementation details of any specific model.
- *Extensibility* enables new models to be added by implementing the component interfaces and registering with the registry; no modifications to existing code are required.
- *Graceful degradation* allows partial model support — a model might provide only a runner (for manual preprocessing) or only an extractor (for analysing external outputs).

**Shared Infrastructure.** All four component base classes inherit from a set of mixins that provide cross-cutting capabilities (Figure 7, lower rows): `ModelComponentMixin` initialises model-specific paths and configuration; `PathResolverMixin` resolves directory structures and provides configuration access; `ShapefileAccessMixin` loads catchment geometry for spatial operations; and `ConfigMixin` provides typed configuration coercion. This mixin architecture ensures consistent behaviour across all model implementations while avoiding code duplication.

**Supported Models.** SYMFLUENCE integrates models spanning the spectrum from parsimonious conceptual formulations to complex process-based representations and data-driven approaches, summarised in Table 6. The `ModelManager` orchestrates the complete execution pipeline for each model: resolving the workflow order, iterating through registered models, and producing standardised results that enable direct intercomparison.

## 3.5.2 Routing Integration

Hydrological models that simulate vertical water balance processes require coupling with routing models to translate hillslope runoff into channel discharge at basin outlets. SYMFLUENCE automates this coupling through dependency resolution and standardised interfaces between runoff-generating and routing components. MizuRoute serves as the primary routing solution for production workflows, implementing three routing methods: Impulse Response Function (IRF) uses a gamma-shaped transfer function for computational efficiency; Kinematic Wave Tracking (KWT) provides more physically based wave propagation; and Diffusive Wave accounts for backwater effects in low-gradient reaches. The mizuRoute preprocessor generates network topology files, HRU-to-segment mappings, and control files from SYMFLUENCE's river network shapefiles.

The `RoutingDecider` determines when routing is required based on configuration and model characteristics. When the `ModelManager` resolves the execution workflow, it automatically inserts routing models after their source models, ensuring correct execution order and preventing configuration errors where users forget to specify routing for distributed simulations.

The coupling between runoff-generating models and routing requires several coordination mechanisms:

- *Runoff variable mapping:* Each model produces runoff under different variable names and units. SUMMA outputs `basRunoff` in mm/timestep; FUSE outputs `q_routed` in mm/day. The routing preprocessor maps these to the expected input variable through model-specific adapters.
- *Spatial correspondence:* Model HRUs must map to routing network segments. The preprocessor generates remapping files that specify which HRUs contribute to which river segments, with area-weighted aggregation when HRU and segment boundaries differ.
- *Temporal alignment:* Model output timesteps must match routing input requirements. The preprocessor handles temporal aggregation or disaggregation as needed.
- *Network topology:* River network connectivity (segment identifiers, downstream references, geometric properties) flows from domain delineation through routing setup. The `MizuRoutePreProcessor` consumes the river network shapefile and generates topology files in mizuRoute's expected format.

A distinctive capability enables routing of lumped model outputs through distributed river networks. When configured with `routing: river_network` for a lumped domain, SYMFLUENCE: (1) delineates subcatchments internally via the full TauDEM workflow; (2) presents a lumped domain to the hydrological model as a single HRU; (3) disaggregates lumped runoff to subcatchments via area weighting; and (4) routes through the full river network. This approach combines the computational efficiency of lumped simulation with the spatial detail of distributed routing, useful for applications where channel routing dynamics matter more than hillslope heterogeneity.
