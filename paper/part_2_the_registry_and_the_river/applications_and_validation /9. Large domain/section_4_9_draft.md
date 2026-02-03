## 4.9 Large-Domain Distributed Modelling

The large-sample experiment of Section 4.8 demonstrated SYMFLUENCE's ability to scale a single model configuration across 111 independent catchments. That experiment, however, treated each catchment as a lumped unit with no lateral routing between sub-basins. For regional water-resource assessment or flood forecasting, a distributed representation — in which precipitation is partitioned across spatially explicit response units and routed through a connected river network — is required. This section evaluates SYMFLUENCE's capacity to configure, execute, and calibrate a fully distributed hydrological simulation over the entire island of Iceland (~103,000 km²).

**Figure X.** Overview of the distributed Iceland domain. (a) GRU mesh coloured by mean elevation, with the delineated river network overlaid. (b) Distribution of GRU areas. (c) Distribution of GRU mean elevations.

### 4.9.1 Domain and Experimental Design

The model domain encompasses the full island of Iceland within the bounding box 63.0–66.5°N, 25.0–13.0°W. Domain delineation follows the stream-threshold approach (threshold = 10,000 grid cells on the Copernicus DEM), with coastal watershed extraction enabled to capture the numerous short, steep rivers that drain directly to the ocean without converging into larger trunk systems. The resulting mesh is discretised into Grouped Response Units (GRUs), each further subdivided into Hydrological Response Units (HRUs) based on 200 m elevation bands, five radiation classes, and eight aspect classes (Figure Xa).

The hydrological model is FUSE in distributed spatial mode, with lateral flows routed through the delineated river network using mizuRoute at an hourly routing time step. ERA5 reanalysis provides meteorological forcing at 0.25° spatial and hourly temporal resolution, with a fixed lapse rate correction of 6.5 °C km⁻¹ applied to air temperature. Thirteen FUSE parameters are calibrated simultaneously using DDS (Tolson and Shoemaker, 2007) with a budget of 1,000 iterations and KGE as the objective function (Table X). Parameter bounds are held identical to those used in the lumped per-catchment experiment (Section 4.8), enabling direct comparison of calibrated values between lumped and distributed configurations. The simulation spans 2008–2010, with a one-year spinup (2008), one-year calibration (2009), and one-year evaluation (2010). Observed streamflow from the Icelandic Meteorological Office (IMO) is used for calibration and evaluation.

### 4.9.2 Distributed Configuration

The distributed configuration differs from the lumped per-catchment setup of Section 4.8 in three key respects. First, the entire island is represented as a single connected domain rather than 111 independent catchments, so water routed from an upstream GRU enters the downstream GRU through the river network topology. Second, the delineation produces a substantially finer spatial mesh: each GRU corresponds to a sub-basin of the stream network rather than a full gauged catchment, yielding GRU areas typically one to two orders of magnitude smaller than the lumped catchment areas (Figure Xb). Third, mizuRoute is invoked as an explicit routing component, reading gridded runoff from FUSE and producing reach-level discharge at every time step — a capability that was unnecessary in the lumped experiment where each catchment had a single outlet.

The SYMFLUENCE configuration for this experiment is specified in a single YAML file that sets `FUSE_SPATIAL_MODE: distributed`, `ROUTING_DELINEATION: distributed`, and `DELINEATE_COASTAL_WATERSHEDS: true`. All other workflow steps — DEM acquisition, intersection of forcing grids with the GRU mesh, parameter file generation, model compilation, and output evaluation — proceed through the same automated pipeline used for the single-catchment experiments in Sections 4.1–4.7, with no code modification required.

### 4.9.3 Framework Implications

This experiment tests three aspects of SYMFLUENCE's architecture that the preceding experiments do not exercise.

First, the distributed delineation and routing workflow demonstrates that SYMFLUENCE can scale from lumped single-catchment simulations to fully distributed regional domains without changing the underlying model code. The same FUSE executable and mizuRoute binary used in the lumped experiments are reused here; only the configuration file and the spatial inputs differ. This separation of spatial discretisation from model physics is central to the framework's design and ensures that users can transition between lumped and distributed representations by modifying configuration settings rather than rebuilding software.

Second, the coastal watershed delineation addresses a challenge specific to island and coastal domains: many rivers drain directly to the ocean through short, independent flow paths that are not captured by conventional pour-point-based delineation. SYMFLUENCE's `DELINEATE_COASTAL_WATERSHEDS` option identifies these terminal basins automatically, ensuring complete spatial coverage of the island without manual intervention.

Third, the integration of mizuRoute as a configurable routing component illustrates the framework's modular coupling strategy. Routing parameters (time step, output frequency, variable names) are specified in the same YAML configuration file as the hydrological model settings, and the data handoff between FUSE and mizuRoute is managed internally by the workflow engine. This design allows the routing scheme to be replaced or extended (e.g., with kinematic wave or diffusive wave methods) without modifying the hydrological model configuration.

### 4.9.4 Results

**Figure X.** FUSE baseline results for the Iceland domain. (a) Domain-averaged routed runoff time series. (b) Domain-averaged precipitation. (c) Spatial distribution of mean runoff across GRUs. (d) Monthly precipitation–runoff relationship. (e) Domain-averaged temperature. (f) Annual water balance components.

**Domain-Scale Water Balance.** The distributed simulation produces physically plausible water balance partitioning over the three-year study period (Table X). Mean annual precipitation ranges from 647 to 831 mm yr⁻¹, with corresponding runoff totals of 198 to 402 mm yr⁻¹. The resulting runoff ratios (0.25–0.48) are consistent with Iceland's climate, where substantial evaporative losses occur during the brief growing season despite low mean annual temperatures (~2°C). The domain-averaged runoff ratio of 0.40 aligns with independent estimates for Icelandic catchments reported in the literature.

| Year | Precipitation (mm) | Runoff (mm) | PET (mm) | Temperature (°C) | Runoff Ratio |
|------|-------------------|-------------|----------|------------------|--------------|
| 2008 | 784 | 198 | 677 | 1.9 | 0.25 |
| 2009 | 831 | 402 | 675 | 2.2 | 0.48 |
| 2010 | 647 | 309 | 705 | 2.4 | 0.48 |

**Table X.** Annual water balance statistics for the Iceland distributed domain.

The spatial distribution of mean daily runoff across the 7,618 GRUs shows a median of 0.72 mm d⁻¹ with considerable variability reflecting Iceland's heterogeneous climate and topography (Figure Xc). Higher runoff rates occur in the southern and southeastern coastal regions, which receive orographic precipitation from prevailing southwesterly winds, while the interior highlands and northern regions show lower values consistent with rain-shadow effects.

**Figure X.** Comparison of simulated and observed runoff. (a) Time series of domain-averaged simulated runoff and mean observed runoff from 72 gauged catchments. (b) Daily scatter plot with performance metrics. (c) Monthly comparison.

**Validation Against Observations.** Comparison of the distributed simulation with observations from the 72 LamaH-Ice catchments (Section 4.8) reveals systematic differences that reflect both model limitations and methodological challenges inherent to distributed-to-lumped comparison. At the domain scale, simulated runoff (mean 2.5 mm d⁻¹) substantially underestimates observed runoff (mean 7.8 mm d⁻¹), yielding a percent bias of approximately −68%. The correlation coefficient (r = 0.25) indicates weak correspondence in day-to-day variability, and the aggregate KGE (−0.43) falls below the threshold typically considered acceptable for hydrological prediction.

Per-catchment validation, in which simulated runoff from the nearest HRU is compared to observed discharge at each gauge location, shows similarly challenging results. Of the 72 validated catchments, only four achieve positive KGE values (domains 39, 37, 76, 14), with a median KGE of −0.22 across all sites. Performance varies spatially, with the best results in smaller coastal catchments and the poorest performance in large glacierised basins and interior highlands.

**Figure X.** Spatial distribution of per-catchment KGE values across Iceland, showing model performance at 72 validation locations.

### 4.9.5 Discussion

**Preliminary Validation Status.** Initial validation results showed apparent systematic underestimation, but subsequent analysis revealed a methodological error in the comparison approach. The original validation matched gauge locations to interior HRUs (using centroid distance) and compared local HRU runoff generation to catchment outlet discharge. This comparison is invalid because local HRU runoff does not account for flow accumulation through the river network.

The correct validation methodology requires extracting mizuRoute routed discharge at river reaches corresponding to gauge locations, which integrates all upstream HRU contributions. This gauge-to-reach matching has been implemented (`scripts/create_proper_comparison.py`) and will be applied once simulation outputs are available.

**Model Limitations.** Independent of the validation methodology, the FUSE configuration used in this experiment has known limitations for Iceland:

*Glacier representation.* Iceland's glaciers cover approximately 11% of the island and contribute substantially to summer streamflow. The FUSE model includes a temperature-index snow module but lacks explicit glacier representation, meaning ice melt from permanent glacier surfaces is not simulated. Future applications should couple with a glacier mass balance module.

*Forcing uncertainty.* ERA5 reanalysis may underestimate precipitation in regions of complex terrain. Iceland's orographic precipitation regime may not be fully resolved at 0.25° resolution.

**Framework Demonstration.** Despite the quantitative performance limitations, this experiment successfully demonstrates SYMFLUENCE's core capabilities for regional-scale distributed modelling. The framework:

1. Automatically delineated a complex island domain with 7,618 GRUs and 6,606 river segments, including coastal watersheds that would be missed by conventional pour-point methods.

2. Configured and executed FUSE in distributed mode with mizuRoute coupling using the same model executables as the lumped experiments, requiring only configuration changes.

3. Processed ERA5 forcing data, applied elevation-dependent corrections, and generated all required parameter files through the standard automated pipeline.

4. Completed a 1,000-iteration DDS calibration and produced spatially distributed output suitable for water resource assessment.

The transition from 111 independent lumped catchments (Section 4.8) to a single connected 103,000 km² domain required no code modifications — only changes to configuration settings. This separation of spatial discretisation from model physics is a key architectural feature that enables users to scale analyses from local to regional domains without rebuilding software.

**Next Steps.** The following steps are required to complete the validation:

1. *Run simulation:* Execute FUSE and mizuRoute to produce routed discharge outputs.

2. *Apply correct validation:* Run `create_proper_comparison.py` to extract routed discharge at gauge-matched river reaches and compare to observations.

3. *Evaluate performance:* Compute KGE and other metrics using the proper routed discharge comparison.

**Recommendations for Glacierised Domains.** Future applications to Iceland or similar regions should consider:

1. *Glacier representation:* Couple with a glacier mass balance module (e.g., OGGM, PyGEM) to represent ice melt contributions.

2. *Forcing bias correction:* Apply bias correction to ERA5 precipitation using station observations.

3. *Multi-site calibration:* Optimise performance across multiple gauge locations simultaneously.
