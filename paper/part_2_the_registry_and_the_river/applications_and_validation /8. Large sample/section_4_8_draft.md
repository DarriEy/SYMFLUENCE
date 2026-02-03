## 4.8 Large-Sample Study

The preceding experiments (Sections 4.1–4.7) demonstrate SYMFLUENCE's modelling capabilities on individual catchments. A framework intended for operational and research use must, however, scale to many catchments with minimal per-basin manual intervention. To evaluate this, we apply a single FUSE configuration template across 111 LamaH-Ice catchments spanning the full hydrological diversity of Iceland (Helgason and Nijssen, 2024). Each catchment is set up, forced, calibrated, and evaluated through the same automated pipeline, differing only in domain identifiers and observation-constrained time periods.

**Figure X.** Overview of the 111 LamaH-Ice study catchments. (a) Catchment boundaries coloured by drainage area (km²), overlaid on the national GRU mesh; dashed red outlines indicate catchments with glacier fraction exceeding 30%. (b) Distribution of catchment area (log scale). (c) Distribution of streamflow record length.

### 4.8.1 Study Domain and Experimental Design

The LamaH-Ice dataset provides delineated catchment boundaries, daily streamflow observations, and catchment attributes for gauged basins across Iceland. Across the 111 catchments, drainage areas span three orders of magnitude (3.8–7,437 km², median 391 km²), mean elevations range from 39 to 1,307 m, and 63 catchments (57%) contain glacierized area — with glacier fractions reaching 93% in the most heavily glacierized basin (Domain 60). Streamflow records range from 4 to 89 years (median 33 years; Figure Xc), reflecting the heterogeneous monitoring history of Iceland's hydrometric network.

All 111 catchments are configured for lumped FUSE modelling with ERA5 reanalysis forcing at hourly resolution (0.25° grid). DDS (Tolson and Shoemaker, 2007) is used for calibration with 1,000 iterations and KGE as the objective function. Thirteen FUSE parameters are calibrated (Table X), with identical parameter bounds across all catchments. Where streamflow records cover the 2005–2014 window, a standardised period split is applied: two-year spinup (2005–2006), five-year calibration (2007–2011), and three-year evaluation (2012–2014); 63 catchments satisfy this criterion. The remaining 48 catchments are assigned catchment-specific periods determined automatically from the available observation window, maintaining the same proportional split of spinup, calibration, and evaluation years.

### 4.8.2 Configuration and Execution

The 111 FUSE configurations are generated programmatically from a single YAML template. A setup script reads each catchment's streamflow record to determine the observation window, computes appropriate time-period splits, derives bounding-box coordinates from the catchment shapefile, and writes the per-domain configuration file. The only fields that vary between configurations are the domain identifier, bounding-box coordinates, and the five time-period settings. All other settings — model structure, parameter bounds, forcing dataset, optimization algorithm and budget — are held constant.

Execution proceeds sequentially via a runner script that invokes the SYMFLUENCE workflow for each domain, skips previously completed catchments, and logs per-domain return codes and elapsed times to a persistent log file. This design allows the campaign to be interrupted and resumed without re-running completed domains — a practical necessity when individual calibration runs span multiple hours.

### 4.8.3 Framework Implications

This experiment exercises three aspects of SYMFLUENCE's architecture that single-catchment studies do not test.

First, the configuration-driven design enables systematic scaling from one catchment to many without code modification. The same YAML schema that defines a single-domain experiment extends to 111 domains through programmatic template instantiation, with domain-specific parameters (observation periods, bounding boxes) derived automatically from the data.

Second, the standardised directory structure and output conventions ensure that results from all catchments are directly comparable and machine-readable. Optimization trajectories, best-fit parameters, and evaluation metrics follow identical naming and formatting conventions regardless of domain, enabling automated cross-catchment analysis without domain-specific logic.

Third, the fault-tolerant execution model — with per-domain logging, skip-on-completion, and resume-from-failure — addresses a practical challenge that arises at scale: individual domains may fail due to data gaps, numerical instabilities, or infrastructure interruptions. The ability to diagnose, fix, and re-run individual failures without repeating the entire campaign is built into the workflow runner rather than requiring external job management infrastructure.
