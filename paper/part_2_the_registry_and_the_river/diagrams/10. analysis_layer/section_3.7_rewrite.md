# Section 3.7 — Analysis Layer (Proposed Rewrite)

> **Note:** This rewrite assumes Figure 10 (the analysis layer architecture diagram)
> is referenced inline. Equations for NSE, KGE, and PBIAS are retained as display math
> — only the surrounding prose is restructured.

---

## 3.7 Analysis Layer

The analysis layer (Figure 10) provides the evaluation, visualization, and diagnostic infrastructure that connects model outputs to scientific interpretation. Where previous layers produce simulations, this layer answers *how good* those simulations are, *why* they succeed or fail, and *whether* they outperform simple reference predictors. The design is organized around four pillars—performance metrics, multi-variable evaluation, visualization, and benchmarking—unified by a cross-cutting registry-and-decorator pattern that enables extension without modifying framework code.

### 3.7.1 Performance Metrics

SYMFLUENCE implements fourteen performance metrics through a centralized `METRIC_REGISTRY` that ensures consistent calculation across all evaluation contexts (Figure 10, top row). Metrics are organized into four complementary categories.

**Efficiency metrics** quantify how well simulated values reproduce observed variability relative to benchmark predictors. The Nash–Sutcliffe Efficiency (NSE; Nash and Sutcliffe, 1970) measures improvement over mean-flow prediction, where values approaching 1 indicate perfect agreement; the log-transformed variant (logNSE; Krause et al., 2005) emphasizes low-flow performance. The Kling–Gupta Efficiency (KGE; Gupta et al., 2009) decomposes performance into correlation (*r*), variability ratio (*α*), and bias ratio (*β*), enabling diagnosis of whether poor performance stems from timing errors, variability misrepresentation, or systematic bias. The modified KGE′ (Kling et al., 2012) reduces sensitivity to flow magnitude by substituting coefficient of variation for standard deviation, while KGE_np (Pool et al., 2018) uses Spearman correlation and flow duration curves for robust assessment under non-Gaussian conditions.

**Error metrics** quantify prediction accuracy in original units. Root Mean Square Error (RMSE) weights larger deviations; Mean Absolute Error (MAE) provides a robust central tendency measure. Normalized variants (NRMSE, MARE) enable cross-site comparison by removing scale dependence.

**Bias metrics** capture systematic over- or underestimation through absolute bias (*μ*_sim − *μ*_obs) and percent bias (PBIAS), where positive values indicate overestimation.

**Correlation metrics** assess linear association via Pearson *r* and the coefficient of determination *R*². Spearman rank correlation provides a non-parametric alternative robust to outliers and non-linear monotonic relationships.

The `METRIC_REGISTRY` stores function references alongside metadata—optimal value, valid range, optimization direction, units, and literature references—and supports multiple aliases (e.g., 'kge', 'KGE', 'kling_gupta') for user convenience. This centralized design guarantees that any metric referenced by name resolves to a single, validated implementation throughout the framework.

### 3.7.2 Multi-Variable Evaluation

Robust model assessment requires evaluation against multiple hydrological variables, since models may reproduce streamflow accurately while misrepresenting internal states such as snow storage or evapotranspiration. SYMFLUENCE implements a registry-based evaluator system (Figure 10, middle rows) where variable-specific evaluators inherit from a common `ModelEvaluator` base class, providing consistent interfaces while accommodating the distinct data formats, unit conventions, and processing requirements of each variable type.

Six evaluator classes are registered via the `EvaluationRegistry`:

- The **`StreamflowEvaluator`** automatically detects output format (standalone, routing, or coupled model), converts mass flux (kg m⁻² s⁻¹) to volumetric flux (m³ s⁻¹), and resolves catchment area through a priority cascade: explicit configuration, model attributes, basin shapefile, catchment shapefile, or geometric calculation.

- The **`ETEvaluator`** supports four observation sources—MODIS MOD16A2 (8-day, 500 m), FLUXCOM (machine-learning gridded, 0.25°), FluxNet towers (in-situ with quality control), and GLEAM (satellite-derived)—handling source-specific unit conversions and quality filters. Multiple aliases ('ET', 'MODIS_ET', 'MOD16', 'FLUXNET') resolve to the same evaluator with appropriate default configurations.

- The **`SnowEvaluator`** assesses snow water equivalent (SWE, continuous, kg m⁻²) and snow-covered area (SCA, fractional, from MODIS/Landsat imagery), applying continuous metrics for SWE and categorical metrics for SCA.

- The **`SoilMoistureEvaluator`** integrates SMAP (~3 km, surface and root zone via depth-weighted averaging), ESA CCI (longest satellite record, 1978–present), and ISMN tower observations at multiple depths. Automatic depth matching with configurable tolerances reconciles model layers against observation depths.

- The **`GroundwaterEvaluator`** operates in two modes: well-based evaluation with automatic datum alignment for reference elevation offsets, and GRACE-based evaluation that sums simulated storage components for comparison against satellite-derived anomalies.

- The **`TWSEvaluator`** compares vertically integrated simulated storage (SWE, canopy water, soil water, aquifer, and optionally glacier mass) against GRACE satellite observations. Linear detrending of glacier mass is critical in glacierized basins, where long-term ice loss dominates the GRACE signal and obscures seasonal dynamics. The evaluator supports multiple GRACE processing centres (JPL, CSR, GSFC).

Evaluators register via decorators (e.g., `@EvaluationRegistry.register('STREAMFLOW')`), enabling loose coupling between the analysis framework and variable-specific implementations. This pattern facilitates extension to new variables—a new evaluator needs only to subclass `ModelEvaluator` and apply the registration decorator.

### 3.7.3 Visualization and Diagnostics

SYMFLUENCE provides publication-ready visualization through a modular plotting architecture (Figure 10, lower left) comprising specialized plotter classes and reusable panel components.

Plotter classes target distinct stages of the modelling workflow. The `DomainPlotter` generates maps of catchment boundaries, discretization units, river networks, and elevation distributions with optional basemap integration. The `ForcingComparisonPlotter` creates side-by-side views of raw gridded forcing and HRU-remapped values with variable-specific colormaps and remapping statistics. The `OptimizationPlotter` tracks calibration progress through convergence curves, parameter evolution traces, and Pareto frontiers. The `ModelComparisonPlotter` generates multi-panel overviews combining time series, flow duration curves, scatter plots, metric tables, monthly boxplots, and residual analysis. The `BenchmarkPlotter` renders performance heatmaps comparing calibrated skill against reference predictors. The `AnalysisPlotter` visualizes parameter sensitivity indices and hydrograph comparisons for top-performing configurations. The `WorkflowDiagnosticPlotter` validates each workflow step—domain definition quality, forcing completeness, observation availability, and model output—identifying data issues before they propagate to calibration. Additional framework plotters (including `DiagnosticPlotter`, `HydrographPlotter`, `ModelResultsPlotter`, and `SnowPlotter`) and model-specific plotters registered via the `PlotterRegistry` extend coverage to specialized use cases.

Plotters compose from six reusable panel types implementing a common rendering interface: `TimeSeriesPanel` (multi-simulation overlay), `FDCPanel` (exceedance probabilities via Weibull plotting position), `ScatterPanel` (observed vs. simulated with 1:1 lines), `MetricsTablePanel` (conditionally colored statistics), `MonthlyBoxplotPanel` (seasonal aggregation), and `ResidualAnalysisPanel` (monthly bias patterns). This compositional architecture enables flexible assembly of custom diagnostic figures while maintaining visual consistency. All plotters output PNG at configurable resolution (default 300 DPI) into a structured directory hierarchy separating domain, forcing, optimization, analysis, and benchmark outputs.

### 3.7.4 Benchmarking and Sensitivity Analysis

Beyond point-wise evaluation, the analysis layer supports systematic assessment of model value and parameter importance (Figure 10, lower right).

The `Benchmarker` class compares calibrated model skill against a hierarchy of reference predictors: mean flow, median flow, monthly climatology, daily climatology, long-term rainfall–runoff ratio, and short-term rainfall–runoff ratio. Following Schaefli and Gupta (2007), benchmarking contextualizes model performance—a calibrated model achieving KGE = 0.7 provides limited value if monthly climatology achieves KGE = 0.65, whereas the same score represents substantial skill if the best benchmark achieves only KGE = 0.3. Benchmark results are computed for both calibration and evaluation periods to assess transferability.

The `SensitivityAnalyzer` implements variance-based methods to identify influential parameters. Total-order Sobol′ indices (Sobol′, 2001) quantify the fraction of output variance attributable to each parameter including all interaction effects. Parameters with near-zero indices can be fixed at nominal values, reducing calibration dimensionality without sacrificing performance. Reliable estimation requires minimum sample sizes (typically >60) with uncertainty bounds on index estimates.

The `BaseStructureEnsembleAnalyzer` enables systematic exploration of model structural choices. For models supporting multiple process representations (e.g., different snow parameterizations, routing schemes, or soil configurations), the analyzer generates all combinations, executes runs, calculates metrics, and identifies optimal configurations. Results are stored in a standardized format for statistical analysis of structure–performance relationships and quantification of structural uncertainty.

Following the evaluator registry pattern, analysis components register through decorators on the `AnalysisRegistry`, which provides factory methods returning appropriate analyzers based on model type with graceful fallback to base implementations when specialized variants are unavailable.
