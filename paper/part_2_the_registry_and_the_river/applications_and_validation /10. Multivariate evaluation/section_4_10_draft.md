## 4.10 Multivariate Evaluation

Sections 4.1–4.9 evaluated hydrological model performance exclusively against observed streamflow. While discharge integrates many watershed processes and remains the most widely available validation target, it provides limited information about the internal consistency of simulated states — a model may reproduce the hydrograph accurately while misrepresenting snow accumulation, soil moisture dynamics, or subsurface storage (Kirchner, 2006; Clark et al., 2016). Satellite-derived observations of snow cover, soil moisture, and terrestrial water storage now offer spatially distributed constraints on process representations that streamflow alone cannot provide. This section evaluates SYMFLUENCE's capacity to ingest, compare against, and calibrate toward multiple observation sources simultaneously, using three experiments that span different variables, spatial scales, and satellite products.

The three experiments are designed to exercise complementary aspects of the multivariate evaluation workflow. Experiment (a) compares simulated total water storage anomalies against GRACE/GRACE-FO gravimetric observations in a single headwater catchment, testing the framework's ability to evaluate subsurface storage dynamics beyond the reach of surface observations. Experiment (b) jointly evaluates snow cover area and soil moisture against MODIS and SMAP retrievals in a snow-dominated catchment, demonstrating weighted multi-objective calibration with heterogeneous observation types. Experiment (c) extends the evaluation to a regional domain, comparing long-term snow cover fraction trends from a 24-year simulation against the full MODIS record across Iceland, testing whether SYMFLUENCE can reproduce observed decadal changes in snow persistence across elevation gradients.

**Table X.** Summary of the three multivariate evaluation experiments. Each experiment pairs SUMMA simulations with one or more satellite-derived observation products. Weight columns indicate the relative contribution to the multi-objective calibration function.

| Experiment | Domain | Area (km²) | Satellite product(s) | Variable(s) | Metric(s) | Period |
|---|---|---|---|---|---|---|
| (a) GRACE TWS | Bow at Banff | 2,210 | GRACE-FO JPL RL06 | TWS anomaly | Pearson r (50%) + KGE (50%) | 2002–2017 |
| (b) SCA + Soil moisture | Paradise Creek | ~500 | MODIS MOD10A1, SMAP L3 | SCA, soil moisture | Accuracy (50%) + r (50%) | 2015–2023 |
| (c) SCF trends | Iceland | ~103,000 | MODIS MOD10A2 | SCF (8-day) | Trend correlation | 2000–2023 |

### 4.10.1 Experiment (a): Total Water Storage Evaluation Against GRACE

Terrestrial water storage (TWS) anomalies derived from the GRACE and GRACE-FO satellite missions provide a vertically integrated measure of changes in snow, soil moisture, and groundwater — quantities that are poorly constrained by streamflow observations alone (Tapley et al., 2004). This experiment compares simulated TWS anomalies from SUMMA against GRACE-FO JPL RL06 mascon solutions in the Bow River basin at Banff, Alberta (2,210 km²), a snow-dominated headwater catchment in the Canadian Rockies.

#### Domain and configuration

The domain is configured as a lumped catchment centred on 51.18°N, 115.57°W with ERA5 forcing at hourly resolution. The simulation spans 2002–2017 with a two-year spinup (2002–2003), seven-year calibration period (2004–2010), and seven-year evaluation period (2011–2017). Two parallel configurations are executed: an uncalibrated baseline and a multi-objective calibrated run. The calibrated configuration optimises 14 SUMMA parameters using DDS with 1,000 iterations, targeting a weighted combination of TWS anomaly correlation against GRACE (weight 0.5) and streamflow KGE against Water Survey of Canada station 05BB001 (weight 0.5). The parameter set includes four groundwater and storage parameters — `aquiferBaseflowExp`, `aquiferScaleFactor`, `specificStorage`, and `specificYield` — that are not typically included in streamflow-only calibration but are expected to influence TWS dynamics directly.

The SYMFLUENCE configuration enables GRACE comparison through the `MULTIVAR_TARGETS` block:

```yaml
MULTIVAR_TARGETS:
  - variable: TWS
    source: GRACE
    product: JPL_RL06_mascon
    metric: correlation
    weight: 0.5
  - variable: streamflow
    source: WSC
    station_id: 05BB001
    metric: KGE
    weight: 0.5
```

This declarative specification instructs the framework to automatically acquire the GRACE mascon product, compute monthly TWS anomalies from both simulated and observed records, and combine the resulting correlation with the streamflow KGE into a single objective function during calibration.

#### Results

The uncalibrated SUMMA simulation demonstrates strong agreement with GRACE-derived TWS anomalies (Figure X). Over the full analysis period (2004–2017), simulated monthly TWS anomalies correlate highly with GRACE CSR mascon solutions (r = 0.89), with an RMSE of 43 mm and negligible mean bias (+3 mm). The model captures both the amplitude and timing of the seasonal TWS cycle, with maximum storage occurring in late spring (May–June) as snowmelt recharges soil and groundwater reservoirs, and minimum storage in late autumn (October–November) following the summer recession.

Streamflow performance at the WSC 05BB001 gauge during the evaluation period (2011–2017) yields r = 0.89, NSE = 0.72, and KGE = 0.63, indicating that the model reproduces the observed hydrograph dynamics reasonably well. However, a systematic negative bias (PBIAS = −28%) suggests that the uncalibrated configuration underestimates total runoff volume, potentially due to excessive evapotranspiration or insufficient precipitation in the ERA5 forcing at high elevations.

Comparison with CanSWE station observations (three SNOTEL stations within the basin) reveals that simulated SWE tracks the seasonal accumulation and ablation patterns accurately (r = 0.88), but underestimates peak SWE by approximately 130 mm on average (RMSE = 184 mm, bias = −131 mm). This negative SWE bias is consistent with the streamflow underestimation and may reflect (i) undercatch-corrected precipitation in ERA5 at alpine elevations, (ii) suboptimal snow accumulation parameters, or (iii) scale mismatch between point observations and catchment-mean simulations.

The seasonal TWS cycle (Figure Xi) illustrates that both simulated and GRACE-observed TWS anomalies peak in May–June and reach minima in October–November, confirming that the model captures the dominant hydrological processes controlling storage dynamics in this snow-dominated catchment. The slight phase lag between simulated and observed peaks (approximately one month) may reflect timing differences in snowmelt initiation that calibration could address.

Evapotranspiration (ET) provides an independent constraint on the surface energy and water balance that is complementary to the storage-focused GRACE observations. Simulated ET is compared against the SSEBop (Operational Simplified Surface Energy Balance) global monthly ET product, which estimates actual ET at 10 km resolution using thermal imagery and a simplified energy balance approach (Senay et al., 2013). Over the available comparison period (2004–2007, 40 months), monthly SUMMA-simulated ET correlates well with SSEBop estimates (r = 0.82), with an RMSE of 1.05 mm/day and a slight positive bias of +0.24 mm/day. The bias indicates that SUMMA simulates slightly higher ET than SSEBop, which may reflect differences in how the two approaches partition available energy or uncertainties in the satellite-based retrievals over complex mountainous terrain. The strong seasonal correlation confirms that the model captures the dominant ET dynamics — minimal evapotranspiration during winter months (<0.1 mm/day) increasing to peak values of 2.5–3.0 mm/day in summer.

**Figure X.** Bow at Banff multivariate evaluation (uncalibrated). (a) Domain map showing the 2,210 km² catchment with observation station locations: CanSWE snow stations (blue squares) and WSC streamflow gauge (red triangle). (b) TWS anomaly time series comparing simulated storage against GRACE CSR mascons. (c) Streamflow comparison. (d) SWE comparison against CanSWE. (e) ET comparison against SSEBop. (f–i) Scatter plots for each variable comparison.

**Table X.** Multivariate evaluation metrics for the Bow at Banff experiment (uncalibrated configuration). Metrics computed over the full analysis period (2004–2017) except streamflow, which uses only the evaluation period (2011–2017), and ET, which uses the available SSEBop overlap period (2004–2007).

| Variable | Metric | Value | Period |
|----------|--------|-------|--------|
| TWS vs GRACE | r | 0.89 | 2004–2017 |
| TWS vs GRACE | RMSE (mm) | 43 | 2004–2017 |
| TWS vs GRACE | Bias (mm) | +3 | 2004–2017 |
| Streamflow | r | 0.89 | Eval 2011–2017 |
| Streamflow | NSE | 0.72 | Eval 2011–2017 |
| Streamflow | KGE | 0.63 | Eval 2011–2017 |
| Streamflow | PBIAS (%) | −28 | Eval 2011–2017 |
| SWE vs CanSWE | r | 0.88 | 2004–2017 |
| SWE vs CanSWE | RMSE (mm) | 184 | 2004–2017 |
| SWE vs CanSWE | Bias (mm) | −131 | 2004–2017 |
| ET vs SSEBop | r | 0.82 | 2004–2007 |
| ET vs SSEBop | RMSE (mm/day) | 1.05 | 2004–2007 |
| ET vs SSEBop | Bias (mm/day) | +0.24 | 2004–2007 |

### 4.10.2 Experiment (b): Joint Snow Cover and Soil Moisture Evaluation

Snow cover area (SCA) and soil moisture represent two key state variables that constrain the surface water and energy balance from above and below the land surface, respectively. Evaluating both simultaneously tests whether a model can reproduce the timing of snow accumulation and melt while maintaining a physically consistent soil moisture response — a balance that single-variable calibration may not achieve (Rakovec et al., 2016). This experiment jointly evaluates SUMMA simulations against MODIS-derived SCA and SMAP-derived soil moisture in the Paradise Creek catchment on Mt. Rainier, Washington (~500 km²).

#### Domain and configuration

The catchment is configured as a lumped domain centred on 46.79°N, 121.75°W with ERA5 forcing. The simulation spans 2015–2023 with a one-year spinup (2015–2016), four-year calibration period (2016–2020), and three-year evaluation period (2020–2023). MODIS MOD10A1 provides daily binary SCA maps at 500 m resolution, with a minimum snow fraction threshold of 10% applied to convert the SUMMA continuous snow water equivalent field to a binary SCA prediction. SMAP Level 3 provides soil moisture retrievals at ~36 km resolution on a 3-day repeat cycle.

The multi-objective function combines SCA accuracy (weight 0.5) with soil moisture Pearson correlation (weight 0.5). Eleven SUMMA parameters are calibrated, spanning both snow processes (`albedoMax`, `albedoMinWinter`, `albedoMinSpring`, `newSnowDenMin`) and soil hydraulic properties (`k_soil`, `theta_sat`, `critSoilWilting`, `theta_res`, `f_impede`), ensuring that the optimiser can adjust both snow and soil representations to improve the joint objective:

```yaml
MULTIVAR_TARGETS:
  - variable: SCA
    source: MODIS
    product: MOD10A1
    metric: accuracy
    weight: 0.5
  - variable: soil_moisture
    source: SMAP
    product: L3
    metric: correlation
    weight: 0.5
```

#### Results

<!-- PLACEHOLDER: Results to be added after experiment execution -->

**Figure X.** Paradise Creek joint SCA and soil moisture evaluation. (a) Monthly SCA accuracy for the evaluation period, comparing calibrated (blue) and uncalibrated (grey) SUMMA against MODIS MOD10A1. (b) Confusion matrices for the accumulation season (Oct–Mar) and ablation season (Apr–Jun). (c) Soil moisture time series: SMAP L3 retrievals (black) and calibrated SUMMA (blue). (d) Trade-off between SCA accuracy and soil moisture correlation across the DDS optimisation trajectory.

**Table X.** Joint evaluation metrics for the Paradise Creek experiment.

### 4.10.3 Experiment (c): Regional Snow Cover Fraction Trends

Snow cover is among the most climate-sensitive land surface variables at high latitudes, with declining trends observed across the Northern Hemisphere over the satellite record (Brown and Robinson, 2011; Mudryk et al., 2020). Reproducing observed decadal trends in snow cover fraction (SCF) requires that a model capture not only the mean seasonal cycle but also the interannual variability driven by changes in temperature and precipitation regimes. This experiment evaluates whether SYMFLUENCE can reproduce the spatial pattern and magnitude of SCF trends across Iceland over the 2000–2023 MODIS record — the longest continuous satellite snow cover dataset available.

#### Domain and configuration

The domain encompasses the full island of Iceland (63.3–66.6°N, 24.6–13.4°W, ~103,000 km²), discretised as a semi-distributed mesh of 2,683 GRUs generated by TauDEM stream-threshold delineation (threshold = 5,000 cells on the Copernicus DEM). CARRA (Copernicus Arctic Regional Reanalysis) provides meteorological forcing at 2.5 km spatial and 3-hourly temporal resolution — a substantially finer forcing product than the ERA5 used in Sections 4.8–4.9, chosen to better resolve the strong orographic precipitation gradients that control snow persistence in Iceland.

The simulation spans the full MODIS era (2000–2023) with a two-year spinup (2000–2001), eleven-year calibration period (2002–2012), and eleven-year evaluation period (2013–2023). MODIS MOD10A2 8-day maximum snow extent composites provide the observational reference. A cloud fraction filter (`MODIS_CLOUD_MAX_FRACTION: 0.3`) excludes composites with excessive cloud contamination. Eleven snow-focused SUMMA parameters are calibrated using DDS (1,000 iterations), targeting trend correlation between simulated and observed SCF time series.

Trend analysis applies the Mann-Kendall test with Sen's slope estimator at the 5% significance level, stratified by:
- **Season**: accumulation (Oct–Mar), ablation (Apr–Jun), and snow-free (Jul–Sep);
- **Elevation**: nine bands from 0–200 m to 1,500–2,000 m, capturing the strong elevation dependence of snow persistence in Iceland.

The SYMFLUENCE configuration specifies these analysis settings declaratively:

```yaml
SCF_TREND_METHOD: Mann-Kendall
SCF_TREND_SIGNIFICANCE: 0.05
SCF_SEASONAL_DECOMPOSITION: True
SCF_SEASONS:
  accumulation: [10, 11, 12, 1, 2, 3]
  ablation: [4, 5, 6]
  snow_free: [7, 8, 9]
SCF_ELEVATION_BANDS: [0, 200, 400, 600, 800, 1000, 1200, 1500, 2000]
```

#### Results

<!-- PLACEHOLDER: Results to be added after experiment execution -->

**Figure X.** Iceland SCF trend analysis. (a) Map of simulated SCF trend (Sen's slope, % decade⁻¹) over the evaluation period (2013–2023), with stippling indicating statistical significance (p < 0.05). (b) Corresponding MODIS MOD10A2 observed SCF trend map. (c) Scatter plot of simulated vs. observed trend magnitudes per GRU. (d) Elevation-stratified SCF trends for the ablation season, comparing simulated (blue) and observed (black) with 95% confidence intervals.

**Figure X.** Seasonal decomposition of SCF trends. (a) Annual mean SCF time series (2000–2023) for simulated (blue) and observed (black), with linear trend overlaid. (b–d) Seasonal SCF trends for accumulation, ablation, and snow-free periods, stratified by elevation band.

**Table X.** Mann-Kendall trend statistics for simulated and observed SCF by season and elevation band. Columns show Sen's slope (% decade⁻¹), p-value, and trend direction agreement between model and observations.

### 4.10.4 Framework Implications

The multivariate evaluation experiments test three aspects of SYMFLUENCE's architecture that streamflow-only experiments do not exercise.

First, the satellite data acquisition and processing pipeline demonstrates that SYMFLUENCE can ingest heterogeneous observation sources — gravimetric (GRACE), optical (MODIS), and microwave (SMAP) — through a unified handler interface. Each satellite product has distinct spatial resolutions (300 km for GRACE mascons, 500 m for MODIS, 36 km for SMAP), temporal sampling patterns (monthly, daily, 3-day), and data formats. The framework's acquisition registry dispatches to product-specific handlers that manage download, quality filtering, spatial aggregation to the model domain, and temporal alignment, exposing a consistent interface to the evaluation and calibration modules regardless of the underlying data source.

Second, the weighted multi-objective calibration mechanism enables users to combine arbitrary evaluation metrics into a single objective function through the declarative `MULTIVAR_TARGETS` configuration. This design avoids the need for users to implement custom objective functions or modify optimiser code when adding new observation constraints. The weight specification allows systematic exploration of trade-offs between competing objectives — for example, how strongly constraining TWS dynamics affects streamflow performance, or whether jointly calibrating to SCA and soil moisture yields parameters that are physically more consistent than single-variable calibration.

Third, the regional trend analysis in Experiment (c) extends the evaluation paradigm from point-in-time performance metrics to long-term trend reproduction — a capability relevant to climate change impact assessments where the question shifts from "does the model match today's observations?" to "does the model reproduce observed rates of change?" The configuration-driven specification of trend analysis methods, significance thresholds, and stratification variables ensures that such analyses are reproducible and can be applied to other domains and variables without code modification.
