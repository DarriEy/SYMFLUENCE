# From Configuration to Prediction: Multi-Model, Multi-Basin Experiments with SYMFLUENCE

**Authors:** Darri Eythorsson, Nicolas Vasquez, Cyril Thébault, Frank Han, Kasra Keshavarz, Wouter Knoben, Dave Casson, Mohammed Ismail Ahmed, Ashley Van Beusekom, Hongli Liu, Befekadu Taddesse Woldegiorgis, Camille Gautier, Katherine Reece, Peter Wagener, Ignacio Aguirre, Paul Coderre, Neharika Bhattarai, Junwei Guo, Shadi Hatami, David Tarboton, James Halgren, Jordan Reed, Steve Burian, Raymond Spiteri, Alain Pietroniro, and Martyn Clark

## Abstract

The companion papers in this series argued that predictive stagnation in computational hydrology reflects an infrastructure deficit rather than a scientific one (Eythorsson et al., 2025a), and described the four-tier layered architecture, registry-based extensibility, and declarative configuration system through which SYMFLUENCE addresses that deficit (Eythorsson et al., 2025b). This paper subjects that architecture to empirical test. Through twelve experiments spanning point-scale flux estimation to regional hydrological simulation across Iceland (~103,000 km²), we demonstrate that the architectural principles described in the companion papers translate into practical experimental capability. We present: (i) domain definition experiments demonstrating scale-invariant workflow execution from a single flux tower to 21,474 hydrological response units; (ii) a six-model ensemble comparing structurally diverse models (SUMMA, FUSE, GR4J, HBV, HYPE, LSTM) on a common testbed, where the ensemble mean (KGE = 0.94) exceeds every individual model; (iii) a forcing ensemble isolating the effect of seven meteorological products on snow simulation, revealing forcing-dependent parameter compensation that undermines transferability; (iv) a twelve-algorithm calibration comparison exposing bimodal parameter regimes and demonstrating that gradient-based methods achieve superior generalization; (v) a 64-member FUSE structural decision ensemble showing that intra-model structural uncertainty rivals inter-model uncertainty; (vi) cross-model sensitivity analysis identifying soil storage and routing as consistently influential processes; (vii) a 111-catchment large-sample application across Iceland; (viii) a fully distributed regional simulation with explicit river routing; (ix) multivariate evaluation against GRACE, MODIS, and SMAP satellite observations; (x) parallel scaling characterization across execution strategies; and (xi) end-to-end data processing pipeline analysis across three orders of magnitude in domain size. The experiments are not intended as definitive hydrological studies; each would benefit from extended evaluation periods and additional basins. Their purpose is to demonstrate that when technical friction is reduced through deliberate architectural design, experiments that would otherwise be prohibitively labor-intensive become routine -- expanding the space of questions that researchers can ask.

---

## 1. Introduction

The first paper in this series (Eythorsson et al., 2025a) documented the impossible generalist problem and argued that the apparent stagnation in hydrological prediction reflects infrastructure fragmentation rather than scientific limits. The second paper (Eythorsson et al., 2025b) described the architectural response: SYMFLUENCE's four-tier layered design, registry-based extensibility, declarative configuration system, CF-Intermediate Format, and scale-invariant workflow orchestration. This third paper asks the question that architecture alone cannot answer: *does it work?*

The experiments presented here are designed to test the architectural claims made in the companion papers against concrete hydrological problems. Each experiment isolates a specific source of uncertainty -- model structure, forcing data, calibration algorithm, intra-model structural decisions, spatial discretization -- while holding all other factors constant through SYMFLUENCE's configuration-driven design. This degree of experimental control is straightforward to describe but difficult to achieve in practice when each factor change requires modifications to bespoke scripts. The architectural investment in declarative configuration, standardized preprocessing, and unified evaluation interfaces is what makes single-factor isolation practical at the scale demonstrated here.

We organize the experiments along two axes. The first axis is *spatial scale*: from point-scale flux tower validation (Paradise SNOTEL), through watershed-scale process modeling (Bow River at Banff, 2,210 km²), to regional-scale distributed simulation (Iceland, ~103,000 km²). The second axis is *experimental design*: single-model calibration, multi-model ensemble, forcing ensemble, algorithm comparison, structural decision analysis, sensitivity screening, large-sample application, multivariate evaluation, and computational scaling.

The results are offered as demonstrations of what becomes practical when infrastructure barriers are lowered, not as definitive hydrological studies. Each experiment would benefit from extended evaluation periods, additional basins, and deeper process-level analysis. Nevertheless, the collection illustrates a central claim of this series: that the bottleneck limiting hydrological prediction is shifting from implementation to inquiry.

### 1.1 Experimental overview

Table 1 summarizes the twelve experiments, the architectural features they exercise, and the sections in which they are described.

**Table 1.** Overview of experiments presented in this paper.

| # | Experiment | Section | Domain | Models | Key Architectural Feature Tested |
|---|---|---|---|---|---|
| 1 | Domain definition | 2 | Paradise, Bow, Iceland | -- | Scale-invariant spatial discretization |
| 2 | Multi-model ensemble | 3 | Bow at Banff | 6 | Registry-based model integration |
| 3 | Forcing ensemble | 4 | Paradise | SUMMA | Configuration-driven factor isolation |
| 4 | Algorithm comparison | 5 | Bow at Banff | HBV | Algorithm registry, JAX integration |
| 5 | Benchmarking | 6 | Bow at Banff | 6 | Integrated evaluation framework |
| 6 | Decision ensemble | 7 | Bow at Banff | FUSE (64) | Structural analysis automation |
| 7 | Sensitivity analysis | 8 | Bow at Banff | 5 | Cross-model process mapping |
| 8 | Large-sample | 9 | Iceland (111 catchments) | FUSE | Batch configuration scaling |
| 9 | Large-domain | 10 | Iceland (full island) | FUSE + mizuRoute | Distributed routing integration |
| 10 | Multivariate evaluation | 11 | Bow, Paradise, Iceland | SUMMA | Satellite data handlers, multi-objective |
| 11 | Parallel scaling | 12 | Bow at Banff | HBV, SUMMA | Execution strategy selection |
| 12 | Data pipeline | 13 | Paradise, Bow, Iceland | -- | CFIF, EASYMORE weight caching |

---

## 2. Domain Definition Across Scales

SYMFLUENCE supports three fundamental spatial modes -- point, watershed, and regional -- enabling consistent workflows from single-site validation to continental-scale simulation. We demonstrate this flexibility through applications at Paradise SNOTEL (Washington, USA), the Bow River at Banff (Alberta, Canada), and the national domain of Iceland. The spatial discretizations considered are listed in Table 2.

**Table 2.** Spatial domains and discretization configurations.

| Domain | Configuration | GRUs | HRUs | Segments |
|---|---|---|---|---|
| Paradise | Point | 1 | 1 | 0 |
| Bow | Lumped | 1 | 1 | 0 |
| Bow | Lumped + elevation bands | 1 | 12 | 0 |
| Bow | Lumped + land classes | 1 | 9 | 0 |
| Bow | Lumped + elevation & aspect | 1 | 94 | 0 |
| Bow | Lumped + distributed routing | 1 | 1 | 49 |
| Bow | Lumped + elevation & distributed routing | 1 | 12 | 49 |
| Bow | Semi-distributed | 49 | 379 | 49 |
| Bow | Semi-distributed + elevation | 49 | 379 | 49 |
| Bow | Semi-distributed + elevation & aspect | 49 | 2,596 | 49 |
| Bow | Distributed | 2,335 | 2,335 | 2,335 |
| Iceland | Semi-distributed | 6,606 | 6,606 | 6,606 |
| Iceland + coastal | Semi-distributed | 7,618 | 7,618 | 6,606 |
| Iceland + coastal | Semi-distributed + elevation | 7,618 | 21,474 | 6,606 |

### 2.1 Point-scale: Paradise SNOTEL

Point-scale applications treat the domain as a single computational unit without lateral routing, appropriate for flux tower validation or snow monitoring stations where spatial heterogeneity is negligible relative to vertical process dynamics. The Paradise SNOTEL station (46.78°N, 121.75°W; elevation 1,560 m) on Mount Rainier receives approximately 2,500 mm annual precipitation, predominantly as snow, making it an ideal testbed for snow process representation.

Configuration requires only pour point coordinates and a bounding box for forcing data acquisition. The framework automatically generates a single GRU domain that intersects with the ERA5 forcing grid cells (0.25° resolution) that overlap the station location. This minimal configuration -- requiring approximately 15 lines of YAML -- produces a complete SUMMA setup for snow water equivalent simulation and validation against SNOTEL observations.

### 2.2 Watershed-scale: Bow River at Banff

Watershed applications represent the most common use case, where spatially distributed processes require explicit representation but domain extent remains computationally tractable. The Bow River basin upstream of Banff (51.17°N, 115.57°W; approximately 2,210 km², 20 ERA5 cells) spans elevations from 1,383 m at the gauge to over 3,436 m at the continental divide, exhibiting strong gradients in precipitation, temperature, and snow dynamics that motivate spatial discretization.

**Figure 1.** Spatial discretization options for the Bow River at Banff across three columns (lumped, semi-distributed, distributed) and three rows of increasing complexity.

Lumped mode treats the entire basin as a single GRU, aggregating all forcing and parameters to basin-average values. This mode minimizes computational cost and parameter dimensionality. Within this single GRU, the framework supports subdivision into HRUs by elevation bands at 200 m intervals (12 HRUs spanning 1,383--3,436 m) or by IGBP land cover classification (9 HRUs). Elevation-band subdivision enables lapse-rate corrections for temperature and precipitation without requiring explicit sub-basin delineation.

Semi-distributed mode partitions the basin into 49 sub-basin GRUs derived from the TDX river network topology, each further subdividable into HRUs. Elevation-band subdivision yields 379 HRUs (~7.7 per GRU), while combined elevation and 8-class aspect subdivision produces 2,596 HRUs (~53 per GRU). The semi-distributed river network (49 segments) enables explicit lateral routing through mizuRoute.

Distributed mode discretizes the basin into 2,335 grid cells at 1 km resolution, each functioning as both a GRU and an HRU, with a corresponding distributed river network.

Hybrid configurations demonstrate that spatial discretization choices for hydrological modeling and routing are independent in SYMFLUENCE and can be mixed freely. A lumped GRU paired with the semi-distributed river network (49 segments) distributes lumped runoff across routing segments. Extending this with lumped elevation-band HRUs (12 HRUs) combined with semi-distributed routing (49 segments) captures elevation-dependent processes -- orographic precipitation gradients, snow accumulation and melt timing -- through lapse-rate corrections applied to each elevation band, while routing the resulting runoff through the spatially explicit river network. Such configurations offer a computationally efficient compromise: vertical process heterogeneity is represented without requiring full sub-basin delineation.

Importantly, all configurations derive from the same pour point and underlying DEM, ensuring consistent domain boundaries while varying only internal discretization.

### 2.3 Regional-scale: Iceland

Regional applications extend watershed concepts to multi-basin domains where consistent forcing and parameter treatment across drainage divides enables coherent large-scale simulation. Iceland (approximately 102,000 km²) presents an ideal demonstration domain: island geography provides unambiguous boundaries, diverse hydroclimatology spans glaciated highlands to coastal lowlands, and the LamaH-ICE dataset provides validation streamflow across multiple gauges.

**Figure 2.** Iceland domain definition at various resolutions: (a) 6,600 river basin GRUs from geofabric delineation, (b) 7,618 GRUs including 1,018 coastal watersheds, (c) 21,474 HRUs from elevation-band subdivision.

Regional configuration specifies a bounding box rather than a pour point, with automatic delineation of all watersheds draining to the coast. The `DELINEATE_COASTAL_WATERSHEDS` option identifies terminal basins draining directly to ocean boundaries, ensuring complete coverage without manual pour point specification. The resulting domain comprises 6,600 GRUs connected by 6,606 river segments, with 1,018 additional coastal GRUs bringing the total to 7,618. Elevation-band subdivision yields 21,474 HRUs across the domain (approximately 2.8 per GRU on average), with elevations spanning from sea level to over 2,000 m. The ERA5 forcing grid for Iceland comprises 618 cells at 0.25° resolution.

### 2.4 Discretization trade-offs

The choice of spatial discretization reflects a trade-off between process representation and computational/parametric complexity. Lumped and semi-distributed modes share parameters across all computational units, maintaining the same calibration dimensionality regardless of HRU count. A lumped configuration with 12 elevation-band HRUs has the same number of free parameters as one with a single HRU -- the elevation bands differ only in their forcing (via lapse-rate adjustments) and static attributes (mean elevation, area), not in calibrated parameter values. Distributed modes can optionally enable spatially varying parameters, though this substantially increases calibration complexity and typically requires regionalization or transfer approaches.

The hybrid configurations illustrate that practitioners need not choose a single discretization paradigm. By decoupling the hydrological response unit structure from the routing network, SYMFLUENCE allows users to independently control vertical complexity (elevation bands, land cover classes, aspect classes) and horizontal complexity (routing network density). For snow-dominated basins like the Bow River, a lumped elevation-band configuration with semi-distributed routing may capture the dominant sources of spatial variability at a fraction of the computational cost of a fully distributed setup.

### 2.5 Forcing grid intersection

Regardless of spatial mode, SYMFLUENCE generates intersection weights mapping forcing grid cells to computational units. For ERA5 (0.25° resolution), the Paradise point domain intersects 9 grid cells; the Bow watershed intersects 20 cells; Iceland requires 618 cells. Intersection weights account for partial overlaps. The intersection geometry is preserved as a shapefile, enabling visualization of forcing grid coverage relative to basin boundaries and supporting diagnosis of scale mismatches.

---

## 3. Multi-Model Ensemble

To demonstrate SYMFLUENCE's capacity for orchestrating structurally diverse models within a unified workflow, we configured an ensemble of six hydrological models for the Bow River at Banff lumped catchment (drainage area 2,210 km²) using ERA5 meteorological forcing. Each model was set up from a near-identical YAML configuration file specifying the domain definition, forcing data paths, observation targets, and calibration settings. Model-specific parameters were the only elements that varied between configurations. All models were calibrated against observed daily streamflow using the Dynamically Dimensioned Search (DDS; Tolson & Shoemaker, 2007) algorithm over the period 2003--2005, with independent evaluation over 2006--2009.

**Table 3.** Multi-model ensemble members and performance.

| Model | Paradigm | *n* | Cal. KGE | Eval. KGE |
|---|---|---|---|---|
| SUMMA | Process-based | 11 | 0.90 | 0.88 |
| FUSE | Flexible conceptual | 13 | 0.90 | 0.88 |
| GR4J | Parsimonious conceptual | 4 | 0.92 | 0.79 |
| HBV | Conceptual | 15 | 0.74 | 0.70 |
| HYPE | Conceptual | 10 | 0.87 | 0.81 |
| LSTM | Machine learning | -- | 0.97 | 0.88 |

The ensemble spans three modeling paradigms -- process-based (SUMMA, FUSE, HYPE), conceptual (GR4J, HBV), and data-driven (LSTM) -- providing a broad sampling of structural uncertainty. SUMMA and FUSE share the mizuRoute river routing scheme but differ in their representations of vertical water and energy fluxes. GR4J achieves strong performance with only four free parameters, while HBV, despite having 15 parameters, yields the lowest calibration KGE (0.74), largely due to a persistent negative bias in mean flow (β = 0.75). The LSTM, trained on the same forcing-streamflow pairs used for calibration, achieves the highest calibration KGE of all models (0.97).

### 3.1 Multi-model hydrograph

**Figure 3.** Simulated and observed hydrographs over the full analysis period: (a) all six models with calibration-evaluation boundary at January 2006; (b) zoom into the April--October 2005 snowmelt season.

All six models reproduce the dominant seasonal cycle of the Bow River, characterized by low winter baseflow (5--15 m³ s⁻¹) and a pronounced snowmelt-driven freshet peaking in June--July (100--300 m³ s⁻¹). During the calibration period, model traces are tightly clustered around the observed hydrograph. During evaluation, the ensemble spread widens -- most noticeably around peak events -- reflecting how each model's structural assumptions about snowmelt dynamics, soil moisture partitioning, and baseflow recession lead to differing responses under conditions not seen during calibration.

The zoomed panel reveals model-specific behavior during the 2005 freshet. HBV consistently underestimates both peak and recession flows, as expected given its low β. GR4J captures the peak magnitude but exhibits sharper, less-damped recession limbs. HYPE shows a delayed and prolonged recession through August--October that diverges from the observed signal. The LSTM tracks the observed hydrograph closely through the rising limb and peak but, like most models, underestimates secondary peaks in the late season. SUMMA and FUSE produce nearly indistinguishable traces throughout, confirming that SYMFLUENCE's workflow yields reproducible results when models share a common routing framework.

### 3.2 KGE decomposition

**Figure 4.** KGE decomposition into correlation (*r*), variability ratio (α), and bias ratio (β) for calibration and evaluation periods.

To diagnose the sources of performance differences, we decompose the KGE into its three components -- Pearson correlation (*r*), variability ratio (α = σ_sim / σ_obs), and bias ratio (β = μ_sim / μ_obs) -- for both calibration and evaluation periods.

All six models maintain high correlation (*r* > 0.87) across both periods, indicating that the timing of hydrological events is consistently well captured regardless of model structure. The primary drivers of KGE variation are α and β. During calibration, most models cluster near the ideal values (α ≈ 1, β ≈ 1), with HBV as the notable exception: its β of 0.75 indicates a 25% underestimation of mean flow. This bias persists into the evaluation period (β = 0.81), pointing to a systematic structural limitation rather than parameter identifiability issues.

During evaluation, all models show α > 1.0, meaning they overestimate flow variability relative to observations. This is most pronounced for HBV (α = 1.22) and HYPE (α = 1.16). GR4J shows the largest shift in β between periods (from 1.01 to 1.12), transitioning from near-unbiased to a 12% positive bias -- likely a consequence of its parsimonious structure limiting its ability to generalize across climate conditions.

The LSTM maintains the most balanced decomposition in evaluation (*r* = 0.93, α = 1.08, β = 1.05), though this does not imply physical realism. SUMMA and FUSE again yield nearly identical decompositions (evaluation *r* = 0.91, α = 1.08, β = 0.99 for both), further confirming workflow consistency.

### 3.3 Ensemble envelope and flow duration curve

**Figure 5.** Ensemble evaluation: (a) ensemble envelope (min--max range) for the evaluation period with ensemble mean and median; (b) flow duration curves for observations, ensemble statistics, and individual models.

The observed hydrograph falls within the ensemble envelope for the large majority of the evaluation period. The ensemble captures peak flows well, though the 2008 annual maximum (~280 m³ s⁻¹) briefly exceeds the upper bound, indicating that all models underestimate the most extreme event in the record. The ensemble mean (KGE = 0.94) and median (KGE = 0.92) both outperform every individual model. This improvement arises from the cancellation of offsetting structural biases: HBV's negative bias and GR4J's positive bias partially neutralize each other in the average, while the correlation component is preserved across all members.

The FDC comparison provides a complementary view. At low exceedance probabilities (<20%, high flows), all models track the observed FDC closely. At intermediate exceedance probabilities (20--80%), the ensemble mean aligns almost exactly with the observed curve. The greatest inter-model spread occurs at high exceedance probabilities (>80%, low flows): HBV drops to near-zero discharge at the lowest flows, while HYPE maintains higher baseflows than observed. The ensemble mean smooths these extremes, tracking the observed low-flow regime more faithfully than any single model.

### 3.4 Implications

Three aspects of these results are relevant to SYMFLUENCE's design objectives. First, the near-identical performance of SUMMA and FUSE demonstrates that SYMFLUENCE's configuration layer produces consistent, reproducible results across models. Second, the diversity of the ensemble illustrates that the model-agnostic design does not restrict users to a narrow class of models -- the same DDS calibration workflow was applied across all six models, with changes confined to a single YAML file per model. Third, the ensemble mean's KGE of 0.94 exceeds the best individual model (LSTM, KGE = 0.88) by a meaningful margin, consistent with the broader multi-model averaging literature (Ajami et al., 2006; Arsenault et al., 2015).

---

## 4. Forcing Ensemble

Whereas Section 3 held the forcing dataset fixed and varied model structure, this experiment holds model structure fixed and varies the atmospheric forcing. Using SUMMA at the Paradise SNOTEL station, we calibrated an identical set of 11 snow-related parameters against observed daily SWE under each of seven forcing datasets. The calibration period spans water years 2016--2018, with independent evaluation over water years 2019--2020. All experiments used DDS optimization of RMSE against SNOTEL SWE, with configuration files differing only in their forcing data paths and identifiers.

**Table 4.** Forcing datasets and SUMMA SWE performance.

| Forcing | Type | Resolution | Cal. KGE | Eval. KGE |
|---|---|---|---|---|
| ERA5 | Global reanalysis | ~31 km | 0.30 | −0.59 |
| AORC | Gridded observations | ~1 km | 0.77 | 0.87 |
| CONUS404 | WRF reanalysis | ~4 km | 0.19 | 0.68 |
| RDRS | Regional reanalysis | ~10 km | 0.94 | 0.60 |
| ACCESS-CM2 | NEX-GDDP CMIP6 | ~25 km | 0.76 | 0.37 |
| GFDL-ESM4 | NEX-GDDP CMIP6 | ~25 km | -- | -- |
| MRI-ESM2-0 | NEX-GDDP CMIP6 | ~25 km | 0.65 | 0.69 |

The seven forcing datasets fall into two categories: four observation-based reanalysis or gridded products -- ERA5, AORC, CONUS404, and RDRS -- and three members of the NEX-GDDP-CMIP6 downscaled climate projections: ACCESS-CM2, GFDL-ESM4, and MRI-ESM2-0.

### 4.1 SWE time series

**Figure 6.** Simulated and observed SWE: (a) four reanalysis-driven simulations versus observed SNOTEL SWE; (b) GDDP-driven simulations with ensemble envelope and mean.

All reanalysis-driven simulations reproduce the dominant seasonal SWE cycle, with accumulation from November through April and melt completing by July. However, the spread among reanalysis products is substantial. ERA5-driven SWE consistently overshoots observed peak SWE by 500--1,000 mm during the calibration period, and by over 1,500 mm in the evaluation winters of 2019 and 2020. This overshoot reflects a known positive precipitation bias in ERA5 at high-elevation Pacific Northwest sites. AORC (~1 km) and CONUS404 (~4 km) track the observed record most faithfully. RDRS captures the general pattern well during calibration but underestimates peak SWE in the evaluation period.

The GDDP-driven simulations show a wider inter-member spread than the reanalysis ensemble, particularly during accumulation seasons. The ensemble envelope expands noticeably during evaluation, consistent with the fact that GDDP products represent climate model free runs rather than observation-constrained reanalyses.

### 4.2 Performance and transferability

**Figure 7.** Performance heatmap and KGE degradation across forcing datasets.

Two key patterns emerge. First, calibration performance is not a reliable predictor of evaluation performance. RDRS achieves the highest calibration KGE (0.94) but degrades by 0.35, while AORC, with a more modest calibration KGE (0.77), actually *improves* during evaluation (KGE = 0.87). This suggests that AORC's higher spatial resolution (1 km) provides forcing fields that are more physically representative of the point-scale SNOTEL site, allowing the calibrated parameter set to generalize without overfitting. RDRS and ERA5, by contrast, deliver forcing at 10--31 km resolution that requires calibration to compensate for systematic biases -- a calibration that transfers poorly to unseen years.

Second, ERA5 is a clear outlier. Its evaluation KGE collapses to −0.59, with an RMSE roughly four times that of AORC, reflecting persistent and amplifying precipitation biases at this maritime mountain site.

### 4.3 Parameter compensation and equifinality

**Figure 8.** Parameter divergence across forcing datasets: (a) Z-score-normalized parameter values; (b) frozen precipitation multiplier versus KGE degradation; (c) composite distortion index versus KGE degradation.

The parameter heatmap reveals pronounced forcing-dependent compensation. The `frozenPrecipMultip` parameter spans a five-fold range across forcings, from 0.84 (CONUS404) to 4.98 (MRI-ESM2-0). Forcings with known precipitation biases drive the calibration toward extreme multiplier values. ERA5 and RDRS both calibrate to multipliers above 4.0, effectively quadrupling their frozen precipitation inputs. AORC and CONUS404, whose finer resolutions better resolve orographic precipitation gradients, require multipliers near unity -- indicating that their forcing fields already provide physically reasonable precipitation amounts.

A similar pattern appears in the melt-related parameters. The melt exponent and albedo decay rate vary by orders of magnitude across forcings, with the most extreme values occurring for ERA5 and RDRS. These compensatory adjustments suggest that the calibration is not recovering the "true" snow physics but rather absorbing forcing biases into parameter values tuned to specific error structures.

### 4.4 Implications

The forcing ensemble highlights three aspects of SYMFLUENCE's design. First, the configuration-driven approach makes it straightforward to isolate the effect of a single experimental factor. Second, the results provide a practical caution against interpreting calibration skill as a measure of forcing quality -- SYMFLUENCE automates split-sample evaluation, making this comparison routine. Third, the GDDP-CMIP6 configurations illustrate extensibility to climate projection workflows: the ten-member GDDP ensemble was generated from a single template configuration using automated substitution of model identifiers, requiring no changes to the core framework.

---

## 5. Calibration Algorithm Comparison

To evaluate SYMFLUENCE's optimization suite, we compare 12 calibration algorithms drawn from six algorithmic families on a common testbed: the HBV model applied to the Bow River at Banff lumped catchment with ERA5 forcing. All algorithms calibrate the same 14 HBV parameters against observed daily streamflow over 2004--2007, with independent evaluation over 2008--2009 and a two-year spinup. KGE serves as the objective function. Each algorithm is allocated a normalized budget of approximately 4,000 function evaluations.

**Table 5.** Calibration and evaluation performance for 12 optimization algorithms. Algorithms sorted by calibration KGE.

| Rank | Algorithm | Family | Cal. KGE | Eval. KGE | KGE Degrad. |
|---|---|---|---|---|---|
| 1 | DDS | Sampling | 0.757 | 0.749 | +0.008 |
| 2 | Nelder-Mead | Direct search | 0.751 | 0.720 | +0.031 |
| 3 | GA | Evolutionary | 0.745 | 0.723 | +0.023 |
| 4 | CMA-ES | Evolutionary | 0.745 | 0.735 | +0.011 |
| 5 | ADAM | Gradient | 0.743 | 0.778 | −0.035 |
| 6 | SA | Stochastic | 0.736 | 0.742 | −0.006 |
| 7 | DE | Evolutionary | 0.734 | 0.740 | −0.006 |
| 8 | SCE-UA | Evolutionary | 0.683 | 0.637 | +0.045 |
| 9 | Bayes. Opt. | Surrogate | 0.678 | 0.687 | −0.009 |
| 10 | PSO | Evolutionary | 0.662 | 0.628 | +0.034 |
| 11 | Basin Hopping | Stochastic | 0.662 | 0.674 | −0.012 |
| 12 | L-BFGS | Gradient | 0.654 | 0.702 | −0.048 |

Two design choices merit explicit discussion. First, gradient-based methods (ADAM, L-BFGS) require a differentiable model. SYMFLUENCE provides this by engaging a smooth HBV formulation (`HBV_SMOOTHING_FACTOR = 15.0`) that replaces threshold operations with soft approximations when the JAX backend is active. The comparison therefore evaluates *algorithm + compatible model formulation* pairs rather than algorithms in isolation. Second, the function evaluation budget is approximate rather than exact: Nelder-Mead may consume up to 8,000 evaluations due to internal simplex reflections, and Bayesian Optimization uses only ~200 true function evaluations but fits a Gaussian process surrogate at each iteration.

### 5.1 Algorithm performance and generalization

**Figure 9.** Algorithm performance: (a) calibration and evaluation KGE sorted by calibration performance; (b) calibration versus evaluation KGE with 1:1 line.

A reference line at KGE = 0.75 divides the field: four algorithms (DDS, Nelder-Mead, GA, CMA-ES) exceed this threshold during calibration. The calibration KGE range is narrow (0.654--0.757, spread of 0.103), indicating that for this basin and model the objective surface is relatively accessible. Six of 12 algorithms fall above the 1:1 line, meaning their parameter sets perform better in evaluation than in calibration. The most striking case is ADAM, which ranks fifth in calibration (KGE = 0.743) but achieves the highest evaluation KGE of any algorithm (0.778). This suggests that the smoothed HBV formulation required by gradient-based optimization may act as an implicit regularizer, producing parameter sets that generalize more robustly.

Conversely, SCE-UA exhibits the largest positive degradation (+0.045), overfitting to the calibration period more than any other algorithm despite its long pedigree in hydrological calibration. All 12 algorithms produce a systematic negative percent bias (PBIAS −23% to −32%), indicating that HBV underestimates mean flow at this site regardless of the parameter set -- a model structural limitation rather than an optimizer deficiency.

### 5.2 Convergence efficiency

**Figure 10.** Convergence trajectories: best KGE as a function of cumulative function evaluations.

DDS reaches KGE = 0.74 within the first 100 function evaluations and plateaus shortly thereafter. ADAM and Nelder-Mead show rapid initial gains, both reaching KGE > 0.70 within 500 evaluations. By contrast, population-based evolutionary methods (GA, CMA-ES) require 1,500--2,500 evaluations to match the same performance level. SCE-UA, Bayesian Optimization, and PSO plateau between KGE = 0.66 and 0.68, suggesting these algorithms would benefit from larger evaluation budgets on this problem.

### 5.3 Parameter equifinality

**Figure 11.** Calibrated parameter values for all 12 algorithms as a normalized heatmap.

The heatmap reveals substantial equifinality: algorithms achieving similar KGE values converge to markedly different parameter sets. This is most evident in the snow module parameters. The top-performing group consistently identifies a high temperature threshold (TT = 1.2--2.2 °C) and a high degree-day factor (CFMAX = 6.8--10.0 mm °C⁻¹ d⁻¹), while the lower-performing group converges to a negative TT (−0.3 to −1.1 °C) and a low CFMAX (1.0--3.4 mm °C⁻¹ d⁻¹). These two regimes represent alternative snow accumulation and melt strategies: a "warm threshold / fast melt" regime versus a "cold threshold / slow melt" regime. Both achieve positive KGE but the warm-threshold regime consistently produces higher values.

Field capacity (FC) shows a similar bifurcation. Several parameters converge to boundary values across multiple algorithms: the snowfall correction factor (SFCF) pins at 1.5 (the upper bound) for 10 of 12 algorithms, suggesting that the feasible range may be too narrow.

### 5.4 Implications

Three findings bear on SYMFLUENCE's design. First, all 12 algorithms are specified through a single YAML parameter with standardized output formats, enabling automated post-processing. Second, the integration of JAX-based automatic differentiation opens HBV to gradient-based optimizers not typically used in hydrology -- ADAM's strong generalization performance suggests gradient methods deserve broader consideration. Third, the equifinality analysis highlights the value of multi-algorithm calibration as a diagnostic tool, revealing the bimodal snow-parameter structure that a single algorithm would not expose.

---

## 6. Benchmarking

To contextualize the multi-model ensemble skill, we evaluate the six calibrated models against a hierarchy of simple reference predictors following Schaefli and Gupta (2007). SYMFLUENCE's `Benchmarker` class computes 19 benchmark flow series from observed streamflow and ERA5 precipitation. After removing invalid benchmarks and consolidating numerically identical flow series, 12 distinct benchmarks remain.

**Figure 12.** Benchmarking results: (a) validation-period KGE of 12 benchmarks versus six calibrated models and ensemble statistics; (b) calibration and validation KGE for benchmarks grouped by category.

### 6.1 Benchmark performance hierarchy

Time-invariant predictors -- the long-term mean and median flow -- perform poorly (KGE = −0.41 and −0.47). Seasonal climatologies, by contrast, capture the dominant snowmelt cycle and achieve substantially higher skill: the daily median flow benchmark attains the highest validation KGE of any reference predictor (0.80). The narrow spread within the seasonal group indicates that the dominant source of predictability at Bow-at-Banff is the annual snowmelt cycle.

Rainfall-runoff benchmarks show mixed performance. At monthly resolution, the rainfall-runoff ratio benchmark achieves KGE = 0.76; at daily resolution it degrades sharply (KGE = 0.23), reflecting the poor correspondence between daily ERA5 precipitation and same-day streamflow in a snowmelt-dominated catchment.

### 6.2 Model ensemble versus benchmarks

Five of six calibrated models exceed all 12 valid benchmarks. LSTM (KGE = 0.88), SUMMA (0.88), and FUSE (0.88) lead by a margin of +0.08 above the best benchmark. HBV is the sole model that falls below the best benchmark (KGE = 0.70 vs. 0.80). The ensemble mean (KGE = 0.94) exceeds the best benchmark by +0.14, confirming that the multi-model ensemble provides substantial predictive value beyond reference models.

**Figure 13.** Benchmark time series: (a) evaluation period with best seasonal benchmark and ensemble envelope; (b) full record with benchmarks of increasing temporal resolution.

### 6.3 Implications

By embedding benchmark evaluation as a standard workflow step, SYMFLUENCE guards against over-interpreting modest KGE values. The daily median benchmark's KGE of 0.80 sets a high bar -- a model achieving KGE = 0.82 contributes only marginal information beyond the seasonal cycle, while the ensemble mean's KGE = 0.94 represents a clear advance. Making such comparisons effortless increases the likelihood that they are routinely reported.

---

## 7. Model Decision Ensemble

Sections 3--5 quantified inter-model, inter-forcing, and inter-calibration uncertainty by treating each model as a monolithic unit. Yet within a single model framework, the choice of process representation can be an equally important source of predictive uncertainty. The Framework for Understanding Structural Errors (FUSE; Clark et al., 2008) exposes these decisions explicitly: each of its nine structural dimensions admits two or more options, yielding 1,728 unique model structures from a common code base.

We select the six most hydrologically meaningful decision dimensions, each with two contrasting options, for a full-factorial design of 2⁶ = 64 combinations.

**Table 6.** Varied FUSE structural decisions.

| Decision | Description | Option A | Option B |
|---|---|---|---|
| ARCH1 | Upper-layer soil architecture | tension1_1 (tension, 2-state) | onestate_1 (single bucket) |
| ARCH2 | Lower-layer soil architecture | tens2pll_2 (tension parallel) | unlimfrc_2 (unlimited) |
| QSURF | Surface runoff generation | arno_x_vic (VIC-style) | prms_varnt (PRMS-style) |
| QPERC | Percolation | perc_f2sat (fraction to saturation) | perc_lower (lower-zone control) |
| ESOIL | Evaporation | sequential | rootweight (root weighting) |
| QINTF | Interflow | intflwnone (none) | intflwsome (active) |

Each of the 64 structures is independently calibrated using SCE-UA with 1,000 function evaluations, optimizing KGE. SYMFLUENCE automates the full loop -- generating all combinations, updating the FUSE decisions file, running the model, extracting performance metrics, and writing a master results CSV -- via a single invocation of the `run_decision_analysis` workflow step.

### 7.1 Performance distribution

**Figure 14.** Performance overview of the 64-member FUSE decision ensemble: (a) KGE distribution; (b) all 64 structures ranked by KGE.

Across all 64 structures, calibrated KGE ranges from −1.89 to 0.86 (mean 0.39, median 0.52, IQR 0.30). Nine structures (14%) yield KGE < 0 -- catastrophic failures despite calibration -- while the top quartile clusters tightly between KGE 0.66 and 0.86. This spread of 2.74 KGE units demonstrates that structural decisions alone can dominate predictive uncertainty. For context, the inter-model ensemble of Section 3 produced a KGE range of comparable magnitude -- structural uncertainty *within* FUSE is of the same order as uncertainty *between* models.

### 7.2 Decision sensitivity

**Figure 15.** Marginal KGE sensitivity per structural decision, ordered by effect size.

Welch's t-tests and one-way ANOVA identify two statistically significant decisions at p < 0.01:

**Percolation** (QPERC: ΔKGE = 0.43, η² = 0.15, p = 0.002). Lower-zone control (`perc_lower`, mean KGE = 0.60) substantially outperforms fraction-to-saturation (`perc_f2sat`, mean KGE = 0.18).

**Interflow** (QINTF: ΔKGE = 0.42, η² = 0.15, p = 0.002). Disabling interflow (`intflwnone`, mean KGE = 0.60) outperforms enabling it (`intflwsome`, mean KGE = 0.18).

The remaining four decisions show no significant marginal effect (p > 0.05, η² < 0.025).

### 7.3 Variance decomposition and interactions

**Figure 16.** Variance decomposition: (a) percentage of total KGE variance attributed to main effects, two-way interactions, and residual; (b) interaction matrix.

A full-factorial Type-I ANOVA decomposes the total KGE variance into main effects (34.1%), two-way interactions (35.7%), and residual (30.2%). The near-parity of main effects and interactions is noteworthy: structural decisions do not act independently. The dominant interaction is Percolation × Interflow (QPERC × QINTF), which alone accounts for 19.1% of total variance -- more than any single main effect.

The nature of this interaction is asymmetric: when interflow is disabled, the percolation scheme has negligible effect (ΔKGE = +0.06); when interflow is active, switching from lower-zone control to fraction-to-saturation reduces mean KGE by 0.91. This implies that the `perc_f2sat` scheme generates unrealistic percolation fluxes that are amplified by the interflow pathway.

### 7.4 Failure-mode analysis

All nine catastrophic structures (KGE < 0) share the combination `perc_f2sat` + `intflwsome` (100% co-occurrence). No structure with either `perc_lower` or `intflwnone` fails. This deterministic failure pattern underscores the importance of evaluating structural interactions rather than decisions in isolation.

### 7.5 Implications

Structural uncertainty is comparable to inter-model uncertainty. Interactions dominate, explaining 35.7% of KGE variance versus 34.1% for all six main effects combined. One-at-a-time sensitivity analyses, which cannot detect interactions, would miss over a third of the structural signal. The 64-member ensemble was executed through a single SYMFLUENCE configuration file and workflow command.

---

## 8. Sensitivity Analysis

Using the five process-based and conceptual models from the Section 3 ensemble (FUSE, GR4J, HBV, HYPE, SUMMA), we computed sensitivity indices from the DDS calibration trajectories generated during that experiment. Rather than running a purpose-designed sampling scheme, we exploited the parameter-objective-function pairs already produced by the optimization iterations -- an approach that SYMFLUENCE automates through its `SensitivityAnalyzer` module. Two screening methods were applied: Spearman rank correlation and RBD-FAST.

To enable cross-model comparison despite differing parameter sets, each parameter was mapped to one of eight hydrological processes (Snow, Evapotranspiration, Soil Storage, Surface Runoff, Percolation, Baseflow, Groundwater Exchange, and Routing).

**Table 7.** Models included in sensitivity screening.

| Model | *n* | Processes Represented | Methods |
|---|---|---|---|
| FUSE | 13 | 6/8 | Correlation, RBD-FAST |
| GR4J | 4 | 3/8 | Correlation, RBD-FAST |
| HBV | 14 | 7/8 | Correlation, RBD-FAST |
| HYPE | 10 | 5/8 | Correlation, RBD-FAST |
| SUMMA | 11 | 4/8 | Correlation only |

### 8.1 Process-level sensitivity

**Figure 17.** Cross-model sensitivity comparison: (a) heatmap of mean normalized sensitivity per process-model combination; (b) radar chart of model-specific sensitivity profiles.

Two processes emerge as consistently sensitive across model structures. **Soil Storage** ranks among the top three most sensitive processes in all four models that represent it (FUSE: 0.88; SUMMA: 0.84; HBV: 0.73; GR4J: 0.70), reflecting the central role of soil water capacity in partitioning precipitation into fast and slow runoff pathways. **Routing** is similarly prominent, dominating in HYPE (0.89) and SUMMA (0.97).

Beyond these shared processes, sensitivity profiles diverge in ways that reflect each model's structural emphasis. FUSE exhibits uniformly high sensitivity across all six represented processes (range 0.69--0.92). HBV shows a wider range (0.36--0.78), with Surface Runoff and Soil Storage dominating. GR4J concentrates its sensitivity on Routing (X3: 1.00) and Groundwater Exchange (X2: 0.85).

Snow parameters show moderate but consistent sensitivity (range 0.54--0.74). Evapotranspiration is weakly to moderately sensitive where it appears, suggesting that ET parameters are less tightly constrained by a streamflow-only calibration objective.

### 8.2 Parameter-level sensitivity

**Figure 18.** Parameter-level sensitivity for each model, colored by hydrological process.

The most striking feature is the contrast in sensitivity structure between models. FUSE exhibits a relatively flat profile: apart from its two least sensitive parameters, all fall within the 0.72--1.00 range, implying that reducing calibration dimensionality would be difficult. GR4J shows clear separation: X3 (Routing store capacity: 1.00) and X2 (Groundwater exchange: 0.87) dominate. HBV displays a graduated pattern with a group of highly sensitive parameters (k0: 1.00, sfcf: 0.99, tt: 0.90, fc: 0.85) contrasting with lower-sensitivity parameters (cwh: 0.38, maxbas: 0.38).

### 8.3 Implications

The sensitivity screening was conducted entirely from existing calibration outputs, requiring no additional model runs. For the five-model ensemble, the full screening executed in under two minutes. The cross-model process mapping reveals that Soil Storage and Routing are consistently influential regardless of how those processes are parameterized. The parameter-level results provide practical guidance for calibration efficiency: in HBV, fixing the four least sensitive parameters would reduce calibration from 14 to 10 dimensions with minimal expected impact on KGE.

---

## 9. Large-Sample Application

A framework intended for operational and research use must scale to many catchments with minimal per-basin manual intervention. We apply a single FUSE configuration template across 111 LamaH-Ice catchments spanning Iceland's full hydrological diversity (Helgason & Nijssen, 2024).

**Figure 19.** Overview of the 111 LamaH-Ice study catchments: (a) catchment boundaries colored by drainage area; (b) distribution of catchment area (log scale); (c) distribution of streamflow record length.

### 9.1 Study domain and experimental design

Across the 111 catchments, drainage areas span three orders of magnitude (3.8--7,437 km², median 391 km²), mean elevations range from 39 to 1,307 m, and 63 catchments (57%) contain glacierized area. Streamflow records range from 4 to 89 years (median 33 years).

All 111 catchments are configured for lumped FUSE modeling with ERA5 reanalysis forcing. DDS is used for calibration with 1,000 iterations. Where streamflow records cover the 2005--2014 window, a standardized period split is applied: two-year spinup (2005--2006), five-year calibration (2007--2011), and three-year evaluation (2012--2014). The remaining 48 catchments are assigned catchment-specific periods determined automatically from the available observation window, maintaining the same proportional split.

### 9.2 Configuration and execution

The 111 FUSE configurations are generated programmatically from a single YAML template. A setup script reads each catchment's streamflow record to determine the observation window, computes appropriate time-period splits, derives bounding-box coordinates from the catchment shapefile, and writes the per-domain configuration file. The only fields that vary are the domain identifier, bounding-box coordinates, and the five time-period settings.

Execution proceeds sequentially via a runner script that skips previously completed catchments, and logs per-domain return codes and elapsed times to a persistent log file. This design allows the campaign to be interrupted and resumed without re-running completed domains.

[Results in progress]

### 9.3 Implications

This experiment tests three aspects of SYMFLUENCE's architecture that single-catchment studies do not. First, the configuration-driven design enables systematic scaling from one catchment to many without code modification. Second, the standardized directory structure ensures that results from all catchments are directly comparable and machine-readable. Third, the fault-tolerant execution model -- with per-domain logging, skip-on-completion, and resume-from-failure -- addresses the practical challenge that individual domains may fail due to data gaps, numerical instabilities, or infrastructure interruptions.

---

## 10. Large-Domain Application

The large-sample experiment treated each catchment as a lumped unit with no lateral routing. For regional water-resource assessment or flood forecasting, a distributed representation is required. This section evaluates SYMFLUENCE's capacity to configure, execute, and calibrate a fully distributed hydrological simulation over the entire island of Iceland (~103,000 km²).

**Figure 20.** Overview of the distributed Iceland domain: (a) GRU mesh colored by mean elevation with river network overlay; (b) distribution of GRU areas; (c) distribution of GRU mean elevations.

### 10.1 Domain and experimental design

The model domain encompasses the full island within 63.0--66.5°N, 25.0--13.0°W. Domain delineation follows the stream-threshold approach, with coastal watershed extraction enabled to capture the numerous short, steep rivers draining directly to the ocean. The resulting mesh is discretized into GRUs, each further subdivided into HRUs based on 200 m elevation bands, five radiation classes, and eight aspect classes.

The hydrological model is FUSE in distributed spatial mode, with lateral flows routed through the delineated river network using mizuRoute at an hourly routing time step. ERA5 provides meteorological forcing. Thirteen FUSE parameters are calibrated using DDS with 1,000 iterations. Parameter bounds are identical to those used in the lumped per-catchment experiment (Section 9), enabling direct comparison.

### 10.2 Distributed configuration

The distributed configuration differs from the lumped setup in three key respects. First, the entire island is represented as a single connected domain rather than 111 independent catchments. Second, each GRU corresponds to a sub-basin of the stream network rather than a full gauged catchment. Third, mizuRoute is invoked as an explicit routing component, reading gridded runoff and producing reach-level discharge at every time step.

The SYMFLUENCE configuration for this experiment is specified in a single YAML file that sets `FUSE_SPATIAL_MODE: distributed`, `ROUTING_DELINEATION: distributed`, and `DELINEATE_COASTAL_WATERSHEDS: true`. All other workflow steps proceed through the same automated pipeline used for single-catchment experiments, with no code modification required.

[Results in progress]

### 10.3 Implications

First, SYMFLUENCE can scale from lumped single-catchment simulations to fully distributed regional domains without changing the underlying model code -- only the configuration file and spatial inputs differ. Second, the coastal watershed delineation addresses a challenge specific to island and coastal domains. Third, the integration of mizuRoute as a configurable routing component illustrates the framework's modular coupling strategy -- routing parameters are specified in the same YAML configuration file, and the data handoff is managed internally.

---

## 11. Multivariate Evaluation

Sections 2--10 evaluated hydrological model performance exclusively against observed streamflow. While discharge integrates many watershed processes, it provides limited information about the internal consistency of simulated states. Satellite-derived observations of snow cover, soil moisture, and terrestrial water storage now offer spatially distributed constraints that streamflow alone cannot provide. This section evaluates SYMFLUENCE's capacity to ingest, compare against, and calibrate toward multiple observation sources simultaneously, through three experiments.

**Table 8.** Summary of the three multivariate evaluation experiments.

| Experiment | Domain | Area (km²) | Data Products | Variable(s) | Period |
|---|---|---|---|---|---|
| (a) GRACE TWS | Bow at Banff | 2,210 | GRACE-FO JPL RL06 | TWS anomaly | 2002--2017 |
| (b) SCA + Soil moisture | Paradise Creek | <1 | MODIS MOD10A1, SMAP L3 | SCA, soil moisture | 2015--2023 |
| (c) SCF trends | Iceland | ~103,000 | MODIS MOD10A2 | SCF (8-day) | 2000--2023 |

### 11.1 Experiment (a): Total water storage evaluation against GRACE

Terrestrial water storage anomalies from GRACE provide a vertically integrated measure of changes in snow, soil moisture, and groundwater. We compare simulated TWS anomalies from SUMMA against GRACE-FO JPL RL06 mascon solutions in the Bow River basin. Two parallel configurations are executed: an uncalibrated baseline and a multi-objective calibrated run targeting a weighted combination of TWS anomaly correlation against GRACE (weight 0.5) and streamflow KGE (weight 0.5). The parameter set includes four groundwater and storage parameters not typically included in streamflow-only calibration.

The SYMFLUENCE configuration enables GRACE comparison through the `MULTIVAR_TARGETS` block, which declaratively instructs the framework to automatically acquire the GRACE mascon product, compute monthly TWS anomalies, and combine the resulting correlation with the streamflow KGE into a single objective function.

[Results in progress]

### 11.2 Experiment (b): Joint snow cover and soil moisture evaluation

Snow cover area and soil moisture represent two key state variables constraining the surface water and energy balance. This experiment jointly evaluates SUMMA simulations against MODIS-derived SCA and SMAP-derived soil moisture at the Paradise Creek catchment. The multi-objective function combines SCA accuracy (weight 0.5) with soil moisture Pearson correlation (weight 0.5). Eleven SUMMA parameters are calibrated spanning both snow and soil hydraulic properties.

[Results in progress]

### 11.3 Experiment (c): Regional snow cover fraction trends

Snow cover is among the most climate-sensitive land surface variables at high latitudes. This experiment evaluates whether SYMFLUENCE can reproduce the spatial pattern and magnitude of SCF trends across Iceland over the 2000--2023 MODIS record. The domain encompasses the full island, discretized as 2,683 GRUs. CARRA reanalysis provides forcing at 2.5 km resolution. Trend analysis applies the Mann-Kendall test with Sen's slope estimator at the 5% significance level, stratified by season and elevation.

[Results in progress]

### 11.4 Implications

The multivariate evaluation experiments test three aspects of SYMFLUENCE's architecture. First, the satellite data acquisition and processing pipeline demonstrates that the framework can ingest heterogeneous observation sources -- gravimetric (GRACE), optical (MODIS), and microwave (SMAP) -- through a unified handler interface. Second, the weighted multi-objective calibration mechanism enables users to combine arbitrary evaluation metrics through declarative configuration. Third, the regional trend analysis extends the evaluation paradigm from point-in-time performance metrics to long-term trend reproduction -- a capability relevant to climate change impact assessments.

---

## 12. Parallel Scaling

SYMFLUENCE implements a three-tier parallel execution architecture that automatically selects among sequential, shared-memory (ProcessPool), and distributed-memory (MPI) execution strategies based on the runtime environment. Additionally, JAX-based models support JIT compilation and GPU offloading, providing model-level acceleration that composes orthogonally with process-level parallelism.

All experiments use the Bow River at Banff testbed with ERA5 forcing, daily timestep, and KGE as the calibration objective. Results are reported for two platforms: a commodity laptop (Apple Silicon, 8--16 logical cores) and an HPC cluster.

[Results in progress]

---

## 13. Data Processing Pipeline

The preceding experiments assumed that forcing data, geospatial attributes, and observational datasets were already preprocessed. In practice, transforming heterogeneous raw data into model-ready inputs constitutes a substantial and error-prone portion of the modeling workflow. This section evaluates SYMFLUENCE's data processing pipeline by tracing its end-to-end transformation chain through three domains: Paradise SNOTEL (point scale, 1 GRU), Bow at Banff (watershed scale, 49 HRUs, 2,210 km²), and Iceland (regional scale, 21,474 HRUs, 103,000 km²).

### 13.1 Pipeline architecture and scaling

The SYMFLUENCE data processing pipeline comprises 16 stages organized as a directed acyclic graph (DAG) with 25 data-flow edges. Stages fall into five categories:

- **Setup** (1 stage): project initialization and configuration validation.
- **Geospatial processing** (5 stages): pour-point creation, domain delineation, discretization, attribute acquisition, and zonal statistics.
- **Forcing preprocessing** (6 stages): ERA5 acquisition, EASYMORE weight generation, weight application, variable standardization to CFIF, elevation lapse-rate correction, and forcing file merging.
- **Observation processing** (3 stages): streamflow, snow, and ET/GRACE retrieval with quality control.
- **Model setup** (1 stage): conversion from CFIF to model-specific input format.

**Figure 21.** Pipeline architecture and data scaling for the three domain experiments: (a) DAG structure; (b) data volume by pipeline category versus HRU count.

The DAG structure enables partial parallelism: the three observation processing stages are independent of the forcing preprocessing chain. Data flow between stages is typed (shapefiles, raster products, NetCDF, CSV, configuration metadata), enforcing interface contracts and enabling output validation before downstream consumption.

The same 16-stage pipeline executes identically for all three domains, with the framework automatically adapting data acquisition extents, remapping matrix dimensions, and observation source selection based on the configuration. Total pipeline output ranges from 12.3 MB (Paradise, 110 files) through 132.9 MB (Bow, 195 files) to 4.3 GB (Iceland, 101 files). Forcing data dominates at all scales.

### 13.2 Forcing data transformation

**Figure 22.** Forcing data transformation for Bow at Banff: (a) ERA5 grid overlaid on HRU polygons; (b) raw ERA5 temperature field; (c) HRU-level temperatures after weight application; (d) basin-averaged temperature at three processing stages; (e) temperature-elevation relationship before and after lapse correction; (f) spatial pattern of lapse-rate correction.

To illustrate the transformation chain, we trace ERA5 temperature through the Bow domain. The ERA5 bounding box comprises a 6 × 7 grid of 42 cells at 0.25° resolution. The EASYMORE intersection algorithm identifies 102 grid-cell × HRU overlaps -- each of the 49 HRUs intersects on average 2.1 ERA5 cells -- and computes area-proportional weights.

Variable transformation converts ERA5-native representations to SYMFLUENCE's CF-Intermediate Format (CFIF): accumulated energy fluxes become instantaneous fluxes, accumulated precipitation becomes precipitation rate, dew-point temperature is converted to specific humidity, and wind components are combined. This standardization layer decouples dataset-specific conventions from model-specific requirements.

Elevation lapse-rate correction adjusts temperature for the difference between grid cell elevation and HRU mean elevation. The mean absolute difference between the raw spatial mean and the basin-weighted average is 2.0 K, with lapse-rate corrections reaching ±3 K for extreme elevation differences.

### 13.3 Scaling analysis

The compression ratio -- raw forcing volume divided by basin-averaged forcing volume -- exhibits a crossover depending on the ratio of target HRUs to source grid cells. At the point scale (1 HRU vs. 9 grid cells), the ratio is 0.12 (spatial aggregation). At the watershed scale (49 HRUs vs. 42 grid cells), the ratio is 0.73 (near 1:1). At the regional scale (21,474 HRUs vs. 954 grid cells), the ratio drops to 0.06: basin-averaged forcing (3.4 GB) is 18× larger than raw forcing (191 MB). This crossover represents a fundamental trade-off in spatial discretization.

### 13.4 Weight caching

The EASYMORE weight-caching architecture becomes increasingly important at larger scales. For the Bow domain, weights (1.1 MB, 102 intersections) are computed once and reused across dozens of calibration trials, model configurations, and evaluation experiments. For Iceland, weights are substantially larger (221 MB, 29,933 intersections) but the amortization benefit is proportionally greater.

### 13.5 Implications

The pipeline's declarative control demonstrates that data preprocessing complexity is absorbed by the framework rather than imposed on the user. The CFIF provides a well-defined interface between the data acquisition layer and the model execution layer that is invariant to scale. The scaling analysis provides quantitative guidance for operational deployments, allowing practitioners to estimate pipeline resource requirements before committing to a particular spatial discretization strategy.

---

## 14. Discussion

The experiments presented in Sections 2--13 are demonstrations of SYMFLUENCE's architectural capabilities rather than definitive hydrological studies. Each experiment would benefit from extended evaluation periods, additional basins, and deeper process-level analysis. Nevertheless, the results collectively illustrate how an integrated workflow architecture shapes the kinds of experiments that are practical to conduct and the patterns that emerge when technical friction is reduced.

### 14.1 From infrastructure to inference

The central argument of this series is that workflow infrastructure is a prerequisite to robust scientific inference in hydrology. The experiments presented here provide preliminary evidence for this claim.

Consider the multi-model ensemble of Section 3. Six structurally diverse models were configured, calibrated, and evaluated against a common set of observations using configuration files that differed only in model-specific parameters. The ensemble mean (KGE = 0.94) exceeded every individual model and every benchmark predictor, consistent with the broader multi-model averaging literature. More instructive than the absolute performance is what the experiment *required*: in a conventional workflow, assembling such an ensemble demands separate preprocessing scripts for each model, ad hoc output reformatting, and manual alignment of evaluation periods. SYMFLUENCE reduces this to configuration changes, making the experiment routine rather than heroic.

Similarly, the forcing ensemble (Section 4) and calibration algorithm comparison (Section 5) each held all factors constant except one, isolating forcing and algorithmic uncertainty respectively. This degree of experimental control is straightforward to describe but difficult to achieve in practice when each factor change requires modifications to bespoke scripts.

These observations do not demonstrate that integrated infrastructure will resolve the predictive stagnation identified in the companion papers. They do suggest that when technical barriers are lowered, experiments that would otherwise be prohibitively labor-intensive become feasible, expanding the space of questions researchers can ask. Whether the resulting experiments yield new scientific understanding depends on the questions posed, not on the infrastructure alone.

### 14.2 Uncertainty partitioning across experimental axes

Sections 3 through 8 each varied a different axis of the modeling workflow: model structure, forcing dataset, calibration algorithm, intra-model structural decisions, and parameter sensitivity. While these experiments were conducted on a limited number of basins and with relatively short evaluation periods, they illustrate how an integrated framework enables comparative analysis across uncertainty sources. A key finding is that intra-model structural uncertainty (Section 7, KGE spread of 2.74 across 64 FUSE structures) rivals inter-model uncertainty (Section 3) -- a result consistent with Clark et al. (2008) but here demonstrated within a unified orchestration framework.

### 14.3 Parameter identifiability and compensation

A recurring theme is the tension between calibration performance and parameter interpretability. The forcing ensemble showed that models can achieve high calibration KGE through parameter values that deviate substantially from their expected physical ranges, particularly when compensating for forcing biases. The algorithm comparison revealed bimodal parameter regimes where structurally different parameter sets produced similar objective function values but differed in their generalization.

These findings reinforce the importance of split-sample evaluation -- which SYMFLUENCE automates -- and of multi-algorithm calibration as a diagnostic tool. A single calibration run with a single algorithm, the standard practice in most studies, would not have revealed the bimodal snow-parameter structure or the forcing-dependent compensation patterns documented here.

### 14.4 The role of architecture in enabling community science

The registry pattern, which allows new models, data sources, and analysis methods to be added without modifying core framework code, addresses a persistent tension in scientific software: the need for stability in production workflows versus extensibility as methods evolve. The experiments were conducted using models spanning Fortran (SUMMA, FUSE), C (VIC), Python (HBV, LSTM), and R (GR4J). The four-component interface (preprocessor, runner, postprocessor, extractor) absorbs this heterogeneity, presenting a uniform surface to the orchestration layer.

The declarative configuration system addresses the reproducibility concerns raised in the companion papers. Each experiment is fully specified by a YAML file that can be version-controlled, shared, and re-executed. These are engineering contributions rather than scientific ones, but the argument of this series is that such contributions are a necessary condition for the kind of systematic, reproducible inquiry that hydrological science requires.

### 14.5 Toward concrescence

The first paper framed the integration challenge through Whitehead's concept of concrescence: the growing together of distinct elements into a unified entity. The experiments here offer an initial test of whether this architectural philosophy translates into practical benefit.

The evidence, while preliminary, suggests that integration yields returns that are not merely additive. The multi-model ensemble's performance exceeding all individual members, the detection of structural interactions invisible to marginal analyses, and the exposure of forcing-dependent parameter compensation across a controlled ensemble are all results that depend on the ability to conduct systematic, multi-factor experiments with minimal manual intervention. These experiments are not impossible without an integrated framework, but the practical barriers to conducting them manually are high enough that they are rarely attempted.

The framework's connection to AI-assisted workflows through INDRA (Eythorsson & Clark, 2025) points toward a further dimension of integration: not merely connecting models, data, and computation, but augmenting the researcher's ability to navigate the resulting experimental space. As the number of models, forcing datasets, discretization strategies, and calibration configurations grows, the combinatorial space of possible experiments expands faster than any individual can explore manually.

### 14.6 Limitations

Several limitations qualify the conclusions drawn from this work.

First, the applications are concentrated on a small number of domains, primarily the Bow River at Banff and the Paradise SNOTEL site. The generalizability of the observed patterns remains to be established across a broader range of hydroclimatic settings.

Second, the evaluation periods are relatively short (2--4 years in most experiments), limiting the ability to assess model robustness under climatic conditions substantially different from the calibration period.

Third, most calibration experiments used a single objective function (KGE against streamflow). Multi-objective and multivariate calibration, which SYMFLUENCE supports, would provide a more complete picture of parameter identifiability and model structural adequacy.

Fourth, while SYMFLUENCE integrates 25 model implementations, the depth of integration varies. Some models (SUMMA, FUSE) have been exercised extensively; others have undergone less thorough validation within the framework.

Fifth, the computational scaling characteristics have not been systematically benchmarked in the experiments presented so far.

Finally, the framework's complexity is itself a limitation. An end-to-end system inevitably introduces abstraction layers that may obscure model behavior from the user. The framework's value proposition assumes that reproducibility and efficiency gains outweigh the cost of operating within a managed environment -- an assumption that may not hold for all use cases.

---

## 15. Conclusion

This paper has subjected the architectural principles described in the companion papers to empirical test through twelve experiments spanning point-scale flux estimation to regional hydrological simulation. The experiments are not intended as definitive hydrological studies; their purpose is to demonstrate that when technical friction is reduced through deliberate architectural design, experiments that would otherwise be prohibitively labor-intensive become routine.

The multi-model ensemble (Section 3) showed that six structurally diverse models, spanning conceptual to data-driven paradigms, can be configured, calibrated, and evaluated through configuration changes alone, with the ensemble mean (KGE = 0.94) exceeding every individual model. The forcing ensemble (Section 4) revealed that calibration performance is not a reliable indicator of forcing quality, and exposed forcing-dependent parameter compensation that undermines transferability. The algorithm comparison (Section 5) identified bimodal parameter regimes and demonstrated that gradient-based methods achieve superior generalization. The structural decision ensemble (Section 7) showed that intra-model structural uncertainty rivals inter-model uncertainty, and that two-way interactions dominate main effects. The sensitivity analysis (Section 8) identified soil storage and routing as consistently influential processes across model structures. The large-sample application (Section 9) demonstrated batch configuration scaling to 111 catchments, and the large-domain application (Section 10) demonstrated fully distributed regional simulation. The data pipeline analysis (Section 13) characterized end-to-end processing across three orders of magnitude in domain size, formalizing the CFIF as a scale-invariant interface between data acquisition and model execution.

Whether SYMFLUENCE ultimately contributes to resolving the predictive stagnation identified in the companion papers depends on how it is used. The framework removes technical barriers; it does not generate scientific hypotheses or interpret results. Its contribution is to shift the bottleneck from implementation to inquiry, creating the conditions under which the community can conduct the systematic, reproducible experiments needed to determine whether detailed process representation improves prediction, or whether simpler approaches suffice. The answer to that question lies not in the architecture but in the experiments it enables.

---

## References

Ajami, N. K., Duan, Q., & Sorooshian, S. (2006). An integrated hydrologic Bayesian multimodel combination framework. *Water Resources Research*, 42(9), W09408. https://doi.org/10.1029/2005WR004745

Arsenault, R., Poulin, A., Côté, P., & Brissette, F. (2014). Comparison of stochastic optimization algorithms in hydrological model calibration. *Journal of Hydrologic Engineering*, 19(7), 1374--1384. https://doi.org/10.1061/(ASCE)HE.1943-5584.0000938

Arsenault, R., Gatien, P., Renaud, B., Brissette, F., & Martel, J.-L. (2015). A comparative analysis of 9 multi-model averaging approaches in hydrological continuous streamflow simulation. *Journal of Hydrology*, 529, 754--767. https://doi.org/10.1016/j.jhydrol.2015.09.001

Clark, M. P., Slater, A. G., Rupp, D. E., Woods, R. A., Vrugt, J. A., Gupta, H. V., Wagener, T., & Hay, L. E. (2008). Framework for Understanding Structural Errors (FUSE): A modular framework to diagnose differences between hydrological models. *Water Resources Research*, 44(12), W00B02. https://doi.org/10.1029/2007WR006735

Clark, M. P., Kavetski, D., & Fenicia, F. (2011). Pursuing the method of multiple working hypotheses for hydrological modeling. *Water Resources Research*, 47(9), W09301. https://doi.org/10.1029/2010WR009827

Clark, M. P., et al. (2015a). A unified approach for process-based hydrological modeling: 1. Modeling concept. *Water Resources Research*, 51(4), 2498--2514. https://doi.org/10.1002/2015WR017198

Clark, M. P., et al. (2015b). A unified approach for process-based hydrological modeling: 2. Model implementation and case studies. *Water Resources Research*, 51(4), 2515--2542. https://doi.org/10.1002/2015WR017200

Craig, J. R., et al. (2020). Flexible watershed simulation with the Raven hydrological modelling framework. *Environmental Modelling & Software*, 129, 104728. https://doi.org/10.1016/j.envsoft.2020.104728

Duan, Q., Sorooshian, S., & Gupta, V. K. (1992). Effective and efficient global optimization for conceptual rainfall-runoff models. *Water Resources Research*, 28(4), 1015--1031. https://doi.org/10.1029/91WR02985

Eythorsson, D., & Clark, M. P. (2025). Toward Automated Scientific Discovery in Hydrology: The Opportunities and Dangers of AI Augmented Research Frameworks. *Hydrological Processes*, 39(1), e70065. https://doi.org/10.1002/hyp.70065

Eythorsson, D., et al. (2025a). Outstanding in every field: The case for shared architectural vision in computational hydrology. *Geoscientific Model Development* [companion paper].

Eythorsson, D., et al. (2025b). The registry and the river: Architectural patterns for community hydrological modeling. *Geoscientific Model Development* [companion paper].

Gharari, S., Clark, M. P., Mizukami, N., Knoben, W. J. M., Wong, J. S., & Pietroniro, A. (2020). Flexible vector-based spatial configurations in land models. *Hydrology and Earth System Sciences*, 24(11), 5953--5971. https://doi.org/10.5194/hess-24-5953-2020

Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria. *Journal of Hydrology*, 377(1--2), 80--91. https://doi.org/10.1016/j.jhydrol.2009.08.003

Helgason, D., & Nijssen, B. (2024). LamaH-Ice: Large-Sample Hydrological Dataset for Iceland. [Dataset].

Hutton, C., et al. (2016). Most computational hydrology is not reproducible, so is it really science? *Water Resources Research*, 52(10), 7548--7555. https://doi.org/10.1002/2016WR019285

Keshavarz, K., Knoben, W. J. M., & Clark, M. P. (2024). Community workflows for advanced reproducibility in hydrologic modeling. *Hydrological Processes*, 38(1), e15044. https://doi.org/10.1002/hyp.15044

Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios. *Journal of Hydrology*, 424--425, 264--277. https://doi.org/10.1016/j.jhydrol.2012.01.011

Knoben, W. J. M. (2024). HydroBM: Hydrological Benchmarking. [Software].

Knoben, W. J. M., et al. (2019). Modular Assessment of Rainfall-Runoff Models Toolbox (MARRMoT) v1.2. *Geoscientific Model Development*, 12(6), 2463--2480. https://doi.org/10.5194/gmd-12-2463-2019

Knoben, W. J. M., et al. (2022). Community Workflows to Advance Reproducibility in Hydrologic Modeling (CWARHM). *Water Resources Research*, 58(11), e2022WR032702. https://doi.org/10.1029/2022WR032702

Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I -- A discussion of principles. *Journal of Hydrology*, 10(3), 282--290. https://doi.org/10.1016/0022-1694(70)90255-6

Nearing, G. S., et al. (2021). What role does hydrological science play in the age of machine learning? *Water Resources Research*, 57(3), e2020WR028091. https://doi.org/10.1029/2020WR028091

Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. *Journal of Hydrology*, 279(1--4), 275--289. https://doi.org/10.1016/S0022-1694(03)00225-7

Schaefli, B., & Gupta, H. V. (2007). Do Nash values have value? *Hydrological Processes*, 21(15), 2075--2080. https://doi.org/10.1002/hyp.6825

Tolson, B. A., & Shoemaker, C. A. (2007). Dynamically dimensioned search algorithm for computationally efficient watershed model calibration. *Water Resources Research*, 43(1), W01413. https://doi.org/10.1029/2005WR004723

Whitehead, A. N. (1929). *Process and Reality: An Essay in Cosmology*. Macmillan, New York.

---

## Notes for co-authors

**Scope:** This paper is the applications and validation companion to "Outstanding in Every Field" (Episode 1) and "The Registry and the River" (Episode 2). Where Episode 1 argues *why* and Episode 2 describes *how*, this paper demonstrates *what* -- the concrete experiments that the architecture enables. The tone is empirical rather than architectural: experimental design, results, interpretation, and limitations.

**Relationship to V4.0 paper:** This manuscript ports and restructures Section 4 (Applications and Validation) of the original V4.0 SYMFLUENCE paper, along with relevant material from Section 5 (Discussion). The Section 4 content has been reorganized into standalone experiment sections with consistent structure (setup, results, implications). The discussion material from V4.0 Section 5 has been adapted for Section 14.

**Results in progress:** Several experiments (Sections 9, 10, 11, 12) have experimental designs documented but results pending. These are marked with "[Results in progress]" placeholders.

**Figures needed:**
1. Spatial discretization options for Bow at Banff -- **Figure 1** (from V4.0 Figure 11)
2. Iceland domain definition -- **Figure 2** (from V4.0 Figure 12)
3. Multi-model hydrographs -- **Figure 3** (from V4.0 Figure 13)
4. KGE decomposition -- **Figure 4** (from V4.0 Figure 14)
5. Ensemble envelope and FDC -- **Figure 5** (from V4.0 Figure 15)
6. SWE time series -- **Figure 6** (from V4.0 Figure 16)
7. Forcing performance heatmap -- **Figure 7** (from V4.0 Figure 17)
8. Parameter divergence -- **Figure 8** (from V4.0 Figure 18)
9. Algorithm performance -- **Figure 9** (from V4.0 Figure 19)
10. Convergence trajectories -- **Figure 10** (from V4.0 Figure 20)
11. Parameter equifinality heatmap -- **Figure 11** (from V4.0 Figure 21)
12. Benchmark results -- **Figure 12** (from V4.0 Figure 22)
13. Benchmark time series -- **Figure 13** (from V4.0 Figure 23)
14. FUSE decision ensemble performance -- **Figure 14** (from V4.0 Figure 24)
15. Decision sensitivity -- **Figure 15** (from V4.0 Figure 25)
16. Variance decomposition -- **Figure 16** (from V4.0 Figure 26)
17. Process sensitivity -- **Figure 17** (from V4.0 Figure 27)
18. Parameter sensitivity -- **Figure 18** (from V4.0 Figure 28)
19. Large-sample catchments -- **Figure 19** (from V4.0 Figure 29)
20. Distributed Iceland domain -- **Figure 20** (from V4.0 Figure 30)
21. Pipeline architecture and scaling -- **Figure 21** (from V4.0 Figure 31)
22. Forcing data transformation -- **Figure 22** (from V4.0 Figure 32)

**Length:** Currently ~12,000 words of body text. For a full GMD paper, this is within range. Results-in-progress sections will add ~3,000--5,000 words when completed.

**Episode 1 and 2 dependencies:** This paper references Episode 1 for the philosophical argument (impossible generalist, infrastructure deficit) and Episode 2 for the architectural description (four-tier design, registry pattern, CFIF, scale-invariance). It should not repeat architectural detail -- only reference it and demonstrate its empirical consequences.
