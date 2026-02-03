## 4.2 Multi-Model Ensemble Streamflow Simulation

To demonstrate SYMFLUENCE's capacity for orchestrating structurally diverse models within a unified workflow, we configured an ensemble of ten hydrological models for the Bow River at Banff lumped catchment (drainage area 2,210 km²) using ERA5 meteorological forcing. Each model was set up from a near-identical YAML configuration file specifying the domain definition, forcing data paths, observation targets, and calibration settings. Model-specific parameters were the only elements that varied between configurations. All ten models were calibrated against observed daily streamflow using the Dynamically Dimensioned Search (DDS; Tolson and Shoemaker, 2007) algorithm over the period 2003–2005, with independent evaluation over 2006–2009.

Seven of the ten models achieved a calibration Kling–Gupta Efficiency (KGE; Gupta et al., 2009) exceeding 0.5 and are retained for the ensemble analysis: SUMMA, FUSE, GR4J, HBV, HYPE, VIC, and LSTM. The remaining three (RHESSys, MESH, and ngen) are excluded as their implementations within SYMFLUENCE are still under active development. Table 1 summarises the included models and their calibration performance.

**Table 1.** Summary of the seven ensemble members. *n* denotes the number of calibrated parameters.

| Model | Type | *n* | Calib. KGE | Eval. KGE | Reference |
|-------|------|-----|-----------|----------|-----------|
| SUMMA | Process-based, multi-physics | 11 | 0.90 | 0.88 | Clark et al. (2015) |
| FUSE  | Process-based, modular | 13 | 0.90 | 0.88 | Clark et al. (2008) |
| GR4J  | Lumped conceptual | 4  | 0.92 | 0.79 | Perrin et al. (2003) |
| HBV   | Semi-distributed conceptual | 15 | 0.74 | 0.70 | Bergström (1995) |
| HYPE  | Semi-distributed, process-based | 10 | 0.87 | 0.81 | Lindström et al. (2010) |
| VIC   | Grid-based, land surface | 13 | 0.81 | 0.71 | Liang et al. (1994) |
| LSTM  | Data-driven (neural network) | — | 0.97 | 0.88 | Hochreiter and Schmidhuber (1997) |

The ensemble spans four modelling paradigms — process-based (SUMMA, FUSE, HYPE), land surface (VIC), conceptual (GR4J, HBV), and data-driven (LSTM) — providing a broad sampling of structural uncertainty. SUMMA and FUSE share the mizuRoute river routing scheme but differ in their representations of vertical water and energy fluxes. GR4J achieves strong performance with only four free parameters, while HBV, despite having 15 parameters, yields the lowest calibration KGE (0.74), largely due to a persistent negative bias in mean flow (β = 0.75). VIC, a grid-based land surface model with 13 calibrated parameters, achieves a calibration KGE of 0.81 with near-unbiased mean flow (β = 1.03) but lower correlation (*r* = 0.81) than most other members, suggesting limitations in capturing event-scale dynamics despite adequate representation of the water balance. The LSTM, trained on the same forcing–streamflow pairs used for calibration, achieves the highest calibration KGE of all models (0.97).

### 4.2.1 Multi-Model Hydrograph

Fig. X presents the simulated and observed hydrographs over the full analysis period. The upper panel (Fig. Xa) shows all seven models alongside the observed record from 2004 to 2009, with the calibration–evaluation boundary marked at January 2006. The lower panel (Fig. Xb) zooms into the April–October 2005 snowmelt season to highlight differences in peak timing and magnitude.

All seven models reproduce the dominant seasonal cycle of the Bow River, which is characterised by low winter baseflow (5–15 m³ s⁻¹) and a pronounced snowmelt-driven freshet peaking in June–July (100–300 m³ s⁻¹). During the calibration period, model traces are tightly clustered around the observed hydrograph. During evaluation, the ensemble spread widens — most noticeably around peak events — reflecting how each model's structural assumptions about snowmelt dynamics, soil moisture partitioning, and baseflow recession lead to differing responses under conditions not seen during calibration.

The zoomed panel reveals model-specific behaviour during the 2005 freshet. HBV (red) consistently underestimates both peak and recession flows, consistent with its low β. GR4J (green) captures the peak magnitude but exhibits sharper, less-damped recession limbs. HYPE (purple) shows a delayed and prolonged recession through August–October that diverges from the observed signal. The LSTM tracks the observed hydrograph closely through the rising limb and peak but, like most models, underestimates secondary peaks in the late season. SUMMA and FUSE produce nearly indistinguishable traces throughout, confirming that SYMFLUENCE's workflow yields reproducible results when models share a common routing framework.

### 4.2.2 KGE Decomposition

To diagnose the sources of performance differences, Fig. Y decomposes the KGE into its three components — Pearson correlation (*r*), variability ratio (α = σ_sim / σ_obs), and bias ratio (β = μ_sim / μ_obs) — for both the calibration and evaluation periods.

All seven models maintain high correlation (*r* > 0.87) across both periods, indicating that the timing of hydrological events is consistently well captured regardless of model structure. The primary drivers of KGE variation are instead α and β.

During calibration, most models cluster near the ideal values (α ≈ 1, β ≈ 1), with HBV as the notable exception: its β of 0.75 indicates a 25% underestimation of mean flow. This bias persists into the evaluation period (β = 0.81), pointing to a systematic structural limitation rather than parameter identifiability issues.

During evaluation, all models show α > 1.0, meaning they overestimate flow variability relative to observations. This is most pronounced for HBV (α = 1.22) and HYPE (α = 1.16), suggesting that both models amplify the high-flow / low-flow contrast when forced outside calibration conditions. GR4J shows the largest shift in β between periods (from 1.01 to 1.12), transitioning from near-unbiased to a 12% positive bias — likely a consequence of its parsimonious structure limiting its ability to generalise across climate conditions.

The LSTM maintains the most balanced decomposition in evaluation (*r* = 0.93, α = 1.08, β = 1.05), though this does not imply physical realism — its components cannot be attributed to specific hydrological processes. SUMMA and FUSE again yield nearly identical decompositions (evaluation *r* = 0.91, α = 1.08, β = 0.99 for both), further confirming workflow consistency.

### 4.2.3 Ensemble Envelope and Flow Duration Curve

Fig. Z evaluates the seven-model ensemble as a whole. The left panel (Fig. Za) shows the ensemble envelope (min–max range across models) for the evaluation period (2006–2009), together with the ensemble mean and median. The right panel (Fig. Zb) compares flow duration curves (FDCs) for the observed record, ensemble statistics, and individual models.

The observed hydrograph falls within the ensemble envelope for the large majority of the evaluation period. The envelope captures peak flows well, though the 2008 annual maximum (~280 m³ s⁻¹) briefly exceeds the upper bound, indicating that all models underestimate the most extreme event in the record. The ensemble mean (KGE = 0.94) and median (KGE = 0.92) both outperform every individual model. This improvement arises from the cancellation of offsetting structural biases: HBV's negative bias and GR4J's positive bias partially neutralise each other in the average, while the correlation component is preserved across all members.

The FDC comparison (Fig. Zb) provides a complementary view of model behaviour across the discharge spectrum. At low exceedance probabilities (<20%, corresponding to high flows), all models and the ensemble track the observed FDC closely. At intermediate exceedance probabilities (20–80%), the ensemble mean aligns almost exactly with the observed curve, while individual models begin to diverge. The greatest inter-model spread occurs at high exceedance probabilities (>80%, low flows): HBV drops to near-zero discharge at the lowest flows, while HYPE maintains higher baseflows than observed. The ensemble mean smooths these extremes, tracking the observed low-flow regime more faithfully than any single model.

### 4.2.4 Framework Implications

Three aspects of these results are relevant to SYMFLUENCE's design objectives.

First, the near-identical performance of SUMMA and FUSE — which share a routing scheme but differ in their process representations — demonstrates that SYMFLUENCE's configuration layer produces consistent, reproducible results across models. The standardised evaluation pipeline writes JSON metric files with identical key structures for all models, enabling the programmatic cross-model analysis shown here without manual post-processing.

Second, the diversity of the ensemble — spanning four-parameter conceptual models to grid-based land surface schemes and deep-learning architectures — illustrates that SYMFLUENCE's model-agnostic design does not restrict the user to a narrow class of models. The same DDS calibration workflow, forced with the same ERA5 data and evaluated against the same streamflow observations, was applied across all seven models with changes confined to a single YAML configuration file per model.

Third, the ensemble results underscore the practical value of multi-model approaches. The ensemble mean KGE exceeds the best individual model by a meaningful margin, consistent with the broader multi-model averaging literature (Ajami et al., 2006; Arsenault et al., 2015). The widening of the ensemble spread during evaluation relative to calibration is consistent with structural uncertainty estimates reported by Clark et al. (2011), and highlights the importance of considering model diversity in operational forecasting contexts. SYMFLUENCE's modular architecture makes it straightforward to extend this ensemble as additional model implementations mature (e.g., MESH, ngen) or as new models are integrated into the framework.
