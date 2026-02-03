## 4.6 Model decision ensemble

### Motivation

Sections 4.2--4.4 quantified inter-model, inter-forcing, and inter-calibration uncertainty by treating each model as a monolithic unit.
Yet within a single model framework the choice of process representation --- the *structural decisions* --- can be an equally important source of predictive uncertainty.
The Framework for Understanding Structural Errors (FUSE; Clark et al., 2008) exposes these decisions explicitly: each of its nine structural dimensions admits two or more options, yielding 1,728 unique model structures from a common code base.
This section uses SYMFLUENCE's `FuseStructureAnalyzer` to systematically evaluate a curated subset of FUSE structural decisions and to partition the resulting performance variance into main effects, two-way interactions, and residual uncertainty.

### Experimental design

We retain the same domain (Bow at Banff, lumped, ERA5 forcing), observation data, calibration period (2001--2010), and evaluation period (2011--2015) as Section 4.2 to enable direct comparison of structural uncertainty against inter-model uncertainty.

Rather than exhaustively evaluating all 1,728 FUSE structures, we select the six most hydrologically meaningful decision dimensions, each with two contrasting options, for a full-factorial design of 2^6 = 64 combinations (Table 1).
Three decisions are held fixed at standard choices: multiplicative rainfall error correction (`RFERR = multiplc_e`), gamma-function routing (`Q_TDH = rout_gamma`), and temperature-index snow model (`SNOWM = temp_index`).

**Table 1.** Varied FUSE structural decisions and options.

| Decision | Description | Option A | Option B |
|----------|-------------|----------|----------|
| ARCH1 | Upper-layer soil architecture | `tension1_1` (tension, 2-state) | `onestate_1` (single bucket) |
| ARCH2 | Lower-layer soil architecture | `tens2pll_2` (tension parallel) | `unlimfrc_2` (unlimited fraction) |
| QSURF | Surface runoff generation | `arno_x_vic` (VIC-style) | `prms_varnt` (PRMS-style) |
| QPERC | Percolation | `perc_f2sat` (fraction to saturation) | `perc_lower` (lower-zone control) |
| ESOIL | Evaporation | `sequential` | `rootweight` (root weighting) |
| QINTF | Interflow | `intflwnone` (none) | `intflwsome` (active) |

Each of the 64 structures is independently calibrated using SCE-UA with 1,000 function evaluations, optimising KGE.
SYMFLUENCE automates the full loop --- generating all combinations, updating the FUSE decisions file, running the model, extracting performance metrics, and writing a master results CSV --- via a single invocation of the `run_decision_analysis` workflow step.

### Results

#### Overall performance spread

Across all 64 structures, calibrated KGE ranges from -1.89 to 0.86 (mean 0.39, median 0.52, IQR 0.30; Figure 1a).
Nine structures (14%) yield KGE < 0 --- catastrophic failures despite calibration --- while the top quartile clusters tightly between KGE 0.66 and 0.86 (Figure 1b).
This spread of 2.74 KGE units across structures sharing the same forcing, domain, and calibration protocol demonstrates that structural decisions alone can dominate predictive uncertainty.

For context, the inter-model ensemble of Section 4.2, which compared entirely different hydrological models (GR4J, HYPE, MESH, SUMMA, FUSE), produced a KGE range of comparable magnitude.
Structural uncertainty *within* FUSE is therefore of the same order as uncertainty *between* models --- a result consistent with Clark et al. (2008) but here demonstrated within the SYMFLUENCE orchestration framework.

#### Decision sensitivity

Welch's t-tests and one-way ANOVA identify two statistically significant decisions at the p < 0.01 level (Figure 2):

1. **Percolation** (QPERC): |delta KGE| = 0.43, eta^2 = 0.15, p = 0.002.
   Lower-zone control (`perc_lower`, mean KGE = 0.60) substantially outperforms fraction-to-saturation (`perc_f2sat`, mean KGE = 0.18).

2. **Interflow** (QINTF): |delta KGE| = 0.42, eta^2 = 0.15, p = 0.002.
   Disabling interflow (`intflwnone`, mean KGE = 0.60) outperforms enabling it (`intflwsome`, mean KGE = 0.18).

The remaining four decisions --- upper-layer architecture (ARCH1), lower-layer architecture (ARCH2), surface runoff (QSURF), and evaporation (ESOIL) --- show no significant marginal effect (p > 0.05, eta^2 < 0.025).
Surface runoff generation (QSURF) is essentially inert, with |delta KGE| = 0.01.

#### Variance decomposition and interactions

A full-factorial Type-I ANOVA decomposes the total KGE variance into main effects (34.1%), two-way interactions (35.7%), and residual (30.2%) (Figure 3a).
The near-parity of main effects and interactions is noteworthy: structural decisions do not act independently.

The dominant interaction is **Percolation x Interflow** (QPERC x QINTF), which alone accounts for 19.1% of total variance (F = 26.6, p < 0.001) --- more than any single main effect (Figure 3b).
The nature of this interaction is asymmetric: when interflow is disabled, the percolation scheme has negligible effect (delta KGE = +0.06); when interflow is active, switching from lower-zone control to fraction-to-saturation reduces mean KGE by 0.91.
This implies that the `perc_f2sat` scheme generates unrealistic percolation fluxes that are amplified by the interflow pathway, creating a structural feedback loop absent from the `perc_lower` scheme.

The second-largest interaction is **Lower-layer architecture x Percolation** (ARCH2 x QPERC, 6.6%, p = 0.004), indicating that the lower soil formulation modulates how percolation affects performance.

#### Failure-mode analysis

All nine catastrophic structures (KGE < 0) share the combination `perc_f2sat` + `intflwsome` (100% co-occurrence, 2.4x enrichment over base rate).
No structure with either `perc_lower` or `intflwnone` fails.
This deterministic failure pattern --- a specific two-decision combination producing model breakdown regardless of the other four decisions --- underscores the importance of evaluating structural interactions rather than decisions in isolation.

The best-performing structure (KGE = 0.86) uses tension-based upper-layer architecture, tension-parallel lower-layer architecture, VIC-style surface runoff, fraction-to-saturation percolation, root weighting evaporation, and no interflow.
Notably, it includes `perc_f2sat`, which is the dominant option in all failures, but avoids the toxic combination with active interflow.

### Discussion

Three findings from this experiment are relevant to the broader SYMFLUENCE framework:

1. **Structural uncertainty is comparable to inter-model uncertainty.**
   The KGE spread across 64 FUSE structures (2.74 units) is of the same order as the spread across the multi-model ensemble of Section 4.2.
   This suggests that model intercomparison studies that treat each model as a fixed entity may underestimate total predictive uncertainty by neglecting intra-model structural variability.

2. **Interactions dominate.**
   Two-way interactions explain 35.7% of KGE variance, exceeding the contribution of all six main effects combined (34.1%).
   The QPERC x QINTF interaction alone (19.1%) is larger than any individual decision.
   One-at-a-time sensitivity analyses, which cannot detect interactions, would miss over a third of the structural signal.

3. **Automation enables systematic exploration.**
   The 64-member ensemble --- each requiring independent calibration --- was executed through a single SYMFLUENCE configuration file and workflow command.
   The `FuseStructureAnalyzer` handles combination generation, decision-file updates, model execution, and metric collection without manual intervention, making full-factorial structural analyses practical for routine application.

### Figures

- **Figure 1.** Performance overview of the 64-member FUSE decision ensemble. (a) KGE distribution with individual structures (points), median (dashed), and interquartile range (shaded). (b) All 64 structures ranked by KGE, coloured from red (worst) to green (best).

- **Figure 2.** Marginal KGE sensitivity per structural decision, ordered by effect size. Paired boxplots show the KGE distribution for each option. Significance stars indicate Welch t-test results (\*\*p < 0.01); |delta| and eta^2 annotated. Only Percolation and Interflow are statistically significant.

- **Figure 3.** Variance decomposition and interaction structure. (a) Percentage of total KGE variance attributed to each main effect, top two-way interactions, and residual (ANOVA Type-I SS). (b) Interaction matrix showing |delta KGE| for main effects (diagonal) and pairwise interaction magnitudes (off-diagonal). The QPERC x QINTF interaction dominates.
