# 4.4 Calibration Ensemble Study

Systematic comparison of 12 optimization algorithms for hydrological model calibration using the HBV model on the Bow River at Banff basin with ERA5 forcing.

## Study Overview

This experiment evaluates the SYMFLUENCE calibration suite by comparing optimization algorithms across multiple paradigms, using a normalized function evaluation budget for fair comparison. The HBV model provides a computationally efficient, JAX-differentiable testbed that enables both gradient-based and derivative-free methods.

### Optimization Algorithms

| Algorithm | Family | Method Type | Key Reference |
|-----------|--------|-------------|---------------|
| **DDS** | Sampling | Single-solution perturbation | Tolson & Shoemaker (2007) |
| **SCE-UA** | Evolutionary | Shuffled complex evolution | Duan et al. (1993) |
| **DE** | Evolutionary | Differential evolution | Storn & Price (1997) |
| **PSO** | Evolutionary | Particle swarm | Kennedy & Eberhart (1995) |
| **GA** | Evolutionary | Genetic algorithm | Holland (1975) |
| **CMA-ES** | Evolutionary | Covariance matrix adaptation | Hansen & Ostermeier (2001) |
| **ADAM** | Gradient | Adaptive moment estimation | Kingma & Ba (2015) |
| **L-BFGS** | Gradient | Quasi-Newton | Liu & Nocedal (1989) |
| **Nelder-Mead** | Direct Search | Simplex | Nelder & Mead (1965) |
| **SA** | Stochastic | Simulated annealing | Kirkpatrick et al. (1983) |
| **Basin Hopping** | Stochastic | Multi-start local opt. | Wales & Doye (1997) |
| **Bayesian Opt.** | Surrogate | Gaussian process surrogate | Snoek et al. (2012) |

### Experimental Design

**Fair comparison via normalized function evaluation budget (~4,000 per algorithm):**

| Algorithm | Iterations | Evals/Iter | Total Evals | Notes |
|-----------|-----------|------------|-------------|-------|
| DDS | 4,000 | 1 | ~4,000 | Single-solution perturbation |
| SCE-UA | 28 | ~145 | ~4,060 | Complex evolution (3 complexes) |
| DE | 200 | 20 | 4,000 | Population-based |
| PSO | 200 | 20 | 4,000 | Population-based |
| GA | 200 | 20 | 4,000 | Population-based |
| CMA-ES | 200 | 20 | 4,000 | Population-based |
| ADAM | 2,000 | ~2 | ~4,000 | Native JAX gradients |
| L-BFGS | 500 | ~8 | ~4,000 | Native gradients + line search |
| Nelder-Mead | 4,000 | ~1-2 | ~4,000-8,000 | Simplex reflections |
| SA | 400 | 10 | 4,000 | 10 steps per temperature |
| Basin Hopping | 80 | ~50 | ~4,000 | 50 local optimization steps/hop |
| Bayesian Opt. | 200 | 1+GP | ~200+ | GP surrogate fitting per iter |

**Experimental objectives:**
1. **Performance**: Which algorithms achieve the best calibration and evaluation metrics?
2. **Convergence**: How efficiently do algorithms approach the optimum?
3. **Equifinality**: Do different algorithms find different parameter sets with similar performance?
4. **Generalization**: Which algorithms produce parameters that transfer well to the evaluation period?
5. **Robustness**: How sensitive are results to random seed initialization?

## Study Structure

```
4. Calibration ensemble/
├── configs/                              # One config per algorithm
│   ├── config_bow_hbv_dds.yaml           # DDS
│   ├── config_bow_hbv_sceua.yaml         # SCE-UA
│   ├── config_bow_hbv_de.yaml            # DE
│   ├── config_bow_hbv_pso.yaml           # PSO
│   ├── config_bow_hbv_ga.yaml            # GA
│   ├── config_bow_hbv_cmaes.yaml         # CMA-ES
│   ├── config_bow_hbv_adam.yaml           # ADAM (smoothed HBV)
│   ├── config_bow_hbv_lbfgs.yaml         # L-BFGS (smoothed HBV)
│   ├── config_bow_hbv_nelder_mead.yaml   # Nelder-Mead
│   ├── config_bow_hbv_sa.yaml            # Simulated Annealing
│   ├── config_bow_hbv_basin_hopping.yaml # Basin Hopping
│   ├── config_bow_hbv_bayesian_opt.yaml  # Bayesian Optimization
│   └── config_bow_hbv_*_seedN.yaml       # Multi-seed variants
├── scripts/
│   ├── generate_configs.py               # Config file generator
│   ├── run_study.py                      # Main execution orchestrator
│   ├── analyze_results.py                # Results analysis & plots
│   └── create_publication_figures.py     # Journal-ready figures
├── results/                              # Study outputs
│   ├── performance_summary.csv           # Algorithm comparison table
│   ├── parameter_comparison.csv          # Calibrated parameters
│   └── plots/                            # All figures
└── README.md                             # This file
```

## Prerequisites

1. **SYMFLUENCE Installation**
   ```bash
   pip install -e ".[hbv]"
   ```

2. **Domain Setup**
   - Bow at Banff lumped domain must exist at: `$SYMFLUENCE_DATA_DIR/Bow_at_Banff_lumped_era5`
   - ERA5 forcing data must be acquired
   - WSC streamflow observations for station 05BB001

## Quick Start

### Step 1: Generate Configuration Files
```bash
cd scripts
python generate_configs.py                    # Single seed (12 configs)
python generate_configs.py --seeds 5          # Multi-seed (60 configs)
python generate_configs.py --budget 8000      # Custom budget
```

### Step 2: Run the Study

**Part 1 - Core algorithm comparison (all 12 algorithms):**
```bash
python run_study.py --part 1
```

**Part 2 - Robustness analysis (6 algorithms x 5 seeds):**
```bash
python run_study.py --part 2 --seeds 5
```

**Part 3 - Gradient vs derivative-free analysis:**
```bash
python run_study.py --part 3
```

**Run specific algorithms only:**
```bash
python run_study.py --algorithm dds,pso,adam --part 1
```

**Dry run (preview commands):**
```bash
python run_study.py --part all --dry-run
```

### Step 3: Analyze Results
```bash
python analyze_results.py --output-dir ../results
```

### Step 4: Generate Publication Figures
```bash
python create_publication_figures.py --format pdf
```

## Common Study Settings

| Parameter | Value |
|-----------|-------|
| **Domain** | Bow River at Banff, Alberta |
| **Pour Point** | 51.1722 N, -115.5717 W |
| **Hydrological Model** | HBV (JAX backend) |
| **Timestep** | Daily (24h) |
| **Warmup** | 365 days |
| **Forcing** | ERA5 |
| **Calibration Period** | 2004-01-01 to 2007-12-31 (4 years) |
| **Evaluation Period** | 2008-01-01 to 2009-12-31 (2 years) |
| **Spinup Period** | 2002-01-01 to 2003-12-31 (2 years) |
| **Calibration Metric** | KGE (Kling-Gupta Efficiency) |
| **Streamflow Station** | WSC 05BB001 |
| **Function Eval Budget** | ~4,000 per algorithm |

### HBV Parameters Calibrated (14 total)
```
tt       - Snow/rain threshold temperature [-3, 3] C
cfmax    - Degree-day factor [1, 10] mm/C/day
sfcf     - Snowfall correction factor [0.5, 1.5]
cfr      - Refreezing coefficient [0, 0.1]
cwh      - Water holding capacity [0, 0.2]
fc       - Field capacity [50, 700] mm
lp       - ET reduction threshold [0.3, 1.0]
beta     - Soil moisture shape [1, 6]
k0       - Fast recession [0.05, 0.99] 1/day
k1       - Slow recession [0.01, 0.5] 1/day
k2       - Baseflow recession [0.0001, 0.1] 1/day
uzl      - Fast flow threshold [0, 100] mm
perc     - Percolation rate [0, 10] mm/day
maxbas   - Routing length [1, 7] days
```

## Expected Outputs

### Figures
| Figure | Description |
|--------|-------------|
| fig1 | Algorithm performance (KGE, NSE, RMSE bar charts) |
| fig2 | Convergence curves (objective vs function evaluations) |
| fig3 | Calibrated parameter distributions (equifinality heatmap) |
| fig4 | Calibration vs evaluation KGE scatter (generalization) |
| fig5 | Multi-seed robustness box plots |
| fig6 | Algorithm family aggregate comparison |
| figS1 | Hydrograph comparison - calibration period |
| figS2 | Hydrograph comparison - evaluation period |

### Tables
| Table | Description |
|-------|-------------|
| performance_summary.csv | All metrics per algorithm (cal + eval) |
| parameter_comparison.csv | Calibrated parameter values per algorithm |

## Notes on Fair Comparison

### Function Evaluation Normalization
Different algorithm families consume different numbers of function evaluations per iteration:
- **Single-solution** (DDS, Nelder-Mead): ~1 eval/iter, use full budget as iterations
- **Population-based** (DE, PSO, GA, CMA-ES): `pop_size` evals/generation, use `budget / pop_size` generations
- **Complex evolution** (SCE-UA): `n_complexes * (2*n_params+1)` evals/iter, use `budget / pop_size` iterations
- **Gradient-based** (ADAM, L-BFGS): Use JAX native autodiff (~2 evals/step); L-BFGS adds line search overhead
- **Stochastic** (SA): `steps_per_temp` evals per temperature level
- **Multi-start** (Basin Hopping): `local_steps` evals per hop via local optimizer
- **Surrogate** (Bayesian Opt.): 1 eval + GP fitting per iteration

### Smoothing for Gradient Methods
ADAM and L-BFGS require a differentiable model. HBV uses smooth approximations of threshold functions (min/max operations) controlled by `HBV_SMOOTHING_FACTOR=15.0`. When the JAX backend is used, SYMFLUENCE automatically provides native gradient callbacks via autodiff, avoiding expensive finite-difference approximations. Derivative-free methods use the standard (non-smooth) HBV. This is a deliberate design choice reflecting real-world usage patterns.

### Random Seeds
Part 2 uses seeds {42, 1042, 2042, 3042, 4042} to assess:
- Median performance (central tendency)
- Interquartile range (typical variability)
- Worst-case performance (tail risk)

## Results Summary (Part 1 - Single Seed)

| Rank | Algorithm | Family | Cal KGE | Eval KGE | KGE Degradation |
|------|-----------|--------|---------|----------|-----------------|
| 1 | DDS | Sampling | 0.757 | 0.749 | +0.008 |
| 2 | Nelder-Mead | Direct Search | 0.751 | 0.720 | +0.031 |
| 3 | GA | Evolutionary | 0.745 | 0.723 | +0.023 |
| 4 | CMA-ES | Evolutionary | 0.745 | 0.735 | +0.010 |
| 5 | ADAM | Gradient | 0.743 | **0.778** | -0.035 |
| 6 | SA | Stochastic | 0.736 | 0.742 | -0.006 |
| 7 | DE | Evolutionary | 0.734 | 0.740 | -0.006 |
| 8 | SCE-UA | Evolutionary | 0.683 | 0.637 | +0.045 |
| 9 | Bayesian Opt. | Surrogate | 0.678 | 0.687 | -0.009 |
| 10 | PSO | Evolutionary | 0.662 | 0.628 | +0.034 |
| 11 | Basin Hopping | Stochastic | 0.662 | 0.674 | -0.012 |
| 12 | L-BFGS | Gradient | 0.654 | 0.702 | -0.048 |

**Key findings:**
- ADAM achieves the best generalization (Eval KGE=0.778) despite ranking 5th in calibration, suggesting the smoothed JAX-differentiable HBV provides implicit regularization
- DDS achieves the best calibration performance with minimal overfitting
- Gradient methods show divergent behaviour: ADAM generalizes well while L-BFGS struggles with line search failures
- Negative KGE degradation (improvement during evaluation) observed for ADAM, SA, DE, Basin Hopping, Bayesian Opt., and L-BFGS

## References

- Tolson, B.A. & Shoemaker, C.A. (2007). Dynamically dimensioned search algorithm. Water Resources Research, 43(1).
- Duan, Q., et al. (1993). Shuffled complex evolution approach. Journal of Optimization Theory and Applications, 76(3).
- Hansen, N. & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. Evolutionary Computation, 9(2).
- Kingma, D.P. & Ba, J. (2015). Adam: A method for stochastic optimization. ICLR.
