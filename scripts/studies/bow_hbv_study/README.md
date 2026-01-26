# Bow at Banff HBV Study

Comprehensive analysis of the HBV hydrological model implementation in SYMFLUENCE using the Bow River at Banff watershed as a test case.

## Study Overview

This study systematically evaluates three key aspects of the HBV model implementation:

### 1. Temporal Resolution (Daily vs Hourly)
- **Objective**: Compare model performance and computational efficiency at different temporal resolutions
- **Configurations**:
  - Daily timestep (24 hours)
  - Hourly timestep (1 hour)
- **Optimizer**: DDS with 4000 iterations
- **Expected Outcomes**:
  - Hourly model should capture sub-daily dynamics better
  - Daily model should be computationally more efficient
  - Performance metrics (KGE, NSE) comparison

### 2. Optimization Algorithm Comparison
- **Objective**: Evaluate calibration performance across different optimization algorithms
- **Algorithms Tested**:
  - **DDS** (Dynamically Dimensioned Search): Gradient-free, adaptive perturbation
  - **PSO** (Particle Swarm Optimization): Population-based, swarm intelligence
  - **DE** (Differential Evolution): Population-based, mutation/crossover
  - **GA** (Genetic Algorithm): Population-based, evolutionary
  - **ADAM** (Adaptive Moment Estimation): Gradient-based (finite-difference gradients)
- **Budget**: 4000 iterations each
- **Timestep**: Daily (24 hours)
- **Expected Outcomes**:
  - Convergence rate comparison
  - Final calibration performance (KGE)
  - Computational efficiency
  - Parameter sensitivity

### 3. Differentiability Analysis
- **Objective**: Assess the impact of smooth approximations and gradient computation methods
- **Sub-studies**:

  **a) Smoothing vs Non-smoothing**
  - Compare HBV with and without smooth threshold functions
  - Impact on gradient quality and optimization performance

  **b) Gradient Computation Methods**
  - **Direct AD (JAX)**: Native automatic differentiation through discrete time-stepping
  - **ODE Adjoint**: Continuous-time formulation with adjoint sensitivity
  - **Finite Differences**: Numerical gradient approximation

  **c) Optimization Performance**
  - ADAM with smoothing (best gradients)
  - DDS without smoothing (gradient-free baseline)
  - Gradient accuracy vs optimization performance tradeoff

## Study Structure

```
bow_hbv_study/
├── configs/                    # Configuration files for each experiment
│   ├── config_bow_hbv_daily_dds.yaml
│   ├── config_bow_hbv_hourly_dds.yaml
│   ├── config_bow_hbv_daily_{pso,de,ga,adam}.yaml
│   ├── config_bow_hbv_daily_dds_{smooth,nosmooth}.yaml
│   └── config_bow_hbv_daily_adam_smooth.yaml
├── scripts/                    # Execution and analysis scripts
│   ├── generate_configs.py    # Generate configuration files
│   ├── run_study.py           # Main study execution script
│   └── analyze_results.py     # Results analysis and visualization
├── results/                    # Study outputs (created during execution)
│   ├── calibration_results/   # Calibration outputs for each experiment
│   ├── plots/                 # Diagnostic plots and comparisons
│   └── summary_report.html    # Interactive summary report
└── README.md                  # This file
```

## Prerequisites

1. **SYMFLUENCE Installation**
   ```bash
   # Install SYMFLUENCE with HBV dependencies
   pip install -e ".[hbv]"
   ```

2. **Domain Setup**
   - Domain data must exist at: `/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_lumped_era5`
   - Forcing data and observed streamflow must be preprocessed

3. **Dependencies**
   ```bash
   pip install jax jaxlib
   pip install diffrax  # For ODE-based gradient comparison
   pip install pandas matplotlib seaborn plotly  # For analysis
   ```

## Quick Start

### Step 1: Generate Configuration Files
```bash
cd scripts
python generate_configs.py
```

This creates all necessary config files in the `configs/` directory.

### Step 2: Run the Study

**Run all study parts:**
```bash
python run_study.py --part all
```

**Run specific parts:**
```bash
# Part 1: Daily vs Hourly
python run_study.py --part 1

# Part 2: Optimizer comparison
python run_study.py --part 2

# Part 3: Differentiability analysis
python run_study.py --part 3

# Gradient comparison only
python run_study.py --part gradients
```

**Dry run (see what will be executed):**
```bash
python run_study.py --part all --dry-run
```

### Step 3: Analyze Results
```bash
python analyze_results.py --output-dir ../results
```

This generates:
- Performance comparison plots
- Convergence curves
- Parameter sensitivity analysis
- Interactive HTML report

## Manual Execution

You can also run experiments manually using the SYMFLUENCE CLI:

### Example: Daily DDS Calibration
```bash
# 1. Preprocessing (if not already done)
symfluence workflow step model_specific_preprocessing \
  --config configs/config_bow_hbv_daily_dds.yaml

# 2. Calibration
symfluence workflow step calibrate_model \
  --config configs/config_bow_hbv_daily_dds.yaml

# 3. Evaluation
symfluence workflow step run_benchmarking \
  --config configs/config_bow_hbv_daily_dds.yaml
```

### Example: Gradient Comparison
```bash
# Compare ODE vs Direct AD vs Finite Differences
python -m symfluence.models.hbv.compare_solvers \
  --n-days 365 \
  --timestep 24 \
  --save-plot results/gradient_comparison.png
```

## Study Configurations

### Common Settings
- **Domain**: Bow at Banff, lumped watershed
- **Calibration Period**: 2004-01-01 to 2007-12-31
- **Evaluation Period**: 2008-01-01 to 2009-12-31
- **Spinup Period**: 2002-01-01 to 2003-12-31
- **Objective Function**: Kling-Gupta Efficiency (KGE)
- **Forcing**: ERA5 reanalysis
- **Observations**: WSC Station 05BB001

### Variable Settings by Experiment

| Experiment | Timestep | Algorithm | Smoothing | Backend |
|------------|----------|-----------|-----------|---------|
| Part 1a    | 24h      | DDS       | No        | JAX     |
| Part 1b    | 1h       | DDS       | No        | JAX     |
| Part 2a    | 24h      | DDS       | No        | JAX     |
| Part 2b    | 24h      | PSO       | No        | JAX     |
| Part 2c    | 24h      | DE        | No        | JAX     |
| Part 2d    | 24h      | GA        | No        | JAX     |
| Part 2e    | 24h      | ADAM      | No        | JAX     |
| Part 3a    | 24h      | DDS       | Yes       | JAX     |
| Part 3b    | 24h      | DDS       | No        | JAX     |
| Part 3c    | 24h      | ADAM      | Yes       | JAX     |

## Expected Results

### 1. Temporal Resolution
- **Performance**: Hourly model expected to achieve higher KGE due to better representation of sub-daily processes
- **Computational Cost**: Daily model ~24x faster per timestep
- **Use Case**: Daily for long-term simulations, hourly for event-based analysis

### 2. Optimization Algorithms
- **DDS**: Robust baseline, good exploration-exploitation balance
- **PSO/DE/GA**: May find different local optima, population diversity beneficial
- **ADAM**: Fastest convergence if gradients are accurate, may struggle without smoothing

### 3. Differentiability
- **Smoothing**: Improves gradient quality but may alter model physics slightly
- **Gradient Methods**:
  - Direct AD (JAX): Fastest, exact gradients for discrete model
  - ODE Adjoint: Memory-efficient, continuous approximation
  - Finite Differences: Slowest, reference for validation

## Interpreting Results

### Key Metrics to Compare

1. **Calibration Performance**
   - Final KGE value
   - NSE, RMSE, Bias
   - Peak flow accuracy
   - Low flow accuracy

2. **Convergence Behavior**
   - Iterations to convergence
   - Total function evaluations
   - Convergence stability

3. **Computational Efficiency**
   - Wall-clock time
   - Function evaluations per second
   - Memory usage

4. **Parameter Distributions**
   - Parameter values and uncertainty
   - Correlation structure
   - Physical realism

### Diagnostic Plots

The analysis script generates:
- **Hydrographs**: Observed vs simulated streamflow
- **Scatter plots**: Observed vs simulated with 1:1 line
- **Convergence curves**: Objective function vs iteration
- **Parameter evolution**: Parameter trajectories over optimization
- **Flow duration curves**: Model performance across flow regimes
- **Seasonal performance**: Monthly KGE/NSE breakdown

## Advanced Usage

### Custom Analysis

Create custom analysis scripts by importing results:

```python
from pathlib import Path
import pandas as pd
import yaml

# Load calibration results
results_dir = Path("results/calibration_results")
for exp_dir in results_dir.glob("study_*"):
    config_path = exp_dir / "config.yaml"
    results_path = exp_dir / "calibration_results.csv"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    results = pd.read_csv(results_path)

    # Your analysis here
    print(f"{exp_dir.name}: Final KGE = {results['kge'].iloc[-1]:.3f}")
```

### Extending the Study

Add new experiments by:

1. Create new config in `configs/`
2. Add to `STUDY_PARTS` in `run_study.py`
3. Run: `python run_study.py --part <new_part_id>`

### Parallel Execution

For faster execution on multi-core systems:

```python
# In run_study.py, modify MPI_PROCESSES in configs
# Or use GNU parallel:
parallel python run_study.py --part {} ::: 1 2 3
```

## Troubleshooting

### Common Issues

1. **"Config file not found"**
   - Run `generate_configs.py` first
   - Check paths in config files match your system

2. **"Forcing data missing"**
   - Ensure preprocessing steps completed: `model_agnostic_preprocessing`
   - Check FORCING_PATH in config

3. **"Calibration fails to converge"**
   - Try increasing NUMBER_OF_ITERATIONS
   - Adjust optimizer hyperparameters (e.g., DDS_R, PSO learning rates)
   - Check parameter bounds are reasonable

4. **"ADAM optimization unstable"**
   - Enable smoothing: `HBV_SMOOTHING: true`
   - Reduce learning rate: `ADAM_LR: 0.001`
   - Check gradients aren't NaN/Inf

### Debugging

Enable debug mode:
```bash
symfluence workflow step calibrate_model \
  --config configs/config_bow_hbv_daily_dds.yaml \
  --debug
```

Check logs:
```bash
tail -f $SYMFLUENCE_DATA_DIR/domain_Bow_at_Banff_lumped_era5/logs/*.log
```

## References

### HBV Model
- Lindström, G., et al. (1997). Development and test of the distributed HBV-96 hydrological model. *Journal of Hydrology*, 201(1-4), 272-288.

### Optimization Algorithms
- Tolson, B.A. and Shoemaker, C.A. (2007). Dynamically dimensioned search algorithm for computationally efficient watershed model calibration. *Water Resources Research*, 43(1).
- Kennedy, J. and Eberhart, R. (1995). Particle swarm optimization. *IEEE International Conference on Neural Networks*.
- Storn, R. and Price, K. (1997). Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341-359.
- Kingma, D.P. and Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.

### Automatic Differentiation
- Baydin, A.G., et al. (2018). Automatic differentiation in machine learning: a survey. *Journal of Machine Learning Research*, 18(153), 1-43.
- Chen, R.T.Q., et al. (2018). Neural ordinary differential equations. *NeurIPS 2018*.

## Contact

For questions or issues:
- GitHub: https://github.com/DarriEy/SYMFLUENCE/issues
- Documentation: https://symfluence.readthedocs.io/

## License

This study is part of SYMFLUENCE and follows the same license.
