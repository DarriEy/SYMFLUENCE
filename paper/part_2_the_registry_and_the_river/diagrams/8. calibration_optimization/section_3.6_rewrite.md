# Section 3.6 — Calibration and Optimization (Proposed Rewrite)

> **Note:** This rewrite assumes Figure 8 (the calibration framework diagram) is
> referenced inline. Tables 7 and 8 (algorithms and objective metrics) are
> retained as-is — only the prose is compressed.

---

## 3.6 Calibration and Optimization

Parameter calibration remains essential for hydrological model application, whether to compensate for structural deficiencies in process representation, to adapt models to local conditions not captured by a priori parameterization, or to quantify predictive uncertainty. SYMFLUENCE provides a comprehensive optimization framework (Figure 8) in which algorithms, objective functions, calibration targets, and execution strategies are decoupled components that can be combined flexibly based on problem requirements.

### 3.6.1 Architecture

The `OptimizationManager` orchestrates calibration workflows, delegating to model-specific optimizers retrieved from the `OptimizerRegistry`. Each optimizer inherits from `BaseModelOptimizer`, which provides parameter management, parallel execution, results tracking, and final evaluation as reusable capabilities. Algorithms implement a common callback interface for solution evaluation, parameter denormalization, and progress logging, enabling them to remain model-agnostic while optimizers handle model-specific details.

Three component registries supply the building blocks for any calibration experiment. The **Algorithm Library** currently provides eighteen algorithms spanning local search (DDS), population-based methods (PSO, DE, SCE-UA, GA), gradient-based optimization (ADAM, L-BFGS), multi-objective strategies (NSGA-II, MOEA/D), and Bayesian/MCMC approaches (DREAM, ABC), summarized in Table 7. The **ObjectiveRegistry** supplies metrics for quantifying simulation–observation agreement (Table 8); for maximization metrics (KGE, NSE), the framework transforms to minimization via `cost = 1 − metric`, enabling consistent algorithm implementations. The **Calibration Target Registry** maps variable types (streamflow, snow/SWE, ET, soil moisture, groundwater, total water storage) to model-specific evaluator implementations through a three-tier lookup: dynamic registry, model-specific overrides, and generic defaults. A `MultivariateTarget` combines multiple variables into a single scalar objective via configurable weighting, enabling multi-variable calibration without algorithm modification.

### 3.6.2 Parameter Normalisation

All algorithms operate in a normalized [0, 1] parameter space, enabling consistent search behavior regardless of parameter magnitudes or units. The `BaseParameterManager` implements bidirectional transformation between normalized and physical spaces. Parameter bounds derive from model-specific sources—SUMMA parses `localParamInfo.txt` and `basinParamInfo.txt`; HBV maintains a central bounds registry; FUSE uses hardcoded ranges—making algorithm implementations fully portable across registered models.

### 3.6.3 Calibration Loop

The calibration workflow proceeds through the iterative cycle shown in Figure 8. After initialization (algorithm selection, iteration budget, population size, and parallelization settings), the optimization loop repeats until convergence or budget exhaustion: (1) the algorithm proposes candidate solutions in normalized space; (2) parameters denormalize to physical values and update model files; (3) models execute—in parallel for population-based methods; (4) objective functions evaluate simulations against observations; and (5) the algorithm updates its internal state based on fitness. Progress logs and checkpoints persist at each iteration.

Final evaluation applies the best parameters to a full-period simulation spanning both calibration and evaluation windows. Metrics computed separately for each period enable detection of overfitting. Results persistence saves optimization history, best parameters, convergence trajectory, and final metrics to JSON and CSV formats.

### 3.6.4 Execution Distribution

Parallel execution employs a strategy pattern with automatic runtime selection (Figure 8, right panel). The **MPI strategy** distributes tasks round-robin across ranks on distributed-memory HPC clusters. The **ProcessPool strategy** uses Python's `ProcessPoolExecutor` for shared-memory parallelism on multi-core workstations. The **Sequential strategy** provides a fallback when parallelism is unavailable. Selection cascades automatically: MPI is attempted first (detected via environment variables), falling back to ProcessPool, then Sequential.

Process isolation prevents file conflicts during parallel evaluation. Each candidate evaluation receives a dedicated directory structure, and configuration files are automatically updated with process-specific paths, ensuring concurrent model executions do not interfere.
