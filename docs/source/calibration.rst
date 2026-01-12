Calibration and Optimization
============================

SYMFLUENCE provides comprehensive calibration capabilities for hydrological models through automated parameter estimation, multi-objective optimization, and robust evaluation frameworks.

Overview
--------

Model calibration in SYMFLUENCE follows a systematic approach:

1. **Parameter Selection** — Define which parameters to calibrate
2. **Objective Functions** — Choose appropriate metrics for optimization
3. **Algorithm Configuration** — Select optimization strategy
4. **Evaluation** — Assess calibrated model performance

Basic Calibration Setup
-----------------------

Configure calibration in your project YAML:

.. code-block:: yaml

   # Calibration periods
   CALIBRATION_PERIOD: "2018-01-01,2018-06-30"
   EVALUATION_PERIOD: "2018-07-01,2018-12-31"
   
   # Parameters to calibrate
   PARAMS_TO_CALIBRATE: [k_snow, fcapil, newSnowDenMin]
   
   # Optimization settings
   OPTIMIZATION_ALGORITHM: DE
   OPTIMIZATION_METRIC: KGE
   POPULATION_SIZE: 48
   NUMBER_OF_ITERATIONS: 30

Parameter Selection
-------------------

**SUMMA Parameters**

Common parameters for calibration:

.. code-block:: yaml

   SETTINGS_SUMMA_PARAMS_TO_CALIBRATE:
     - k_snow          # snow thermal conductivity
     - fcapil          # capillary fringe thickness
     - newSnowDenMin   # minimum new snow density
     - theta_sat       # soil porosity
     - theta_res       # residual soil moisture
     - vGn_alpha       # van Genuchten alpha parameter
     - vGn_n           # van Genuchten n parameter

**FUSE Parameters**

For FUSE model calibration:

.. code-block:: yaml

   SETTINGS_FUSE_PARAMS_TO_CALIBRATE:
     - alpha           # baseflow recession parameter
     - beta            # percolation parameter  
     - k_storage       # storage coefficient
     - qbrate_2c       # baseflow rate
     - percfrac        # percolation fraction

**NextGen Parameters**

Noah-OWP parameters:

.. code-block:: yaml

   NGEN_NOAH_PARAMS_TO_CALIBRATE:
     - bexp            # pore size distribution
     - dksat           # saturated hydraulic conductivity
     - psisat          # saturated soil potential
     - refkdt          # reference infiltration parameter

Optimization Algorithms
-----------------------

**Differential Evolution (DE)**

Robust global optimizer. Recommended for most applications.

.. code-block:: yaml

   OPTIMIZATION_ALGORITHM: DE
   POPULATION_SIZE: 48
   NUMBER_OF_ITERATIONS: 30
   DE_STRATEGY: best1bin
   DE_MUTATION_FACTOR: 0.5
   DE_CROSSOVER_PROB: 0.7

**Dynamically Dimensioned Search (DDS)**

Efficient for high-dimensional problems.

.. code-block:: yaml

   OPTIMIZATION_ALGORITHM: DDS
   NUMBER_OF_ITERATIONS: 1000
   DDS_R: 0.2

**Particle Swarm Optimization (PSO)**

Good for continuous optimization problems.

.. code-block:: yaml

   OPTIMIZATION_ALGORITHM: PSO
   POPULATION_SIZE: 30
   NUMBER_OF_ITERATIONS: 50
   PSO_INERTIA: 0.9
   PSO_COGNITIVE: 2.0
   PSO_SOCIAL: 2.0

**Multi-Objective (NSGA-II)**

For multiple competing objectives.

.. code-block:: yaml

   OPTIMIZATION_ALGORITHM: NSGA-II
   OPTIMIZATION_METRICS: [KGE, NSE, PBIAS]
   POPULATION_SIZE: 100
   NUMBER_OF_ITERATIONS: 50

Objective Functions
-------------------

**Single Objective**

.. code-block:: yaml

   OPTIMIZATION_METRIC: KGE

Available metrics:
- **KGE** — Kling-Gupta Efficiency (recommended)
- **NSE** — Nash-Sutcliffe Efficiency  
- **RMSE** — Root Mean Square Error
- **PBIAS** — Percent Bias
- **R2** — Coefficient of Determination

**Multi-Objective**

.. code-block:: yaml

   OPTIMIZATION_METRICS: [KGE, NSE, PBIAS]
   MULTI_OBJECTIVE_WEIGHTS: [0.5, 0.3, 0.2]

**Custom Objectives**

Define custom objective functions:

.. code-block:: yaml

   CUSTOM_OBJECTIVE: peak_weighted_kge
   PEAK_WEIGHT_THRESHOLD: 0.9  # 90th percentile
   PEAK_WEIGHT_FACTOR: 2.0

Advanced Calibration
--------------------

**Distributed Calibration**

Calibrate spatially distributed parameters:

.. code-block:: yaml

   DISTRIBUTED_CALIBRATION: true
   SPATIAL_AGGREGATION: hru  # or 'basin', 'elevation_bands'
   PARAMETER_REGIONALIZATION: 
     - elevation
     - slope
     - land_cover

**Multi-Model Calibration**

Calibrate multiple models simultaneously:

.. code-block:: yaml

   MODELS_TO_CALIBRATE: [SUMMA, FUSE]
   ENSEMBLE_WEIGHTING: performance  # or 'equal', 'bayesian'

**Temporal Calibration**

Different parameters for different seasons:

.. code-block:: yaml

   TEMPORAL_CALIBRATION:
     winter: [k_snow, newSnowDenMin]
     summer: [theta_sat, vGn_alpha]
   SEASON_DEFINITIONS:
     winter: "12-01,03-31"
     summer: "06-01,09-30"

Calibration Execution
---------------------

**Command Line**

.. code-block:: bash

   # Run calibration step
   symfluence workflow step calibrate_model --config my_project.yaml

   # Run full workflow including calibration
   symfluence workflow run --config my_project.yaml

   # Check workflow status
   symfluence workflow status --config my_project.yaml

**Python API**

.. code-block:: python

   from symfluence import Configuration, Calibrator
   
   # Load configuration
   config = Configuration('my_project.yaml')
   
   # Initialize calibrator
   calibrator = Calibrator(config)
   
   # Run calibration
   results = calibrator.optimize()
   
   # Get best parameters
   best_params = results.best_parameters
   best_score = results.best_score

Results and Evaluation
----------------------

**Output Files**

Calibration produces:

- ``calibration_results.csv`` — Parameter evolution
- ``best_parameters.yaml`` — Optimal parameter set
- ``objective_history.png`` — Convergence plot
- ``parameter_sensitivity.csv`` — Sensitivity analysis

**Performance Metrics**

.. code-block:: yaml

   EVALUATION_METRICS:
     - KGE
     - NSE  
     - RMSE
     - PBIAS
     - R2
     - peak_timing_error
     - volume_error

**Validation**

Always validate on independent period:

.. code-block:: yaml

   VALIDATION_PERIOD: "2019-01-01,2019-12-31"
   SPLIT_SAMPLE_TEST: true

Best Practices
--------------

1. **Parameter Bounds**
   
   Set realistic parameter ranges:
   
   .. code-block:: yaml
   
      PARAMETER_BOUNDS:
        k_snow: [0.01, 1.0]
        theta_sat: [0.3, 0.6]
        
2. **Convergence Criteria**
   
   .. code-block:: yaml
   
      CONVERGENCE_TOLERANCE: 1e-6
      STAGNATION_GENERATIONS: 10
      
3. **Computational Efficiency**
   
   .. code-block:: yaml
   
      PARALLEL_CALIBRATION: true
      N_CORES: 16
      BATCH_SIZE: 8

4. **Robustness Testing**
   
   .. code-block:: yaml
   
      MONTE_CARLO_RUNS: 100
      BOOTSTRAP_VALIDATION: true

Troubleshooting
---------------

**Common Issues**

- **Slow convergence**: Increase population size or iterations
- **Parameter bounds**: Check realistic ranges for your domain
- **Memory issues**: Reduce batch size or use distributed computing
- **Poor performance**: Verify observation data quality

**Debugging**

.. code-block:: yaml

   DEBUG_CALIBRATION: true
   SAVE_INTERMEDIATE_RESULTS: true
   PLOT_PARAMETER_EVOLUTION: true

Example Workflows
-----------------

**Basic Single-Objective**

.. code-block:: yaml

   CALIBRATION_PERIOD: "2015-01-01,2017-12-31"
   EVALUATION_PERIOD: "2018-01-01,2020-12-31"
   PARAMS_TO_CALIBRATE: [k_snow, fcapil, theta_sat]
   OPTIMIZATION_ALGORITHM: DE
   OPTIMIZATION_METRIC: KGE
   POPULATION_SIZE: 30
   NUMBER_OF_ITERATIONS: 50

**Multi-Objective with Validation**

.. code-block:: yaml

   CALIBRATION_PERIOD: "2010-01-01,2015-12-31"
   EVALUATION_PERIOD: "2016-01-01,2018-12-31"
   VALIDATION_PERIOD: "2019-01-01,2021-12-31"
   PARAMS_TO_CALIBRATE: [alpha, beta, k_storage, qbrate_2c]
   OPTIMIZATION_ALGORITHM: NSGA-II
   OPTIMIZATION_METRICS: [KGE, NSE, PBIAS]
   POPULATION_SIZE: 100
   NUMBER_OF_ITERATIONS: 100
   SPLIT_SAMPLE_TEST: true

---

**See Also**

- :doc:`configuration` — Complete parameter reference
- :doc:`troubleshooting` — Calibration troubleshooting and diagnostics
- :doc:`api` — Programmatic calibration control
