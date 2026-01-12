=========================================
API Reference
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

The SYMFLUENCE Python API provides programmatic access to the full workflow, from project setup through calibration and analysis. The primary entry point is the ``SYMFLUENCE`` class, which coordinates manager components through the ``WorkflowOrchestrator``.

Quick Start
===========

Basic Usage
-----------

.. code-block:: python

   from symfluence import SYMFLUENCE

   # Initialize from configuration file
   conf = SYMFLUENCE("my_project.yaml")

   # Run complete workflow
   conf.run_workflow()

   # Or run individual steps
   conf.setup_project()
   conf.define_domain()
   conf.acquire_forcings()
   conf.run_models()

Step-by-Step Execution
----------------------

.. code-block:: python

   from symfluence import SYMFLUENCE

   # Initialize
   conf = SYMFLUENCE("config.yaml")

   # 1. Project Setup
   conf.setup_project()
   conf.create_pour_point()

   # 2. Domain Definition
   conf.acquire_attributes()
   conf.define_domain()
   conf.discretize_domain()

   # 3. Data Acquisition
   conf.process_observed_data()
   conf.acquire_forcings()
   conf.run_model_agnostic_preprocessing()

   # 4. Model Execution
   conf.preprocess_models()
   conf.run_models()
   conf.postprocess_results()

   # 5. Calibration (optional)
   conf.calibrate_model()

   # 6. Analysis
   conf.run_benchmarking()
   conf.run_sensitivity_analysis()

Configuration Access
--------------------

.. code-block:: python

   from symfluence import SYMFLUENCE
   from symfluence.core.config import SymfluenceConfig

   # Load typed configuration directly
   config = SymfluenceConfig.from_file("config.yaml")

   # Access typed attributes
   print(f"Domain: {config.domain.name}")
   print(f"Model: {config.model.hydrological_model}")
   print(f"Start: {config.experiment.time_start}")

   # Initialize SYMFLUENCE with typed config
   conf = SYMFLUENCE(config)

Core API
========

SYMFLUENCE (Main Class)
-----------------------

The primary interface for all SYMFLUENCE operations.

.. autoclass:: symfluence.core.system.SYMFLUENCE
   :members:
   :undoc-members:
   :show-inheritance:

**Initialization:**

.. code-block:: python

   from symfluence import SYMFLUENCE

   # From YAML file path
   conf = SYMFLUENCE("path/to/config.yaml")

   # From SymfluenceConfig object
   from symfluence.core.config import SymfluenceConfig
   config = SymfluenceConfig.from_file("config.yaml")
   conf = SYMFLUENCE(config)

   # From dictionary
   config_dict = {"DOMAIN_NAME": "test", ...}
   conf = SYMFLUENCE(config_dict)

**Core Methods:**

.. code-block:: python

   # Full workflow execution
   conf.run_workflow(force_run=False)

   # Workflow status
   status = conf.get_workflow_status()
   # Returns: {
   #   "total_steps": 15,
   #   "completed_steps": 8,
   #   "pending_steps": 7,
   #   "step_details": [...]
   # }

Manager Classes
===============

SYMFLUENCE uses a manager-based architecture where each major subsystem has a dedicated manager class.

Project Manager
---------------

Handles project initialization and structure.

.. automodule:: symfluence.project.project_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

.. code-block:: python

   from symfluence.project.project_manager import ProjectManager

   pm = ProjectManager(config, logger)

   # Setup project directory structure
   pm.setup_project()

   # Create pour point from coordinates
   pm.create_pour_point()

   # Get project information
   info = pm.get_project_info()

Domain Manager
--------------

Manages domain definition and discretization.

.. automodule:: symfluence.geospatial.domain_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

.. code-block:: python

   from symfluence.geospatial.domain_manager import DomainManager

   dm = DomainManager(config, logger)

   # Define domain boundaries
   dm.define_domain()

   # Discretize into HRUs/GRUs
   dm.discretize_domain()

   # Get domain statistics
   stats = dm.get_domain_statistics()

Data Manager
------------

Coordinates data acquisition and preprocessing.

.. automodule:: symfluence.data.data_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

.. code-block:: python

   from symfluence.data.data_manager import DataManager

   data_mgr = DataManager(config, logger)

   # Acquire geospatial attributes
   data_mgr.acquire_attributes()

   # Process observed streamflow data
   data_mgr.process_observed_data()

   # Acquire forcing data (ERA5, RDRS, etc.)
   data_mgr.acquire_forcings()

   # Run model-agnostic preprocessing
   data_mgr.run_model_agnostic_preprocessing()

Model Manager
-------------

Coordinates model preprocessing, execution, and postprocessing.

.. automodule:: symfluence.models.model_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

.. code-block:: python

   from symfluence.models.model_manager import ModelManager

   mm = ModelManager(config, logger)

   # Preprocess for all configured models
   mm.preprocess_models()

   # Run model simulations
   mm.run_models()

   # Extract and format results
   mm.postprocess_results()

   # Get available models
   models = mm.get_available_models()

Optimization Manager
--------------------

Handles calibration and optimization.

.. automodule:: symfluence.optimization.optimization_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

.. code-block:: python

   from symfluence.optimization.optimization_manager import OptimizationManager

   opt = OptimizationManager(config, logger)

   # Run calibration
   results = opt.run_calibration()

   # Get best parameters
   best_params = results["best_parameters"]
   best_score = results["best_score"]

Analysis Manager
----------------

Performs model evaluation and analysis.

.. automodule:: symfluence.evaluation.analysis_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

.. code-block:: python

   from symfluence.evaluation.analysis_manager import AnalysisManager

   am = AnalysisManager(config, logger)

   # Run benchmarking analysis
   am.run_benchmarking()

   # Run sensitivity analysis
   am.run_sensitivity_analysis()

   # Run decision analysis
   am.run_decision_analysis()

Workflow Orchestrator
---------------------

Manages workflow step execution and dependencies.

.. automodule:: symfluence.project.workflow_orchestrator
   :members:
   :undoc-members:
   :show-inheritance:

**Usage:**

.. code-block:: python

   from symfluence.project.workflow_orchestrator import WorkflowOrchestrator

   orchestrator = WorkflowOrchestrator(config, logger, managers)

   # Run full workflow
   orchestrator.run_workflow()

   # Run specific step
   orchestrator.run_step("calibrate_model")

   # Get workflow status
   status = orchestrator.get_workflow_status()

Model Registry
==============

The Model Registry enables plugin-style model integration.

.. automodule:: symfluence.models.registry
   :members:
   :undoc-members:
   :show-inheritance:

Registering Models
------------------

.. code-block:: python

   from symfluence.models.registry import ModelRegistry

   # Register preprocessor
   @ModelRegistry.register_preprocessor('MY_MODEL')
   class MyPreProcessor:
       def __init__(self, config, logger):
           self.config = config
           self.logger = logger

       def run_preprocessing(self):
           # Preprocessing logic
           pass

   # Register runner
   @ModelRegistry.register_runner('MY_MODEL', method_name='run_my_model')
   class MyRunner:
       def __init__(self, config, logger, reporting_manager=None):
           self.config = config
           self.logger = logger

       def run_my_model(self):
           # Model execution logic
           pass

   # Register postprocessor
   @ModelRegistry.register_postprocessor('MY_MODEL')
   class MyPostProcessor:
       def __init__(self, config, logger, reporting_manager=None):
           self.config = config
           self.logger = logger

       def extract_streamflow(self):
           # Result extraction logic
           pass

Querying Registry
-----------------

.. code-block:: python

   from symfluence.models.registry import ModelRegistry

   # List registered models
   models = ModelRegistry.list_models()
   # ['SUMMA', 'FUSE', 'GR', 'HYPE', 'NGEN', ...]

   # Get specific components
   preprocessor_cls = ModelRegistry.get_preprocessor('SUMMA')
   runner_cls = ModelRegistry.get_runner('SUMMA')
   postprocessor_cls = ModelRegistry.get_postprocessor('SUMMA')

   # Check if model is registered
   is_registered = ModelRegistry.is_registered('MY_MODEL')

Optimization API
================

Base Optimizer
--------------

.. automodule:: symfluence.optimization.optimizers.base_model_optimizer
   :members:
   :undoc-members:
   :show-inheritance:

DDS Optimizer
-------------

Dynamically Dimensioned Search optimizer.

.. automodule:: symfluence.optimization.optimizers.dds_optimizer
   :members:
   :undoc-members:
   :show-inheritance:

**Usage:**

.. code-block:: python

   from symfluence.optimization.optimizers.dds_optimizer import DDSOptimizer

   optimizer = DDSOptimizer(config, logger)
   results = optimizer.optimize()

   print(f"Best score: {results['best_score']}")
   print(f"Best params: {results['best_parameters']}")

Algorithm Selection
-------------------

.. code-block:: python

   # Configure algorithm in YAML
   # OPTIMIZATION_ALGORITHM: DDS  # or DE, PSO, SCE-UA, NSGA-II

   # Programmatic algorithm selection
   from symfluence.optimization.optimization_manager import OptimizationManager

   opt = OptimizationManager(config, logger)

   # Available algorithms
   algorithms = ['DDS', 'DE', 'PSO', 'SCE-UA', 'NSGA-II', 'ADAM', 'LBFGS']

Data Acquisition
================

Acquisition Service
-------------------

.. automodule:: symfluence.data.acquisition.acquisition_service
   :members:
   :undoc-members:
   :show-inheritance:

**Available Data Sources:**

.. code-block:: python

   # Forcing datasets
   forcing_sources = [
       'ERA5',        # ECMWF reanalysis
       'ERA5-Land',   # High-resolution land reanalysis
       'RDRS',        # Regional Deterministic Reforecast System
       'CARRA',       # Copernicus Arctic Regional Reanalysis
       'AORC',        # Analysis of Record for Calibration
       'CONUS404',    # CONUS 404 dataset
       'HRRR',        # High-Resolution Rapid Refresh
       'EM-Earth',    # EM-Earth reanalysis
       'NEX-GDDP',    # NASA climate projections
   ]

   # Observation datasets
   obs_sources = [
       'USGS',        # US Geological Survey streamflow
       'WSC',         # Water Survey of Canada
       'GRDC',        # Global Runoff Data Centre
       'MODIS',       # Remote sensing products
       'GRACE',       # Gravity recovery data
   ]

Acquisition Handlers
--------------------

.. code-block:: python

   from symfluence.data.acquisition import AcquisitionRegistry

   # Get available handlers
   handlers = AcquisitionRegistry.list_handlers()

   # Get specific handler
   era5_handler = AcquisitionRegistry.get_handler('ERA5')

Geospatial Operations
=====================

Domain Discretization
---------------------

.. automodule:: symfluence.geospatial.discretization.core
   :members:
   :undoc-members:
   :show-inheritance:

**Discretization Methods:**

.. code-block:: python

   # Available discretization approaches
   methods = [
       'lumped',           # Single unit
       'GRUs',             # Grouped Response Units
       'elevation',        # Elevation bands
       'radiation',        # Radiation-based
       'combined',         # Multiple criteria
   ]

Evaluation
==========

Evaluators
----------

.. automodule:: symfluence.evaluation.evaluators.base
   :members:
   :undoc-members:
   :show-inheritance:

**Available Evaluators:**

.. code-block:: python

   from symfluence.evaluation.evaluators import (
       StreamflowEvaluator,
       ETEvaluator,
       SnowEvaluator,
       SoilMoistureEvaluator,
       GroundwaterEvaluator,
       TWSEvaluator,
   )

   # Initialize evaluator
   evaluator = StreamflowEvaluator(config, project_dir, logger)

   # Evaluate simulation
   metrics = evaluator.evaluate(sim_dir)
   # Returns: {'KGE': 0.85, 'NSE': 0.82, 'RMSE': 12.5, ...}

Metrics
-------

.. code-block:: python

   # Available metrics
   metrics = [
       'KGE',      # Kling-Gupta Efficiency
       'KGEnp',    # Non-parametric KGE
       'NSE',      # Nash-Sutcliffe Efficiency
       'RMSE',     # Root Mean Square Error
       'MAE',      # Mean Absolute Error
       'PBIAS',    # Percent Bias
       'R2',       # Coefficient of Determination
   ]

Reporting
=========

Reporting Manager
-----------------

.. automodule:: symfluence.reporting.reporting_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Visualization Methods:**

.. code-block:: python

   from symfluence.reporting.reporting_manager import ReportingManager

   rm = ReportingManager(config, logger)

   # Generate domain map
   rm.plot_domain_map()

   # Generate hydrograph
   rm.plot_hydrograph(observed, simulated)

   # Generate calibration convergence plot
   rm.plot_calibration_convergence(results)

   # Generate sensitivity analysis plot
   rm.visualize_sensitivity_analysis(sensitivity_results)

Configuration
=============

SymfluenceConfig
----------------

.. automodule:: symfluence.core.config
   :members:
   :undoc-members:
   :show-inheritance:

**Loading and Using Configuration:**

.. code-block:: python

   from symfluence.core.config import SymfluenceConfig, ensure_typed_config

   # Load from file
   config = SymfluenceConfig.from_file("config.yaml")

   # From dictionary
   config = SymfluenceConfig(**config_dict)

   # Ensure typed config (for mixed dict/config inputs)
   config = ensure_typed_config(maybe_dict_or_config)

   # Access configuration values
   domain = config.domain.name
   model = config.model.hydrological_model

   # Convert to dictionary
   flat_dict = config.to_dict(flatten=True)

Utilities
=========

Path Management
---------------

.. code-block:: python

   from symfluence.data.path_manager import PathManager

   pm = PathManager(config)

   # Access standard paths
   project_dir = pm.project_dir
   forcing_dir = pm.forcing_dir
   simulations_dir = pm.simulations_dir
   observations_dir = pm.observations_dir

Logging
-------

.. code-block:: python

   from symfluence.project.logging_manager import LoggingManager

   # Initialize logging
   log_mgr = LoggingManager(config)
   logger = log_mgr.get_logger("my_module")

   # Log messages
   logger.info("Processing started")
   logger.warning("Optional data not found")
   logger.error("Critical failure")

Error Handling
--------------

.. code-block:: python

   from symfluence.core.exceptions import (
       SymfluenceError,           # Base exception
       ConfigurationError,        # Config issues
       DataAcquisitionError,      # Data download failures
       ModelExecutionError,       # Model run failures
       ValidationError,           # Validation failures
   )

   try:
       conf.run_workflow()
   except ConfigurationError as e:
       print(f"Configuration problem: {e}")
   except ModelExecutionError as e:
       print(f"Model failed: {e}")
   except SymfluenceError as e:
       print(f"General error: {e}")

Advanced Usage
==============

Custom Workflow
---------------

.. code-block:: python

   from symfluence import SYMFLUENCE

   conf = SYMFLUENCE("config.yaml")

   # Run subset of steps
   conf.setup_project()
   conf.define_domain()

   # Skip to model execution (assumes data exists)
   conf.preprocess_models()
   conf.run_models()

   # Custom post-processing
   results = conf.postprocess_results()

   # Access internal managers
   model_mgr = conf.managers['model']
   data_mgr = conf.managers['data']

Parallel Execution
------------------

.. code-block:: python

   # Configure in YAML
   # MPI_PROCESSES: 8
   # PARALLEL_CALIBRATION: true

   # Or programmatically
   config_dict['MPI_PROCESSES'] = 8
   config_dict['PARALLEL_CALIBRATION'] = True

   conf = SYMFLUENCE(config_dict)
   conf.calibrate_model()  # Uses parallel execution

Batch Processing
----------------

.. code-block:: python

   from symfluence import SYMFLUENCE
   from pathlib import Path

   # Process multiple domains
   config_files = Path("configs/").glob("*.yaml")

   for config_file in config_files:
       print(f"Processing {config_file.name}")
       conf = SYMFLUENCE(str(config_file))
       conf.run_workflow()

References
==========

- :doc:`getting_started` — High-level workflow tutorial
- :doc:`configuration` — Configuration parameter reference
- :doc:`config_system` — Configuration system architecture
- :doc:`developer_guide` — Extending SYMFLUENCE
- :doc:`examples` — Example workflows and use cases
