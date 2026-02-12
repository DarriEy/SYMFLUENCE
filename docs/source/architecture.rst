=========================================
Architecture Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

SYMFLUENCE is built on a modular, manager-based architecture with clear separation of concerns. The framework follows established design patterns to enable extensibility, maintainability, and loose coupling between components.

**Core Principles:**

- **Manager Pattern**: Subsystems coordinated through dedicated manager classes
- **Registry Pattern**: Models self-register using decorators for plugin extensibility
- **Mixin Pattern**: Shared functionality distributed through composable mixins
- **Typed Configuration**: Pydantic models for validation and type safety
- **Lazy Loading**: Components instantiated on-demand for efficiency

System Architecture Diagram
===========================

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                           SYMFLUENCE Class                              │
   │                    (Primary Entry Point / Facade)                       │
   └───────────────────────────────┬─────────────────────────────────────────┘
                                   │
                                   ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                      WorkflowOrchestrator                               │
   │              (Step Coordination & Execution Control)                    │
   └───────────────────────────────┬─────────────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
   ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
   │  DataManager  │      │ ModelManager  │      │ Optimization  │
   │               │      │               │      │   Manager     │
   └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
           │                      │                      │
           ▼                      ▼                      ▼
   ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
   │  Acquisition  │      │ ModelRegistry │      │  Optimizers   │
   │   Services    │      │   (Plugin)    │      │  (DE/DDS/     │
   │               │      │               │      │   ADAM/PSO)   │
   └───────────────┘      └───────────────┘      └───────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
                 ▼               ▼               ▼
          ┌──────────┐    ┌──────────┐    ┌──────────┐
          │Preproc   │    │ Runner   │    │Postproc  │
          │(per      │    │(per      │    │(per      │
          │ model)   │    │ model)   │    │ model)   │
          └──────────┘    └──────────┘    └──────────┘

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                     Cross-Cutting Concerns                              │
   ├─────────────────────┬─────────────────────┬─────────────────────────────┤
   │   DomainManager     │  ReportingManager   │    AnalysisManager          │
   │   (Geospatial)      │  (Visualization)    │    (Evaluation)             │
   └─────────────────────┴─────────────────────┴─────────────────────────────┘

Directory Structure
===================

.. code-block:: text

   symfluence/
   ├── core/                      # Core framework infrastructure
   │   ├── system.py             # SYMFLUENCE main class (entry point)
   │   ├── base_manager.py       # Abstract base for all managers
   │   ├── config/               # Configuration system
   │   │   ├── models.py         # SymfluenceConfig (Pydantic)
   │   │   ├── transformers.py   # Flat ↔ nested conversion
   │   │   └── validators.py     # Custom validation rules
   │   ├── exceptions.py         # Custom exception hierarchy
   │   ├── constants.py          # Unit conversions, defaults
   │   └── mixins/               # Shared functionality mixins
   │
   ├── project/                   # Project & workflow management
   │   ├── project_manager.py    # Directory structure setup
   │   ├── workflow_orchestrator.py  # Step execution engine
   │   ├── logging_manager.py    # Logging configuration
   │   └── manager_factory.py    # Lazy manager instantiation
   │
   ├── data/                      # Data acquisition & preprocessing
   │   ├── data_manager.py       # Data operations facade
   │   ├── acquisition/          # Cloud data acquisition
   │   │   ├── services/         # CDS, GEE, S3 handlers
   │   │   └── handlers/         # Dataset-specific logic
   │   ├── preprocessing/        # Model-agnostic preprocessing
   │   └── utils/                # Spatial, NetCDF, archive utils
   │
   ├── geospatial/               # Domain definition & discretization
   │   ├── domain_manager.py     # Spatial operations facade
   │   ├── geofabric/            # River network extraction
   │   └── discretization/       # HRU generation methods
   │
   ├── models/                    # Hydrological model integrations
   │   ├── model_manager.py      # Model execution coordinator
   │   ├── registry.py           # Plugin registration system
   │   ├── base/                 # Base classes for model components
   │   │   ├── base_preprocessor.py
   │   │   ├── base_runner.py
   │   │   └── base_postprocessor.py
   │   └── {model}/              # Model-specific implementations
   │       ├── preprocessor.py
   │       ├── runner.py
   │       ├── postprocessor.py
   │       └── config.py
   │
   ├── optimization/             # Calibration & optimization
   │   ├── optimization_manager.py
   │   ├── optimizers/           # Algorithm implementations
   │   │   ├── algorithms/       # DE, DDS, PSO, ADAM, L-BFGS
   │   │   └── base_model_optimizer.py
   │   ├── parameter_managers/   # Model-specific parameter handling
   │   └── workers/              # Parallel evaluation workers
   │
   ├── evaluation/               # Performance metrics & analysis
   │   ├── analysis_manager.py
   │   ├── metrics/              # KGE, NSE, RMSE, etc.
   │   └── evaluators/           # Streamflow, snow, etc.
   │
   └── reporting/                # Visualization & output
       ├── reporting_manager.py
       └── plotters/             # Specialized plot generators

Core Design Patterns
====================

Manager Pattern
---------------

Each major subsystem has a dedicated manager class that:

- Coordinates multiple services/components
- Provides a high-level API to the orchestrator
- Inherits from ``BaseManager`` for consistent behavior
- Lazy-loads dependencies for efficiency

**BaseManager provides:**

.. code-block:: python

   class BaseManager(ConfigurableMixin, ABC):
       """Abstract base for all SYMFLUENCE managers."""

       def __init__(self, config, logger, reporting_manager=None):
           # Auto-convert dict to typed config
           self._config = SymfluenceConfig(**config) if isinstance(config, dict) else config
           self.logger = logger
           self.reporting_manager = reporting_manager
           self._initialize_services()  # Subclass hook

       def _execute_workflow(self, items, handler, operation_name):
           """Standardized batch processing with error handling."""

       def _safe_visualize(self, viz_func, *args, **kwargs):
           """Safe visualization with error handling."""

**Key Managers:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Manager
     - Responsibility
   * - ``DataManager``
     - Data acquisition, merging, preprocessing
   * - ``DomainManager``
     - Catchment delineation, HRU discretization
   * - ``ModelManager``
     - Model preprocessing, execution, postprocessing
   * - ``OptimizationManager``
     - Parameter calibration, algorithm orchestration
   * - ``AnalysisManager``
     - Performance evaluation, metrics calculation
   * - ``ReportingManager``
     - Visualization, plot generation

Registry Pattern (Plugin System)
--------------------------------

The ``ModelRegistry`` enables extensible model architecture without modifying core code:

**Registration via Decorators:**

.. code-block:: python

   from symfluence.models.registry import ModelRegistry

   @ModelRegistry.register_preprocessor('MYMODEL')
   class MyModelPreprocessor(BaseModelPreProcessor):
       def run_preprocessing(self):
           # Model-specific input preparation
           pass

   @ModelRegistry.register_runner('MYMODEL', method_name='run_mymodel')
   class MyModelRunner(BaseModelRunner):
       def run_mymodel(self):
           # Execute model
           pass

   @ModelRegistry.register_postprocessor('MYMODEL')
   class MyModelPostprocessor:
       def extract_streamflow(self):
           # Parse output files
           pass

**Component Discovery:**

.. code-block:: python

   # ModelManager queries registry
   preprocessor_cls = ModelRegistry.get_preprocessor('SUMMA')
   runner_cls = ModelRegistry.get_runner('SUMMA')
   method_name = ModelRegistry.get_runner_method('SUMMA')  # 'run_summa'

   # Instantiate and execute
   preprocessor = preprocessor_cls(config, logger)
   preprocessor.run_preprocessing()

**Registry Contents:**

.. code-block:: python

   ModelRegistry._preprocessors   # {model_name: class}
   ModelRegistry._runners         # {model_name: class}
   ModelRegistry._postprocessors  # {model_name: class}
   ModelRegistry._visualizers     # {model_name: function}
   ModelRegistry._runner_methods  # {model_name: 'run_method_name'}

**Benefits:**

- Loose coupling: Framework doesn't import model modules directly
- Easy extension: Add models without framework changes
- Third-party plugins: External packages can register components
- Testing: Mock components replace production implementations

Mixin Pattern
-------------

Shared functionality is distributed through composable mixins:

**ConfigurableMixin:**

.. code-block:: python

   class ConfigurableMixin:
       """Provides config access and common properties."""

       @property
       def config(self) -> SymfluenceConfig:
           return self._config

       @property
       def project_dir(self) -> Path:
           return Path(self.config.root.data_dir) / "domain" / self.config.domain.name

       @property
       def experiment_id(self) -> str:
           return self.config.domain.experiment_id

       def _get_config_value(self, accessor, default):
           """Safe config access with fallback."""
           try:
               value = accessor()
               return value if value is not None else default
           except (AttributeError, KeyError):
               return default

**MizuRouteConfigMixin:**

.. code-block:: python

   class MizuRouteConfigMixin:
       """Adds mizuRoute routing capabilities to runners."""

       @property
       def mizu_settings_path(self):
           return self._get_config_value(
               lambda: self.config.path.mizu_settings,
               self.project_dir / 'settings' / 'mizuroute'
           )

       def _run_mizuroute(self, spatial_config, model_name):
           """Execute mizuRoute routing."""
           # Shared routing logic
           pass

**Usage:**

.. code-block:: python

   class HBVRunner(BaseModelRunner, UnifiedModelExecutor, MizuRouteConfigMixin):
       """HBV runner with routing support via mixin."""
       pass

Typed Configuration (Pydantic)
------------------------------

Configuration uses hierarchical Pydantic models for validation:

**Structure:**

.. code-block:: python

   class SymfluenceConfig(BaseModel):
       """Root configuration model."""
       root: RootConfig
       domain: DomainConfig
       forcing: ForcingConfig
       model: ModelConfig
       optimization: OptimizationConfig
       paths: PathConfig
       logging: LoggingConfig

   class DomainConfig(BaseModel):
       name: str
       experiment_id: str = "default"
       definition_method: Literal['polygon', 'delineate', 'merit_basins']
       discretization: str = "GRUs"

   class ModelConfig(BaseModel):
       hydrological_model: str
       summa: Optional[SummaConfig] = None
       fuse: Optional[FuseConfig] = None
       hbv: Optional[HBVConfig] = None
       # ... other models

**Loading & Validation:**

.. code-block:: python

   # From YAML file
   config = SymfluenceConfig.from_file("config.yaml")

   # With CLI overrides
   config = SymfluenceConfig.from_file(
       "config.yaml",
       overrides={'FORCING_DATASET': 'ERA5'},
       use_env=True,  # Allow environment variables
       validate=True  # Run validation rules
   )

   # Convert to flat dict (backward compatibility)
   flat_config = config.to_dict(flatten=True)

**Validation Rules:**

.. code-block:: python

   @field_validator('forcing_dataset')
   @classmethod
   def validate_dataset(cls, v):
       valid = ['ERA5', 'RDRS', 'CARRA', 'CERRA', 'Daymet']
       if v not in valid:
           raise ValueError(f"Invalid dataset: {v}")
       return v

Data Flow
=========

Complete Workflow
-----------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    1. PROJECT INITIALIZATION                            │
   │                                                                         │
   │  ProjectManager.setup()                                                 │
   │  ├── Create directory structure                                        │
   │  ├── Validate configuration                                            │
   │  └── Initialize logging                                                 │
   └───────────────────────────────────┬─────────────────────────────────────┘
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    2. DOMAIN DEFINITION                                 │
   │                                                                         │
   │  DomainManager.define_domain()                                          │
   │  ├── Polygon: Load shapefile → extract pour point                       │
   │  ├── Delineate: DEM + pour point → watershed delineation                │
   │  └── MERIT: Basin ID → pre-computed catchment                           │
   │                                                                         │
   │  DomainManager.discretize()                                             │
   │  ├── Elevation bands: Split by elevation                                │
   │  ├── Radiation: Split by aspect/slope                                   │
   │  └── Combined: Multi-attribute discretization                           │
   └───────────────────────────────────┬─────────────────────────────────────┘
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    3. DATA ACQUISITION                                  │
   │                                                                         │
   │  DataManager.acquire_forcing()                                          │
   │  ├── ERA5: CDS API → download → subset to domain                        │
   │  ├── Observations: GRDC/USGS → align to time window                     │
   │  └── Attributes: Shapefiles → compute basin properties                  │
   │                                                                         │
   │  DataManager.merge_forcing()                                            │
   │  ├── Temporal alignment across datasets                                 │
   │  ├── Spatial interpolation to HRUs                                      │
   │  └── Unit conversion and standardization                                │
   └───────────────────────────────────┬─────────────────────────────────────┘
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    4. MODEL PREPROCESSING                               │
   │                                                                         │
   │  ModelManager.preprocess(model_name)                                    │
   │  ├── Registry lookup: ModelRegistry.get_preprocessor(model_name)        │
   │  ├── Instantiate: preprocessor = PreprocessorClass(config, logger)      │
   │  └── Execute: preprocessor.run_preprocessing()                          │
   │                                                                         │
   │  Model-specific operations:                                             │
   │  ├── SUMMA: Generate forcingFileList.txt, attributes.nc                 │
   │  ├── HBV: Create forcing CSV/NetCDF with PET calculation                │
   │  └── GR: Create R-compatible forcing files                              │
   └───────────────────────────────────┬─────────────────────────────────────┘
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    5. MODEL EXECUTION                                   │
   │                                                                         │
   │  ModelManager.run_models()                                              │
   │  ├── Registry lookup: ModelRegistry.get_runner(model_name)              │
   │  ├── Get method: ModelRegistry.get_runner_method(model_name)            │
   │  ├── Instantiate: runner = RunnerClass(config, logger)                  │
   │  └── Execute: getattr(runner, method_name)()                            │
   │                                                                         │
   │  Execution patterns:                                                    │
   │  ├── Process-based: SUMMA, FUSE (subprocess call)                       │
   │  ├── In-memory: HBV, LSTM (Python/JAX execution)                        │
   │  └── R interface: GR (rpy2 bridge)                                      │
   └───────────────────────────────────┬─────────────────────────────────────┘
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    6. POSTPROCESSING & EVALUATION                       │
   │                                                                         │
   │  ModelManager.postprocess(model_name)                                   │
   │  ├── Registry lookup: ModelRegistry.get_postprocessor(model_name)       │
   │  ├── Extract streamflow from model outputs                              │
   │  └── Standardize to common format (CSV, NetCDF)                         │
   │                                                                         │
   │  AnalysisManager.evaluate()                                             │
   │  ├── Load simulations and observations                                  │
   │  ├── Calculate metrics (KGE, NSE, RMSE, etc.)                           │
   │  └── Generate evaluation report                                         │
   └───────────────────────────────────┬─────────────────────────────────────┘
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    7. CALIBRATION (Optional)                            │
   │                                                                         │
   │  OptimizationManager.run_calibration()                                  │
   │  ├── Initialize optimizer (DE, DDS, ADAM, PSO, NSGA-II)                 │
   │  ├── Create parameter manager for model                                 │
   │  ├── Spawn workers for parallel evaluation                              │
   │  └── Iterate until convergence                                          │
   │                                                                         │
   │  Per iteration:                                                         │
   │  ├── Optimizer proposes parameters                                      │
   │  ├── Worker applies parameters to model                                 │
   │  ├── Worker runs model                                                  │
   │  ├── Worker extracts metrics                                            │
   │  └── Optimizer updates based on fitness                                 │
   └─────────────────────────────────────────────────────────────────────────┘

Model Component Lifecycle
-------------------------

.. code-block:: text

   1. Registration (import time):
      ┌─────────────────────────────────────────────────────┐
      │  @ModelRegistry.register_runner('SUMMA', 'run_summa')│
      │  class SUMMARunner: ...                              │
      │                                                      │
      │  → ModelRegistry._runners['SUMMA'] = SUMMARunner     │
      │  → ModelRegistry._runner_methods['SUMMA'] = 'run_summa'│
      └─────────────────────────────────────────────────────┘

   2. Discovery (workflow time):
      ┌─────────────────────────────────────────────────────┐
      │  runner_cls = ModelRegistry.get_runner('SUMMA')     │
      │  method = ModelRegistry.get_runner_method('SUMMA')  │
      │                                                      │
      │  → runner_cls = SUMMARunner                         │
      │  → method = 'run_summa'                              │
      └─────────────────────────────────────────────────────┘

   3. Instantiation (per-execution):
      ┌─────────────────────────────────────────────────────┐
      │  runner = runner_cls(config, logger, reporting_mgr) │
      │                                                      │
      │  → runner.project_dir = /data/domain/my_basin       │
      │  → runner.experiment_id = 'calibration_001'         │
      └─────────────────────────────────────────────────────┘

   4. Execution:
      ┌─────────────────────────────────────────────────────┐
      │  result = getattr(runner, method)()                 │
      │                                                      │
      │  → runner.run_summa()                               │
      │  → subprocess.run(['summa.exe', config_file])       │
      │  → return output_directory                          │
      └─────────────────────────────────────────────────────┘

Extending SYMFLUENCE
====================

Adding a New Model
------------------

1. **Create model directory:**

   .. code-block:: bash

      mkdir src/symfluence/models/mymodel
      touch src/symfluence/models/mymodel/__init__.py
      touch src/symfluence/models/mymodel/preprocessor.py
      touch src/symfluence/models/mymodel/runner.py
      touch src/symfluence/models/mymodel/postprocessor.py
      touch src/symfluence/models/mymodel/config.py

2. **Implement preprocessor:**

   .. code-block:: python

      # preprocessor.py
      from symfluence.models.base import BaseModelPreProcessor
      from symfluence.models.registry import ModelRegistry

      @ModelRegistry.register_preprocessor('MYMODEL')
      class MyModelPreprocessor(BaseModelPreProcessor):

          def _get_model_name(self) -> str:
              return "MYMODEL"

          def run_preprocessing(self) -> bool:
              self.logger.info("Preprocessing for MYMODEL")
              # Load forcing, convert to model format, write files
              self.create_directories()
              forcing = self._load_forcing()
              self._write_model_inputs(forcing)
              return True

3. **Implement runner:**

   .. code-block:: python

      # runner.py
      from symfluence.models.base import BaseModelRunner
      from symfluence.models.registry import ModelRegistry

      @ModelRegistry.register_runner('MYMODEL', method_name='run_mymodel')
      class MyModelRunner(BaseModelRunner):

          def _get_model_name(self) -> str:
              return "MYMODEL"

          def run_mymodel(self):
              self.logger.info("Running MYMODEL")
              # Execute model (subprocess, library call, etc.)
              result = subprocess.run([self.executable, self.config_file])
              return self.output_dir if result.returncode == 0 else None

4. **Implement postprocessor:**

   .. code-block:: python

      # postprocessor.py
      from symfluence.models.base.standard_postprocessor import StandardModelPostprocessor
      from symfluence.models.registry import ModelRegistry

      @ModelRegistry.register_postprocessor('MYMODEL')
      class MyModelPostprocessor(StandardModelPostprocessor):
          model_name = "MYMODEL"
          output_file_pattern = "{domain}_mymodel_output.nc"
          streamflow_variable = "discharge"
          streamflow_unit = "cms"

5. **Register in __init__.py:**

   .. code-block:: python

      # src/symfluence/models/__init__.py
      from . import mymodel  # Add this line

6. **Add configuration schema:**

   .. code-block:: python

      # config.py
      from pydantic import BaseModel, Field

      class MyModelConfig(BaseModel):
          parameter_a: float = Field(default=1.0, ge=0.0)
          parameter_b: str = Field(default='default')
          spatial_mode: Literal['lumped', 'distributed'] = 'lumped'

Adding a New Optimization Algorithm
-----------------------------------

.. code-block:: python

   # src/symfluence/optimization/optimizers/algorithms/myalgo.py
   from .base_algorithm import BaseAlgorithm

   class MyAlgorithm(BaseAlgorithm):
       """Custom optimization algorithm."""

       def __init__(self, config, logger, **kwargs):
           super().__init__(config, logger, **kwargs)
           # Algorithm-specific initialization

       def optimize(self, objective_function, bounds, **kwargs):
           """Run optimization."""
           # Implement optimization loop
           for iteration in range(self.max_iterations):
               # Propose new parameters
               # Evaluate objective
               # Update population
               pass
           return best_params, best_fitness

Adding a New Data Handler
-------------------------

.. code-block:: python

   # src/symfluence/data/acquisition/handlers/mydata.py
   from .base import BaseDataHandler

   class MyDataHandler(BaseDataHandler):
       """Handler for MyData dataset."""

       dataset_name = "MYDATA"
       variables = ['precipitation', 'temperature']

       def acquire(self, spatial_extent, time_range):
           """Download data for domain and time range."""
           # Connect to data source
           # Download files
           # Return path to downloaded data
           pass

       def preprocess(self, raw_data_path):
           """Convert to standard format."""
           # Regrid, unit conversion, variable renaming
           pass

Configuration System Details
============================

Hierarchical Structure
----------------------

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: my_basin
   EXPERIMENT_ID: calibration_001

   # Nested under 'forcing'
   FORCING_DATASET: ERA5
   FORCING_START_YEAR: 2010
   FORCING_END_YEAR: 2020

   # Nested under 'model.summa'
   SUMMA_SPATIAL_MODE: distributed
   SUMMA_ROUTING_INTEGRATION: mizuRoute

   # Nested under 'optimization'
   OPTIMIZATION_ALGORITHM: DE
   OPTIMIZATION_MAX_ITERATIONS: 5000

**Transformation to typed config:**

.. code-block:: python

   # Flat dict (legacy format)
   flat = {'DOMAIN_NAME': 'my_basin', 'FORCING_DATASET': 'ERA5', ...}

   # Converted to hierarchical
   config = SymfluenceConfig.from_dict(flat)
   config.domain.name  # 'my_basin'
   config.forcing.dataset  # 'ERA5'
   config.model.summa.spatial_mode  # 'distributed'

Environment Variable Support
----------------------------

.. code-block:: bash

   export SYMFLUENCE_DATA_DIR=/data/symfluence
   export ERA5_CDS_API_KEY=xxxxx
   export SUMMA_EXE=/opt/summa/bin/summa.exe

.. code-block:: python

   config = SymfluenceConfig.from_file("config.yaml", use_env=True)
   # Environment variables override YAML values

Error Handling
==============

Exception Hierarchy
-------------------

.. code-block:: text

   SymfluenceError                    # Base exception
   ├── ConfigurationError             # Invalid configuration
   │   ├── ConfigValidationError      # Pydantic validation failure
   │   └── ConfigFileError            # Missing/unreadable config file
   ├── DataError                      # Data acquisition/processing issues
   │   ├── DataAcquisitionError       # Failed to download
   │   └── DataValidationError        # Invalid data format
   ├── ModelError                     # Model execution issues
   │   ├── ModelExecutionError        # Runtime failure
   │   └── ModelConfigError           # Missing model files
   └── OptimizationError              # Calibration issues
       └── ConvergenceError           # Failed to converge

Context Manager Pattern
-----------------------

.. code-block:: python

   from symfluence.core.exceptions import symfluence_error_handler

   with symfluence_error_handler(
       "Model execution",
       self.logger,
       error_type=ModelExecutionError
   ):
       # Protected code block
       result = subprocess.run(cmd)
       if result.returncode != 0:
           raise RuntimeError(f"Model failed: {result.stderr}")

Testing Architecture
====================

Test Organization
-----------------

.. code-block:: text

   tests/
   ├── unit/                    # Fast, isolated tests
   │   ├── core/               # Config, exceptions, utils
   │   ├── models/             # Model components
   │   │   ├── test_hbv_model.py
   │   │   └── test_summa_preprocessor.py
   │   └── optimization/       # Optimizers, workers
   ├── integration/            # Component interaction tests
   │   ├── calibration/        # End-to-end calibration
   │   ├── domain/             # Domain definition
   │   └── preprocessing/      # Data pipeline
   └── e2e/                    # Full workflow tests

Test Fixtures
-------------

.. code-block:: python

   @pytest.fixture
   def mock_config():
       """Provide test configuration."""
       return SymfluenceConfig(
           root=RootConfig(data_dir='/tmp/test'),
           domain=DomainConfig(name='test_basin'),
           ...
       )

   @pytest.fixture
   def sample_forcing():
       """Provide sample forcing data."""
       return xr.Dataset({
           'pr': (['time'], np.random.rand(365)),
           'temp': (['time'], np.random.rand(365) * 20),
       })

Additional Resources
====================

**Internal Documentation:**

- :doc:`developer_guide`: Adding new models, testing
- :doc:`configuration`: Full configuration reference
- :doc:`api`: API reference

**Design Pattern References:**

- Registry Pattern: Gang of Four
- Factory Pattern: Creational design patterns
- Mixin Pattern: Python composition patterns
- Pydantic: https://docs.pydantic.dev/

**Contributing:**

- `Contribution Guidelines <https://github.com/DarriEy/SYMFLUENCE/blob/main/CONTRIBUTING.md>`_
- GitHub: https://github.com/DarriEy/SYMFLUENCE
