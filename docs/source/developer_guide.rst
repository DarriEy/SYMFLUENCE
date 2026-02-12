Developer Guide
===============

This guide provides technical documentation for developers who want to extend SYMFLUENCE, add new models, or contribute to the codebase.

---

Architecture Overview
---------------------

SYMFLUENCE follows a modular, manager-based architecture with clear separation of concerns:

**Core Components**

.. code-block:: text

   symfluence/
   ├── core/                    # Core system and configuration
   │   ├── system.py           # Main SYMFLUENCE class
   │   ├── base_manager.py     # Base class for all managers
   │   ├── config/             # Typed configuration models
   │   └── exceptions.py       # Custom exception hierarchy
   ├── project/                 # Project and workflow management
   │   ├── project_manager.py  # Project initialization
   │   └── workflow_orchestrator.py  # Step orchestration
   ├── data/                    # Data acquisition and preprocessing
   │   ├── data_manager.py     # Data operations facade
   │   ├── acquisition/        # Cloud data acquisition
   │   └── preprocessing/      # Model-agnostic preprocessing
   ├── geospatial/             # Domain definition and discretization
   │   ├── domain_manager.py   # Domain operations
   │   └── discretization/     # HRU generation
   ├── models/                  # Model integrations
   │   ├── model_manager.py    # Model execution coordination
   │   ├── registry.py         # Plugin registration system
   │   └── {model}/            # Model-specific implementations
   ├── optimization/           # Calibration and optimization
   │   ├── optimization_manager.py
   │   └── optimizers/         # Algorithm implementations
   ├── evaluation/             # Performance metrics and analysis
   │   └── analysis_manager.py
   └── reporting/              # Visualization and output
       └── reporting_manager.py

**Design Patterns**

1. **Manager Pattern**: Each major subsystem has a manager class that coordinates operations
2. **Registry Pattern**: Models self-register using decorators (see :doc:`api`)
3. **Mixin Pattern**: Common functionality shared through mixins
4. **Typed Configuration**: Pydantic models for configuration validation

---

Adding a New Hydrological Model
--------------------------------

SYMFLUENCE uses a registry-based plugin system that makes adding new models straightforward.

Step 1: Create Model Directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new directory under ``src/symfluence/models/``:

.. code-block:: bash

   mkdir src/symfluence/models/mymodel
   touch src/symfluence/models/mymodel/__init__.py
   touch src/symfluence/models/mymodel/preprocessor.py
   touch src/symfluence/models/mymodel/runner.py
   touch src/symfluence/models/mymodel/postprocessor.py

Step 2: Implement Preprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a preprocessor that inherits from ``BaseModelPreProcessor``:

.. code-block:: python

   # src/symfluence/models/mymodel/preprocessor.py
   from symfluence.models.base import BaseModelPreProcessor
   from symfluence.models.registry import ModelRegistry

   @ModelRegistry.register_preprocessor('MYMODEL')
   class MyModelPreProcessor(BaseModelPreProcessor):
       \"\"\"
       Preprocessor for MyModel.

       Converts generic forcing data into MyModel-specific input format.
       \"\"\"

       def _get_model_name(self) -> str:
           return "MYMODEL"

       def __init__(self, config, logger):
           super().__init__(config, logger)
           # Initialize model-specific paths and settings
           self.model_input_dir = self.project_dir / 'forcing' / 'MYMODEL_input'
           self.model_input_dir.mkdir(parents=True, exist_ok=True)

       def run_preprocessing(self):
           \"\"\"Main preprocessing entry point.\"\"\"
           self.logger.info("Starting MyModel preprocessing")

           # 1. Load forcing data
           forcing_data = self.load_forcing_data()

           # 2. Transform to model format
           model_input = self.transform_forcing(forcing_data)

           # 3. Generate configuration files
           self.write_config_files()

           # 4. Write model input files
           self.write_model_inputs(model_input)

           self.logger.info("MyModel preprocessing complete")

Step 3: Implement Runner
~~~~~~~~~~~~~~~~~~~~~~~~~

Create a runner that executes the model:

.. code-block:: python

   # src/symfluence/models/mymodel/runner.py
   from pathlib import Path
   from symfluence.models.registry import ModelRegistry

   @ModelRegistry.register_runner('MYMODEL', method_name='run_mymodel')
   class MyModelRunner:
       \"\"\"Runner for MyModel execution.\"\"\"

       def __init__(self, config, logger, reporting_manager=None):
           self.config = config
           self.logger = logger
           self.reporting_manager = reporting_manager
           self.project_dir = Path(config.root.data_dir) / "domain" / config.domain.name

       def run_mymodel(self):
           \"\"\"Execute MyModel simulation.\"\"\"
           self.logger.info("Running MyModel")

           # Build command
           executable = self.get_executable_path()
           config_file = self.project_dir / 'settings' / 'MYMODEL' / 'config.txt'

           cmd = [str(executable), str(config_file)]

           # Execute
           import subprocess
           result = subprocess.run(cmd, capture_output=True, text=True)

           if result.returncode != 0:
               self.logger.error(f"MyModel failed: {result.stderr}")
               raise RuntimeError("MyModel execution failed")

           self.logger.info("MyModel execution complete")

Step 4: Implement Postprocessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a postprocessor to extract results:

.. code-block:: python

   # src/symfluence/models/mymodel/postprocessor.py
   import pandas as pd
   from pathlib import Path
   from symfluence.models.registry import ModelRegistry

   @ModelRegistry.register_postprocessor('MYMODEL')
   class MyModelPostProcessor:
       \"\"\"Postprocessor for MyModel results extraction.\"\"\"

       def __init__(self, config, logger, reporting_manager=None):
           self.config = config
           self.logger = logger
           self.reporting_manager = reporting_manager
           self.project_dir = Path(config.root.data_dir) / "domain" / config.domain.name

       def extract_streamflow(self):
           \"\"\"Extract streamflow from MyModel outputs.\"\"\"
           self.logger.info("Extracting MyModel streamflow")

           # Read model output
           output_file = self.project_dir / 'simulations' / 'mymodel_output.csv'
           df = pd.read_csv(output_file)

           # Standardize format
           results = pd.DataFrame({
               'datetime': pd.to_datetime(df['time']),
               'MYMODEL_discharge_cms': df['flow']
           })
           results.set_index('datetime', inplace=True)

           # Save to standard location
           results_file = self.project_dir / 'results' / f"{self.config.domain.experiment_id}_results.csv"
           results.to_csv(results_file)

           self.logger.info(f"Results saved to {results_file}")

Step 5: Register Model
~~~~~~~~~~~~~~~~~~~~~~~

Update ``src/symfluence/models/__init__.py`` to import your model:

.. code-block:: python

   # Existing imports...
   from . import summa
   from . import fuse
   from . import gr
   from . import hype
   from . import mymodel  # Add this line

   __all__ = ['summa', 'fuse', 'gr', 'hype', 'mymodel']

Step 6: Add Configuration Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add model-specific configuration to ``src/symfluence/core/config/models.py``:

.. code-block:: python

   class MyModelConfig(BaseModel):
       \"\"\"MyModel-specific configuration.\"\"\"
       parameter_a: float = 1.0
       parameter_b: float = 2.0
       use_feature_x: bool = False

   class ModelConfig(BaseModel):
       \"\"\"Model configuration section.\"\"\"
       hydrological_model: Optional[str] = None
       summa: Optional[SummaConfig] = None
       fuse: Optional[FuseConfig] = None
       gr: Optional[GRConfig] = None
       hype: Optional[HypeConfig] = None
       mymodel: Optional[MyModelConfig] = None  # Add this

Step 7: Test Your Model
~~~~~~~~~~~~~~~~~~~~~~~~

Create tests for your model:

.. code-block:: bash

   touch tests/unit/models/test_mymodel_preprocessor.py
   touch tests/integration/models/test_mymodel_integration.py

Example test:

.. code-block:: python

   # tests/unit/models/test_mymodel_preprocessor.py
   import pytest
   from symfluence.models.mymodel.preprocessor import MyModelPreProcessor

   def test_mymodel_preprocessor_initialization(mock_config, mock_logger):
       preprocessor = MyModelPreProcessor(mock_config, mock_logger)
       assert preprocessor._get_model_name() == "MYMODEL"
       assert preprocessor.model_input_dir.exists()

Step 8: Documentation
~~~~~~~~~~~~~~~~~~~~~~

Add your model to the documentation:

1. Update ``docs/source/configuration.rst`` with model-specific parameters
2. Add example configuration in ``src/symfluence/resources/config_templates/``
3. Document in ``docs/source/api.rst`` if API changes

---

Extending Functionality
-----------------------

Adding a New Optimization Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create algorithm class in ``src/symfluence/optimization/optimizers/``:

.. code-block:: python

   from .base_model_optimizer import BaseModelOptimizer

   class MyOptimizer(BaseModelOptimizer):
       \"\"\"My custom optimization algorithm.\"\"\"

       def optimize(self):
           \"\"\"Run optimization.\"\"\"
           # Implementation

2. Register in ``src/symfluence/optimization/optimization_manager.py``

3. Add configuration support

4. Add tests

Adding a New Data Source
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create handler in ``src/symfluence/data/acquisition/handlers/``
2. Inherit from ``BaseDataHandler``
3. Implement ``acquire()`` and ``validate()`` methods
4. Register with ``AcquisitionService``

Adding New Discretization Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create class in ``src/symfluence/geospatial/discretization/attributes/``
2. Follow existing patterns (elevation.py, radiation.py)
3. Register in discretization core
4. Add configuration parameters

---

Testing Guidelines
------------------

SYMFLUENCE uses pytest with multiple test levels:

**Test Organization**

.. code-block:: text

   tests/
   ├── unit/                # Fast, isolated tests
   │   ├── core/
   │   ├── models/
   │   └── optimization/
   ├── integration/         # Component interaction tests
   │   ├── calibration/
   │   ├── domain/
   │   └── preprocessing/
   └── e2e/                # End-to-end workflow tests

**Running Tests**

.. code-block:: bash

   # All tests
   pytest

   # Unit tests only
   pytest tests/unit/

   # Specific module
   pytest tests/unit/models/test_summa_preprocessor.py

   # With coverage
   pytest --cov=symfluence --cov-report=html

   # Specific markers
   pytest -m "not slow"
   pytest -m "requires_data"

**Test Markers**

.. code-block:: python

   @pytest.mark.slow  # Long-running tests
   @pytest.mark.requires_data  # Needs external data
   @pytest.mark.requires_binaries  # Needs model executables
   @pytest.mark.integration  # Integration test
   @pytest.mark.e2e  # End-to-end test

**Writing Good Tests**

.. code-block:: python

   import pytest
   from symfluence.models.summa.preprocessor import SummaPreProcessor

   @pytest.fixture
   def sample_config():
       \"\"\"Provide test configuration.\"\"\"
       return {
           'DOMAIN_NAME': 'test_domain',
           'FORCING_DATASET': 'ERA5',
           # ... other required parameters
       }

   def test_preprocessor_creates_output_directory(sample_config, tmp_path):
       \"\"\"Test that preprocessor creates required directories.\"\"\"
       # Arrange
       config = sample_config.copy()
       config['SYMFLUENCE_DATA_DIR'] = str(tmp_path)

       # Act
       preprocessor = SummaPreProcessor(config, logger)
       preprocessor.run_preprocessing()

       # Assert
       assert (tmp_path / 'forcing' / 'SUMMA_input').exists()

---

Code Style and Standards
-------------------------

**Python Style**

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use Black for formatting (configuration in ``pyproject.toml``)

**Docstring Format**

Use Google-style docstrings:

.. code-block:: python

   def calculate_metrics(observed: np.ndarray, simulated: np.ndarray) -> Dict[str, float]:
       \"\"\"
       Calculate performance metrics for model evaluation.

       Args:
           observed: Array of observed values
           simulated: Array of simulated values

       Returns:
           Dictionary containing metric names and values

       Raises:
           ValueError: If arrays have different lengths

       Example:
           >>> obs = np.array([1, 2, 3])
           >>> sim = np.array([1.1, 2.1, 2.9])
           >>> metrics = calculate_metrics(obs, sim)
           >>> print(metrics['nse'])
           0.95
       \"\"\"

**Import Order**

.. code-block:: python

   # Standard library
   from pathlib import Path
   from typing import Dict, Any

   # Third-party
   import numpy as np
   import pandas as pd

   # Local imports
   from symfluence.core.base_manager import BaseManager
   from symfluence.models.registry import ModelRegistry

---

Configuration System
--------------------

For detailed configuration patterns, see :doc:`configuration`.

Key points:

- Uses Pydantic for type validation
- Immutable configuration objects
- Typed configuration models in ``src/symfluence/core/config/models.py``
- Legacy dict support for backward compatibility

---

Contribution Workflow
---------------------

See the `Contribution Guidelines <https://github.com/DarriEy/SYMFLUENCE/blob/main/CONTRIBUTING.md>`_ for complete information.

**Quick Start**

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/my-feature``
3. Make changes and add tests
4. Run tests: ``pytest``
5. Format code: ``black src/``
6. Commit with descriptive message
7. Push and create pull request to ``develop``

**Pull Request Checklist**

- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Docstrings added/updated
- [ ] Type hints included
- [ ] CHANGELOG.md updated
- [ ] Code formatted with Black

---

Release Process
---------------

SYMFLUENCE follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

**Creating a Release**

1. Update version in ``pyproject.toml``
2. Update CHANGELOG.md
3. Create release branch: ``git checkout -b release/v0.X.0``
4. Run full test suite
5. Merge to ``main``
6. Tag release: ``git tag -a v0.X.0 -m "Release v0.X.0"``
7. Push tags: ``git push --tags``
8. GitHub Actions handles PyPI deployment

---

Additional Resources
--------------------

**Internal Documentation**

- :doc:`api` — API reference with autodoc
- :doc:`configuration` — Configuration system usage
- TESTING.md — Comprehensive testing guide (in tests/ directory)

**External Resources**

- `SUMMA Documentation <https://summa.readthedocs.io/>`_
- `FUSE Documentation <https://naddor.github.io/fuse/>`_
- `Pydantic Documentation <https://docs.pydantic.dev/>`_
- `Pytest Documentation <https://docs.pytest.org/>`_

**Community**

- GitHub Issues: https://github.com/DarriEy/SYMFLUENCE/issues
- GitHub Discussions: https://github.com/DarriEy/SYMFLUENCE/discussions
- `Contributing Guide <https://github.com/DarriEy/SYMFLUENCE/blob/main/CONTRIBUTING.md>`_

---

Getting Help
------------

**For Development Questions:**

1. Check existing documentation and examples
2. Search GitHub issues for similar questions
3. Ask in GitHub Discussions
4. Open an issue with ``[dev]`` tag

**For Bug Reports:**

Include:
- SYMFLUENCE version
- Python version
- Operating system
- Minimal reproducible example
- Full error traceback
- Steps to reproduce
