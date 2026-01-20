=========================================
Configuration System Architecture
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

SYMFLUENCE uses a layered configuration system that combines flexibility with type safety. Configuration flows through multiple stages from YAML files to validated, typed Python objects.

**Design Goals:**

- Type-safe configuration with runtime validation
- Backwards compatibility with legacy configurations
- Environment variable overrides for deployment flexibility
- Clear error messages with actionable suggestions
- Hierarchical organization matching workflow stages

Configuration Flow
==================

Configuration processing follows a pipeline:

.. code-block:: text

   ┌─────────────────┐
   │  YAML File      │  Raw configuration file
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  Load & Parse   │  yaml.safe_load()
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  Normalize      │  Key aliases, type coercion
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  Env Override   │  SYMFLUENCE_* variables
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  Validate       │  Pydantic schema validation
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  SymfluenceConfig │  Typed configuration object
   └─────────────────┘

Core Components
===============

SymfluenceConfig
----------------

The primary configuration class using Pydantic models:

.. code-block:: python

   from symfluence.core.config import SymfluenceConfig

   # Load from file
   config = SymfluenceConfig.from_file("my_project.yaml")

   # Access typed attributes
   domain_name = config.domain.name
   model = config.model.hydrological_model
   start_time = config.experiment.time_start

   # Convert to flat dictionary (legacy compatibility)
   config_dict = config.to_dict(flatten=True)

**Hierarchical Structure:**

.. code-block:: python

   SymfluenceConfig
   ├── root           # Root paths (data_dir, code_dir)
   ├── domain         # Domain settings (name, definition, discretization)
   ├── experiment     # Experiment settings (id, time range)
   ├── forcing        # Forcing data settings (dataset, variables)
   ├── model          # Model configuration (hydrological, routing)
   │   ├── summa      # SUMMA-specific settings
   │   ├── fuse       # FUSE-specific settings
   │   ├── gr         # GR-specific settings
   │   ├── hype       # HYPE-specific settings
   │   ├── ngen       # NGEN-specific settings
   │   └── ...
   ├── optimization   # Calibration settings
   ├── analysis       # Evaluation and analysis settings
   └── output         # Output and logging settings

Configuration Loader
--------------------

The ``config_loader`` module handles the loading pipeline:

.. code-block:: python

   from symfluence.core.config.config_loader import (
       normalize_config,
       validate_config
   )

   # Normalize raw config
   raw_config = yaml.safe_load(open("config.yaml"))
   normalized = normalize_config(raw_config)

   # Validate
   errors = validate_config(normalized)
   if errors:
       print(errors)

Defaults Registry
-----------------

Model-specific defaults are managed centrally:

.. code-block:: python

   from symfluence.core.config import DefaultsRegistry

   # Get defaults for a model
   summa_defaults = DefaultsRegistry.get_defaults("SUMMA")

   # Register custom defaults
   DefaultsRegistry.register("MY_MODEL", {
       "PARAM_A": 1.0,
       "PARAM_B": "default_value"
   })

Key Normalization
=================

All configuration keys are normalized for consistency:

Uppercase Conversion
--------------------

Keys are converted to uppercase:

.. code-block:: yaml

   # Input (any case)
   domain_name: my_basin
   Domain_Name: my_basin
   DOMAIN_NAME: my_basin

   # All become:
   DOMAIN_NAME: my_basin

Alias Mapping
-------------

Legacy and alternative names are mapped:

.. code-block:: text

   Legacy Name                  →  Current Name
   ─────────────────────────────────────────────
   CONFLUENCE_DATA_DIR          →  SYMFLUENCE_DATA_DIR
   CONFLUENCE_CODE_DIR          →  SYMFLUENCE_CODE_DIR
   OPTIMISATION_METHODS         →  OPTIMIZATION_METHODS
   OPTIMISATION_TARGET          →  OPTIMIZATION_TARGET
   GR_SPATIAL                   →  GR_SPATIAL_MODE
   OPTIMIZATION_ALGORITHM       →  ITERATIVE_OPTIMIZATION_ALGORITHM

Type Coercion
-------------

String values are coerced to appropriate types:

.. code-block:: yaml

   # Booleans
   FORCE_RUN: "true"     # → True
   FORCE_RUN: "yes"      # → True
   FORCE_RUN: "1"        # → True
   FORCE_RUN: "false"    # → False

   # None values
   OPTIONAL_PARAM: "none"  # → None
   OPTIONAL_PARAM: "null"  # → None
   OPTIONAL_PARAM: ""      # → None

   # Numbers
   POPULATION_SIZE: "48"     # → 48 (int)
   LEARNING_RATE: "0.001"    # → 0.001 (float)

   # Lists
   PARAMS_TO_CALIBRATE: "k_snow,theta_sat,fcapil"
   # → ["k_snow", "theta_sat", "fcapil"]

Environment Variables
=====================

Configuration can be overridden via environment variables:

Setting Overrides
-----------------

Use the ``SYMFLUENCE_`` prefix:

.. code-block:: bash

   # Override domain name
   export SYMFLUENCE_DOMAIN_NAME="production_basin"

   # Override model
   export SYMFLUENCE_HYDROLOGICAL_MODEL="SUMMA"

   # Override experiment ID
   export SYMFLUENCE_EXPERIMENT_ID="run_$(date +%Y%m%d)"

   # Override numeric values
   export SYMFLUENCE_POPULATION_SIZE="100"

   # Override boolean
   export SYMFLUENCE_FORCE_RUN_ALL_STEPS="true"

Precedence
----------

Environment variables take highest precedence:

.. code-block:: text

   1. Environment variables (highest)
   2. YAML configuration file
   3. Model-specific defaults
   4. System defaults (lowest)

Validation
==========

Required Fields
---------------

Eight fields are mandatory:

.. code-block:: yaml

   # Core paths
   SYMFLUENCE_DATA_DIR: /path/to/data
   SYMFLUENCE_CODE_DIR: /path/to/code

   # Domain identification
   DOMAIN_NAME: bow_at_banff
   EXPERIMENT_ID: run_001

   # Temporal extent
   EXPERIMENT_TIME_START: "2018-01-01 00:00"
   EXPERIMENT_TIME_END: "2018-12-31 23:00"

   # Spatial configuration
   DOMAIN_DEFINITION_METHOD: semidistributed
   SUBSET_FROM_GEOFABRIC: true
   SUB_GRID_DISCRETIZATION: GRUs

   # Model selection
   HYDROLOGICAL_MODEL: SUMMA
   FORCING_DATASET: ERA5

Type Validation
---------------

Pydantic validates field types:

.. code-block:: python

   # These will fail validation:
   POPULATION_SIZE: "not_a_number"  # Expected int
   EXPERIMENT_TIME_START: "invalid"  # Expected datetime
   HYDROLOGICAL_MODEL: "UNKNOWN"     # Not in allowed values

Enum Validation
---------------

Certain fields have restricted values:

.. code-block:: yaml

   # HYDROLOGICAL_MODEL must be one of:
   # SUMMA, FUSE, GR, HYPE, NGEN, LSTM, GNN, MESH, RHESSYS

   # FORCING_DATASET must be one of:
   # ERA5, ERA5-Land, RDRS, CARRA, AORC, CONUS404, HRRR, ...

   # DOMAIN_DEFINITION_METHOD must be one of:
   # point, lumped, semidistributed, distributed

Error Messages
--------------

Validation errors include helpful context:

.. code-block:: text

   ======================================================================
   Configuration Validation Failed
   ======================================================================

   Missing Required Fields:
   ----------------------------------------------------------------------
     ✗ DOMAIN_NAME
       Required field not provided

   Invalid Field Values:
   ----------------------------------------------------------------------
     ✗ HYDROLOGICAL_MODEL
       Value: "UNKNOWN_MODEL"
       Expected: One of [SUMMA, FUSE, GR, HYPE, NGEN, ...]

   Possible Typos:
   ----------------------------------------------------------------------
     ? DOMIAN_NAME → Did you mean DOMAIN_NAME?

   ======================================================================
   See documentation: https://symfluence.readthedocs.io/configuration
   Template available: symfluence init --template
   ======================================================================

Working with Configuration
==========================

Loading Configuration
---------------------

.. code-block:: python

   from symfluence.core.config import SymfluenceConfig

   # From file
   config = SymfluenceConfig.from_file("project.yaml")

   # From dictionary
   config = SymfluenceConfig(**config_dict)

   # With validation
   config = SymfluenceConfig.model_validate(config_dict)

Accessing Values
----------------

.. code-block:: python

   # Typed access (recommended)
   domain = config.domain.name
   model = config.model.hydrological_model
   start = config.experiment.time_start

   # Flat dictionary access (legacy)
   config_dict = config.to_dict(flatten=True)
   domain = config_dict["DOMAIN_NAME"]

   # Model-specific settings
   if config.model.summa:
       filemanager = config.model.summa.filemanager

Modifying Configuration
-----------------------

.. code-block:: python

   # Create modified copy
   new_config = config.model_copy(update={
       "domain": {"name": "new_domain"}
   })

   # Update experiment
   config.experiment.id = "new_experiment"

Converting Formats
------------------

.. code-block:: python

   # To flat dictionary
   flat_dict = config.to_dict(flatten=True)

   # To hierarchical dictionary
   nested_dict = config.model_dump()

   # To YAML
   import yaml
   yaml_str = yaml.dump(config.to_dict(flatten=True))

Best Practices
==============

1. **Use Typed Config**: Prefer ``SymfluenceConfig`` over raw dictionaries
2. **Validate Early**: Validate configuration at startup
3. **Environment for Secrets**: Use env vars for paths and credentials
4. **Version Control**: Keep configuration in version control
5. **Templates**: Start from provided templates
6. **Documentation**: Document custom parameters

Example Configuration
---------------------

.. code-block:: yaml

   # ============================================
   # SYMFLUENCE Project Configuration
   # ============================================

   # --- Core Paths ---
   SYMFLUENCE_DATA_DIR: /data/projects
   SYMFLUENCE_CODE_DIR: /opt/symfluence

   # --- Domain Definition ---
   DOMAIN_NAME: bow_at_banff
   EXPERIMENT_ID: calibration_2024
   POUR_POINT_COORDS: "51.1784/-115.5708"
   BOUNDING_BOX_COORDS: "51.5/-116.0/50.8/-115.0"

   # --- Temporal Settings ---
   EXPERIMENT_TIME_START: "2018-01-01 00:00"
   EXPERIMENT_TIME_END: "2020-12-31 23:00"
   CALIBRATION_PERIOD: "2018-01-01,2019-12-31"
   EVALUATION_PERIOD: "2020-01-01,2020-12-31"

   # --- Spatial Configuration ---
   DOMAIN_DEFINITION_METHOD: semidistributed
   SUBSET_FROM_GEOFABRIC: true
   SUB_GRID_DISCRETIZATION: GRUs

   # --- Model Selection ---
   HYDROLOGICAL_MODEL: SUMMA
   ROUTING_MODEL: MIZUROUTE
   FORCING_DATASET: ERA5

   # --- Optimization ---
   OPTIMIZATION_ALGORITHM: DE
   OPTIMIZATION_METRIC: KGE
   POPULATION_SIZE: 48
   NUMBER_OF_ITERATIONS: 50

See Also
========

- :doc:`configuration` — Complete parameter reference
- :doc:`developer_guide` — Extending the configuration system
- :doc:`api` — Programmatic configuration access
