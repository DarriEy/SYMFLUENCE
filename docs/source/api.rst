API Reference
=============

Overview
--------
The SYMFLUENCE Python API mirrors the CLI and provides programmatic access to the full workflow.
The primary entry point is the ``SYMFLUENCE`` class, which coordinates manager components and the
``WorkflowOrchestrator``-driven step sequence.

Quick Start
-----------
.. code-block:: python

   from symfluence import SYMFLUENCE

   conf = SYMFLUENCE("my_project.yaml")
   conf.run_workflow()  # executes the orchestrated end-to-end pipeline

Managers and Responsibilities
-----------------------------
Internally, SYMFLUENCE composes several managers, coordinated by the workflow orchestrator:

- ``project`` — project structure, pour point creation
- ``domain`` — domain definition, discretization
- ``data`` — attributes/forcing acquisition, observed data, model-agnostic preprocessing
- ``model`` — model-specific preprocessing, simulation runs, post-processing
- ``optimization`` — calibration and emulation
- ``analysis`` — benchmarking, decision and sensitivity analyses

Core Methods (by Stage)
-----------------------

Project Setup
~~~~~~~~~~~~~
.. code-block:: python

   conf.setup_project()          # project manager
   conf.create_pour_point()      # project manager

Domain and Data
~~~~~~~~~~~~~~~
.. code-block:: python

   conf.acquire_attributes()     # data manager
   conf.define_domain()          # domain manager
   conf.discretize_domain()      # domain manager

Observed/Forcing Data & Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   conf.process_observed_data()                 # data manager
   conf.acquire_forcings()                      # data manager
   conf.run_model_agnostic_preprocessing()      # data manager

Model Execution and Post-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   conf.preprocess_models()      # model manager
   conf.run_models()             # model manager
   conf.postprocess_results()    # model manager

Optimization and Emulation
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   conf.calibrate_model()        # optimization manager
   conf.run_emulation()          # optimization manager

Analyses (Optional)
~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   conf.run_benchmarking()       # analysis manager
   conf.run_decision_analysis()  # analysis manager
   conf.run_sensitivity_analysis()  # analysis manager

End-to-End Orchestration
------------------------
The orchestrated pipeline executes steps in order, skipping completed steps unless forced.

.. code-block:: python

   conf.run_workflow(force_run=False)  # force_run=True to recompute all steps

Configuration flags affecting orchestration include:
- ``FORCE_RUN_ALL_STEPS`` (YAML) and ``force_run`` (API) to recompute outputs
- ``STOP_ON_ERROR`` to stop or continue on failure
- ``MPI_PROCESSES`` and model-level parallel settings for scalable runs

Status and Monitoring
---------------------
Programmatically query progress using the orchestrator's status:

.. code-block:: python

   status = conf.get_workflow_status()
   # returns {total_steps, completed_steps, pending_steps, step_details: [...]}

Logging
-------
Set verbosity in YAML (e.g., ``LOG_LEVEL: INFO``) and inspect logs under
``_workLog_<domain_name>/`` for step-by-step diagnostics.

Extending SYMFLUENCE
--------------------
SYMFLUENCE is designed for extensibility. The most common extension point is adding support for a new hydrological model.

Adding a New Model
~~~~~~~~~~~~~~~~~~
As of v0.5.6, SYMFLUENCE uses a **Model Registry** system. To add a new model:

1. Create a new utility module in ``src/symfluence/models/``.
2. Use the ``ModelRegistry`` decorators to register your preprocessor, runner, and postprocessor classes:

   .. code-block:: python

      from .registry import ModelRegistry

      @ModelRegistry.register_preprocessor('MY_MODEL')
      class MyPreProcessor:
          def __init__(self, config, logger): ...
          def run_preprocessing(self): ...

      @ModelRegistry.register_runner('MY_MODEL', method_name='run_my_model')
      class MyRunner:
          def __init__(self, config, logger): ...
          def run_my_model(self): ...

      @ModelRegistry.register_postprocessor('MY_MODEL')
      class MyPostProcessor:
          def __init__(self, config, logger): ...
          def extract_streamflow(self): ...

3. Import your module in ``src/symfluence/models/__init__.py`` to ensure registration.

The ``ModelManager`` will then automatically support your model if it is listed in the ``HYDROLOGICAL_MODEL`` configuration parameter.

Other Extensions
~~~~~~~~~~~~~~~~
- **Optimization strategies:** Add new strategies under ``src/symfluence/optimization/`` and expose them via the ``OptimizationManager``.
- **Analyses:** Create new analysis modules in ``src/symfluence/evaluation/`` and wire them into the orchestrated sequence in ``WorkflowOrchestrator``.

Core Classes
------------

SYMFLUENCE (Main Class)
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: symfluence.core.system.SYMFLUENCE
   :members:
   :undoc-members:
   :show-inheritance:

Manager Classes
---------------

Project Manager
~~~~~~~~~~~~~~~

.. automodule:: symfluence.project.project_manager
   :members:
   :undoc-members:
   :show-inheritance:

Data Manager
~~~~~~~~~~~~

.. automodule:: symfluence.data.data_manager
   :members:
   :undoc-members:
   :show-inheritance:

Model Manager
~~~~~~~~~~~~~

.. automodule:: symfluence.models.model_manager
   :members:
   :undoc-members:
   :show-inheritance:

Workflow Orchestrator
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: symfluence.project.workflow_orchestrator
   :members:
   :undoc-members:
   :show-inheritance:

Model Registry
--------------

.. automodule:: symfluence.models.registry
   :members:
   :undoc-members:
   :show-inheritance:

Optimization
------------

Base Optimizer
~~~~~~~~~~~~~~

.. automodule:: symfluence.optimization.optimizers.base_model_optimizer
   :members:
   :undoc-members:
   :show-inheritance:

DDS Optimizer
~~~~~~~~~~~~~

.. automodule:: symfluence.optimization.optimizers.dds_optimizer
   :members:
   :undoc-members:
   :show-inheritance:

Data Acquisition
----------------

Acquisition Service
~~~~~~~~~~~~~~~~~~~

.. automodule:: symfluence.data.acquisition.acquisition_service
   :members:
   :undoc-members:
   :show-inheritance:

Geospatial Operations
---------------------

Domain Discretization
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: symfluence.geospatial.discretization.core
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation
----------

Base Evaluator
~~~~~~~~~~~~~~

.. automodule:: symfluence.evaluation.evaluators.base
   :members:
   :undoc-members:
   :show-inheritance:

References
----------
- :doc:`getting_started` — high-level orchestration and usage
- :doc:`configuration` — configuration schema and examples
- :doc:`examples` — runnable tutorials
