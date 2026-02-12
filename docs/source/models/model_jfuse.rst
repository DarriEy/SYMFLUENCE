.. _models_jfuse:

=========================================
JFuse Model Guide (Experimental)
=========================================

.. warning::

   **EXPERIMENTAL MODULE**: JFuse is under active development and the API may change
   without notice. For production use, consider the stable :doc:`model_fuse` module instead.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

JFuse (JAX-based FUSE) is a **differentiable hydrological model** built on the JAX
framework. It provides native automatic differentiation capabilities for efficient
gradient-based parameter calibration, with optional GPU acceleration and JIT compilation.

**Key Capabilities:**

- Differentiable implementation using JAX autodiff
- JIT compilation for fast execution
- GPU acceleration support (optional)
- Multiple model structures with gradient-optimized configurations
- Native gradient computation for ADAM, L-BFGS optimization
- Compatible with evolutionary algorithms

**Typical Applications:**

- Gradient-based hydrological model calibration
- Hybrid physics-ML research
- Sensitivity analysis via automatic differentiation
- Large-scale parameter studies with GPU acceleration
- Uncertainty quantification through gradient information

**Spatial Scales:** Catchment to regional

**Temporal Resolution:** Daily

Model Physics and Structure
===========================

Mathematical Foundation
-----------------------

JFuse implements the FUSE modular framework with JAX arrays, providing:

1. **Upper Zone (Tension + Free Storage):**

   - TENSION2_FREE architecture for gradient support
   - Overflow smoothing for numerical stability

2. **Lower Zone:**

   - Single or dual reservoir configurations
   - Nonlinear baseflow with guaranteed gradients

3. **Surface Runoff:**

   - UZ_PARETO distribution (gradient-friendly)
   - Saturation area dynamics

4. **Snow Module:**

   - Temperature thresholds for rain/snow
   - Variable melt factors (MFMAX, MFMIN)

Model Configurations
--------------------

JFuse provides optimized configurations for gradient-based calibration:

**prms_gradient (Default, Recommended)**

- 14 calibration parameters with guaranteed non-zero gradients
- Upper arch: TENSION2_FREE
- Lower arch: SINGLE_NOEVAP
- Baseflow: NONLINEAR

**max_gradient**

- Maximum parameter coverage (~21 parameters)
- Sacramento-based architecture
- PARALLEL_LINEAR baseflow

Available Structures
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Structure
     - Description
   * - prms_gradient
     - Optimized for gradient-based calibration (default)
   * - max_gradient
     - Maximum parameter coverage
   * - prms
     - Standard PRMS architecture
   * - sacramento
     - Sacramento Soil Moisture Accounting
   * - topmodel
     - TOPMODEL-style structure
   * - vic
     - Variable Infiltration Capacity

Configuration in SYMFLUENCE
===========================

Model Selection
---------------

To use JFuse in your configuration:

.. code-block:: yaml

   HYDROLOGICAL_MODEL: JFUSE

Key Configuration Parameters
----------------------------

Model Structure and Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - JFUSE_MODEL_CONFIG_NAME
     - prms_gradient
     - Model configuration (prms_gradient, max_gradient, etc.)
   * - JFUSE_SPATIAL_MODE
     - auto
     - Spatial mode (lumped, distributed, auto)
   * - JFUSE_N_HRUS
     - 1
     - Number of HRUs (distributed mode)
   * - JFUSE_ENABLE_ROUTING
     - false
     - Enable internal Muskingum-Cunge routing
   * - JFUSE_WARMUP_DAYS
     - 365
     - Spinup period in days
   * - JFUSE_TIMESTEP_DAYS
     - 1.0
     - Model timestep (days)

Runtime Configuration
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - JFUSE_JIT_COMPILE
     - true
     - Enable JAX JIT compilation
   * - JFUSE_USE_GPU
     - false
     - Use GPU acceleration (requires JAX CUDA)
   * - JFUSE_ENABLE_SNOW
     - true
     - Enable snow processes

Calibration Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - JFUSE_USE_GRADIENT_CALIBRATION
     - true
     - Use gradient-based optimization
   * - JFUSE_CALIBRATION_METRIC
     - KGE
     - Objective function (KGE, NSE)
   * - JFUSE_PARAMS_TO_CALIBRATE
     - (14 params)
     - Comma-separated parameter names

Default Calibration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The prms_gradient configuration uses 14 parameters with guaranteed gradients:

.. code-block:: yaml

   JFUSE_PARAMS_TO_CALIBRATE: >
     S1_max,S2_max,ku,ki,ks,n,Ac_max,b,
     f_rchr,T_rain,T_melt,MFMAX,MFMIN,smooth_frac

Parameter bounds:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Parameter
     - Bounds
     - Description
   * - S1_max
     - [10, 500]
     - Upper zone storage capacity (mm)
   * - S2_max
     - [50, 2000]
     - Lower zone storage capacity (mm)
   * - ku
     - [0.01, 0.99]
     - Upper layer drainage coefficient
   * - ki
     - [0.001, 0.5]
     - Interflow coefficient
   * - ks
     - [0.0001, 0.1]
     - Baseflow coefficient
   * - n
     - [0.1, 5.0]
     - TOPMODEL exponent
   * - Ac_max
     - [0.01, 0.99]
     - Max saturated area fraction
   * - b
     - [0.01, 3.0]
     - Surface runoff shape
   * - f_rchr
     - [0.01, 0.99]
     - Recharge fraction
   * - T_rain
     - [-3.0, 3.0]
     - Rain/snow threshold (C)
   * - T_melt
     - [-5.0, 5.0]
     - Melt threshold (C)
   * - MFMAX
     - [1.0, 8.0]
     - Maximum melt factor (mm/C/day)
   * - MFMIN
     - [0.5, 4.0]
     - Minimum melt factor (mm/C/day)
   * - smooth_frac
     - [0.01, 0.5]
     - Overflow smoothing fraction

Initial States
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - JFUSE_INITIAL_S1
     - 0.0
     - Initial upper storage (mm)
   * - JFUSE_INITIAL_S2
     - 50.0
     - Initial lower storage (mm)
   * - JFUSE_INITIAL_SNOW
     - 0.0
     - Initial snow storage (mm SWE)

Routing Configuration
^^^^^^^^^^^^^^^^^^^^^

For distributed mode with internal routing:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Parameter
     - Default
     - Description
   * - JFUSE_ROUTING_SUBSTEP_METHOD
     - adaptive
     - Substep method (fixed, adaptive)
   * - JFUSE_ROUTING_MAX_SUBSTEPS
     - 10
     - Maximum routing substeps (1-100)
   * - JFUSE_DEFAULT_MANNINGS_N
     - 0.035
     - Manning's roughness coefficient
   * - JFUSE_DEFAULT_CHANNEL_SLOPE
     - 0.001
     - Default channel slope

Input Requirements
==================

Forcing Data
------------

JFuse requires daily forcing data:

.. code-block:: text

   precipitation  : Precipitation [mm/day]
   temperature    : Air temperature [C]
   pet            : Potential evapotranspiration [mm/day]

The preprocessor handles:

- Variable name resolution (pr/precip, tas/temp, etc.)
- Unit conversion (K to C, mm/s to mm/day)
- Temporal resampling (hourly to daily)
- PET calculation via Hamon method if not provided

Output Specifications
=====================

Lumped Mode
-----------

**Files:**

- ``{domain}_jfuse_output.csv`` - CSV format
- ``{domain}_jfuse_output.nc`` - NetCDF format

**Variables:**

.. code-block:: text

   streamflow     : Streamflow [m3/s]
   runoff         : Runoff [mm/day]

Distributed Mode
----------------

**Files:**

- ``{domain}_{experiment}_runs_def.nc`` - Per-HRU NetCDF

**Variables:**

.. code-block:: text

   gruId          : GRU identifier
   runoff         : Runoff per HRU [mm/day]

Usage Examples
==============

Basic Configuration
-------------------

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: my_basin
   HYDROLOGICAL_MODEL: JFUSE

   # Time settings
   EXPERIMENT_TIME_START: "2015-01-01"
   EXPERIMENT_TIME_END: "2020-12-31"

   # JFuse configuration
   JFUSE_MODEL_CONFIG_NAME: prms_gradient
   JFUSE_SPATIAL_MODE: lumped
   JFUSE_JIT_COMPILE: true

   # Forcing
   FORCING_DATASET: ERA5

Gradient-Based Calibration with ADAM
------------------------------------

.. code-block:: yaml

   # Enable gradient calibration
   JFUSE_USE_GRADIENT_CALIBRATION: true
   JFUSE_CALIBRATION_METRIC: KGE

   # ADAM optimizer settings
   OPTIMIZATION_ALGORITHM: ADAM
   NUMBER_OF_ITERATIONS: 500

Run calibration:

.. code-block:: bash

   symfluence workflow step calibrate_model --config config.yaml

GPU-Accelerated Calibration
---------------------------

.. code-block:: yaml

   # Enable GPU (requires JAX with CUDA)
   JFUSE_USE_GPU: true
   JFUSE_JIT_COMPILE: true

   # Large population for GPU efficiency
   POPULATION_SIZE: 256
   NUMBER_OF_ITERATIONS: 1000

Python API Usage
----------------

.. code-block:: python

   from symfluence.models.jfuse import (
       JFUSERunner,
       JFUSEPreProcessor,
       JFUSEWorker
   )
   import logging

   logger = logging.getLogger('jfuse')

   # Preprocess forcing
   preprocessor = JFUSEPreProcessor(config, logger)
   preprocessor.run_preprocessing()

   # Run simulation
   runner = JFUSERunner(config, logger)
   output_dir = runner.run_jfuse()

   # Calibration with gradients
   worker = JFUSEWorker(config, logger)
   worker.initialize()

   # Check gradient support
   if worker.supports_native_gradients():
       # Compute gradients
       grads = worker.compute_gradient(params, metric='kge')

       # Or combined evaluation
       loss, grads = worker.evaluate_with_gradient(params, metric='kge')

       # Check gradient coverage
       coverage = worker.check_gradient_coverage(list(params.keys()))

Installation and Dependencies
=============================

Core Dependencies
-----------------

.. code-block:: text

   jfuse          : JFuse Python library
   jax            : JAX for autodiff
   jaxlib         : JAX runtime
   numpy          : Array operations
   equinox        : PyTree manipulation

Installation
------------

.. code-block:: bash

   # Basic installation
   pip install jfuse

   # With gradient support (CPU)
   pip install jfuse jax jaxlib

   # With GPU support (CUDA 11.x)
   pip install jfuse jax[cuda11_cudnn82]

Checking Installation
---------------------

.. code-block:: python

   from symfluence.models.jfuse import check_jfuse_installation

   status = check_jfuse_installation()
   print(f"JFuse installed: {status['jfuse_installed']}")
   print(f"JAX installed: {status['jax_installed']}")
   print(f"Native gradients: {status['native_gradients_available']}")

Known Limitations
=================

1. **Experimental Status:**

   - API may change without notice
   - Use stable FUSE for production

2. **Distributed Routing:**

   - Internal routing not fully integrated
   - Use external mizuRoute for distributed applications

3. **Gradient Coverage:**

   - Not all parameters have non-zero gradients for all structures
   - Use ``check_gradient_coverage()`` to verify
   - prms_gradient configuration optimized for gradients

4. **JAX Dependency:**

   - Gradients unavailable without JAX
   - Falls back to finite differences

5. **Memory:**

   - JAX arrays kept in memory during simulation
   - Large distributed runs may require significant RAM

Comparison with CFuse
=====================

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - JFuse
     - CFuse
   * - Backend
     - JAX
     - PyTorch + Enzyme
   * - GPU Support
     - Native JAX CUDA
     - PyTorch CUDA
   * - JIT Compilation
     - Native JAX
     - torch.compile
   * - Gradient Quality
     - Consistent
     - Depends on Enzyme
   * - Installation
     - Simpler (pip)
     - Requires CMake build

Troubleshooting
===============

Common Issues
-------------

**Error: "jfuse module not found"**

.. code-block:: bash

   pip install jfuse

**Error: "No GPU found"**

JAX defaults to CPU. For GPU:

.. code-block:: bash

   # Check JAX devices
   python -c "import jax; print(jax.devices())"

   # Install CUDA version
   pip install jax[cuda11_cudnn82]

**Slow first run**

JIT compilation occurs on first execution. Subsequent runs are faster.

**Zero gradients for some parameters**

Use ``check_gradient_coverage()`` or switch to ``max_gradient`` configuration:

.. code-block:: yaml

   JFUSE_MODEL_CONFIG_NAME: max_gradient

Additional Resources
====================

- :doc:`model_fuse` - Stable FUSE documentation
- :doc:`model_cfuse` - PyTorch-based CFuse (experimental)
- :doc:`../calibration` - Calibration workflows
- :doc:`../configuration` - Full parameter reference
