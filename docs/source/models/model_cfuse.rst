.. _models_cfuse:

=========================================
CFuse Model Guide (Experimental)
=========================================

.. warning::

   **EXPERIMENTAL MODULE**: CFuse is under active development and the API may change
   without notice. For production use, consider the stable :doc:`model_fuse` module instead.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

CFuse (Conceptual FUSE) is a **differentiable, PyTorch-based implementation** of the FUSE
(Framework for Understanding Structural Errors) hydrological model. It supports automatic
differentiation (AD) for gradient-based calibration, enabling efficient parameter optimization
using modern machine learning techniques.

**Key Capabilities:**

- Differentiable model implementation using PyTorch
- Native gradient computation via Enzyme AD (optional)
- Multiple model structures (PRMS, Sacramento, TOPMODEL, VIC, ARNO)
- Lumped and distributed spatial modes
- Gradient-based optimization (ADAM, L-BFGS)
- Compatible with evolutionary optimization algorithms

**Typical Applications:**

- Research on differentiable hydrological modeling
- Gradient-based parameter calibration
- Hybrid physics-ML modeling experiments
- Model structure sensitivity analysis
- Rapid prototyping of calibration workflows

**Spatial Scales:** Catchment to regional

**Temporal Resolution:** Daily (sub-daily experimental)

**Repository:** https://github.com/DarriEy/cFUSE

Model Physics and Structure
===========================

Mathematical Foundation
-----------------------

CFuse implements the FUSE modular framework with PyTorch tensors, enabling:

1. **Upper Zone Storage:**

   - Tension and free storage components
   - Configurable overflow and drainage

2. **Lower Zone Storage:**

   - Single or dual reservoir options
   - Linear and nonlinear baseflow

3. **Surface Runoff:**

   - Saturation excess (TOPMODEL-style)
   - Infiltration excess (Horton-style)
   - Pareto distribution options

4. **Snow Module:**

   - Temperature-based rain/snow partitioning
   - Degree-day melt with variable melt factors

Available Model Structures
--------------------------

CFuse supports five pre-defined model structures:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Structure
     - Description
   * - prms
     - PRMS-style model with tension storage (recommended for gradients)
   * - sacramento
     - Sacramento Soil Moisture Accounting structure
   * - topmodel
     - TOPMODEL-style saturation excess runoff
   * - vic
     - Variable Infiltration Capacity structure
   * - arno
     - ARNO/VIC baseflow formulation

Spatial Modes
-------------

**Lumped Mode:**

- Single catchment simulation
- All forcing averaged to catchment scale
- Fastest execution

**Distributed Mode:**

- Per-HRU simulation with batch processing
- Forcing extracted per spatial unit
- Optional internal routing

Configuration in SYMFLUENCE
===========================

Model Selection
---------------

To use CFuse in your configuration:

.. code-block:: yaml

   HYDROLOGICAL_MODEL: CFUSE

Key Configuration Parameters
----------------------------

Model Structure and Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - CFUSE_MODEL_STRUCTURE
     - prms
     - Model structure (prms, sacramento, topmodel, vic, arno)
   * - CFUSE_SPATIAL_MODE
     - auto
     - Spatial mode (lumped, distributed, auto)
   * - CFUSE_ENABLE_SNOW
     - true
     - Enable snow accumulation and melt
   * - CFUSE_WARMUP_DAYS
     - 365
     - Number of spinup days (discarded)
   * - CFUSE_TIMESTEP_DAYS
     - 1.0
     - Model timestep in days (0.01-1.0)
   * - CFUSE_DEVICE
     - cpu
     - PyTorch device (cpu, cuda)

Gradient Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - CFUSE_USE_NATIVE_GRADIENTS
     - true
     - Use Enzyme AD for gradient computation
   * - CFUSE_USE_GRADIENT_CALIBRATION
     - true
     - Use gradient-based optimization
   * - CFUSE_CALIBRATION_METRIC
     - KGE
     - Objective function (KGE, NSE)

Calibration Parameters
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - CFUSE_PARAMS_TO_CALIBRATE
     - (14 params)
     - Comma-separated parameter list
   * - CFUSE_SPATIAL_PARAMS
     - false
     - Enable per-HRU parameter calibration

Default calibration parameters (14-parameter set):

- **Storage**: S1_max, S2_max
- **Drainage**: ku, ki, ks, n
- **Saturation**: Ac_max, b
- **Recharge**: f_rchr
- **Snow**: T_rain, T_melt, MFMAX, MFMIN
- **Numerical**: smooth_frac

Initial States
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - CFUSE_INITIAL_S1
     - 50.0
     - Initial upper storage (mm)
   * - CFUSE_INITIAL_S2
     - 200.0
     - Initial lower storage (mm)
   * - CFUSE_INITIAL_SNOW
     - 0.0
     - Initial snow storage (mm SWE)

Input Requirements
==================

Forcing Data
------------

CFuse requires daily forcing data:

.. code-block:: text

   precipitation  : Precipitation [mm/day]
   temperature    : Air temperature [C] (auto-converts from K)
   pet            : Potential evapotranspiration [mm/day] (calculated if missing)

The preprocessor automatically:

- Handles multiple variable naming conventions
- Converts units (K to C, mm/s to mm/day)
- Resamples hourly data to daily
- Calculates PET via Hamon method if not provided

Observation Data
----------------

For calibration:

.. code-block:: text

   streamflow     : Observed streamflow [m3/s or mm/day]

Output Specifications
=====================

Lumped Mode Output
------------------

**Files:**

- ``{domain}_cfuse_output.csv`` - CSV with datetime, streamflow
- ``{domain}_cfuse_output.nc`` - NetCDF with full output

**Variables:**

.. code-block:: text

   streamflow_cms    : Streamflow [m3/s]
   streamflow_mm_day : Streamflow [mm/day]
   runoff            : Total runoff [mm/day]

Distributed Mode Output
-----------------------

**Files:**

- ``{domain}_{experiment}_runs_def.nc`` - NetCDF with per-HRU runoff

**Variables:**

.. code-block:: text

   gruId    : GRU identifier
   runoff   : Runoff per HRU [mm/day]

Usage Examples
==============

Basic Configuration
-------------------

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: my_basin
   HYDROLOGICAL_MODEL: CFUSE

   # Time period
   EXPERIMENT_TIME_START: "2015-01-01"
   EXPERIMENT_TIME_END: "2020-12-31"

   # CFuse configuration
   CFUSE_MODEL_STRUCTURE: prms
   CFUSE_SPATIAL_MODE: lumped
   CFUSE_WARMUP_DAYS: 365

   # Forcing
   FORCING_DATASET: ERA5

Gradient-Based Calibration
--------------------------

.. code-block:: yaml

   # Enable gradient optimization
   CFUSE_USE_GRADIENT_CALIBRATION: true
   CFUSE_USE_NATIVE_GRADIENTS: true
   CFUSE_CALIBRATION_METRIC: KGE

   # Use ADAM optimizer
   OPTIMIZATION_ALGORITHM: ADAM

   # Or L-BFGS
   OPTIMIZATION_ALGORITHM: LBFGS

Run the calibration:

.. code-block:: bash

   symfluence workflow step calibrate_model --config config.yaml

Python API Usage
----------------

.. code-block:: python

   from symfluence.models.cfuse import (
       CFUSERunner,
       CFUSEPreProcessor,
       CFUSEWorker
   )
   import logging

   logger = logging.getLogger('cfuse')

   # Preprocess forcing data
   preprocessor = CFUSEPreProcessor(config, logger)
   preprocessor.run_preprocessing()

   # Run simulation
   runner = CFUSERunner(config, logger)
   output_dir = runner.run_cfuse()

   # For calibration with gradients
   worker = CFUSEWorker(config, logger)
   worker.initialize()

   if worker.supports_native_gradients():
       loss, gradients = worker.evaluate_with_gradient(
           params={'S1_max': 300.0, 'S2_max': 1500.0, ...},
           metric='kge'
       )

Installation and Dependencies
=============================

Core Dependencies
-----------------

.. code-block:: text

   cfuse         : Main Python module
   cfuse_core    : C++ compiled module
   torch         : PyTorch for gradient computation
   numpy         : Array operations

Optional Dependencies
---------------------

.. code-block:: text

   Enzyme AD     : For native automatic differentiation
   CUDA          : For GPU acceleration

Installation
------------

From repository:

.. code-block:: bash

   git clone https://github.com/DarriEy/cFUSE.git
   cd cFUSE
   mkdir build && cd build
   cmake .. -DDFUSE_BUILD_PYTHON=ON [-DDFUSE_USE_ENZYME=ON]
   make -j$(nproc)
   cd ..
   pip install -e . --no-deps

Checking Installation
---------------------

.. code-block:: python

   from symfluence.models.cfuse import check_cfuse_installation

   status = check_cfuse_installation()
   print(f"CFuse installed: {status['cfuse_installed']}")
   print(f"Native gradients: {status['native_gradients_available']}")

Known Limitations
=================

1. **Experimental Status:**

   - API may change without notice
   - Not recommended for production workflows
   - Use stable FUSE module for critical applications

2. **Enzyme AD:**

   - Requires exact Clang version match
   - Falls back to numerical gradients if unavailable
   - GPU support requires CUDA-compatible PyTorch

3. **Distributed Mode:**

   - Per-HRU calibration may have edge cases
   - External routing recommended (mizuRoute)

4. **Output:**

   - Only daily output currently supported
   - Sub-daily output experimental

Comparison with Stable FUSE
===========================

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Feature
     - CFuse (Experimental)
     - FUSE (Stable)
   * - Gradients
     - Native (PyTorch/Enzyme)
     - Numerical only
   * - Optimization
     - ADAM, L-BFGS, evolutionary
     - Evolutionary only
   * - API Stability
     - May change
     - Stable
   * - Production Ready
     - No
     - Yes
   * - GPU Support
     - Yes (optional)
     - No

For production use, consider the stable :doc:`model_fuse` module.

Troubleshooting
===============

Common Issues
-------------

**Error: "cfuse module not found"**

.. code-block:: bash

   # Verify installation
   python -c "import cfuse; print(cfuse.__version__)"

   # Reinstall if needed
   pip install -e /path/to/cFUSE --no-deps

**Error: "Enzyme AD not available"**

The model will fall back to numerical gradients. For native gradients:

1. Install Enzyme AD with matching Clang version
2. Rebuild cFUSE with ``-DDFUSE_USE_ENZYME=ON``

**Slow gradient computation**

- Ensure JIT compilation is enabled
- Use CUDA device if available: ``CFUSE_DEVICE: cuda``
- Reduce warmup period for testing

Additional Resources
====================

- :doc:`model_fuse` - Stable FUSE module documentation
- :doc:`../calibration` - Calibration workflows
- :doc:`../configuration` - Full parameter reference
- GitHub: https://github.com/DarriEy/cFUSE
