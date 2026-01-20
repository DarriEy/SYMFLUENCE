.. _routing_droute:

=======================
dRoute Routing Model
=======================

.. warning::
    **EXPERIMENTAL** - The dRoute integration is in active development and should be
    used at your own risk. The API may change without notice in future releases.

Overview
========

dRoute is a C++ river routing library with Python bindings that enables:

- **Automatic differentiation** for gradient-based calibration
- **Multiple routing methods** (Muskingum-Cunge, IRF, Lag, Diffusive Wave, KWT)
- **Native Python API** for fast in-memory routing (no subprocess overhead)
- **mizuRoute-compatible** network topology format for seamless model switching

Key Features
------------

1. **Gradient-Based Calibration**: Native AD support via CoDiPack or Enzyme enables
   efficient gradient computation for routing parameters, enabling ~15x faster
   calibration compared to finite differences.

2. **Multiple Routing Schemes**:

   - ``muskingum_cunge``: Well-tested Muskingum-Cunge method (default)
   - ``irf``: Impulse Response Function routing
   - ``lag``: Simple lag routing
   - ``diffusive_wave``: Diffusive wave approximation
   - ``kwt``: Kinematic Wave Tracking

3. **mizuRoute Compatibility**: Uses the same network topology format as mizuRoute,
   allowing easy switching between routing models without re-preprocessing.

Installation
============

dRoute requires separate installation of the C++ library with Python bindings.

Using SYMFLUENCE Tool Installer
-------------------------------

.. code-block:: bash

    symfluence tools install droute

This will:

1. Clone the dRoute repository
2. Build with CMake (Python bindings + AD enabled)
3. Install Python package

Manual Installation
-------------------

.. code-block:: bash

    # Clone repository
    git clone https://github.com/your-org/droute.git
    cd droute

    # Build with CMake
    mkdir build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DDROUTE_BUILD_PYTHON=ON \
        -DDROUTE_ENABLE_AD=ON \
        -DDROUTE_AD_BACKEND=codipack

    make -j$(nproc)
    make install

    # Install Python bindings
    pip install .

Verify installation:

.. code-block:: bash

    python -c "import droute; print(droute.__version__)"

Configuration
=============

Basic Configuration
-------------------

.. code-block:: yaml

    model:
      hydrological_model: SUMMA
      routing_model: DROUTE

      droute:
        routing_method: muskingum_cunge
        routing_dt: 3600  # seconds
        topology_format: netcdf

Configuration Options
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``execution_mode``
     - ``python``
     - Execution mode: ``python`` (preferred) or ``subprocess``
   * - ``routing_method``
     - ``muskingum_cunge``
     - Routing scheme: ``muskingum_cunge``, ``irf``, ``lag``, ``diffusive_wave``, ``kwt``
   * - ``routing_dt``
     - ``3600``
     - Routing timestep in seconds
   * - ``enable_gradients``
     - ``false``
     - Enable AD for gradient-based calibration
   * - ``ad_backend``
     - ``codipack``
     - AD backend: ``codipack`` or ``enzyme``
   * - ``topology_format``
     - ``netcdf``
     - Topology file format: ``netcdf``, ``geojson``, ``csv``
   * - ``from_model``
     - ``default``
     - Source model for runoff input

Gradient-Based Calibration
--------------------------

To enable gradient-based calibration:

.. code-block:: yaml

    model:
      routing_model: DROUTE
      droute:
        enable_gradients: true
        ad_backend: codipack
        params_to_calibrate: velocity,diffusivity

    optimization:
      algorithm: L-BFGS  # or Adam, gradient_descent

Network Topology
================

dRoute uses mizuRoute-compatible NetCDF topology files. If you have already run
mizuRoute preprocessing, dRoute can reuse the existing topology.

Required Topology Variables
---------------------------

- ``segId``: Segment identifiers
- ``downSegId``: Downstream segment identifiers
- ``slope``: Segment slopes
- ``length``: Segment lengths (m)
- ``hruId``: HRU identifiers
- ``hruToSegId``: HRU-to-segment mapping
- ``area``: HRU areas (m²)

Optional:

- ``width``: Channel widths (m) - estimated from contributing area if not provided

Usage
=====

Standard Workflow
-----------------

.. code-block:: python

    from symfluence.models.droute import DRoutePreProcessor, DRouteRunner

    # Preprocessing (reuses mizuRoute topology if available)
    preprocessor = DRoutePreProcessor(config, logger)
    preprocessor.run_preprocessing()

    # Run routing
    runner = DRouteRunner(config, logger)
    output_path = runner.run_droute()

Calibration with Gradients
--------------------------

.. code-block:: python

    from symfluence.models.droute.calibration import DRouteWorker

    worker = DRouteWorker(config, logger)

    # Check if gradients are available
    if worker.supports_native_gradients():
        # Efficient gradient computation via AD
        loss, gradients = worker.evaluate_with_gradient(params, metric='kge')
    else:
        # Fallback to standard evaluation
        metrics = worker.calculate_metrics(output_dir, config)

Switching from mizuRoute
========================

To switch from mizuRoute to dRoute:

1. Change routing model in config:

   .. code-block:: yaml

       model:
         routing_model: DROUTE  # was: MIZUROUTE

2. Run preprocessing (will reuse existing mizuRoute topology):

   .. code-block:: bash

       symfluence preprocess

3. Run model:

   .. code-block:: bash

       symfluence run

The network topology from mizuRoute will be automatically detected and converted.

Troubleshooting
===============

Import Error: droute not found
------------------------------

dRoute Python bindings are not installed. Either:

1. Install using: ``symfluence tools install droute``
2. Build from source with ``-DDROUTE_BUILD_PYTHON=ON``
3. Install pre-built wheel: ``pip install droute``

AD/Gradient computation not available
-------------------------------------

dRoute was not compiled with AD support. Rebuild with:

.. code-block:: bash

    cmake .. -DDROUTE_ENABLE_AD=ON -DDROUTE_AD_BACKEND=codipack

No topology file found
----------------------

dRoute requires a network topology file. Either:

1. Run mizuRoute preprocessing first to generate topology
2. Provide a custom topology file in config

Disabling Experimental Warning
------------------------------

To disable the experimental warning when importing dRoute:

.. code-block:: bash

    export SYMFLUENCE_DISABLE_EXPERIMENTAL=1

Note: This will disable ALL experimental modules.

References
==========

- Cunge, J.A. (1969). On the Subject of a Flood Propagation Method
  (Muskingum Method). Journal of Hydraulic Research.
- mizuRoute: https://github.com/ESCOMP/mizuRoute
- CoDiPack: https://github.com/SciCompKL/CoDiPack
- Enzyme: https://enzyme.mit.edu/

See Also
========

- :ref:`routing_mizuroute` - Alternative routing model
- :ref:`calibration` - Calibration framework documentation
- :ref:`models_summa` - SUMMA model (common source for routing)
