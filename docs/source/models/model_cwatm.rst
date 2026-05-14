CWatM Model Guide
=================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

CWatM (Community Water Model) is a distributed hydrological model developed at
IIASA for regional to global water availability studies. The SYMFLUENCE
integration provides initial support for preparing forcing, writing a CWatM
settings file, launching the model, and extracting standard discharge outputs.

**Key Capabilities:**

- Distributed gridded hydrological simulation
- NetCDF and TSS discharge output extraction
- Optional PET calculation controlled through CWatM settings
- Installation through ``symfluence binary install cwatm``

.. note::

   The 0.8.3 integration generates forcing files and ``settings.ini`` but still
   relies on an installed CWatM checkout and its static CWatM-Earth parameter
   dataset.

Model Selection
===============

Use CWatM as the hydrological model:

.. code-block:: yaml

   HYDROLOGICAL_MODEL: CWATM
   CWATM_INSTALL_PATH: default
   CWATM_EXE: run_cwatm.py
   SETTINGS_CWATM_PATH: default
   CWATM_CONFIG_FILE: settings.ini
   CWATM_SPATIAL_MODE: distributed

Installation
============

Install CWatM through the binary/tool installer:

.. code-block:: bash

   symfluence binary install cwatm

By default, SYMFLUENCE installs CWatM under:

.. code-block:: text

   $SYMFLUENCE_DATA_DIR/installs/cwatm

You can point to an existing checkout instead:

.. code-block:: yaml

   CWATM_INSTALL_PATH: /path/to/CWatM
   CWATM_EXE: run_cwatm.py

Configuration Parameters
========================

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``CWATM_INSTALL_PATH``
     - ``default``
     - CWatM install directory. ``default`` resolves to ``installs/cwatm``.
   * - ``CWATM_EXE``
     - ``run_cwatm.py``
     - CWatM runner script.
   * - ``SETTINGS_CWATM_PATH``
     - ``default``
     - Project settings directory for CWatM files.
   * - ``CWATM_CONFIG_FILE``
     - ``settings.ini``
     - Main CWatM INI file.
   * - ``CWATM_RESOLUTION``
     - ``30min``
     - Nominal grid resolution label.
   * - ``CWATM_SPATIAL_MODE``
     - ``distributed``
     - Spatial mode for the model adapter.
   * - ``EXPERIMENT_OUTPUT_CWATM``
     - ``default``
     - Output path override.
   * - ``CWATM_OUTPUT_DIR``
     - ``default``
     - CWatM output directory override.
   * - ``CWATM_SPINUP_YEARS``
     - ``0``
     - Number of spinup years.
   * - ``CWATM_CALC_EVAPORATION``
     - ``False``
     - Whether CWatM should calculate evaporation internally.
   * - ``CWATM_PET_METHOD``
     - ``hamon``
     - PET method used when evaporation is calculated internally.
   * - ``CWATM_PARAMS_TO_CALIBRATE``
     - ``SnowMeltCoef,crop_correct,...``
     - Comma-separated calibration parameter names.
   * - ``CWATM_TIMEOUT``
     - ``14400``
     - Maximum runtime in seconds.

Expected Layout
===============

The preprocessor writes, and the runner reads, the INI file here:

.. code-block:: text

   <project_dir>/settings/CWATM/settings.ini

It executes:

.. code-block:: bash

   python run_cwatm.py settings.ini -q

Outputs are read from:

.. code-block:: text

   <project_dir>/simulations/<EXPERIMENT_ID>/CWATM/

Result Extraction
=================

The result extractor searches for standard CWatM discharge files first:

- ``discharge_daily.tss``
- ``discharge_daily.nc``
- ``discharge*.nc``
- ``*discharge*.nc``

For NetCDF discharge outputs, SYMFLUENCE looks for ``discharge``, ``Qsim``, or
``Q`` variables. Distributed discharge grids are reduced with a maximum
aggregation for streamflow.

Minimal Example
===============

.. code-block:: yaml

   DOMAIN_NAME: example_basin
   EXPERIMENT_ID: run_1
   HYDROLOGICAL_MODEL: CWATM
   ROUTING_MODEL: none

   CWATM_INSTALL_PATH: default
   CWATM_EXE: run_cwatm.py
   CWATM_CONFIG_FILE: settings.ini
   CWATM_TIMEOUT: 14400

Run the model step once the CWatM settings file exists:

.. code-block:: bash

   symfluence workflow step run_model --config config_example_basin.yaml
