Noah-MP Model Guide
===================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

Noah-MP is a standalone land surface model. SYMFLUENCE integrates the
NOAA-OWP ``noah-owp-modular`` implementation as a 1-D column model with
Fortran-namelist configuration, ASCII forcing input, and NetCDF output.

**Key Capabilities:**

- Standalone Noah-MP executable management
- Configuration schema for common physics options
- NetCDF runoff, ET, snow, soil moisture, and energy output extraction
- Installation through ``symfluence binary install noahmp``

.. note::

   Noah-MP is a land surface column model. In SYMFLUENCE, ``streamflow`` output
   is represented by total runoff where available, not routed basin discharge.

Model Selection
===============

Use Noah-MP as the hydrological model:

.. code-block:: yaml

   HYDROLOGICAL_MODEL: NOAHMP
   NOAHMP_INSTALL_PATH: default
   NOAHMP_EXE: noah_owp_modular.exe
   SETTINGS_NOAHMP_PATH: default
   NOAHMP_NAMELIST_FILE: namelist.input

Installation
============

Install noah-owp-modular through the binary/tool installer:

.. code-block:: bash

   symfluence binary install noahmp

The build requires a Fortran compiler and NetCDF-Fortran. By default,
SYMFLUENCE installs the model under:

.. code-block:: text

   $SYMFLUENCE_DATA_DIR/installs/noah-owp-modular

You can point to an existing build instead:

.. code-block:: yaml

   NOAHMP_INSTALL_PATH: /path/to/noah-owp-modular
   NOAHMP_EXE: noah_owp_modular.exe

Configuration Parameters
========================

.. list-table::
   :header-rows: 1
   :widths: 32 20 48

   * - Parameter
     - Default
     - Description
   * - ``NOAHMP_INSTALL_PATH``
     - ``default``
     - noah-owp-modular install directory.
   * - ``NOAHMP_EXE``
     - ``noah_owp_modular.exe``
     - Executable name under the model ``run/`` directory.
   * - ``SETTINGS_NOAHMP_PATH``
     - ``default``
     - Project settings directory for Noah-MP files.
   * - ``NOAHMP_NAMELIST_FILE``
     - ``namelist.input``
     - Main Fortran namelist file.
   * - ``NOAHMP_FORCING_FILE``
     - ``forcing.txt``
     - Forcing input file name.
   * - ``NOAHMP_OUTPUT_FILE``
     - ``output.nc``
     - NetCDF output file name.
   * - ``NOAHMP_TIMESTEP``
     - ``3600``
     - Model timestep in seconds. Supported values are 900, 1800, and 3600.
   * - ``NOAHMP_NSOIL``
     - ``4``
     - Number of soil layers.
   * - ``NOAHMP_NSNOW``
     - ``3``
     - Number of snow layers.
   * - ``NOAHMP_RUNOFF_OPTION``
     - ``1``
     - Runoff physics option.
   * - ``NOAHMP_SPINUP_LOOPS``
     - ``0``
     - Number of spinup loops.
   * - ``NOAHMP_TIMEOUT``
     - ``7200``
     - Maximum runtime in seconds.

Expected Layout
===============

The runner expects the namelist here:

.. code-block:: text

   <project_dir>/settings/NOAHMP/namelist.input

It executes the model from that settings directory using:

.. code-block:: text

   <NOAHMP_INSTALL_PATH>/run/noah_owp_modular.exe

Outputs are read from:

.. code-block:: text

   <project_dir>/simulations/<EXPERIMENT_ID>/NOAHMP/

Result Extraction
=================

The result extractor searches for ``output.nc`` and extracts:

- ``streamflow`` / ``runoff`` from ``SFCRNOFF`` and ``UGDRNOFF``
- ``et`` from ``ECAN``, ``ETRAN``, ``EDIR``, or ``LH``
- ``snow`` from ``SNEQV``, ``SNOWH``, and related snow flux variables
- ``soil_moisture`` from ``SMC``, ``SH2O``, or ``ZWT``
- ``energy`` from ``LH``, ``FSH``, ``GH``, ``FSA``, ``FIRA``, ``TV``, ``TG``, or ``T2M``

Minimal Example
===============

.. code-block:: yaml

   DOMAIN_NAME: column_site
   EXPERIMENT_ID: run_1
   HYDROLOGICAL_MODEL: NOAHMP
   ROUTING_MODEL: none

   NOAHMP_INSTALL_PATH: default
   NOAHMP_EXE: noah_owp_modular.exe
   NOAHMP_NAMELIST_FILE: namelist.input
   NOAHMP_TIMESTEP: 3600
   NOAHMP_TIMEOUT: 7200

Run the model step once the Noah-MP namelist and forcing files exist:

.. code-block:: bash

   symfluence workflow step run_model --config config_column_site.yaml
