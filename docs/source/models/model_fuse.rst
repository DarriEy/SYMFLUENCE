=========================================
FUSE Model Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

FUSE (Framework for Understanding Structural Errors) is a modular hydrological modeling framework that allows systematic evaluation of model structural uncertainty. FUSE combines different model components to create alternative model structures, enabling users to test hypotheses about dominant hydrological processes.

**Key Capabilities:**

- 1,248 unique model structures from component combinations
- Flexible spatial modes (lumped, semi-distributed, distributed)
- Elevation band discretization for snow modeling
- Multiple PET calculation methods
- Structure ensemble modeling
- Optional routing integration (mizuRoute)
- Synthetic hydrograph generation for testing
- Modular component-based architecture

**Typical Applications:**

- Model structural uncertainty quantification
- Comparative hydrology studies
- Snow hydrology with elevation gradients
- Process hypothesis testing
- Ensemble streamflow prediction
- Benchmarking other models

**Spatial Scales:** Point to basin (tested up to 10,000 km²)

**Temporal Resolution:** Daily (primarily) to sub-daily

Model Structure and Components
==============================

FUSE Philosophy
--------------

Unlike traditional fixed-structure models, FUSE builds models from modular components. Each component represents a conceptual choice for a hydrological process:

**Four Core Model Components:**

1. **Upper soil layer architecture** (2 options)
2. **Lower soil layer architecture** (2 options)
3. **Base flow** (2 options)
4. **Surface runoff** (4 options)

This creates 2 × 2 × 2 × 4 = 32 parent structures.

**Additional Decisions:**

- Evapotranspiration (3 options)
- Interflow (2 options)
- Time delay mechanisms (2 options)
- Percolation (3 options)
- ...and more

Total combinations: **1,248 unique model structures**

Common FUSE Structures
---------------------

Several pre-defined structures replicate well-known models:

.. code-block:: text

   Structure 60:  TOPMODEL-like structure
   Structure 230: SACRAMENTO-like structure
   Structure 342: ARNOVI-like structure
   Structure 426: PRMS-like structure
   Structure 900: VIC-like structure

**In SYMFLUENCE**, structure is specified via model ID (60-1248).

Model Components Explained
--------------------------

**Upper Soil Store:**

- ``tens1a``: Single tension storage
- ``tens1b``: Cascading tension storages

**Lower Soil Store:**

- ``fixedsiz_lower``: Fixed-size lower zone
- ``tens2pll_2``: Tension pool with parallel stores

**Baseflow:**

- ``qb_prms``: Non-linear baseflow (PRMS-style)
- ``qb_topmodel``: Exponential baseflow (TOPMODEL-style)

**Surface Runoff:**

- ``arno_x_vic``: ARNO/VIC infiltration excess
- ``prms_variant``: PRMS saturation excess
- ``tmdl_param``: TOPMODEL saturation
- ``fix_y_frac``: Fixed fraction runoff

Spatial Modes in SYMFLUENCE
===========================

FUSE in SYMFLUENCE supports three spatial modes:

Lumped Mode
----------

Single HRU represents entire basin:

.. code-block:: yaml

   FUSE_SPATIAL_MODE: lumped

   # Simple basin-wide model
   # Fastest execution
   # Good for conceptual studies

Semi-Distributed Mode
--------------------

Multiple subcatchments, each with single HRU:

.. code-block:: yaml

   FUSE_SPATIAL_MODE: semi_distributed

   # Subcatchments from delineation
   # Routed with mizuRoute
   # Balances complexity and performance

Distributed Mode
---------------

HRUs within each subcatchment (typically elevation bands):

.. code-block:: yaml

   FUSE_SPATIAL_MODE: distributed

   DISCRETIZATION:
     elevation_bands:
       n_bands: 5  # 5 elevation zones per subcatchment

   # Detailed snow modeling
   # Accounts for elevation gradients
   # Most accurate for mountainous basins

Configuration in SYMFLUENCE
===========================

Model Selection
--------------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: FUSE

Key Configuration Parameters
----------------------------

Installation and Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - FUSE_INSTALL_PATH
     - default
     - Path to FUSE installation
   * - FUSE_EXE
     - fuse.exe
     - FUSE executable name
   * - FUSE_TIMEOUT
     - 3600
     - Execution timeout (seconds)
   * - SETTINGS_FUSE_PATH
     - default
     - Working directory for FUSE
   * - SETTINGS_FUSE_FILEMANAGER
     - default
     - FUSE file manager path

Spatial Configuration
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - FUSE_SPATIAL_MODE
     - distributed
     - Spatial discretization mode
   * - FUSE_N_ELEVATION_BANDS
     - 1
     - Number of elevation bands per subcatchment
   * - FUSE_SUBCATCHMENT_DIM
     - longitude
     - Dimension for subcatchment extraction

Model Structure
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - FUSE_DECISION_OPTIONS
     - {}
     - Model structure ID or component decisions
   * - FUSE_FILE_ID
     - null
     - Custom identifier for output files

Routing Integration
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - FUSE_ROUTING_INTEGRATION
     - default
     - Enable mizuRoute integration
   * - ROUTING_MODEL
     - null
     - Set to 'mizuRoute' for routing

Calibration
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - SETTINGS_FUSE_PARAMS_TO_CALIBRATE
     - null
     - Comma-separated list of parameters
   * - EXPERIMENT_OUTPUT_FUSE
     - default
     - Output directory for experiments

Model Structure Selection
========================

Using Structure ID
-----------------

Simplest approach - specify a structure number:

.. code-block:: yaml

   FUSE_DECISION_OPTIONS:
     smodl:
       model_id: 60  # TOPMODEL-like structure

Using Component Decisions
-------------------------

Explicitly define each component:

.. code-block:: yaml

   FUSE_DECISION_OPTIONS:
     smodl:
       rferr: 'additive_e'        # Rainfall error model
       arch1: 'onestate_1'        # Upper layer architecture
       arch2: 'tens2pll_2'        # Lower layer architecture
       qsurf: 'arno_x_vic'        # Surface runoff
       qperc: 'perc_f2sat'        # Percolation
       esoil: 'sequential'        # Evaporation
       qintf: 'intflwnone'        # Interflow
       q_tdh: 'rout_gamma'        # Time delay histogram

Structure Ensemble
------------------

Test multiple structures simultaneously:

.. code-block:: yaml

   FUSE_DECISION_OPTIONS:
     ensemble:
       - model_id: 60   # TOPMODEL
       - model_id: 230  # SACRAMENTO
       - model_id: 900  # VIC
       - model_id: 426  # PRMS

   # SYMFLUENCE will run all structures and compare performance

Input File Specifications
=========================

FUSE uses ASCII control files and NetCDF forcing data.

Forcing Data (NetCDF)
--------------------

**File:** ``<basin_name>_input_<mode>.nc``

**Required variables (distributed mode):**

.. code-block:: text

   hru         : HRU/elevation band dimension
   gru         : Subcatchment dimension
   time        : Time coordinate

   pptrate     : Precipitation rate [kg m-2 s-1]
   airtemp     : Air temperature [K]
   spechum     : Specific humidity [kg kg-1]
   SWRadAtm    : Shortwave radiation [W m-2]
   LWRadAtm    : Longwave radiation [W m-2]
   windspd     : Wind speed [m s-1]
   airpres     : Air pressure [Pa]

**Required variables (lumped mode):**

.. code-block:: text

   time        : Time coordinate
   pptrate     : Basin-averaged precipitation [kg m-2 s-1]
   pet         : Potential evapotranspiration [kg m-2 s-1]

Control Files (ASCII)
--------------------

**File:** ``fm_catch.txt`` (File Manager)

Contains paths to all FUSE input/output files:

.. code-block:: text

   CONTROL_FILE      ./input/fuse_control.txt
   FORCING_PATH      ./forcing/
   OUTPUT_PATH       ./output/
   MBANDS_INFO       ./input/elev_bands.txt

**File:** ``fuse_control.txt`` (Control File)

Specifies model configuration:

.. code-block:: text

   ! Model structure
   SMODL 60

   ! Simulation period
   SSTRT 2015 01 01 00
   SFINISH 2020 12 31 23

   ! Timestep (seconds)
   DT 86400

Elevation Bands File
-------------------

**File:** ``elev_bands.txt`` (for distributed mode)

.. code-block:: text

   ! Elevation band information
   ! nBands  lowerElev  upperElev  midElev  z_weight  area_frac
   5         500        1000       750      0.2       0.15
   5         1000       1500       1250     0.2       0.25
   5         1500       2000       1750     0.2       0.30
   5         2000       2500       2250     0.2       0.20
   5         2500       3000       2750     0.2       0.10

Output File Specifications
==========================

FUSE outputs NetCDF files with states and fluxes.

Standard Output (NetCDF)
-----------------------

**File:** ``<basin>_<structure>_<mode>_output.nc``

**Output variables:**

.. code-block:: text

   time              : Time coordinate

   # Fluxes [mm/day or mm/timestep]
   total_q           : Total runoff
   eff_ppt           : Effective precipitation
   evap_1            : Evaporation from upper layer
   evap_2            : Evaporation from lower layer
   q_sf              : Surface runoff
   q_if              : Interflow
   q_perc            : Percolation
   q_base            : Baseflow

   # States [mm]
   tens_1            : Upper tension storage
   tens_2            : Lower tension storage
   free_1            : Upper free storage
   free_2            : Lower free storage
   watr_1            : Upper water storage
   watr_2            : Lower water storage

Routed Output (with mizuRoute)
-----------------------------

**File:** ``<basin>_routed_streamflow.nc``

.. code-block:: text

   time              : Time coordinate
   seg               : River segment ID

   dlayRunoff        : IRF-routed runoff [m³/s]
   instRunoff        : Instantaneous runoff [m³/s]
   KWTroutedRunoff   : KWT-routed runoff [m³/s]

Model-Specific Workflows
========================

Basic FUSE Workflow (Lumped)
----------------------------

Simplest FUSE application:

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: my_basin
   HYDROLOGICAL_MODEL: FUSE

   # Lumped mode
   FUSE_SPATIAL_MODE: lumped
   DOMAIN_DEFINITION_METHOD: polygon
   CATCHMENT_SHP_PATH: ./basin_boundary.shp

   # Structure selection
   FUSE_DECISION_OPTIONS:
     smodl:
       model_id: 60  # TOPMODEL structure

   # Forcing
   FORCING_DATASET: ERA5
   FORCING_START_YEAR: 2010
   FORCING_END_YEAR: 2015

Run:

.. code-block:: bash

   symfluence workflow run --config config.yaml

Distributed FUSE with Elevation Bands
------------------------------------

For snow-dominated basins:

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: mountain_basin
   HYDROLOGICAL_MODEL: FUSE

   # Distributed mode with elevation bands
   FUSE_SPATIAL_MODE: distributed
   FUSE_N_ELEVATION_BANDS: 7

   # Delineate domain
   DOMAIN_DEFINITION_METHOD: semidistributed
   POUR_POINT_COORDS: [-120.5, 48.3]

   # Create elevation discretization
   DISCRETIZATION:
     elevation_bands:
       n_bands: 7
       method: equal_area  # Or: equal_interval

   # Structure selection
   FUSE_DECISION_OPTIONS:
     smodl:
       model_id: 900  # VIC-like structure (good for snow)

   # Optional routing
   ROUTING_MODEL: mizuRoute
   FUSE_ROUTING_INTEGRATION: mizuRoute

Structure Ensemble Workflow
--------------------------

Compare multiple model structures:

.. code-block:: yaml

   # config.yaml
   HYDROLOGICAL_MODEL: FUSE

   # Test 4 different structures
   FUSE_DECISION_OPTIONS:
     ensemble:
       - model_id: 60   # TOPMODEL
       - model_id: 230  # SACRAMENTO
       - model_id: 342  # ARNOVI
       - model_id: 900  # VIC

   # Calibrate each structure
   OPTIMIZATION_ALGORITHM: DDS
   OPTIMIZATION_MAX_ITERATIONS: 1000

   # Common parameters to calibrate
   SETTINGS_FUSE_PARAMS_TO_CALIBRATE: "maxwatr_1,maxwatr_2,baserte,rtfrac1,qbrate_2a"

Result: Best-performing structure identified for your basin.

Multi-Site Distributed FUSE
---------------------------

For large river networks:

.. code-block:: yaml

   # config.yaml
   DOMAIN_DEFINITION_METHOD: merit_basins
   MERIT_BASIN_IDS: [10234, 10235, 10236, 10237]  # 4 sub-basins

   FUSE_SPATIAL_MODE: distributed
   FUSE_N_ELEVATION_BANDS: 5

   # Enable routing to combine subcatchments
   ROUTING_MODEL: mizuRoute
   FUSE_ROUTING_INTEGRATION: mizuRoute

   # Structure and calibration
   FUSE_DECISION_OPTIONS:
     smodl:
       model_id: 426  # PRMS structure

   SETTINGS_FUSE_PARAMS_TO_CALIBRATE: "maxwatr_1,maxwatr_2,baserte"

Calibration Strategies
=====================

Recommended Parameters
---------------------

**Universal parameters (all structures):**

.. code-block:: yaml

   SETTINGS_FUSE_PARAMS_TO_CALIBRATE: "maxwatr_1,maxwatr_2,baserte,rtfrac1"

Parameters explained:

- ``maxwatr_1``: Maximum storage in upper layer [mm] (10 - 500)
- ``maxwatr_2``: Maximum storage in lower layer [mm] (50 - 5000)
- ``baserte``: Baseflow rate parameter (0.001 - 1000)
- ``rtfrac1``: Fraction of roots in upper layer (0.01 - 0.99)

**Snow-focused basins (with elevation bands):**

.. code-block:: yaml

   SETTINGS_FUSE_PARAMS_TO_CALIBRATE: "maxwatr_1,maxwatr_2,baserte,rtfrac1,tempf,snofall_t"

Additional parameters:

- ``snofall_t``: Snowfall temperature threshold [°C] (-2 to +2)
- ``tempf``: Temperature index for snowmelt [mm/°C/day] (0.5 - 10)

**Structure-specific parameters:**

For TOPMODEL-like structures (e.g., 60):

.. code-block:: yaml

   SETTINGS_FUSE_PARAMS_TO_CALIBRATE: "maxwatr_1,maxwatr_2,baserte,loglamb,tishape"

- ``loglamb``: Log of topographic index decay parameter
- ``tishape``: Topographic index histogram shape

For VIC-like structures (e.g., 900):

.. code-block:: yaml

   SETTINGS_FUSE_PARAMS_TO_CALIBRATE: "maxwatr_1,maxwatr_2,baserte,binf_p,ds_p"

- ``binf_p``: Infiltration parameter for VIC
- ``ds_p``: Baseflow parameter

Parameter Bounds
---------------

Standard ranges:

.. code-block:: python

   # Storage parameters
   maxwatr_1:    [10, 500]       # mm
   maxwatr_2:    [50, 5000]      # mm
   fracten:      [0.01, 0.95]    # fraction tension storage

   # Routing parameters
   baserte:      [0.001, 1000]   # baseflow rate
   rtfrac1:      [0.01, 0.99]    # root fraction
   rtfrac2:      [0.01, 0.99]    # root fraction layer 2

   # Runoff parameters
   percfrac:     [0.01, 0.99]    # percolation fraction
   fprimqb:      [0.001, 10]     # baseflow exponent
   qbrate_2a:    [0.001, 0.25]   # baseflow coefficient

   # Snow parameters (if applicable)
   snofall_t:    [-2, 2]         # °C
   tempf:        [0.5, 10]       # mm/°C/day

Structure Ensemble Calibration
------------------------------

When using structure ensembles:

1. **Calibrate each structure independently:**

   .. code-block:: yaml

      # Each structure gets its own parameter set
      OPTIMIZATION_MAX_ITERATIONS: 1000

2. **Compare performance:**

   - Evaluate on calibration period
   - Test on validation period
   - Check for overfitting

3. **Select or weight structures:**

   - Use best structure
   - Or create weighted ensemble based on performance

Known Limitations
================

1. **Primarily Daily Timestep:**

   - FUSE designed for daily modeling
   - Sub-daily possible but less tested
   - Flash flood applications limited

2. **Limited Process Representation:**

   - No explicit glacier modeling
   - Simple snow module (degree-day)
   - No detailed energy balance
   - Limited groundwater dynamics

3. **Routing:**

   - No internal routing in lumped mode
   - Requires mizuRoute for distributed applications
   - Unit hydrograph approach in some structures

4. **Parameter Identifiability:**

   - Some structures have >20 parameters
   - High-dimensional calibration challenging
   - Parameter equifinality common

5. **Computational Considerations:**

   - Structure ensembles multiply compute time
   - 1,248 structures = extensive testing required
   - Distributed mode with many bands can be slow

Troubleshooting
==============

Common Issues
-------------

**Error: "FUSE executable not found"**

.. code-block:: yaml

   # Solution: Verify installation
   FUSE_INSTALL_PATH: /absolute/path/to/fuse/bin
   FUSE_EXE: fuse.exe

**Error: "Invalid model structure ID"**

.. code-block:: yaml

   # Solution: Use valid structure (60-1248)
   FUSE_DECISION_OPTIONS:
     smodl:
       model_id: 60  # Valid structure

**Error: "Dimension mismatch in forcing data"**

Check forcing file dimensions:

.. code-block:: bash

   ncdump -h forcing_input.nc

   # For distributed mode, verify:
   # - hru dimension matches n_elevation_bands
   # - gru dimension matches n_subcatchments

**Error: "FUSE simulation crashed - NaN values"**

Common causes:

1. **Extreme parameters** → Adjust calibration bounds
2. **Missing forcing data** → Check for gaps in precipitation/PET
3. **Very small timestep** → Use daily timestep
4. **Invalid structure combination** → Use pre-defined structure ID

**Elevation bands not created**

.. code-block:: yaml

   # Solution: Ensure distributed mode + discretization
   FUSE_SPATIAL_MODE: distributed
   FUSE_N_ELEVATION_BANDS: 5

   DISCRETIZATION:
     elevation_bands:
       n_bands: 5

**Routing output missing**

.. code-block:: yaml

   # Solution: Enable routing integration
   ROUTING_MODEL: mizuRoute
   FUSE_ROUTING_INTEGRATION: mizuRoute

Performance Tips
---------------

**Speed up calibration:**

1. Use lumped mode during initial testing
2. Reduce number of elevation bands (3-5 sufficient for most basins)
3. Start with single structure, not ensemble
4. Use shorter calibration period (3-5 years)

**Improve accuracy:**

1. Use distributed mode for heterogeneous basins
2. Add elevation bands for snow-dominated regions
3. Test multiple structures to find best fit
4. Calibrate on long period (8-10 years) if data available

**Debug slowly:**

.. code-block:: bash

   # Run FUSE manually to see detailed output
   cd <project_dir>/settings/FUSE/
   ../../installs/fuse/bin/fuse.exe fm_catch.txt

Additional Resources
===================

**FUSE Documentation:**

- Original paper: Clark et al. (2008) - https://doi.org/10.1029/2007WR006735
- Code repository: https://github.com/naddor/fuse
- Structure catalog: See FUSE documentation for complete structure list

**SYMFLUENCE-specific:**

- :doc:`../configuration`: Full parameter reference
- :doc:`../calibration`: Calibration workflows
- :doc:`model_summa`: Comparison with SUMMA
- :doc:`../troubleshooting`: General troubleshooting

**Publications:**

- Clark et al. (2008): "Framework for Understanding Structural Errors (FUSE):
  A modular framework to diagnose differences between hydrological models"
- Henn et al. (2015): "Spatial dependence of FUSE model parameters across basins"
- Knoben et al. (2020): "Modular Assessment of Rainfall-Runoff Models Toolbox (MARRMoT)"

**Example Workflow:**

.. code-block:: bash

   # View distributed FUSE example
   symfluence examples list | grep FUSE

   # Copy example config
   symfluence examples copy bow_river_fuse ./my_fuse_project
