=========================================
NGEN Model Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

NGEN (NOAA Next Generation Water Resources Modeling Framework) is a modular, interoperable hydrological modeling framework developed by NOAA's Office of Water Prediction. NGEN uses the Basic Model Interface (BMI) to couple multiple process modules, enabling flexible model configuration and community-driven model development.

**Key Capabilities:**

- Modular BMI-compliant process models
- Multiple formulation combinations (CFE, Noah-OWP-M, TOPMODEL, LSTM, etc.)
- Catchment-based spatial structure (HY_Features)
- T-Route routing integration
- JSON-based configuration
- Python and C++ module support
- Flexible time stepping
- National Water Model (NWM) compatibility

**Typical Applications:**

- Operational flood forecasting (National Water Model)
- Research model intercomparison
- Process module testing and development
- Continental-scale hydrology
- Data assimilation experiments
- Ensemble streamflow prediction

**Spatial Scales:** Catchment (1-100 km²) to continental

**Temporal Resolution:** Hourly (operational) to sub-hourly

NGEN Architecture
=================

Basic Model Interface (BMI)
---------------------------

NGEN uses BMI to standardize model coupling:

**BMI Functions:**

- ``initialize()``: Set up model
- ``update()``: Advance one time step
- ``get_value()``: Retrieve state/flux
- ``set_value()``: Update state/flux
- ``finalize()``: Clean up

This allows any BMI-compliant model to plug into NGEN.

Core Modules
-----------

**1. CFE (Conceptual Functional Equivalent)**

- Conceptual rainfall-runoff model
- Inspired by NWM/Sacramento
- Soil moisture accounting
- Fast, simple, robust

**2. Noah-OWP-M (Noah-MP for OWP)**

- Land surface model
- Energy and water balance
- Snow physics
- Vegetation dynamics
- More complex than CFE

**3. PET Modules**

- ``pet_fao56``: FAO-56 Penman-Monteith
- ``pet_priestley_taylor``: Simplified PET
- ``pet_hargreaves``: Temperature-based PET

**4. TOPMODEL**

- Topographic index-based model
- Variable source area concept

**5. LSTM (Machine Learning)**

- Data-driven runoff prediction
- Can replace physics-based modules

**6. T-Route**

- Channel routing
- Diffusive wave or Muskingum-Cunge

Formulation Combinations
------------------------

NGEN's power is combining modules:

**Example 1: CFE + PET**

.. code-block:: text

   Forcing → PET Module → CFE Module → Runoff → T-Route

**Example 2: Noah-OWP-M (standalone)**

.. code-block:: text

   Forcing → Noah-OWP-M (includes internal PET) → Runoff → T-Route

**Example 3: TOPMODEL + LSTM**

.. code-block:: text

   Forcing → TOPMODEL → LSTM Correction → Runoff

Configuration in SYMFLUENCE
===========================

Model Selection
--------------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: NGEN

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
   * - NGEN_INSTALL_PATH
     - default
     - Path to NGEN installation
   * - NGEN_EXE
     - ngen
     - NGEN executable name
   * - NGEN_ACTIVE_CATCHMENT_ID
     - null
     - Specific catchment to simulate (or null for all)

Module Selection
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - NGEN_MODULES_TO_CALIBRATE
     - null
     - Which modules to calibrate (CFE, NOAH, PET)

Calibration Parameters
^^^^^^^^^^^^^^^^^^^^^

**CFE Module:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - NGEN_CFE_PARAMS_TO_CALIBRATE
     - null
     - CFE parameters (maxsmc, Ksat, b, etc.)

**Noah-OWP-M Module:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - NGEN_NOAH_PARAMS_TO_CALIBRATE
     - null
     - Noah parameters (REFKDT, SMCMAX, etc.)

**PET Module:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - NGEN_PET_PARAMS_TO_CALIBRATE
     - null
     - PET parameters (albedo, etc.)

Input File Specifications
=========================

NGEN uses JSON for configuration and NetCDF/CSV for data.

Realization Configuration (JSON)
--------------------------------

**File:** ``realization.json``

Defines which modules to use and how they connect:

.. code-block:: json

   {
     "global": {
       "formulations": [
         {
           "name": "bmi_c",
           "params": {
             "model_type_name": "CFE",
             "library_file": "./extern/cfe/cmake_build/libcfemodel.so",
             "init_config": "./config/cfe_config.json",
             "main_output_variable": "Q_OUT",
             "variables_names_map": {
               "QINSUR": "surface_runoff",
               "QSUB": "subsurface_runoff"
             },
             "uses_forcing_file": false
           }
         },
         {
           "name": "bmi_c",
           "params": {
             "model_type_name": "PET",
             "library_file": "./extern/evapotranspiration/cmake_build/libpetmodel.so",
             "init_config": "./config/pet_config.json",
             "main_output_variable": "water_potential_evaporation_flux"
           }
         }
       ]
     },
     "time": {
       "start_time": "2015-01-01 00:00:00",
       "end_time": "2020-12-31 23:00:00",
       "output_interval": 3600
     },
     "catchments": {
       "cat-123": {
         "formulations": [
           {
             "name": "bmi_c",
             "params": {"model_type_name": "CFE"}
           }
         ]
       }
     }
   }

CFE Configuration (JSON)
------------------------

**File:** ``cfe_config.json``

.. code-block:: json

   {
     "forcing_file": "cat_123_forcing.csv",
     "soil_params": {
       "depth": 2.0,
       "b": 4.05,
       "mult": 1000.0,
       "satdk": 0.00000338,
       "satpsi": 0.355,
       "slop": 0.1,
       "smcmax": 0.439,
       "wltsmc": 0.066
     },
     "max_gw_storage": 0.01,
     "Cgw": 0.01,
     "expon": 6.0,
     "gw_storage": 0.05,
     "alpha_fc": 0.33,
     "soil_storage": 0.05,
     "K_nash": 0.03,
     "K_lf": 0.01,
     "nash_storage": [0.0, 0.0, 0.0],
     "giuh_ordinates": [0.06, 0.51, 0.28, 0.12, 0.03]
   }

Noah-OWP-M Configuration (JSON)
------------------------------

**File:** ``noah_config.json``

.. code-block:: json

   {
     "forcing_file": "cat_123_forcing.csv",
     "soil_params": {
       "smcmax": [0.439, 0.421, 0.434, 0.476],
       "smcwlt": [0.010, 0.028, 0.047, 0.084],
       "smcref": [0.283, 0.249, 0.236, 0.219],
       "bexp": [4.26, 4.74, 5.33, 5.33],
       "satdk": [1.07e-5, 1.41e-5, 5.23e-6, 2.81e-6],
       "satpsi": [0.355, 0.363, 0.355, 0.257]
     },
     "veg_params": {
       "shdfac": 0.7,
       "nroot": 4,
       "refkdt": 3.0,
       "z0": 0.5,
       "czil": 0.1,
       "lai": 4.5,
       "csoil": 2.00e+06
     }
   }

Forcing Data (CSV)
------------------

**File:** ``cat_<ID>_forcing.csv``

.. code-block:: text

   time,APCP_surface,DLWRF_surface,DSWRF_surface,PRES_surface,SPFH_2maboveground,TMP_2maboveground,UGRD_10maboveground,VGRD_10maboveground
   2015-01-01 00:00:00,0.0,250.3,0.0,101325,0.005,275.15,2.3,-1.1
   2015-01-01 01:00:00,0.5,248.1,0.0,101320,0.0051,274.95,2.5,-1.0
   ...

**Required variables:**

- ``APCP_surface``: Precipitation [mm/hr]
- ``TMP_2maboveground``: Air temperature [K]
- ``SPFH_2maboveground``: Specific humidity [kg/kg]
- ``DSWRF_surface``: Shortwave radiation [W/m²]
- ``DLWRF_surface``: Longwave radiation [W/m²]
- ``UGRD_10maboveground``: U-wind [m/s]
- ``VGRD_10maboveground``: V-wind [m/s]
- ``PRES_surface``: Pressure [Pa]

Output File Specifications
==========================

NGEN outputs NetCDF or CSV files depending on configuration.

Catchment Output (CSV)
----------------------

**File:** ``cat_<ID>_output.csv``

.. code-block:: text

   Time,Q_OUT,SOIL_STORAGE,GW_STORAGE,GIUH_RUNOFF,NASH_FLOW,DEEP_GW_FLOW
   2015-01-01 00:00:00,0.45,0.234,0.012,0.10,0.25,0.05
   2015-01-01 01:00:00,0.42,0.232,0.011,0.08,0.24,0.05
   ...

**Variables (CFE example):**

- ``Q_OUT``: Total outflow [m³/s or mm/hr]
- ``SOIL_STORAGE``: Soil moisture storage [m or mm]
- ``GW_STORAGE``: Groundwater storage [m or mm]
- ``GIUH_RUNOFF``: GIUH routed runoff
- ``NASH_FLOW``: Nash cascade flow
- ``DEEP_GW_FLOW``: Deep groundwater loss

Aggregated Output (NetCDF)
--------------------------

**File:** ``ngen_output.nc``

Aggregated results across all catchments:

.. code-block:: text

   dimensions:
     time = UNLIMITED
     catchment = 5000

   variables:
     float streamflow(time, catchment)
     float soil_moisture(time, catchment)
     float snow_water_equiv(time, catchment)

Model-Specific Workflows
========================

Basic NGEN with CFE
------------------

Simplest NGEN setup:

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: test_catchment
   HYDROLOGICAL_MODEL: NGEN

   # Domain definition (single catchment or multiple)
   DOMAIN_DEFINITION_METHOD: semidistributed
   POUR_POINT_COORDS: [-105.5, 40.2]

   # NGEN modules
   NGEN_MODULES_TO_CALIBRATE: CFE

   # CFE parameters
   NGEN_CFE_PARAMS_TO_CALIBRATE: "maxsmc,Ksat,b,expon"

   # Forcing
   FORCING_DATASET: ERA5
   FORCING_START_YEAR: 2015
   FORCING_END_YEAR: 2020

Run:

.. code-block:: bash

   symfluence workflow run --config config.yaml

NGEN with Noah-OWP-M
-------------------

For detailed land surface modeling:

.. code-block:: yaml

   # config.yaml
   HYDROLOGICAL_MODEL: NGEN

   # Use Noah-OWP-M formulation
   NGEN_MODULES_TO_CALIBRATE: NOAH

   # Noah parameters
   NGEN_NOAH_PARAMS_TO_CALIBRATE: "REFKDT,SMCMAX,BEXP"

   # Noah requires all 8 forcing variables
   FORCING_DATASET: ERA5  # Provides complete forcings

Multi-Catchment NGEN
-------------------

For river basin modeling:

.. code-block:: yaml

   # Delineate multiple catchments
   DOMAIN_DEFINITION_METHOD: merit_basins
   MERIT_BASIN_IDS: [500123, 500124, 500125, 500126]  # 4 catchments

   HYDROLOGICAL_MODEL: NGEN

   # Enable T-Route routing
   ROUTING_MODEL: troute  # NGEN's routing module

   # CFE + PET formulation
   NGEN_MODULES_TO_CALIBRATE: "CFE,PET"

National Scale NGEN
------------------

Following National Water Model approach:

.. code-block:: yaml

   # National-scale application
   DOMAIN_DEFINITION_METHOD: nwm_hydrofabric
   NWM_VERSION: "2.1"
   NWM_DOMAIN: "CONUS"  # Continental US

   HYDROLOGICAL_MODEL: NGEN

   # Standard NWM formulation
   NGEN_MODULES_TO_CALIBRATE: CFE

   # Parallel execution essential
   NUM_PROCESSES: 256

Calibration Strategies
=====================

CFE Parameters
-------------

.. code-block:: yaml

   NGEN_CFE_PARAMS_TO_CALIBRATE: "maxsmc,Ksat,b,expon,Cgw,alpha_fc"

**Parameter descriptions:**

- ``maxsmc``: Maximum soil moisture content [-] (0.3-0.6)
- ``Ksat``: Saturated hydraulic conductivity [m/s] (1e-7 to 1e-4)
- ``b``: Pore size distribution index [-] (2-12)
- ``expon``: GIUH exponent [-] (3-10)
- ``Cgw``: Groundwater coefficient [1/hr] (0.001-0.1)
- ``alpha_fc``: Field capacity fraction [-] (0.2-0.8)

**Recommended bounds:**

.. code-block:: python

   maxsmc:    [0.3, 0.6]        # Soil moisture at saturation
   Ksat:      [1e-7, 1e-4]      # Hydraulic conductivity [m/s]
   b:         [2.0, 12.0]       # Brooks-Corey exponent
   expon:     [3.0, 10.0]       # GIUH shape parameter
   Cgw:       [0.001, 0.1]      # Groundwater recession [1/hr]
   alpha_fc:  [0.2, 0.8]        # Field capacity fraction
   K_lf:      [0.001, 0.1]      # Lateral flow coefficient
   K_nash:    [0.01, 0.5]       # Nash cascade coefficient

Noah-OWP-M Parameters
--------------------

.. code-block:: yaml

   NGEN_NOAH_PARAMS_TO_CALIBRATE: "REFKDT,SMCMAX,BEXP"

**Common parameters:**

.. code-block:: python

   REFKDT:    [1.0, 6.0]        # Infiltration parameter
   SMCMAX:    [0.3, 0.6]        # Max soil moisture (layer-specific)
   BEXP:      [2.0, 12.0]       # Clapp-Hornberger exponent
   SATDK:     [1e-7, 1e-4]      # Saturated conductivity
   CZIL:      [0.01, 0.5]       # Surface roughness parameter

PET Parameters
-------------

.. code-block:: yaml

   NGEN_PET_PARAMS_TO_CALIBRATE: "alpha,beta"

.. code-block:: python

   alpha:     [0.8, 1.2]        # PET multiplier
   beta:      [-0.2, 0.2]       # PET additive adjustment

Calibration Tips
---------------

1. **Start with CFE** (simplest, 6-8 parameters)
2. **Use DDS or PSO** (efficient for 10-20 dimensional problems)
3. **Hourly forcings recommended** for NGEN
4. **Consider multi-site calibration** across catchment network
5. **Validate routing** if using multiple catchments

Known Limitations
================

1. **Complexity:**

   - Steep learning curve
   - JSON configuration verbose
   - Multiple files per catchment

2. **Module Dependencies:**

   - Requires compiling BMI modules
   - Version compatibility issues
   - Platform-specific builds

3. **Data Format:**

   - NGEN expects specific forcing variable names
   - HY_Features catchment structure required
   - T-Route network topology needed

4. **Computational Cost:**

   - Hourly timestep slower than daily models
   - BMI overhead
   - Large national domains require HPC

5. **Documentation:**

   - Rapidly evolving framework
   - Limited examples outside NWM
   - Module-specific docs scattered

Troubleshooting
==============

Common Issues
-------------

**Error: "NGEN executable not found"**

.. code-block:: yaml

   # Verify NGEN installation
   NGEN_INSTALL_PATH: /absolute/path/to/ngen/bin
   NGEN_EXE: ngen

**Error: "BMI module library not found"**

Solution: Ensure all BMI modules are compiled:

.. code-block:: bash

   # Check for CFE library
   ls $NGEN_INSTALL_PATH/extern/cfe/cmake_build/libcfemodel.so

   # If missing, compile modules
   cd $NGEN_INSTALL_PATH/extern/cfe
   mkdir cmake_build && cd cmake_build
   cmake ..
   make

**Error: "Realization configuration error"**

Check JSON syntax:

.. code-block:: bash

   # Validate JSON
   python -m json.tool realization.json

**Error: "Forcing variable not found"**

NGEN expects specific variable names. Verify:

.. code-block:: bash

   # Check forcing CSV headers
   head -1 cat_123_forcing.csv

   # Should match NGEN expectations:
   # APCP_surface, TMP_2maboveground, etc.

**Poor simulation results**

1. **Check forcings:**

   .. code-block:: bash

      # Plot forcing time series
      python
      >>> import pandas as pd
      >>> df = pd.read_csv('cat_123_forcing.csv')
      >>> df.plot(subplots=True)

2. **Verify catchment attributes** (area, slope, soil properties)

3. **Adjust calibration bounds**

4. **Check for missing data gaps**

**Routing not working**

.. code-block:: yaml

   # Ensure T-Route is enabled
   ROUTING_MODEL: troute

   # Verify network topology file exists
   # <project_dir>/domain/routing/troute_network.json

Performance Tips
---------------

**Speed up calibration:**

1. Calibrate single catchment first
2. Transfer parameters to similar catchments
3. Use shorter calibration period (3-5 years)
4. Start with CFE (simpler than Noah-OWP-M)

**Improve accuracy:**

1. Use all 8 forcing variables (not simplified)
2. Calibrate with hourly data
3. Multi-objective optimization (flow + states)
4. Include snow observations if applicable

Additional Resources
===================

**NGEN Documentation:**

- GitHub: https://github.com/NOAA-OWP/ngen
- Wiki: https://github.com/NOAA-OWP/ngen/wiki
- BMI documentation: https://bmi.readthedocs.io

**Modules:**

- CFE: https://github.com/NOAA-OWP/cfe
- Noah-OWP-M: https://github.com/NOAA-OWP/noah-owp-modular
- T-Route: https://github.com/NOAA-OWP/t-route

**HY_Features:**

- Hydrofabric: https://github.com/NOAA-OWP/hydrofabric

**SYMFLUENCE-specific:**

- :doc:`../configuration`: NGEN parameter reference
- :doc:`../calibration`: Calibration workflows
- :doc:`model_summa`: Comparison with physics-based models
- :doc:`model_lstm`: Comparison with data-driven approaches
- :doc:`../troubleshooting`: General troubleshooting

**Publications:**

- Johnson et al. (2023): "Next Generation Water Resources Modeling Framework"
  (upcoming/draft publications)

- National Water Model documentation: https://water.noaa.gov/about/nwm

**Training:**

- NOAA-OWP workshops and webinars
- Community forums: https://github.com/NOAA-OWP/ngen/discussions
