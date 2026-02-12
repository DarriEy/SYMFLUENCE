=========================================
RHESSys Model Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

RHESSys (Regional Hydro-Ecologic Simulation System) is a distributed, process-based ecohydrological model developed at the University of North Carolina. RHESSys integrates hydrological, ecological, and biogeochemical processes to simulate water, carbon, and nitrogen cycling across landscapes with complex topography and heterogeneous vegetation.

**Key Capabilities:**

- Coupled water, carbon, and nitrogen cycles
- Distributed hillslope-scale processes
- Topographically-driven lateral fluxes
- Vegetation growth and succession
- Detailed soil carbon and nitrogen dynamics
- Wildfire simulation (WMFire module)
- Road network effects on hydrology
- Multi-scale hierarchical spatial structure

**Typical Applications:**

- Mountainous watershed hydrology
- Forest ecology and management impacts
- Fire effects on hydrology and ecosystem
- Climate change impacts on coupled systems
- Land use change scenarios
- Nutrient export from watersheds
- Ecohydrological research

**Spatial Scales:** Small watersheds (1 km²) to regional (1,000 km²)

**Temporal Resolution:** Daily (standard) to sub-daily

Model Structure
===============

Spatial Hierarchy
-----------------

RHESSys uses a nested spatial structure:

.. code-block:: text

   Basin
   └─ Zones (e.g., elevation bands, climate zones)
       └─ Hillslopes
           └─ Patches (land cover units)
               └─ Canopy Strata (vegetation layers)

**Patch:** Homogeneous land cover unit (e.g., forest stand, meadow)

**Hillslope:** Topographically-defined drainage unit

**Zone:** Climate or elevation zone

**Basin:** Entire watershed

Vertical Structure
------------------

Within each patch, vertical processes occur through:

**Canopy Layers:**

- Overstory (trees)
- Understory (shrubs)
- Ground cover (grasses, herbs)

**Soil Layers:**

- Litter
- Organic horizons
- Mineral soil (multiple layers possible)

**Routing:**

- Saturated subsurface flow (topography-driven)
- Surface runoff
- Channel routing

Process Representation
----------------------

**Hydrology:**

- Topography-based lateral flow (TOPMODEL concepts)
- Variable source area dynamics
- Soil moisture redistribution
- Snowmelt (temperature-index or energy balance)
- Evapotranspiration (Penman-Monteith)
- Infiltration and percolation

**Carbon:**

- Photosynthesis (Farquhar model)
- Autotrophic and heterotrophic respiration
- Litterfall and decomposition
- Soil organic matter dynamics
- Net Primary Productivity (NPP)

**Nitrogen:**

- Plant uptake
- Mineralization and immobilization
- Nitrification and denitrification
- Leaching
- Atmospheric deposition

**Optional Modules:**

- **WMFire:** Wildfire spread and effects
- **Roads:** Road hydrological effects
- **Urban:** Urban hydrology

Configuration in SYMFLUENCE
===========================

Model Selection
---------------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: RHESSys

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
   * - RHESSYS_INSTALL_PATH
     - default
     - Path to RHESSys executable
   * - RHESSYS_EXE
     - rhessys
     - RHESSys executable name
   * - SETTINGS_RHESSYS_PATH
     - default
     - RHESSys working directory
   * - RHESSYS_TIMEOUT
     - 7200
     - Execution timeout (seconds)

Configuration Files
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - EXPERIMENT_OUTPUT_RHESSYS
     - default
     - Output directory
   * - FORCING_RHESSYS_PATH
     - default
     - Forcing data path
   * - RHESSYS_WORLD_TEMPLATE
     - world.template
     - World file template
   * - RHESSYS_FLOW_TEMPLATE
     - flow.template
     - Flow table template

Calibration Parameters
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Parameter
     - Default
     - Description
   * - RHESSYS_PARAMS_TO_CALIBRATE
     - sat_to_gw_coeff,gw_loss_coeff,m,Ksat_0,porosity_0,soil_depth,snow_melt_Tcoef
     - Hydrological parameters
   * - RHESSYS_SKIP_CALIBRATION
     - true
     - Skip calibration (use defaults)

Wildfire Module
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - RHESSYS_USE_WMFIRE
     - false
     - Enable WMFire wildfire module
   * - WMFIRE_INSTALL_PATH
     - installs/wmfire/lib
     - WMFire library path
   * - WMFIRE_LIB
     - libwmfire.so
     - WMFire shared library

Input File Specifications
=========================

RHESSys uses ASCII text files for configuration and inputs.

World File
----------

**File:** ``worldfile``

Defines the spatial structure and initial states:

.. code-block:: text

   basin_ID  1
   x  500000.0
   y  4500000.0
   latitude  40.5
   basin_parm_ID  101
   num_zones  1

   zone_ID  1
   x  500000.0
   y  4500000.0
   z  1500.0
   zone_parm_ID  201
   num_hillslopes  3

   hillslope_ID  1
   x  500100.0
   y  4500100.0
   z  1450.0
   hillslope_parm_ID  301
   area  50000.0
   num_patches  5

   patch_ID  1
   x  500110.0
   y  4500110.0
   z  1440.0
   soil_parm_ID  401
   landuse_parm_ID  501
   area  10000.0
   soil_depth  1.5
   Ksat_0  10.0
   porosity_0  0.45
   num_canopy_strata  1

   canopy_strata_ID  1
   veg_parm_ID  601
   lai  4.5
   cover_fraction  0.8

Flow Table
----------

**File:** ``flowtable``

Defines topographic connectivity and routing:

.. code-block:: text

   patchID  hillslopeID  zoneID  basinID  gamma  area_m2  total_gamma  next_patchID
   1        1            1       1        0.25   10000    0.25         2
   2        1            1       1        0.35   15000    0.60         3
   3        1            1       1        0.40   12000    1.00         -999

**Columns:**

- ``gamma``: Routing weight (fraction of upslope area)
- ``total_gamma``: Cumulative upslope weight
- ``next_patchID``: Downslope patch (-999 = outlet)

Forcing Data (Climate)
----------------------

**File:** ``climate_station_###``

Daily meteorological data:

.. code-block:: text

   year  month  day  rain_mm  tmax_C  tmin_C  tavg_C  vpd_Pa
   2015  1      1    5.2      8.5     -2.3    3.1     450
   2015  1      2    0.0      7.2     -1.8    2.7     520
   2015  1      3    12.4     4.1     -3.5    0.3     380
   ...

**Required variables:**

- ``rain_mm``: Precipitation [mm/day]
- ``tmax_C, tmin_C, tavg_C``: Temperatures [°C]
- ``vpd_Pa``: Vapor pressure deficit [Pa]

**Optional (for full energy balance):**

- ``Kdown_direct``: Direct shortwave radiation [W/m²]
- ``Kdown_diffuse``: Diffuse shortwave radiation [W/m²]
- ``wind_m/s``: Wind speed [m/s]

Parameter Files
---------------

**Vegetation parameters:**

- Photosynthesis parameters (Vmax, Jmax)
- Allocation coefficients
- Phenology settings
- Mortality rates

**Soil parameters:**

- Hydraulic properties (Ksat, porosity, field capacity)
- Organic matter content
- Depth profiles

Output File Specifications
==========================

RHESSys produces ASCII time series outputs.

Basin Output
------------

**File:** ``<basin>_basin.daily``

Basin-averaged daily fluxes:

.. code-block:: text

   year  month  day  streamflow_mm  precip_mm  ET_mm  snowpack_mm  sat_deficit_mm
   2015  1      1    2.5            5.2        0.8    150.2        234.5
   2015  1      2    2.3            0.0        0.9    148.8        232.1
   2015  1      3    3.8            12.4       0.7    152.3        225.6
   ...

**Key variables:**

- ``streamflow_mm``: Basin streamflow [mm/day]
- ``precip_mm``: Basin precipitation [mm/day]
- ``ET_mm``: Evapotranspiration [mm/day]
- ``snowpack_mm``: Snow water equivalent [mm]
- ``sat_deficit_mm``: Saturation deficit [mm]

Carbon and Nitrogen Output
--------------------------

**File:** ``<basin>_basin.monthly``

.. code-block:: text

   year  month  GPP_gC  NPP_gC  NEP_gC  heteroResp_gC  N_uptake_gN  N_leached_gN
   2015  1      45.2    22.3    8.5     13.8           1.2          0.3
   2015  2      52.1    26.8    10.2    16.6           1.5          0.2
   ...

**Variables:**

- ``GPP_gC``: Gross Primary Productivity [gC/m²/month]
- ``NPP_gC``: Net Primary Productivity [gC/m²/month]
- ``NEP_gC``: Net Ecosystem Productivity [gC/m²/month]
- ``heteroResp_gC``: Heterotrophic respiration [gC/m²/month]
- ``N_uptake_gN``: Nitrogen uptake [gN/m²/month]
- ``N_leached_gN``: Nitrogen leaching [gN/m²/month]

Patch-Level Output
------------------

**File:** ``<basin>_patch.daily``

Spatially distributed output (optional, large files):

.. code-block:: text

   year  month  day  patchID  streamflow  ET  LAI  soilC_kgC/m2
   2015  1      1    1        2.3         0.8  4.5  12.3
   2015  1      1    2        2.5         0.9  3.8  10.5
   2015  1      1    3        2.1         0.7  5.2  15.2
   ...

Model-Specific Workflows
========================

Basic RHESSys Workflow
----------------------

Hydrology-focused application:

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: mountain_watershed
   HYDROLOGICAL_MODEL: RHESSys

   # Domain
   DOMAIN_DEFINITION_METHOD: semidistributed
   POUR_POINT_COORDS: [-120.5, 44.2]  # Oregon Cascades

   # Forcing
   FORCING_DATASET: ERA5
   FORCING_START_YEAR: 2010
   FORCING_END_YEAR: 2020

   # RHESSys configuration
   RHESSYS_INSTALL_PATH: /path/to/rhessys
   RHESSYS_PARAMS_TO_CALIBRATE: "sat_to_gw_coeff,gw_loss_coeff,m,Ksat_0,soil_depth"

Run:

.. code-block:: bash

   symfluence workflow run --config config.yaml

Forest Ecosystem Application
----------------------------

Full ecohydrological simulation:

.. code-block:: yaml

   # config.yaml
   HYDROLOGICAL_MODEL: RHESSys

   # Enable carbon and nitrogen cycles
   RHESSYS_SIMULATE_CARBON: true
   RHESSYS_SIMULATE_NITROGEN: true

   # Detailed vegetation representation
   DISCRETIZATION:
     landclass:
       sources: ['ForestInventory', 'MODIS']
       n_classes: 12  # Different forest types + other land covers

   # Calibrate both hydro and eco parameters
   RHESSYS_PARAMS_TO_CALIBRATE: "sat_to_gw_coeff,m,Ksat_0,Vmax,Jmax,allocation_leaf"

Fire Impact Assessment
----------------------

Simulate wildfire effects:

.. code-block:: yaml

   # config.yaml
   HYDROLOGICAL_MODEL: RHESSys

   # Enable WMFire module
   RHESSYS_USE_WMFIRE: true
   WMFIRE_INSTALL_PATH: /path/to/wmfire/lib

   # Pre-fire calibration period
   CALIBRATION_PERIOD: [2005, 2010]

   # Post-fire simulation
   SIMULATION_PERIOD: [2010, 2020]  # Fire occurred in 2010

   # Fire impact on parameters (handled by WMFire)
   # - Reduced canopy cover
   # - Increased runoff
   # - Soil water repellency

Climate Change Scenarios
------------------------

Multi-decade projections:

.. code-block:: yaml

   # Baseline
   FORCING_DATASET: historical_climate
   FORCING_PERIOD: [1980, 2010]

   # Future scenarios
   FORCING_DATASET: CMIP6_SSP585
   FORCING_PERIOD: [2040, 2070]

   # RHESSys will simulate vegetation response to changing climate
   RHESSYS_SIMULATE_CARBON: true  # Dynamic vegetation

Calibration Strategies
======================

Hydrological Parameters
-----------------------

Key parameters for streamflow calibration:

.. code-block:: yaml

   RHESSYS_PARAMS_TO_CALIBRATE: "sat_to_gw_coeff,gw_loss_coeff,m,Ksat_0,porosity_0,soil_depth,snow_melt_Tcoef"

**Parameter descriptions:**

- ``sat_to_gw_coeff``: Fraction of drainage to deep groundwater [-] (0.0-1.0)
- ``gw_loss_coeff``: Groundwater loss rate [1/day] (0.0-0.5)
- ``m``: Soil drainage exponent (TOPMODEL m parameter) [m] (0.01-10.0)
- ``Ksat_0``: Saturated hydraulic conductivity at surface [m/day] (0.001-10.0)
- ``porosity_0``: Soil porosity at surface [-] (0.3-0.7)
- ``soil_depth``: Soil depth [m] (0.1-5.0)
- ``snow_melt_Tcoef``: Snow melt temperature coefficient [mm/°C/day] (1.0-10.0)

**Recommended bounds:**

.. code-block:: python

   sat_to_gw_coeff:  [0.0, 1.0]      # Fraction to deep GW
   gw_loss_coeff:    [0.0, 0.5]      # GW loss rate [1/day]
   m:                [0.01, 10.0]    # Drainage exponent [m]
   Ksat_0:           [0.001, 10.0]   # Hydraulic conductivity [m/day]
   porosity_0:       [0.3, 0.7]      # Soil porosity [-]
   soil_depth:       [0.1, 5.0]      # Soil depth [m]
   snow_melt_Tcoef:  [1.0, 10.0]     # Melt factor [mm/°C/day]

Ecological Parameters
---------------------

For carbon/nitrogen modeling:

.. code-block:: yaml

   # Photosynthesis
   RHESSYS_PARAMS_TO_CALIBRATE: "Vmax,Jmax,stomatal_slope,stomatal_intercept"

   # Allocation
   RHESSYS_PARAMS_TO_CALIBRATE: "allocation_leaf,allocation_root,allocation_stem"

**Bounds:**

.. code-block:: python

   # Photosynthesis (vary by species)
   Vmax:               [20, 100]     # Max carboxylation [µmol/m²/s]
   Jmax:               [40, 200]     # Max electron transport [µmol/m²/s]

   # Allocation (fractions, must sum to 1.0)
   allocation_leaf:    [0.2, 0.5]
   allocation_root:    [0.2, 0.5]
   allocation_stem:    [0.1, 0.4]

Calibration Approach
--------------------

**Multi-objective recommended:**

.. code-block:: yaml

   OPTIMIZATION_ALGORITHM: NSGA2

   OPTIMIZATION_METRICS:
     - KGE              # Streamflow
     - RMSE_snow        # Snow (if data available)
     - MAE_NPP          # Net Primary Productivity (if data available)

**Phased calibration:**

1. **Hydrology first** (streamflow, snow)
2. **Then ecology** (fix hydro params, calibrate carbon/nitrogen)

Known Limitations
=================

1. **Computational Cost:**

   - Distributed model with complex processes = slow
   - Carbon/nitrogen cycles add significant overhead
   - Large watersheds (>100 km²) can take hours/days
   - Not suitable for real-time operational forecasting

2. **Data Requirements:**

   - Detailed DEM and flow routing
   - Vegetation parameters for each species
   - Soil profiles (depth, properties)
   - Climate data (daily minimum)

3. **Complexity:**

   - Steep learning curve
   - Many parameters (100+ for full eco model)
   - World file and flow table construction tedious
   - Difficult to debug

4. **Spatial Resolution:**

   - Patch-based structure limits resolution
   - Very fine resolution (e.g., 10m patches) impractical
   - Typically 30m-100m patches

5. **Calibration Challenges:**

   - High-dimensional parameter space
   - Interactions between hydro and eco processes
   - Limited observational data for eco variables (GPP, NPP)

6. **Model Stability:**

   - Can be sensitive to parameter combinations
   - Numerical instability with extreme values
   - Careful spinup required (especially carbon pools)

Troubleshooting
===============

Common Issues
-------------

**Error: "RHESSys executable not found"**

.. code-block:: yaml

   # Verify installation
   RHESSYS_INSTALL_PATH: /absolute/path/to/rhessys/bin
   RHESSYS_EXE: rhessys

**Error: "World file format error"**

Check world file structure:

.. code-block:: bash

   # Verify world file syntax
   head -50 worldfile

   # Common issues:
   # - Missing required fields
   # - Incorrect nesting (zones, hillslopes, patches, strata)
   # - Mismatched IDs

**Error: "Flow table connectivity error"**

.. code-block:: bash

   # Verify flow table
   cat flowtable

   # Check:
   # - All patches have downstream connection (or -999 for outlet)
   # - Gamma values reasonable (0-1)
   # - Total_gamma sums correctly

**Simulation crashes - NaN values**

1. **Check parameter ranges** (especially Ksat, porosity)
2. **Verify forcing data** (no missing values, extremes)
3. **Increase spinup period** (carbon pools need equilibration)
4. **Reduce timestep** (if sub-daily)

**Unrealistic streamflow**

.. code-block:: yaml

   # Check key parameters:
   # - m (should be 0.1-5.0 for most cases)
   # - Ksat_0 (0.01-5.0 m/day typical)
   # - soil_depth (0.5-3.0 m common)

   # Recalibrate:
   RHESSYS_PARAMS_TO_CALIBRATE: "m,Ksat_0,sat_to_gw_coeff,soil_depth"

**Carbon pools unrealistic**

1. **Extend spinup** (100+ years for stable carbon)
2. **Check vegetation parameters** (GPP reasonable?)
3. **Verify climate forcing** (temperature, radiation)

**WMFire not working**

.. code-block:: yaml

   # Verify WMFire library compiled and accessible
   RHESSYS_USE_WMFIRE: true
   WMFIRE_INSTALL_PATH: /path/to/wmfire/lib
   WMFIRE_LIB: libwmfire.so

.. code-block:: bash

   # Check library exists
   ls /path/to/wmfire/lib/libwmfire.so

Performance Tips
================

**Speed up execution:**

1. **Hydrology only** (disable carbon/nitrogen)
2. **Coarser patches** (aggregate to 100m instead of 30m)
3. **Fewer hillslopes** (simplify topography if acceptable)
4. **Daily timestep** (avoid sub-daily if not needed)

**Improve accuracy:**

1. **Fine spatial resolution** (30m patches)
2. **Detailed hillslope representation**
3. **Full carbon and nitrogen cycles**
4. **Sub-daily timestep** (if data supports)
5. **Long spinup** (200+ years for stable carbon)

**Debug efficiently:**

.. code-block:: bash

   # Run RHESSys manually to see detailed output
   cd <project_dir>/settings/RHESSys/
   ../../installs/rhessys/rhessys -st 2015 1 1 1 -ed 2016 1 1 1 -w world -whdr world.hdr -r flow -t tecfile

Additional Resources
====================

**RHESSys Documentation:**

- Official site: https://github.com/RHESSys/RHESSys
- Wiki: https://github.com/RHESSys/RHESSys/wiki
- User guide: https://github.com/RHESSys/RHESSys/wiki/Documentation

**Publications:**

- Tague & Band (2004): "RHESSys: Regional Hydro-Ecologic Simulation System"
  https://doi.org/10.1016/j.envsoft.2003.10.005

- Tague et al. (2013): "Deep groundwater mediates streamflow response to climate variability"
  https://doi.org/10.1002/grl.50690

- Son & Tague (2019): "Hydrologic and biogeochemical controls on nitrogen cycling"
  https://doi.org/10.1002/hyp.13565

**WMFire:**

- Kennedy et al. (2017): "Modeling wildfire and hydrology"

**SYMFLUENCE-specific:**

- :doc:`../configuration`: RHESSys parameter reference
- :doc:`../calibration`: Calibration strategies
- :doc:`model_summa`: Comparison with physical models
- :doc:`../troubleshooting`: General troubleshooting

**Tools:**

- **RHESSys GIS preprocessing:** Tools for creating world files and flow tables from GIS data
- **GRASS GIS integration:** Many users use GRASS for RHESSys preprocessing

**Community:**

- RHESSys Google Group: RHESSys users and developers
- GitHub Issues: https://github.com/RHESSys/RHESSys/issues

**Training:**

- UNC workshops (periodic)
- Online tutorials: https://github.com/RHESSys/RHESSys/wiki/Tutorials
