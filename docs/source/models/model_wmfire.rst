.. _models_wmfire:

=========================================
WM-Fire Model Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

WM-Fire is a **wildfire spread simulation module** integrated with RHESSys (Regional
Hydro-Ecologic Simulation System). It simulates fire spread across landscapes based
on fuel availability, moisture conditions, topography, and wind dynamics, enabling
coupled ecohydrology-fire modeling.

**Key Capabilities:**

- Spatially explicit fire spread simulation
- Coupling with RHESSys litter carbon and soil moisture
- Multiple fuel moisture models (Nelson 2000)
- Configurable ignition points and spread parameters
- Fire perimeter validation against observations
- GeoTIFF and RHESSys-compatible output formats

**Typical Applications:**

- Watershed-scale wildfire modeling
- Fire effects on ecohydrology
- Fuel management scenario analysis
- Fire risk assessment
- Post-fire hydrological response studies

**Spatial Resolution:** 10-200m (recommended 30m)

**Temporal Resolution:** 1-24 hours

**Reference:** Kennedy, M.C., McKenzie, D., Tague, C., Dugger, A.L. (2017).
"Balancing uncertainty and complexity to incorporate fire spread in an eco-hydrological model."
International Journal of Wildland Fire, 26(8): 706-718.

Model Physics and Structure
===========================

Fire Spread Dynamics
--------------------

WM-Fire models fire spread probability based on:

1. **Fuel Load:**

   - Derived from RHESSys litter carbon pools (litr1c-litr4c)
   - Weighted by decomposition rate (faster = more flammable)

2. **Fuel Moisture:**

   - Nelson (2000) equilibrium moisture equations
   - 1hr, 10hr, 100hr, 1000hr timelag fuels
   - Temperature and humidity dependent

3. **Topographic Effects:**

   - Slope acceleration of fire spread
   - Aspect influence on fuel drying

4. **Wind Dynamics:**

   - Log-normal wind speed distribution
   - von Mises-Fisher directional distribution

Fuel Calculation
----------------

Fuel load is calculated from RHESSys litter carbon:

.. code-block:: text

   Fuel = sum(litter_pool[i] * weight[i] * carbon_to_fuel_ratio)

   Pool weights (default):
   - litr1c (labile):       0.35  (leaves, fast decomposing)
   - litr2c (cellulose):    0.30  (medium decomposing)
   - litr3c (lignin):       0.25  (woody, slow decomposing)
   - litr4c (recalcitrant): 0.10  (very slow decomposing)

Moisture Dynamics
-----------------

The Nelson (2000) equilibrium moisture model:

.. code-block:: text

   EMC = f(relative_humidity, temperature)

   Moisture update:
   mc(t+dt) = emc + (mc(t) - emc) * exp(-dt/timelag)

Fuel size classes:

.. list-table::
   :header-rows: 1
   :widths: 15 20 25 40

   * - Class
     - Timelag
     - Diameter
     - Application
   * - 1hr
     - 1 hour
     - <6mm
     - Fine fuels, leaves
   * - 10hr
     - 10 hours
     - 6-25mm
     - Small twigs
   * - 100hr
     - 100 hours
     - 25-75mm
     - Medium branches
   * - 1000hr
     - 1000 hours
     - >75mm
     - Large logs

Configuration in SYMFLUENCE
===========================

Enabling WM-Fire
----------------

WM-Fire is enabled as part of RHESSys configuration:

.. code-block:: yaml

   HYDROLOGICAL_MODEL: RHESSYS
   RHESSYS_USE_WMFIRE: true

Key Configuration Parameters
----------------------------

Grid Configuration
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - WMFIRE_GRID_RESOLUTION
     - 30
     - Fire grid resolution in meters (10-200)
   * - WMFIRE_TIMESTEP_HOURS
     - 24
     - Simulation timestep in hours (1-24)

Fuel and Moisture
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Parameter
     - Default
     - Description
   * - WMFIRE_NDAYS_AVERAGE
     - 30.0
     - Fuel moisture averaging window (days)
   * - WMFIRE_FUEL_SOURCE
     - static
     - Fuel source (static, rhessys_litter)
   * - WMFIRE_MOISTURE_SOURCE
     - static
     - Moisture source (static, rhessys_soil)
   * - WMFIRE_CARBON_TO_FUEL_RATIO
     - 2.0
     - Carbon to fuel conversion (1.0-5.0)

Ignition Configuration
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Parameter
     - Default
     - Description
   * - WMFIRE_IGNITION_SHAPEFILE
     - null
     - Path to ignition point shapefile
   * - WMFIRE_IGNITION_POINT
     - null
     - Ignition coordinates as "lat/lon"
   * - WMFIRE_IGNITION_DATE
     - null
     - Ignition date (YYYY-MM-DD)
   * - WMFIRE_IGNITION_NAME
     - ignition
     - Ignition point name

Ignition priority:

1. Shapefile (if specified and exists)
2. Coordinates (if specified)
3. Random (ignition_col=-1, ignition_row=-1 in fire.def)

Fire Perimeter Validation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Parameter
     - Default
     - Description
   * - WMFIRE_PERIMETER_SHAPEFILE
     - null
     - Observed fire perimeter shapefile
   * - WMFIRE_PERIMETER_DIR
     - null
     - Directory with perimeter shapefiles

Output Configuration
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - WMFIRE_WRITE_GEOTIFF
     - true
     - Write GeoTIFF outputs for visualization

Spread Coefficients
^^^^^^^^^^^^^^^^^^^

Optional overrides (use null for defaults from literature):

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - WMFIRE_LOAD_K1
     - 3.9
     - Fuel load coefficient
   * - WMFIRE_LOAD_K2
     - 0.07
     - Fuel load sensitivity
   * - WMFIRE_MOISTURE_K1
     - 3.8
     - Moisture coefficient
   * - WMFIRE_MOISTURE_K2
     - 0.27
     - Moisture sensitivity

Fire Definition Parameters
==========================

The fire.def file contains all WM-Fire parameters. Key categories:

Grid Dimensions
---------------

.. code-block:: text

   n_rows          : Grid rows (must match fire grids)
   n_cols          : Grid columns (must match fire grids)

Spread Coefficients
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Coefficient
     - k1
     - k2
     - Purpose
   * - load_k1/k2
     - 3.9
     - 0.07
     - Fuel load effect on spread
   * - slope_k1/k2
     - 0.91
     - 1.0
     - Topographic slope effect
   * - moisture_k1/k2
     - 3.8
     - 0.27
     - Fuel moisture effect
   * - winddir_k1/k2
     - 0.87
     - 0.48
     - Wind direction/speed effect

Wind Parameters
---------------

.. code-block:: text

   mean_log_wind   : Mean of log(wind) = 0.494
   sd_log_wind     : Std dev of log(wind) = 0.654

   # von Mises-Fisher distribution
   mean1_rvm       : First mean direction
   mean2_rvm       : Second mean direction
   kappa1_rvm      : First concentration
   kappa2_rvm      : Second concentration
   p_rvm           : Mixture probability = 0.411

Output Files
============

Fire Grids
----------

**Patch Grid:**

- ``patch_grid.txt`` - RHESSys text format
- ``patch_grid.tif`` - GeoTIFF (if enabled)

**DEM Grid:**

- ``dem_grid.txt`` - RHESSys text format
- ``dem_grid.tif`` - GeoTIFF (if enabled)

Fire Definition
---------------

- ``fire.def`` - RHESSys fire parameter file

Validation Outputs
------------------

- ``comparison_map.png`` - Visual comparison of simulated vs observed
- Metrics: IoU, Dice coefficient, commission/omission rates

Usage Examples
==============

Basic Fire Simulation
---------------------

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: fire_basin
   HYDROLOGICAL_MODEL: RHESSYS

   # Enable WM-Fire
   RHESSYS_USE_WMFIRE: true

   # Grid configuration
   WMFIRE_GRID_RESOLUTION: 30
   WMFIRE_TIMESTEP_HOURS: 24

   # Ignition point
   WMFIRE_IGNITION_POINT: "51.2096/-115.7539"
   WMFIRE_IGNITION_DATE: "2014-07-15"
   WMFIRE_IGNITION_NAME: "Lightning_Strike"

Fire with Observed Perimeter
----------------------------

.. code-block:: yaml

   # Ignition from shapefile
   WMFIRE_IGNITION_SHAPEFILE: /data/fires/ignition_2014.shp

   # Validation perimeter
   WMFIRE_PERIMETER_SHAPEFILE: /data/fires/perimeter_2014.shp

   # Output
   WMFIRE_WRITE_GEOTIFF: true

Python API Usage
----------------

.. code-block:: python

   from symfluence.models.wmfire import (
       FireGridManager,
       FireDefGenerator,
       FuelCalculator,
       FuelMoistureModel,
       IgnitionManager,
       FirePerimeterValidator
   )
   import geopandas as gpd

   # 1. Create fire grids from catchment
   grid_manager = FireGridManager(config)
   catchment_gdf = gpd.read_file('catchment.shp')
   patch_grid, dem_grid = grid_manager.create_fire_grid(catchment_gdf)

   # 2. Calculate fuel from litter pools
   fuel_calc = FuelCalculator(carbon_to_fuel_ratio=2.0)
   litter_pools = {'litr1c': 0.5, 'litr2c': 1.0, 'litr3c': 1.5, 'litr4c': 0.5}
   fuel_load = fuel_calc.calculate_fuel_load(litter_pools)
   load_k1, load_k2 = fuel_calc.calculate_load_coefficients(fuel_load)

   # 3. Calculate moisture
   moisture_model = FuelMoistureModel(fuel_class='10hr')
   emc = moisture_model.equilibrium_moisture(rh=30, temp_c=25)
   moist_k1, moist_k2 = moisture_model.calculate_moisture_coefficients(0.15)

   # 4. Setup ignition
   ign_manager = IgnitionManager(config)
   ignition = ign_manager.get_ignition_point()
   if ignition:
       ign_row, ign_col = ign_manager.convert_to_grid_indices(
           ignition, patch_grid.transform, patch_grid.crs,
           patch_grid.nrows, patch_grid.ncols
       )

   # 5. Generate fire.def
   gen = FireDefGenerator(config)
   gen.write_fire_def(
       'fire.def',
       patch_grid,
       fuel_stats={'load_k1': load_k1, 'load_k2': load_k2},
       moisture_stats={'moisture_k1': moist_k1, 'moisture_k2': moist_k2},
       ignition_row=ign_row,
       ignition_col=ign_col
   )

   # 6. Export grids
   patch_grid.to_geotiff('patch_grid.tif')
   dem_grid.to_geotiff('dem_grid.tif')

Perimeter Validation
--------------------

.. code-block:: python

   # Load perimeters
   validator = FirePerimeterValidator()
   observed = validator.load_perimeters('observed_perimeter.shp')
   simulated = gpd.read_file('simulated_perimeter.shp')

   # Compare
   metrics = validator.compare_perimeters(simulated, observed)
   print(f"IoU: {metrics['iou']:.3f}")
   print(f"Dice: {metrics['dice']:.3f}")
   print(f"Simulated: {metrics['simulated_area_ha']:.1f} ha")
   print(f"Observed: {metrics['observed_area_ha']:.1f} ha")

   # Create comparison map
   validator.create_comparison_map(
       simulated, observed,
       output_path='comparison.png',
       title='Fire Perimeter Comparison'
   )

Validation Metrics
==================

The perimeter validator provides:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric
     - Description
   * - IoU (Jaccard)
     - Intersection / Union
   * - Dice Coefficient
     - 2 * Intersection / (Simulated + Observed)
   * - Area Ratio
     - Simulated Area / Observed Area
   * - Commission Rate
     - False positives (simulated but not observed)
   * - Omission Rate
     - False negatives (observed but not simulated)

Dependencies
============

Core Dependencies
-----------------

.. code-block:: text

   numpy          : Array operations
   geopandas      : Spatial data handling (optional)
   shapely        : Geometry operations
   rasterio       : GeoTIFF I/O (optional)
   pyproj         : Coordinate transformations

Build Dependencies
------------------

For the WM-Fire C++ library:

.. code-block:: text

   boost          : Boost headers
   g++/clang      : C++ compiler

Installation
------------

Boost installation:

.. code-block:: bash

   # macOS
   brew install boost

   # Ubuntu/Debian
   apt-get install libboost-dev

   # conda
   conda install -c conda-forge boost

The WM-Fire library is automatically built during RHESSys setup.

Known Limitations
=================

1. **Fuel/Moisture Sources:**

   - Currently static values only
   - Dynamic rhessys_litter and rhessys_soil sources planned

2. **Fire Effects:**

   - Can simulate spread
   - Feedback to RHESSys (burn severity) in development

3. **Resolution Trade-offs:**

   - <30m: Computationally intensive
   - >100m: Loss of spatial detail

4. **Multi-Fire:**

   - Single fire event assumed
   - Limited multi-fire scenario support

Troubleshooting
===============

Common Issues
-------------

**Error: "rasterio not available"**

Falls back to point-sampling for grid creation:

.. code-block:: bash

   pip install rasterio

**Error: "Boost headers not found"**

.. code-block:: bash

   # Set BOOST_ROOT
   export BOOST_ROOT=/path/to/boost

   # Or install via package manager
   brew install boost  # macOS

**Grid dimension mismatch**

Ensure patch_grid and dem_grid have same dimensions:

.. code-block:: python

   print(f"Patch: {patch_grid.nrows} x {patch_grid.ncols}")
   print(f"DEM: {dem_grid.nrows} x {dem_grid.ncols}")

Additional Resources
====================

- :doc:`model_rhessys` - RHESSys model documentation
- :doc:`../configuration` - Full parameter reference
- Kennedy et al. (2017): https://doi.org/10.1071/WF16080
