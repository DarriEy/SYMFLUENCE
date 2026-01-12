=========================================
T-Route Routing Model Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

T-Route (NOAA-OWP t-route) is a high-performance channel routing model developed by NOAA's Office of Water Prediction. It is designed for operational river routing in the National Water Model (NWM) and supports efficient routing through large-scale river networks using vectorized computations.

**Key Capabilities:**

- Muskingum-Cunge routing with diffusive wave approximation
- High-performance vectorized computations
- Native integration with NGEN framework
- Support for reservoir and lake operations
- Efficient handling of large networks (NHDPlus scale)
- Python-based with Cython acceleration
- Real-time operational forecasting support

**Typical Applications:**

- NGEN hydrological model routing
- SUMMA runoff routing
- Operational streamflow forecasting
- NHDPlus network routing
- Continental-scale flood forecasting
- Multi-model ensemble routing

**Spatial Scales:** Sub-basin to continental (NHDPlus network)

**Temporal Resolution:** Sub-hourly to hourly

Routing Method
==============

T-Route implements a Muskingum-Cunge routing scheme:

Muskingum-Cunge Method
----------------------

**Method:** Variable-parameter Muskingum-Cunge

**Characteristics:**

- Physically-based wave propagation
- Diffusive wave approximation
- Accounts for channel geometry
- Variable celerity and diffusivity
- Handles backwater effects (limited)

**Core Equations:**

.. code-block:: text

   Q(x+Δx, t+Δt) = C1*Q(x, t+Δt) + C2*Q(x, t) + C3*Q(x+Δx, t) + C4*q_lateral

   where:
     C1, C2, C3, C4 = Muskingum coefficients (K, X dependent)
     K = reach travel time
     X = weighting factor (0 to 0.5)
     q_lateral = lateral inflow from hillslope runoff

**Parameters:**

- Channel length (m)
- Channel slope (m/m)
- Manning's roughness coefficient (n)
- Channel bottom width (m)
- Channel side slope
- Compound channel geometry (optional)

**Advantages:**

- Computationally efficient
- Handles complex networks
- Mass conservative
- Suitable for operational forecasting

Integration with SYMFLUENCE
===========================

T-Route integrates with SYMFLUENCE through the unified model framework, providing seamless routing of runoff from any supported hydrological model.

Workflow Position
-----------------

T-Route operates after hydrological model execution:

.. code-block:: text

   1. Hydrological Model (SUMMA/NGEN/FUSE/GR/HYPE)
      ↓
   2. Runoff Output (q_lateral per HRU)
      ↓
   3. T-Route Preprocessing (topology, remapping)
      ↓
   4. T-Route Execution
      ↓
   5. Routed Streamflow (at outlet/gages)

Configuration
=============

Basic Setup
-----------

Enable T-Route in your configuration YAML:

.. code-block:: yaml

   # Model Selection
   HYDROLOGICAL_MODEL: SUMMA    # or NGEN, FUSE, GR, HYPE
   ROUTING_MODEL: TROUTE

   # T-Route Settings
   TROUTE_FROM_MODEL: SUMMA     # Source model for runoff
   SETTINGS_TROUTE_TOPOLOGY: troute_topology.nc
   SETTINGS_TROUTE_CONFIG_FILE: troute_config.yml

Network Topology
----------------

T-Route requires a network topology file in NetCDF format with NWM-compatible variable names:

.. code-block:: yaml

   # Required variables in topology file:
   # - comid: Unique segment ID
   # - to_node: Downstream segment ID
   # - length: Segment length (meters)
   # - slope: Channel slope (m/m)
   # - link_id_hru: Segment ID for HRU discharge
   # - hru_area_m2: HRU contributing area

Channel Parameters
------------------

Configure channel geometry:

.. code-block:: yaml

   # Default channel parameters
   TROUTE_MANNINGS_N: 0.06
   TROUTE_BOTTOM_WIDTH: 10.0
   TROUTE_SIDE_SLOPE: 1.0
   TROUTE_CHANNEL_DEPTH: 5.0

Computational Settings
----------------------

.. code-block:: yaml

   # Parallel execution
   TROUTE_PARALLEL_MODE: true
   TROUTE_CPU_POOL: 4

   # Time stepping
   TROUTE_DT: 300              # Routing timestep (seconds)
   TROUTE_SUBSTEPPING: true

Input Requirements
==================

T-Route requires specific input files that SYMFLUENCE generates automatically:

1. Network Topology File
------------------------

NetCDF file containing river network structure:

- Segment connectivity (comid, to_node)
- Channel geometry (length, slope)
- HRU-to-segment mapping
- Channel parameters (Manning's n, widths)

2. Runoff Forcing File
----------------------

NetCDF file with lateral inflows from hydrological model:

.. code-block:: text

   Variables:
     q_lateral(time, segment)  # Lateral inflow [m³/s]

   Dimensions:
     time: simulation timesteps
     segment: river segments matching topology

3. Configuration File
---------------------

YAML configuration specifying:

- Network topology path
- Forcing file path
- Output location
- Routing parameters
- Computational options

Output Files
============

T-Route produces routed streamflow output:

Standard Outputs
----------------

.. code-block:: text

   simulations/<experiment_id>/troute/
   ├── flowveldepth_<timestamp>.nc   # Streamflow, velocity, depth
   └── channel_restart_<timestamp>.nc # Restart file

Output Variables
----------------

.. code-block:: yaml

   # Primary output variables:
   # - streamflow: Routed discharge (m³/s)
   # - velocity: Flow velocity (m/s)
   # - depth: Flow depth (m)

   # Per segment, per timestep

Preprocessing Details
=====================

The TRoutePreProcessor handles:

1. **Topology Generation**: Creates NetCDF topology from SYMFLUENCE shapefiles
2. **Variable Mapping**: Maps SYMFLUENCE column names to T-Route conventions
3. **Parameter Assignment**: Sets default channel parameters
4. **Configuration Generation**: Creates T-Route YAML config file

Shapefile Requirements
----------------------

SYMFLUENCE uses river network and basin shapefiles:

.. code-block:: yaml

   # River network shapefile columns (configurable):
   RIVER_NETWORK_SHP_SEGID: seg_id       # Segment ID
   RIVER_NETWORK_SHP_DOWNSEGID: dseg_id  # Downstream segment
   RIVER_NETWORK_SHP_LENGTH: Length      # Segment length
   RIVER_NETWORK_SHP_SLOPE: Slope        # Channel slope

   # Basin shapefile columns:
   RIVER_BASIN_SHP_HRU_TO_SEG: HRU_seg   # HRU to segment mapping
   RIVER_BASIN_SHP_AREA: area_m2         # HRU area

Coupling with Hydrological Models
=================================

SUMMA Coupling
--------------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: SUMMA
   ROUTING_MODEL: TROUTE
   TROUTE_FROM_MODEL: SUMMA

   # SUMMA output variable mapping
   # averageInstantRunoff → q_lateral

NGEN Coupling
-------------

T-Route is the native routing component for NGEN:

.. code-block:: yaml

   HYDROLOGICAL_MODEL: NGEN
   ROUTING_MODEL: TROUTE
   TROUTE_FROM_MODEL: NGEN

   # Direct coupling via NGEN realization

FUSE/GR/HYPE Coupling
---------------------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: FUSE  # or GR, HYPE
   ROUTING_MODEL: TROUTE
   TROUTE_FROM_MODEL: FUSE

   # Runoff extraction from model outputs

Troubleshooting
===============

Common Issues
-------------

**T-Route module not found:**

.. code-block:: bash

   # Install t-route
   pip install nwm-routing

   # Or from source:
   git clone https://github.com/NOAA-OWP/t-route.git
   cd t-route && pip install -e .

**Topology file errors:**

- Verify segment IDs are unique
- Check downstream connectivity (no orphan segments)
- Ensure all HRUs map to valid segments

**Runoff variable mismatch:**

- Check TROUTE_FROM_MODEL matches your hydrological model
- Verify runoff variable name in source output
- Ensure time dimensions align

**Memory issues with large networks:**

.. code-block:: yaml

   # Enable chunked processing
   TROUTE_CHUNK_SIZE: 1000
   TROUTE_MEMORY_EFFICIENT: true

Performance Tips
----------------

1. **Use parallel mode** for networks > 1000 segments
2. **Match timesteps** between hydro model and routing
3. **Pre-compute topology** for repeated runs
4. **Use restart files** for long simulations

References
==========

**T-Route Documentation:**

- GitHub: https://github.com/NOAA-OWP/t-route
- NOAA-OWP: https://water.noaa.gov/

**Technical References:**

- Cunge, J.A. (1969). On the subject of a flood propagation computation method (Muskingum method). Journal of Hydraulic Research.
- NOAA National Water Model documentation

**Related SYMFLUENCE Documentation:**

- :doc:`mizuroute` — Alternative routing model
- :doc:`../models/model_ngen` — NGEN integration
- :doc:`../models/model_summa` — SUMMA coupling
- :doc:`../configuration` — Full configuration reference
