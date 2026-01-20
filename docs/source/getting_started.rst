Getting Started
===============

This guide walks you through setting up and running your first SYMFLUENCE workflow — from creating a configuration file to executing a full hydrological simulation.

---

Quick Start
-----------

1. **Create your configuration**

   Copy and edit the provided template:

   .. code-block:: bash

      cp src/symfluence/resources/config_templates/config_template.yaml my_project.yaml

2. **Define your study domain**

   You can specify:
   - A pour point (latitude/longitude)
   - A bounding box
   - An existing shapefile

   Example (YAML):

   .. code-block:: yaml

      # Pour point coordinates
      POUR_POINT_COORDS: 51.1722/-115.5717

      # Or bounding box (lat_max/lon_min/lat_min/lon_max)
      BOUNDING_BOX_COORDS: 52.0/-116.0/50.0/-114.0

3. **Run the workflow**

   .. code-block:: bash

      # Run project setup
      symfluence workflow step setup_project --config my_project.yaml

      # Run full workflow
      symfluence workflow run --config my_project.yaml

      # Or run individual steps
      symfluence workflow steps setup_project calibrate_model --config my_project.yaml

---

Complete Walkthrough
---------------------

This section provides a step-by-step walkthrough of a complete SYMFLUENCE workflow with expected outputs and verification steps.

**Example Domain**: Bow River at Banff, Alberta, Canada

Prerequisites
~~~~~~~~~~~~~

1. SYMFLUENCE installed (see :doc:`installation`)
2. At least 10 GB free disk space
3. Internet connection for data acquisition
4. 2-4 hours for complete workflow (depending on system)

Step 1: Initialize Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create project directory
   export SYMFLUENCE_DATA_DIR="/path/to/data"
   cd $SYMFLUENCE_DATA_DIR

   # Initialize configuration from template
   symfluence project init bow_river

**Expected Output:**

.. code-block:: text

   ✓ Created project directory: bow_river/
   ✓ Copied configuration template: bow_river/config.yaml
   ✓ Project initialized successfully

**Verification:**

.. code-block:: bash

   ls bow_river/
   # Should show: config.yaml

**Estimated Time:** < 1 minute

Step 2: Configure Your Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit the configuration file:

.. code-block:: bash

   nano bow_river/config.yaml

Key parameters to set:

.. code-block:: yaml

   # Domain definition
   DOMAIN_NAME: "Bow_at_Banff"
   POUR_POINT_COORDS: "51.1722/-115.5717"
   DOMAIN_DEFINITION_METHOD: "semidistributed"

   # Time period (start small for testing)
   EXPERIMENT_TIME_START: "2018-01-01"
   EXPERIMENT_TIME_END: "2018-12-31"
   SPINUP_PERIOD: "2017-10-01,2017-12-31"

   # Model selection
   HYDROLOGICAL_MODEL: "SUMMA"

   # Forcing data
   FORCING_DATASET: "ERA5"

**Verification:**

.. code-block:: bash

   # Validate configuration
   symfluence config validate --config bow_river/config.yaml

**Expected Output:**

.. code-block:: text

   ✓ YAML syntax valid
   ✓ All required parameters present
   ✓ Parameter types correct
   ✓ Date ranges valid
   Configuration validation passed

**Estimated Time:** 5-10 minutes

Step 3: Set Up Project Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   symfluence workflow step setup_project --config bow_river/config.yaml

**Expected Output:**

.. code-block:: text

   [INFO] Starting project setup
   [INFO] Creating directory structure
   [INFO] Initializing logging system
   ✓ Created: attributes/
   ✓ Created: forcing/
   ✓ Created: shapefiles/
   ✓ Created: settings/
   ✓ Created: simulations/
   ✓ Created: results/
   ✓ Created: _workLog_Bow_at_Banff/
   [INFO] Project setup complete

**Verification:**

.. code-block:: bash

   ls $SYMFLUENCE_DATA_DIR/domain/Bow_at_Banff/
   # Should show: attributes/ forcing/ shapefiles/ settings/ etc.

**Estimated Time:** < 1 minute

Step 4: Delineate Domain
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   symfluence workflow step define_domain --config bow_river/config.yaml

**Expected Output:**

.. code-block:: text

   [INFO] Starting domain delineation
   [INFO] Downloading DEM data for region
   [INFO] Running TauDEM flow accumulation
   [INFO] Identifying stream network (threshold: 7500 cells)
   [INFO] Delineating catchment from pour point
   ✓ Catchment area: 2210 km²
   ✓ Stream length: 45 km
   ✓ Mean elevation: 1650 m
   [INFO] Saved catchment shapefile: shapefiles/catchment/Bow_at_Banff.shp
   [INFO] Domain delineation complete

**Verification:**

.. code-block:: bash

   # Check shapefile exists
   ls $SYMFLUENCE_DATA_DIR/domain/Bow_at_Banff/shapefiles/catchment/
   # Should show: Bow_at_Banff.shp, .shx, .dbf, .prj

   # Check catchment statistics in log
   tail -20 _workLog_Bow_at_Banff/domain_delineation.log

**Estimated Time:** 5-15 minutes

Step 5: Discretize Domain
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   symfluence workflow step discretize_domain --config bow_river/config.yaml

**Expected Output:**

.. code-block:: text

   [INFO] Starting domain discretization
   [INFO] Discretization method: elevation
   [INFO] Elevation band size: 400 m
   [INFO] Generating elevation bands
   ✓ Created 8 HRUs
   ✓ Elevation range: 1300-2900 m
   ✓ Saved: shapefiles/catchment/Bow_at_Banff_HRUs_elevation.shp
   [INFO] Discretization complete

**Verification:**

.. code-block:: bash

   # Visualize HRUs (if QGIS installed)
   qgis shapefiles/catchment/Bow_at_Banff_HRUs_elevation.shp

   # Or check HRU count
   ogrinfo -al shapefiles/catchment/Bow_at_Banff_HRUs_elevation.shp | grep "Feature Count"

**Estimated Time:** 5-10 minutes

Step 6: Acquire Attribute Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   symfluence workflow step acquire_attributes --config bow_river/config.yaml

**Expected Output:**

.. code-block:: text

   [INFO] Acquiring attribute data
   [INFO] Downloading DEM (Copernicus 30m)
   ✓ Downloaded: attributes/elevation/dem/Bow_at_Banff_elv.tif
   [INFO] Downloading land cover (MODIS)
   ✓ Downloaded: attributes/landclass/Bow_at_Banff_land_classes.tif
   [INFO] Downloading soil data (USDA-NRCS)
   ✓ Downloaded: attributes/soilclass/Bow_at_Banff_soil_classes.tif
   [INFO] Attribute acquisition complete

**Verification:**

.. code-block:: bash

   ls attributes/*/
   # Should show DEM, land cover, and soil files

   # Check DEM metadata
   gdalinfo attributes/elevation/dem/Bow_at_Banff_elv.tif

**Estimated Time:** 10-30 minutes (depends on network speed)

Step 7: Acquire Forcing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   symfluence workflow step acquire_forcings --config bow_river/config.yaml

**Expected Output:**

.. code-block:: text

   [INFO] Acquiring forcing data: ERA5
   [INFO] Time period: 2017-10-01 to 2018-12-31
   [INFO] Variables: airtemp, pptrate, spechum, windspd, SWRadAtm, LWRadAtm
   [INFO] Downloading data from CDS
   ✓ Downloaded: forcing/raw_data/ERA5_201710.nc
   ✓ Downloaded: forcing/raw_data/ERA5_201711.nc
   ...
   ✓ Downloaded 15 monthly files
   [INFO] Forcing acquisition complete

**Verification:**

.. code-block:: bash

   ls forcing/raw_data/
   # Should show ERA5_*.nc files

   # Check one file
   ncdump -h forcing/raw_data/ERA5_201801.nc | head -20

**Estimated Time:** 20-60 minutes (depends on CDS queue)

Step 8: Preprocess Data
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   symfluence workflow step run_model_agnostic_preprocessing --config bow_river/config.yaml

**Expected Output:**

.. code-block:: text

   [INFO] Starting model-agnostic preprocessing
   [INFO] Merging forcing files
   [INFO] Spatial remapping to catchment
   [INFO] Temporal interpolation
   [INFO] Variable unit conversions
   ✓ Processed: forcing/basin_averaged_forcing/Bow_at_Banff_ERA5_merged.nc
   [INFO] Preprocessing complete

**Verification:**

.. code-block:: bash

   # Check merged file
   ncdump -h forcing/basin_averaged_forcing/Bow_at_Banff_ERA5_merged.nc

   # Verify time coverage
   ncdump -v time forcing/basin_averaged_forcing/Bow_at_Banff_ERA5_merged.nc | tail -10

**Estimated Time:** 10-20 minutes

Step 9: Model-Specific Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   symfluence workflow step preprocess_models --config bow_river/config.yaml

**Expected Output:**

.. code-block:: text

   [INFO] Starting SUMMA preprocessing
   [INFO] Processing forcing data to SUMMA format
   [INFO] Generating attribute files
   ✓ Created: forcing/SUMMA_input/Bow_at_Banff_forcing.nc
   ✓ Created: forcing/SUMMA_input/Bow_at_Banff_attributes.nc
   [INFO] Creating SUMMA configuration files
   ✓ Created: settings/SUMMA/fileManager.txt
   ✓ Created: settings/SUMMA/decisionsFile.txt
   ✓ Created: settings/SUMMA/outputControl.txt
   [INFO] SUMMA preprocessing complete

**Verification:**

.. code-block:: bash

   ls forcing/SUMMA_input/
   ls settings/SUMMA/

**Estimated Time:** 5-15 minutes

Step 10: Run Model
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   symfluence workflow step run_models --config bow_river/config.yaml

**Expected Output:**

.. code-block:: text

   [INFO] Starting SUMMA execution
   [INFO] Running 8 HRUs
   HRU 1/8 complete (12.5%)
   HRU 2/8 complete (25.0%)
   ...
   HRU 8/8 complete (100%)
   [INFO] SUMMA execution complete
   [INFO] Adding MIZUROUTE to workflow (dependency of SUMMA)
   [INFO] Running MIZUROUTE
   ✓ Routing complete
   [INFO] Model execution complete

**Verification:**

.. code-block:: bash

   # Check SUMMA output
   ls simulations/*/SUMMA/

   # Check mizuRoute output
   ls simulations/*/mizuRoute/

**Estimated Time:** 10-30 minutes

Step 11: Extract Results
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   symfluence workflow step postprocess_results --config bow_river/config.yaml

**Expected Output:**

.. code-block:: text

   [INFO] Post-processing SUMMA results
   [INFO] Extracting streamflow
   ✓ Saved: results/baseline_01_results.csv
   [INFO] Generating plots
   ✓ Created: results/streamflow_timeseries.png
   [INFO] Calculating baseline metrics
   ============================================================
   BASELINE MODEL PERFORMANCE (before calibration)
   ============================================================
     SUMMA:
       KGE  = 0.4521
       KGE' = 0.4312
       NSE  = 0.3987
       Bias = +12.3%
       Valid data points: 365
     Note: KGE < 0.5 suggests calibration may significantly improve results
   ============================================================

**Verification:**

.. code-block:: bash

   # View results
   head results/baseline_01_results.csv

   # View plot
   open results/streamflow_timeseries.png  # macOS
   # or
   xdg-open results/streamflow_timeseries.png  # Linux

**Estimated Time:** 2-5 minutes

Summary of Expected Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After completing all steps, your project directory should contain:

.. code-block:: text

   domain/Bow_at_Banff/
   ├── attributes/
   │   ├── elevation/dem/Bow_at_Banff_elv.tif
   │   ├── landclass/Bow_at_Banff_land_classes.tif
   │   └── soilclass/Bow_at_Banff_soil_classes.tif
   ├── forcing/
   │   ├── raw_data/ERA5_*.nc (15 files)
   │   ├── basin_averaged_forcing/Bow_at_Banff_ERA5_merged.nc
   │   └── SUMMA_input/Bow_at_Banff_forcing.nc
   ├── shapefiles/
   │   ├── catchment/Bow_at_Banff.shp
   │   └── catchment/Bow_at_Banff_HRUs_elevation.shp
   ├── settings/
   │   ├── SUMMA/fileManager.txt
   │   └── mizuRoute/mizuRoute_control.txt
   ├── simulations/
   │   └── baseline_01/
   │       ├── SUMMA/output/
   │       └── mizuRoute/output/
   ├── results/
   │   ├── baseline_01_results.csv
   │   └── streamflow_timeseries.png
   └── _workLog_Bow_at_Banff/
       ├── system.log
       ├── domain_delineation.log
       ├── data_acquisition.log
       └── model_run.log

**Total Disk Usage:** ~8-10 GB

**Total Time:** 1.5-3 hours (mostly data acquisition)

Troubleshooting Common Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter issues, see:

- **Data acquisition fails**: Check internet connection and CDS credentials (:doc:`installation`)
- **Domain delineation fails**: Verify pour point coordinates and TauDEM installation (:doc:`troubleshooting`)
- **Model execution fails**: Check log files in ``_workLog_*/`` (:doc:`troubleshooting`)
- **Poor baseline performance**: Normal before calibration; see :doc:`calibration`

Next Steps
~~~~~~~~~~

After completing this walkthrough:

1. **Calibrate your model**: See :doc:`calibration` for parameter optimization
2. **Try different models**: Change ``HYDROLOGICAL_MODEL`` to FUSE, GR, or HYPE
3. **Extend time period**: Increase simulation length for longer analysis
4. **Add observation data**: Include streamflow observations for validation
5. **Explore examples**: See :doc:`examples` for advanced workflows

---

Example: Bow River Watershed
----------------------------

.. literalinclude:: ../../src/symfluence/resources/config_templates/config_template.yaml
   :language: yaml
   :caption: Example configuration (excerpt)
   :lines: 1-40

This example performs:
1. Domain delineation and forcing data setup
2. Model configuration (e.g., SUMMA, FUSE, NextGen, GR4J, LSTM)
3. Simulation execution
4. Routing using mizuRoute
5. Output evaluation and visualization

---

Understanding the Workflow
--------------------------

Each SYMFLUENCE run follows a structured pipeline:

1. **Domain Definition** — delineate watershed or region
2. **Data Acquisition** — retrieve and preprocess forcing datasets (ERA5, Daymet, etc.)
3. **Model Setup** — configure supported models
4. **Simulation** — execute model runs
5. **Routing & Evaluation** — route flows and compute diagnostics
6. **Reporting** — generate plots, metrics, and summaries

Each step can be called individually using the workflow command: ``symfluence workflow step <step_name>`` (e.g., ``workflow step setup_project`` or ``workflow step calibrate_model``).

---

Next Steps
----------

- Explore the :doc:`configuration` structure in detail
- Try the progressive :doc:`examples` for advanced applications
- Visit :doc:`troubleshooting` for setup or runtime guidance
