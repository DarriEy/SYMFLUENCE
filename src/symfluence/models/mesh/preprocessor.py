"""
MESH model preprocessor.

Handles data preparation using meshflow library for MESH model setup.
Uses meshflow exclusively for all preprocessing - both lumped and distributed modes.
"""

import os
from typing import Dict, Any
from pathlib import Path
import shutil

try:
    from meshflow.core import MESHWorkflow
    MESHFLOW_AVAILABLE = True
except ImportError as e:
    import logging
    logging.debug(f"meshflow import failed: {e}. MESH preprocessing will be limited.")
    MESHFLOW_AVAILABLE = False

    # Fallback placeholder
    class MESHWorkflow:
        def __init__(self, **kwargs):
            logging.debug("MESHWorkflow placeholder - meshflow not available")
            pass
        def run(self, save_path=None):
            pass
        def save(self, output_dir):
            pass


from ..base import BaseModelPreProcessor
from ..mixins import ObservationLoaderMixin
from ..registry import ModelRegistry
from symfluence.core.exceptions import ConfigurationError, ModelExecutionError


@ModelRegistry.register_preprocessor('MESH')
class MESHPreProcessor(BaseModelPreProcessor, ObservationLoaderMixin):
    """
    Preprocessor for the MESH model.

    Handles data preparation using meshflow library for MESH model setup.
    All preprocessing is done through meshflow for both lumped and distributed modes.

    Attributes:
        config: Configuration settings for MESH
        logger: Logger object for recording processing information
        project_dir: Directory for the current project
        setup_dir: Directory for MESH setup files (inherited)
        domain_name: Name of the domain being processed (inherited)
    """

    def _get_model_name(self) -> str:
        """Return model name for MESH."""
        return "MESH"

    def __init__(self, config: Dict[str, Any], logger: Any):
        # Initialize base class (handles common paths)
        super().__init__(config, logger)

        # MESH-specific catchment path (uses river basins instead of catchment)
        self.catchment_path = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')

        # Phase 3: Use typed config when available
        if self.config:
            self.catchment_name = self.config.paths.river_basins_name
            if self.catchment_name == 'default':
                self.catchment_name = f"{self.domain_name}_riverBasins_{self.config.domain.definition_method}.shp"
        else:
            self.catchment_name = self.config_dict.get('RIVER_BASINS_NAME')
            if self.catchment_name == 'default':
                self.catchment_name = f"{self.domain_name}_riverBasins_{self.config_dict.get('DOMAIN_DEFINITION_METHOD')}.shp"

        # River network paths
        self.rivers_path = self.get_river_network_path().parent
        self.rivers_name = self.get_river_network_path().name

    def _get_spatial_mode(self) -> str:
        """
        Determine MESH spatial mode from configuration.

        Returns:
            'lumped' or 'distributed'
        """
        # Check explicit setting first
        spatial_mode = self.config_dict.get('MESH_SPATIAL_MODE', 'auto')

        if spatial_mode != 'auto':
            return spatial_mode

        # Auto-detect from DOMAIN_DEFINITION_METHOD
        domain_method = self.config_dict.get('DOMAIN_DEFINITION_METHOD', 'lumped')

        if domain_method in ['point', 'lumped']:
            return 'lumped'
        elif domain_method in ['delineate', 'semi_distributed', 'distributed']:
            return 'distributed'

        return 'lumped'  # Default fallback

    def run_preprocessing(self):
        """
        Run the complete MESH preprocessing workflow.

        Uses the template method pattern from BaseModelPreProcessor.
        """
        self.logger.info("Starting MESH preprocessing")
        return self.run_preprocessing_template()

    def _pre_setup(self) -> None:
        """MESH-specific pre-setup: create meshflow config (template hook)."""
        self._meshflow_config = self._create_meshflow_config()

    def _prepare_forcing(self) -> None:
        """MESH-specific forcing data preparation using meshflow (template hook)."""
        spatial_mode = self._get_spatial_mode()
        self.logger.info(f"MESH spatial mode: {spatial_mode}")

        # Use meshflow for all preprocessing
        self._run_meshflow(self._meshflow_config)

    def _create_model_configs(self) -> None:
        """Create MESH-specific configuration files (template hook)."""
        self.logger.info("Creating MESH configuration files")

        # Create run options (always regenerate to ensure correct flags)
        # Unless meshflow already created one
        run_options_path = self.forcing_dir / "MESH_input_run_options.ini"
        if not run_options_path.exists():
            self._create_run_options()
        else:
            self.logger.info(f"Using existing run options from meshflow")

        # Create CLASS and hydrology parameter files (only if not already created by meshflow)
        class_params_path = self.forcing_dir / "MESH_parameters_CLASS.ini"
        if not class_params_path.exists():
            self._create_class_parameters()
        else:
            self.logger.info(f"Using existing CLASS parameters from meshflow")

        hydro_params_path = self.forcing_dir / "MESH_parameters_hydrology.ini"
        if not hydro_params_path.exists():
            self._create_hydrology_parameters()
        else:
            self.logger.info(f"Using existing hydrology parameters from meshflow")

        # Create streamflow input file with gauge location
        streamflow_path = self.forcing_dir / "MESH_input_streamflow.txt"
        if not streamflow_path.exists():
            self._create_streamflow_input()
        else:
            self.logger.info(f"Using existing streamflow input file")

        # Copy additional settings files from setup_dir
        self._copy_settings_to_forcing()

    def _create_meshflow_config(self) -> Dict[str, Any]:
        """Create configuration dictionary for meshflow."""

        def _get_config_value(key: str, default_value):
            value = self.config_dict.get(key)
            if value is None or value == 'default':
                return default_value
            return value

        # meshflow expects: standard_name -> actual_file_variable_name
        # These match the variable names in our basin_averaged_data files
        default_forcing_vars = {
            "air_pressure": "airpres",
            "specific_humidity": "spechum",
            "air_temperature": "airtemp",
            "wind_speed": "windspd",
            "precipitation": "pptrate",
            "shortwave_radiation": "SWRadAtm",
            "longwave_radiation": "LWRadAtm",
        }

        # Units from source data
        default_forcing_units = {
            "air_pressure": 'Pa',
            "specific_humidity": 'kg/kg',
            "air_temperature": 'K',
            "wind_speed": 'm/s',
            "precipitation": 'm/s',
            "shortwave_radiation": 'W/m^2',
            "longwave_radiation": 'W/m^2',
        }

        # Target units for MESH
        default_forcing_to_units = {
            "air_pressure": 'Pa',
            "specific_humidity": 'kg/kg',
            "air_temperature": 'K',
            "wind_speed": 'm/s',
            "precipitation": 'mm/s',
            "shortwave_radiation": 'W/m^2',
            "longwave_radiation": 'W/m^2',
        }

        # NALCMS 2020 landcover classes (integer keys for meshflow compatibility)
        default_landcover_classes = {
            1: 'Temperate or sub-polar needleleaf forest',
            2: 'Sub-polar taiga needleleaf forest',
            3: 'Tropical or sub-tropical broadleaf evergreen forest',
            4: 'Tropical or sub-tropical broadleaf deciduous forest',
            5: 'Temperate or sub-polar broadleaf deciduous forest',
            6: 'Mixed forest',
            7: 'Tropical or sub-tropical shrubland',
            8: 'Temperate or sub-polar shrubland',
            9: 'Tropical or sub-tropical grassland',
            10: 'Temperate or sub-polar grassland',
            11: 'Sub-polar or polar shrubland-lichen-moss',
            12: 'Sub-polar or polar grassland-lichen-moss',
            13: 'Sub-polar or polar barren-lichen-moss',
            14: 'Wetland',
            15: 'Cropland',
            16: 'Barren lands',
            17: 'Urban',
            18: 'Water',
            19: 'Snow and Ice',
        }

        # ddb_vars maps standard names -> input shapefile column names
        # Note: 'rank', 'next', 'subbasin_area', and 'landclass' are handled internally by meshflow
        default_ddb_vars = {
            'river_slope': 'Slope',
            'river_length': 'Length',
            'river_class': 'strmOrder',  # River class (Strahler order)
        }

        default_ddb_units = {
            'river_slope': 'm/m',
            'river_length': 'm',
            'rank': 'dimensionless',
            'next': 'dimensionless',
            'gru': 'dimensionless',
            'subbasin_area': 'm^2',
        }

        default_ddb_min_values = {
            'river_slope': 1e-6,
            'river_length': 1e-3,
            'subbasin_area': 1e-3,
        }

        # Build forcing files path
        forcing_files_path = Path(
            _get_config_value(
                'MESH_FORCING_PATH',
                self.project_dir / 'forcing' / 'basin_averaged_data',
            )
        )
        forcing_files_glob = str(forcing_files_path / '*.nc')

        # Landcover stats file
        landcover_stats_path = _get_config_value('MESH_LANDCOVER_STATS_PATH', None)
        if landcover_stats_path:
            landcover_path = Path(landcover_stats_path)
        else:
            landcover_file = _get_config_value(
                'MESH_LANDCOVER_STATS_FILE',
                'modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv',
            )
            landcover_dir = Path(
                _get_config_value(
                    'MESH_LANDCOVER_STATS_DIR',
                    self.project_dir / 'attributes' / 'gistool-outputs',
                )
            )
            landcover_path = landcover_dir / landcover_file

        # Dynamically detect GRU classes from landcover stats
        detected_gru_classes = self._detect_gru_classes(landcover_path)
        self.logger.info(f"Detected GRU classes in landcover: {detected_gru_classes}")

        # Get simulation dates with spinup support
        time_window = self.get_simulation_time_window()
        # Add spinup period (default 365 days) for model stabilization
        spinup_days = int(_get_config_value('MESH_SPINUP_DAYS', 365))

        if time_window:
            from datetime import timedelta
            analysis_start, end_date = time_window
            # Simulation starts earlier to allow spinup
            sim_start = analysis_start - timedelta(days=spinup_days)
            forcing_start_date = sim_start.strftime('%Y-%m-%d %H:%M:%S')
            sim_start_date = sim_start.strftime('%Y-%m-%d %H:%M:%S')
            sim_end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
            self.logger.info(
                f"MESH simulation: {sim_start_date} to {sim_end_date} "
                f"(spinup: {spinup_days} days before analysis start {analysis_start.strftime('%Y-%m-%d')})"
            )
        else:
            forcing_start_date = '2001-01-01 00:00:00'
            sim_start_date = '2001-01-01 00:00:00'
            sim_end_date = '2010-12-31 23:00:00'

        # GRU (Group Response Unit) mapping - maps landcover classes to CLASS parameter types
        # This is critical for meshflow to generate proper CLASS parameters
        # Full NALCMS mapping (used to build dynamic mapping)
        full_gru_mapping = {
            0: 'needleleaf',  # Unknown -> default to needleleaf
            1: 'needleleaf',  # Temperate or sub-polar needleleaf forest
            2: 'needleleaf',  # Sub-polar taiga needleleaf forest
            3: 'broadleaf',   # Tropical or sub-tropical broadleaf evergreen forest
            4: 'broadleaf',   # Tropical or sub-tropical broadleaf deciduous forest
            5: 'broadleaf',   # Temperate or sub-polar broadleaf deciduous forest
            6: 'broadleaf',   # Mixed forest
            7: 'grass',       # Tropical or sub-tropical shrubland
            8: 'grass',       # Temperate or sub-polar shrubland
            9: 'grass',       # Tropical or sub-tropical grassland
            10: 'grass',      # Temperate or sub-polar grassland
            11: 'grass',      # Sub-polar or polar shrubland-lichen-moss
            12: 'grass',      # Sub-polar or polar grassland-lichen-moss
            13: 'barrenland', # Sub-polar or polar barren-lichen-moss
            14: 'water',      # Wetland
            15: 'crops',      # Cropland
            16: 'barrenland', # Barren lands
            17: 'urban',      # Urban
            18: 'water',      # Water
            19: 'water',      # Snow and Ice
        }

        # Only include GRU classes that exist in the landcover data
        # meshflow will fail if we include non-existent GRU classes
        if detected_gru_classes:
            default_gru_mapping = {k: full_gru_mapping.get(k, 'needleleaf')
                                   for k in detected_gru_classes}
        else:
            # Fallback to all classes if detection failed
            default_gru_mapping = full_gru_mapping

        # Settings for meshflow (matching meshflow example structure)
        default_settings = {
            'core': {
                'forcing_files': 'single',  # 'single' for combined file, 'multiple' for per-variable
                'forcing_start_date': forcing_start_date,
                'simulation_start_date': sim_start_date,
                'simulation_end_date': sim_end_date,
                'forcing_time_zone': 'UTC',
                'output_path': 'results',
            },
            'class_params': {
                'measurement_heights': {
                    'wind_speed': 10.0,
                    'specific_humidity': 2.0,
                    'air_temperature': 2.0,
                    'roughness_length': 50.0,
                },
                'copyright': {
                    'author': 'University of Calgary',
                    'location': 'SYMFLUENCE',
                },
                # GRU mapping - critical for CLASS parameter generation
                'grus': _get_config_value('MESH_GRU_MAPPING', default_gru_mapping),
            },
            'hydrology_params': {
                # Note: routing is a LIST of dicts - one per river class
                # meshflow iterates with enumerate() expecting a list
                'routing': [
                    {
                        'r2n': 0.4,    # WF_R2 - Channel roughness factor
                        'r1n': 0.02,   # WF_R1 - Overland flow roughness
                        'pwr': 2.37,   # PWR - Power exponent
                        'flz': 0.001,  # FLZ - Lower zone fraction
                    },
                ],
                # Note: hydrology is a dict keyed by GRU class number
                'hydrology': {},  # Use defaults for each GRU
            },
            'run_options': {
                'flags': {
                    'etc': {
                        'RUNMODE': 'runclass',  # runclass = CLASS + Routing
                    },
                },
            },
        }

        config = {
            'riv': str(self.rivers_path / self.rivers_name),
            'cat': str(self.catchment_path / self.catchment_name),
            'landcover': str(landcover_path),
            'forcing_files': forcing_files_glob,
            'forcing_vars': _get_config_value('MESH_FORCING_VARS', default_forcing_vars),
            'forcing_units': _get_config_value('MESH_FORCING_UNITS', default_forcing_units),
            'forcing_to_units': _get_config_value('MESH_FORCING_TO_UNITS', default_forcing_to_units),
            'main_id': _get_config_value('MESH_MAIN_ID', 'GRU_ID'),
            'ds_main_id': _get_config_value('MESH_DS_MAIN_ID', 'DSLINKNO'),
            'landcover_classes': _get_config_value('MESH_LANDCOVER_CLASSES', default_landcover_classes),
            'ddb_vars': _get_config_value('MESH_DDB_VARS', default_ddb_vars),
            'ddb_units': _get_config_value('MESH_DDB_UNITS', default_ddb_units),
            'ddb_to_units': _get_config_value('MESH_DDB_TO_UNITS', default_ddb_units),
            'ddb_min_values': _get_config_value('MESH_DDB_MIN_VALUES', default_ddb_min_values),
            'gru_dim': _get_config_value('MESH_GRU_DIM', 'NGRU'),
            'hru_dim': _get_config_value('MESH_HRU_DIM', 'subbasin'),
            'outlet_value': _get_config_value('MESH_OUTLET_VALUE', -9999),  # meshflow example uses -9999
            'settings': _get_config_value('MESH_SETTINGS', default_settings),
        }
        return config

    def _run_meshflow(self, config: Dict[str, Any]) -> None:
        """
        Run meshflow to generate MESH input files.

        Uses meshflow for both lumped and distributed modes.
        meshflow generates: MESH_drainage_database.nc, MESH_forcing.nc, and config files.
        """
        if not MESHFLOW_AVAILABLE:
            raise ModelExecutionError(
                "meshflow is not available. Please install it with: "
                "pip install git+https://github.com/CH-Earth/meshflow.git@main"
            )

        # Check required files exist
        required_files = [config.get('riv'), config.get('cat')]
        missing_files = [f for f in required_files if f and not Path(f).exists()]
        if missing_files:
            raise ConfigurationError(
                f"MESH preprocessing requires these files: {missing_files}. "
                "Run geospatial preprocessing first."
            )

        # Prepare working copies of shapefiles (to avoid modifying originals)
        riv_copy = self.forcing_dir / f"temp_{Path(config['riv']).name}"
        cat_copy = self.forcing_dir / f"temp_{Path(config['cat']).name}"

        self._copy_shapefile(config['riv'], riv_copy)
        self._copy_shapefile(config['cat'], cat_copy)

        # Sanitize shapefiles for meshflow compatibility
        self._sanitize_shapefile(str(riv_copy))
        self._sanitize_shapefile(str(cat_copy))

        # Fix outlet segment (meshflow requires DSLINKNO = outlet_value for outlet)
        outlet_value = config.get('outlet_value', -9999)  # meshflow example uses -9999
        self._fix_outlet_segment(str(riv_copy), outlet_value=outlet_value)

        # Prepare landcover stats if needed
        landcover_path = config.get('landcover', '')
        if landcover_path and Path(landcover_path).exists():
            sanitized_landcover = self._sanitize_landcover_stats(landcover_path)
            config['landcover'] = sanitized_landcover
        else:
            self.logger.warning(f"Landcover file not found: {landcover_path}")

        # Update config with working copies
        config['riv'] = str(riv_copy)
        config['cat'] = str(cat_copy)

        # Clean output directory
        output_files = [
            self.forcing_dir / "MESH_forcing.nc",
            self.forcing_dir / "MESH_drainage_database.nc",
        ]
        for f in output_files:
            if f.exists():
                f.unlink()

        try:
            import meshflow
            self.logger.info(f"Using meshflow version: {getattr(meshflow, '__version__', 'unknown')}")

            # Initialize MESHWorkflow
            self.logger.info("Initializing MESHWorkflow with config")
            self.logger.debug(f"Config keys: {list(config.keys())}")
            workflow = MESHWorkflow(**config)

            # Try full workflow first (generates forcing, ddb, and params)
            try:
                self.logger.info("Running full meshflow workflow")
                workflow.run(save_path=str(self.forcing_dir))
                workflow.save(output_dir=str(self.forcing_dir))
                self.logger.info("Full meshflow workflow completed successfully")

            except Exception as run_error:
                # Fallback: Use meshflow for DDB only, prepare forcing separately
                self.logger.warning(f"Full meshflow workflow failed ({run_error}), falling back to DDB-only mode")

                # Initialize routing and drainage database
                self.logger.info("Running meshflow for drainage database only")
                workflow.init()  # Sets up routing
                workflow.init_ddb()  # Creates drainage database

                # Save DDB output
                ddb_path = self.forcing_dir / "MESH_drainage_database.nc"
                workflow.ddb.to_netcdf(ddb_path)
                self.logger.info(f"Created drainage database: {ddb_path}")

                # Try to generate parameter files via meshflow's render_configs API
                # The proper workflow is:
                # 1. init_class(return_dict=True) -> class_dict
                # 2. init_hydrology(return_dict=True) -> hydro_dict
                # 3. init_options(return_dict=True) -> options_dict
                # 4. render_configs(class_dicts, hydrology_dicts, options_dict) -> sets text attributes
                try:
                    self.logger.info("Attempting to generate CLASS parameters via meshflow")
                    class_dict = workflow.init_class(return_dict=True)
                    self.logger.debug(f"CLASS dict keys: {list(class_dict.keys()) if class_dict else 'None'}")
                except Exception as class_error:
                    self.logger.warning(f"meshflow CLASS init failed: {class_error}")
                    class_dict = None

                try:
                    self.logger.info("Attempting to generate hydrology parameters via meshflow")
                    hydro_dict = workflow.init_hydrology(return_dict=True)
                    self.logger.debug(f"Hydrology dict keys: {list(hydro_dict.keys()) if hydro_dict else 'None'}")
                except Exception as hydro_error:
                    self.logger.warning(f"meshflow hydrology init failed: {hydro_error}")
                    hydro_dict = None

                try:
                    self.logger.info("Attempting to generate run options via meshflow")
                    options_dict = workflow.init_options(return_dict=True)
                    self.logger.debug(f"Options dict keys: {list(options_dict.keys()) if options_dict else 'None'}")
                except Exception as opt_error:
                    self.logger.warning(f"meshflow options init failed: {opt_error}")
                    options_dict = None

                # Render configs to text files if we have the required dicts
                if class_dict and hydro_dict and options_dict:
                    try:
                        self.logger.info("Rendering meshflow configs to text")
                        # Define process details for routing parameters
                        # These tell the template which routing params to include
                        process_details = {
                            'routing': ['r2n', 'r1n', 'pwr', 'flz'],
                            'hydrology': [],  # No GRU-dependent hydrology params
                        }
                        workflow.render_configs(
                            class_dicts=class_dict,
                            hydrology_dicts=hydro_dict,
                            options_dict=options_dict,
                            process_details=process_details
                        )

                        # Save CLASS parameters
                        if hasattr(workflow, 'class_text') and workflow.class_text:
                            class_path = self.forcing_dir / "MESH_parameters_CLASS.ini"
                            with open(class_path, 'w') as f:
                                f.write(workflow.class_text)
                            self.logger.info(f"Created CLASS parameters via meshflow: {class_path}")

                        # Save hydrology parameters
                        if hasattr(workflow, 'hydrology_text') and workflow.hydrology_text:
                            hydro_path = self.forcing_dir / "MESH_parameters_hydrology.ini"
                            with open(hydro_path, 'w') as f:
                                f.write(workflow.hydrology_text)
                            self.logger.info(f"Created hydrology parameters via meshflow: {hydro_path}")

                        # Save run options
                        if hasattr(workflow, 'options_text') and workflow.options_text:
                            run_path = self.forcing_dir / "MESH_input_run_options.ini"
                            with open(run_path, 'w') as f:
                                f.write(workflow.options_text)
                            self.logger.info(f"Created run options via meshflow: {run_path}")

                    except Exception as render_error:
                        self.logger.warning(f"meshflow render_configs failed: {render_error}")
                else:
                    self.logger.warning("Could not generate all required dicts for render_configs")

                # Prepare forcing separately (more robust for our data format)
                self.logger.info("Preparing forcing data directly")
                self._prepare_forcing_direct(config)

            # Post-process for MESH compatibility
            self._postprocess_meshflow_output()

            self.logger.info("meshflow preprocessing completed successfully")

        except Exception as e:
            self.logger.error(f"meshflow preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            raise ModelExecutionError(f"meshflow preprocessing failed: {e}")

    def _prepare_forcing_direct(self, config: Dict[str, Any]) -> None:
        """
        Prepare MESH forcing directly from basin-averaged data.

        This bypasses meshflow's CDO-based forcing prep which has issues
        with frequency inference on multi-file datasets.
        """
        import xarray as xr
        import numpy as np
        import pandas as pd
        from netCDF4 import Dataset as NC4Dataset
        import glob

        forcing_files = sorted(glob.glob(config.get('forcing_files', '')))
        if not forcing_files:
            raise ModelExecutionError("No forcing files found")

        self.logger.info(f"Loading {len(forcing_files)} forcing files")

        # Load and combine forcing files
        ds = xr.open_mfdataset(forcing_files, combine='by_coords', parallel=False)

        # Variable mapping from source to MESH names
        forcing_vars = config.get('forcing_vars', {})
        var_rename = {v: k for k, v in forcing_vars.items()}  # Reverse mapping

        # MESH variable name mapping
        mesh_var_names = {
            'air_pressure': 'PRES',
            'specific_humidity': 'QA',
            'air_temperature': 'TA',
            'wind_speed': 'UV',
            'precipitation': 'PRE',
            'shortwave_radiation': 'FSIN',
            'longwave_radiation': 'FLIN',
        }

        # Find spatial dimension
        hru_dim = config.get('hru_dim', 'hru')
        spatial_dim = None
        for dim in ['hru', 'subbasin', 'N', 'gru', 'GRU_ID']:
            if dim in ds.dims:
                spatial_dim = dim
                break

        if spatial_dim is None:
            raise ModelExecutionError("Could not find spatial dimension in forcing data")

        n_spatial = ds.dims[spatial_dim]
        self.logger.info(f"Found spatial dimension '{spatial_dim}' with {n_spatial} elements")

        # Create MESH forcing file
        forcing_path = self.forcing_dir / "MESH_forcing.nc"
        with NC4Dataset(forcing_path, 'w', format='NETCDF4') as ncfile:
            # Create dimensions
            ncfile.createDimension('time', None)  # unlimited
            ncfile.createDimension('N', n_spatial)

            # Create time variable
            time_data = ds['time'].values
            var_time = ncfile.createVariable('time', 'f8', ('time',))
            var_time.long_name = 'time'
            var_time.standard_name = 'time'
            var_time.units = 'hours since 1900-01-01'
            var_time.calendar = 'gregorian'

            # Convert time to hours since reference
            reference = pd.Timestamp('1900-01-01')
            time_hours = np.array([(pd.Timestamp(t) - reference).total_seconds() / 3600.0
                                   for t in time_data])
            var_time[:] = time_hours

            # Create N coordinate
            var_n = ncfile.createVariable('N', 'i4', ('N',))
            var_n[:] = np.arange(1, n_spatial + 1)

            # Create forcing variables
            for src_var, standard_name in var_rename.items():
                if src_var in ds:
                    mesh_name = mesh_var_names.get(standard_name, src_var)
                    var_data = ds[src_var].values

                    # Ensure shape is (time, spatial)
                    if var_data.ndim == 2:
                        if var_data.shape[0] != len(time_data):
                            var_data = var_data.T

                    var = ncfile.createVariable(mesh_name, 'f4', ('time', 'N'), fill_value=-9999.0)
                    var.long_name = standard_name.replace('_', ' ')
                    var.units = self._get_var_units(mesh_name)
                    var[:] = var_data
                    self.logger.debug(f"Created {mesh_name} from {src_var}")

            # Global attributes
            ncfile.title = 'MESH Forcing Data'
            ncfile.Conventions = 'CF-1.6'
            ncfile.history = f'Created by SYMFLUENCE on {pd.Timestamp.now()}'

        self.logger.info(f"Created MESH forcing file: {forcing_path}")
        ds.close()

    def _postprocess_meshflow_output(self) -> None:
        """
        Post-process meshflow output for MESH compatibility.

        - Rename 'subbasin' dimension to 'N' (MESH expects 'N')
        - Rename forcing variables to MESH 1.5 naming convention
        - Create split forcing files for distributed mode
        """
        import xarray as xr
        import numpy as np

        # Rename 'subbasin' to 'N' in all NetCDF files
        for nc_file in [
            self.forcing_dir / "MESH_forcing.nc",
            self.forcing_dir / "MESH_drainage_database.nc"
        ]:
            if nc_file.exists():
                try:
                    with xr.open_dataset(nc_file) as ds:
                        rename_dict = {}
                        if 'subbasin' in ds.dims:
                            rename_dict['subbasin'] = 'N'
                        if rename_dict:
                            ds_renamed = ds.rename(rename_dict)
                            temp_path = nc_file.with_suffix('.tmp.nc')
                            ds_renamed.to_netcdf(temp_path)
                            os.replace(temp_path, nc_file)
                            self.logger.info(f"Renamed 'subbasin' to 'N' in {nc_file.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to rename dimension in {nc_file.name}: {e}")

        # Rename forcing variables for MESH 1.5 compatibility
        forcing_nc = self.forcing_dir / "MESH_forcing.nc"
        if forcing_nc.exists():
            try:
                with xr.open_dataset(forcing_nc) as ds:
                    # MESH 1.5 expected variable names
                    rename_map = {
                        'airpres': 'PRES',
                        'spechum': 'QA',
                        'airtemp': 'TA',
                        'windspd': 'UV',
                        'pptrate': 'PRE',
                        'SWRadAtm': 'FSIN',
                        'LWRadAtm': 'FLIN',
                        # Alternative names
                        'air_pressure': 'PRES',
                        'specific_humidity': 'QA',
                        'air_temperature': 'TA',
                        'wind_speed': 'UV',
                        'precipitation': 'PRE',
                        'shortwave_radiation': 'FSIN',
                        'longwave_radiation': 'FLIN',
                    }

                    existing_rename = {k: v for k, v in rename_map.items() if k in ds.variables}
                    if existing_rename:
                        ds_renamed = ds.rename(existing_rename)

                        # Ensure dimension order is (time, N)
                        if 'time' in ds_renamed.dims and 'N' in ds_renamed.dims:
                            ds_renamed = ds_renamed.transpose('time', 'N', ...)

                        temp_path = forcing_nc.with_suffix('.tmp.nc')
                        ds_renamed.to_netcdf(temp_path, unlimited_dims=['time'])
                        os.replace(temp_path, forcing_nc)
                        self.logger.info(f"Renamed forcing variables for MESH 1.5")
            except Exception as e:
                self.logger.warning(f"Failed to rename forcing variables: {e}")

        # Create split forcing files for distributed mode
        self._create_split_forcing_files()

        # Add missing variables to drainage database if needed
        self._ensure_ddb_completeness()

        # Fix hydrology file to include WF_R2 (required by MESH for channel routing)
        self._fix_hydrology_wf_r2()

        # Fix CLASS initial conditions (snow parameters for winter simulations)
        self._fix_class_initial_conditions()

    def _create_split_forcing_files(self) -> None:
        """Create individual forcing files per variable for MESH distributed mode."""
        import xarray as xr
        import numpy as np
        from netCDF4 import Dataset as NC4Dataset
        import pandas as pd

        forcing_nc = self.forcing_dir / "MESH_forcing.nc"
        if not forcing_nc.exists():
            self.logger.warning("MESH_forcing.nc not found, skipping split file creation")
            return

        # MESH 1.5 variable mapping to file names
        var_to_file = {
            'FSIN': 'basin_shortwave.nc',
            'FLIN': 'basin_longwave.nc',
            'PRES': 'basin_pres.nc',
            'TA': 'basin_temperature.nc',
            'QA': 'basin_humidity.nc',
            'UV': 'basin_wind.nc',
            'PRE': 'basin_rain.nc',
        }

        try:
            with xr.open_dataset(forcing_nc) as ds:
                n_dim = 'N' if 'N' in ds.dims else 'subbasin' if 'subbasin' in ds.dims else None
                if not n_dim:
                    self.logger.warning("No spatial dimension found in forcing file")
                    return

                n_size = ds.dims[n_dim]
                time_data = ds['time'].values if 'time' in ds else None

                for mesh_var, filename in var_to_file.items():
                    if mesh_var in ds:
                        var_path = self.forcing_dir / filename
                        var_data = ds[mesh_var].values

                        # Create NetCDF file
                        with NC4Dataset(var_path, 'w', format='NETCDF4') as ncfile:
                            # Create dimensions
                            ncfile.createDimension('time', None)  # unlimited
                            ncfile.createDimension('N', n_size)

                            # Create data variable
                            var = ncfile.createVariable(mesh_var, 'f4', ('time', 'N'), fill_value=-9999.0)
                            var.long_name = self._get_var_long_name(mesh_var)
                            var.units = self._get_var_units(mesh_var)
                            var[:] = var_data

                            # Create time variable
                            if time_data is not None:
                                var_time = ncfile.createVariable('time', 'f8', ('time',))
                                var_time.long_name = 'time'
                                var_time.standard_name = 'time'
                                var_time.units = 'hours since 1900-01-01'
                                var_time.calendar = 'gregorian'

                                reference = pd.Timestamp('1900-01-01')
                                time_hours = [(pd.Timestamp(t) - reference).total_seconds() / 3600.0
                                              for t in time_data]
                                var_time[:] = time_hours

                            # Create N coordinate
                            var_n = ncfile.createVariable('N', 'i4', ('N',))
                            var_n[:] = np.arange(1, n_size + 1)

                        self.logger.debug(f"Created {filename}")

                self.logger.info(f"Created split forcing files for MESH distributed mode")

        except Exception as e:
            self.logger.warning(f"Failed to create split forcing files: {e}")

    def _get_var_long_name(self, var: str) -> str:
        """Get long name for MESH variable."""
        names = {
            'FSIN': 'downward shortwave radiation',
            'FLIN': 'downward longwave radiation',
            'PRES': 'air pressure',
            'TA': 'air temperature',
            'QA': 'specific humidity',
            'UV': 'wind speed',
            'PRE': 'precipitation rate',
        }
        return names.get(var, var)

    def _get_var_units(self, var: str) -> str:
        """Get units for MESH variable."""
        units = {
            'FSIN': 'W m-2',
            'FLIN': 'W m-2',
            'PRES': 'Pa',
            'TA': 'K',
            'QA': 'kg kg-1',
            'UV': 'm s-1',
            'PRE': 'kg m-2 s-1',
        }
        return units.get(var, '1')

    def _ensure_ddb_completeness(self) -> None:
        """Ensure drainage database has all required variables for MESH."""
        import xarray as xr
        import numpy as np

        ddb_nc = self.forcing_dir / "MESH_drainage_database.nc"
        if not ddb_nc.exists():
            return

        try:
            with xr.open_dataset(ddb_nc) as ds:
                n_dim = 'N' if 'N' in ds.dims else 'subbasin' if 'subbasin' in ds.dims else None
                if not n_dim:
                    return

                n_size = ds.dims[n_dim]
                modified = False

                # Add IREACH if missing (reservoir ID, 0 = no reservoir)
                if 'IREACH' not in ds:
                    ds['IREACH'] = xr.DataArray(
                        np.zeros(n_size, dtype=np.int32),
                        dims=[n_dim],
                        attrs={'long_name': 'Reservoir ID', '_FillValue': -1}
                    )
                    modified = True
                    self.logger.info("Added IREACH to drainage database")

                # Add IAK if missing (river class)
                if 'IAK' not in ds:
                    ds['IAK'] = xr.DataArray(
                        np.ones(n_size, dtype=np.int32),
                        dims=[n_dim],
                        attrs={'long_name': 'River class', '_FillValue': -1}
                    )
                    modified = True
                    self.logger.info("Added IAK to drainage database")

                # Add AL (side length) for routing calculations
                # MESH uses this for flow timing - sqrt(GridArea) gives representative length
                # NOTE: coordinates must be "lon lat" (not "lat lon") for MESH to assign the variable
                if 'AL' not in ds and 'GridArea' in ds:
                    grid_area = ds['GridArea'].values
                    side_length = np.sqrt(grid_area)
                    ds['AL'] = xr.DataArray(
                        side_length,
                        dims=[n_dim],
                        attrs={
                            'long_name': 'Side length of grid',
                            'units': 'm',
                            'coordinates': 'lon lat',  # Must match MESH expected order
                            '_FillValue': np.nan  # Use NaN like other MESH variables
                        }
                    )
                    modified = True
                    self.logger.info(f"Added AL (side length) to drainage database: min={side_length.min():.1f}m, max={side_length.max():.1f}m")

                # Add DA (drainage area) if missing - compute from routing topology
                if 'DA' not in ds and 'GridArea' in ds and 'Next' in ds and 'Rank' in ds:
                    grid_area = ds['GridArea'].values
                    next_arr = ds['Next'].values.astype(int)
                    rank_arr = ds['Rank'].values.astype(int)

                    # Initialize DA with local GridArea
                    da = grid_area.copy()

                    # Compute accumulated drainage area by routing order
                    # Simple iterative approach: keep accumulating until no changes
                    for _ in range(n_size):  # Max iterations = n_size
                        changed = False
                        for i in range(n_size):
                            if next_arr[i] > 0:  # Not an outlet
                                # Find downstream index
                                ds_idx = np.where(rank_arr == next_arr[i])[0]
                                if len(ds_idx) > 0:
                                    ds_idx = ds_idx[0]
                                    # DA at downstream should include this cell's area
                                    new_da = da[ds_idx] + grid_area[i]
                                    if new_da != da[ds_idx]:
                                        da[ds_idx] = new_da
                                        changed = True
                        if not changed:
                            break

                    ds['DA'] = xr.DataArray(
                        da,
                        dims=[n_dim],
                        attrs={
                            'long_name': 'Drainage area',
                            'units': 'm**2',
                            'coordinates': 'lon lat',  # Must match MESH expected order
                            '_FillValue': np.nan  # Use NaN like other MESH variables
                        }
                    )
                    modified = True
                    self.logger.info(f"Added DA (drainage area) to drainage database: max={da.max()/1e6:.1f} km²")

                if modified:
                    temp_path = ddb_nc.with_suffix('.tmp.nc')
                    ds.to_netcdf(temp_path)
                    os.replace(temp_path, ddb_nc)

        except Exception as e:
            self.logger.warning(f"Failed to ensure DDB completeness: {e}")

    def _fix_hydrology_wf_r2(self) -> None:
        """
        Ensure WF_R2 is in the hydrology file.

        MESH requires WF_R2 for channel routing even when using new routing (R2N).
        meshflow only outputs R2N, so we need to add WF_R2 if missing.
        """
        hydro_path = self.forcing_dir / "MESH_parameters_hydrology.ini"
        if not hydro_path.exists():
            return

        try:
            with open(hydro_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')

            # Check if WF_R2 is already present
            if 'WF_R2' in content:
                self.logger.debug("WF_R2 already present in hydrology file")
                return

            # Find where R2N is and add WF_R2 before it
            new_lines = []
            r2n_found = False
            for i, line in enumerate(lines):
                if line.startswith('R2N') and not r2n_found:
                    # Extract R2N values to use for WF_R2
                    # R2N line format: "R2N    0.400    0.400 ..."
                    parts = line.split()
                    if len(parts) >= 2:
                        # Use R2N values for WF_R2 as well (or use a default)
                        r2n_values = parts[1:]  # Get all values
                        # Create WF_R2 line with same values
                        wf_r2_line = "WF_R2  " + "    ".join(r2n_values) + "                                                # channel roughness (old routing)"
                        new_lines.append(wf_r2_line)
                        r2n_found = True

                        # Also update the number of parameters (line before R2N)
                        # Find the line with "# Number of channel routing parameters"
                        for j in range(len(new_lines) - 1, -1, -1):
                            if "Number of channel routing parameters" in new_lines[j]:
                                # Parse the current count and increment
                                count_line = new_lines[j]
                                # Extract number from start of line
                                import re
                                match = re.match(r'\s*(\d+)', count_line)
                                if match:
                                    old_count = int(match.group(1))
                                    new_count = old_count + 1
                                    new_lines[j] = count_line.replace(
                                        f'{old_count:8d}' if old_count >= 10 else f'       {old_count}',
                                        f'{new_count:8d}' if new_count >= 10 else f'       {new_count}',
                                        1
                                    )
                                break

                new_lines.append(line)

            if r2n_found:
                with open(hydro_path, 'w') as f:
                    f.write('\n'.join(new_lines))
                self.logger.info("Added WF_R2 to hydrology parameters file")

        except Exception as e:
            self.logger.warning(f"Failed to add WF_R2 to hydrology file: {e}")

    def _fix_class_initial_conditions(self) -> None:
        """
        Fix CLASS initial conditions for proper snow simulation.

        meshflow generates zero initial conditions (SNO=0, ALBS=0, RHOS=0) which
        causes CLASS snow energy balance failures, especially for winter simulations.

        This fixes line 19 of each GRU (RCAN/SCAN/SNO/ALBS/RHOS/GRO) to have
        reasonable winter initial conditions.
        """
        class_path = self.forcing_dir / "MESH_parameters_CLASS.ini"
        if not class_path.exists():
            return

        try:
            with open(class_path, 'r') as f:
                lines = f.readlines()

            # Determine initial snow conditions based on simulation start month
            # For mountain basins starting in winter, use reasonable initial snow
            time_window = self.get_simulation_time_window()
            if time_window:
                start_month = time_window[0].month
            else:
                start_month = 1  # Default to January

            # Set initial conditions based on season
            # Winter months (Nov-Apr) should have snow
            if start_month in [11, 12, 1, 2, 3, 4]:
                initial_sno = 50.0    # 50 mm SWE initial snow
                initial_albs = 0.80   # Fresh snow albedo
                initial_rhos = 300.0  # Snow density (kg/m³)
            else:
                # Summer - minimal snow except at high elevations
                initial_sno = 10.0    # Small amount for numerical stability
                initial_albs = 0.60   # Aged snow albedo
                initial_rhos = 350.0  # Denser snow

            modified = False
            new_lines = []

            for line in lines:
                # Match line 19 pattern: ends with "19 RCAN/SCAN/SNO/ALBS/RHOS/GRO"
                if '19 RCAN/SCAN/SNO/ALBS/RHOS/GRO' in line:
                    # Parse the current values
                    # Format: "   0.000   0.000   0.000   0.000   0.000   1.000    19 RCAN/SCAN/SNO/ALBS/RHOS/GRO"
                    parts = line.split()
                    if len(parts) >= 8:  # 6 values + "19" + description
                        try:
                            rcan = float(parts[0])  # Keep RCAN
                            scan = float(parts[1])  # Keep SCAN
                            # sno = float(parts[2])  # Replace
                            # albs = float(parts[3])  # Replace
                            # rhos = float(parts[4])  # Replace
                            gro = float(parts[5])   # Keep GRO

                            # Construct new line with fixed values
                            new_line = (
                                f"   {rcan:.3f}   {scan:.3f}   {initial_sno:.1f}   "
                                f"{initial_albs:.2f}   {initial_rhos:.1f}   {gro:.3f}                             "
                                f"19 RCAN/SCAN/SNO/ALBS/RHOS/GRO\n"
                            )
                            new_lines.append(new_line)
                            modified = True
                            continue
                        except (ValueError, IndexError) as e:
                            self.logger.warning(f"Failed to parse CLASS line 19: {e}")
                            new_lines.append(line)
                            continue

                new_lines.append(line)

            if modified:
                with open(class_path, 'w') as f:
                    f.writelines(new_lines)
                self.logger.info(
                    f"Fixed CLASS initial conditions: SNO={initial_sno}mm, "
                    f"ALBS={initial_albs}, RHOS={initial_rhos}kg/m³"
                )
            else:
                self.logger.debug("No CLASS initial conditions to fix")

        except Exception as e:
            self.logger.warning(f"Failed to fix CLASS initial conditions: {e}")

    def _create_run_options(self) -> None:
        """Create MESH_input_run_options.ini file."""
        run_options_path = self.forcing_dir / "MESH_input_run_options.ini"

        spatial_mode = self._get_spatial_mode()

        # Get simulation times with spinup support
        time_window = self.get_simulation_time_window()
        # Add spinup period for model stabilization (same as in _create_meshflow_config)
        spinup_days = int(self._get_config('MESH_SPINUP_DAYS', 365))

        if time_window:
            import pandas as pd
            from datetime import timedelta
            analysis_start, end_time = time_window
            # Simulation starts earlier to allow spinup
            start_time = pd.Timestamp(analysis_start - timedelta(days=spinup_days))
            end_time = pd.Timestamp(end_time)
        else:
            import pandas as pd
            start_time = pd.Timestamp("2004-01-01 01:00")
            end_time = pd.Timestamp("2004-01-05 23:00")

        # Set drainage database format based on spatial mode
        # MESH 1.5 NetCDF4 format for both modes
        if spatial_mode == 'distributed':
            shd_flag = 'nc_subbasin'  # NetCDF subbasin format (1D)
        else:
            shd_flag = 'nc'  # NetCDF for lumped

        content = f"""MESH input run options file                             # comment line 1                                | *
##### Control Flags #####                               # comment line 2                                | *
----#                                                   # comment line 3                                | *
   14                                                   # Number of control flags                       | I5
SHDFILEFLAG         {shd_flag}                          # Drainage database format (nc_subbasin for distributed)
BASINFORCINGFLAG    nc                                  # Forcing file format (nc = NetCDF)
RUNMODE             runclass                            # Run mode (runclass = CLASS + Routing)
INPUTPARAMSFORMFLAG ini                                 # Parameter file format (ini = .ini files)
RESUMEFLAG          off                                 # Resume from state (off=No)
SAVERESUMEFLAG      off                                 # Save final state (off=No)
TIMESTEPFLAG        60                                  # Time step in minutes (default 60)
OUTFIELDSFLAG       default                             # Output fields (all, none, default)
BASINRUNOFFFLAG     ts                                  # Runoff output format (ts = time series)
LOCATIONFLAG        1                                   # Centroid location
PBSMFLAG            off                                 # Blowing snow (off)
BASEFLOWFLAG        wf_lzs                              # Baseflow formulation
INTERPOLATIONFLAG   0                                   # Interpolation (0=No)
METRICSSPINUP       {spinup_days}                       # Spinup days to exclude from calibration metrics
##### Output Grid selection #####                       #15 comment line 15                             | *
----#                                                   #16 comment line 16                             | *
    0   #Maximum 5 points                               #17 Number of output grid points                | I5
---------#---------#---------#---------#---------#      #18 comment line 18                             | *
         1                                              #19 Grid number                                 | 5I10
         1                                              #20 Land class                                  | 5I10
./                                                      #21 Output directory                            | 5A10
##### Output Directory #####                            #22 comment line 22                             | *
---------#                                              #23 comment line 23                             | *
./                                                      #24 Output Directory for total-basin files      | A10
##### Simulation Run Times #####                        #25 comment line 25                             | *
---#---#---#---#                                        #26 comment line 26                             | *
{start_time.year:04d} {start_time.dayofyear:03d} {start_time.hour:3d}   0
{end_time.year:04d} {end_time.dayofyear:03d} {end_time.hour:3d}   0
"""
        with open(run_options_path, 'w') as f:
            f.write(content)
        self.logger.info(f"Created {run_options_path} (spatial_mode={spatial_mode}, shd_flag={shd_flag})")

    def _create_class_parameters(self) -> None:
        """Create MESH CLASS parameters file with defaults for all GRU types."""
        import json
        import xarray as xr

        # Get number of GRU classes from drainage database
        ddb_path = self.forcing_dir / "MESH_drainage_database.nc"
        if not ddb_path.exists():
            self.logger.warning("Drainage database not found, cannot create CLASS parameters")
            return

        try:
            with xr.open_dataset(ddb_path) as ds:
                ngru = ds.dims.get('NGRU', 1)
        except Exception as e:
            self.logger.warning(f"Failed to read DDB: {e}")
            ngru = 1

        self.logger.info(f"Creating CLASS parameters for {ngru} GRU classes")

        # Load default CLASS params from meshflow
        try:
            from meshflow.utility import DEFAULT_CLASS_PARAMS
            with open(DEFAULT_CLASS_PARAMS, 'r') as f:
                defaults = json.load(f)
            class_defaults = defaults.get('class_defaults', {})
        except Exception:
            # Fallback to hardcoded defaults
            class_defaults = {
                'veg': {'line5': {'fcan': 1, 'lamx': 1.45}, 'line6': {'lnz0': -1.3, 'lamn': 1.2}},
                'soil': {'line14': {'sand1': 50, 'sand2': 50, 'sand3': 50}},
            }

        # Create the CLASS ini file
        class_path = self.forcing_dir / "MESH_parameters_CLASS.ini"

        with open(class_path, 'w') as f:
            f.write(";; CLASS parameter file\n")
            f.write(";; Generated by SYMFLUENCE for MESH 1.5\n\n")
            f.write(f"NMELT 1    ! number of landcover classes for MESH\n")
            f.write(f"NGRU {ngru}    ! number of GRUs\n\n")

            # Write header
            f.write(";; Vegetation parameters\n")

            # Reference heights
            f.write("[REF_HEIGHTS]\n")
            f.write("ZRFM 10.0  ! Reference height for wind speed\n")
            f.write("ZRFH 2.0   ! Reference height for temperature/humidity\n")
            f.write("ZBLD 50.0  ! Blending height\n\n")

            # GRU parameters (one set per GRU)
            for gru in range(1, ngru + 1):
                f.write(f";; GRU {gru}\n")
                f.write(f"[GRU_{gru}]\n")
                # Vegetation
                veg = class_defaults.get('veg', {})
                f.write(f"FCAN 1.0 LAMX {veg.get('line5', {}).get('lamx', 1.45)}\n")
                f.write(f"LNZ0 {veg.get('line6', {}).get('lnz0', -1.3)} LAMN {veg.get('line6', {}).get('lamn', 1.2)}\n")
                f.write(f"ALVC 0.045 CMAS 4.5\n")
                f.write(f"ALIC 0.16 ROOT 1.09\n")
                f.write(f"RSMN 145 QA50 36\n")
                f.write(f"VPDA 0.8 VPDB 1.05\n")
                f.write(f"PSGA 100 PSGB 5\n")
                # Soil
                f.write(f"DRN 1 SDEP 2.5 FARE 1 DD 50\n")
                f.write(f"XSLP 0.03 XDRAINH 0.35 MANN 0.1 KSAT 0.05\n")
                f.write(f"SAND 50 50 50\n")
                f.write(f"CLAY 20 20 20\n")
                f.write(f"ORGM 0 0 0\n")
                # Prognostic
                f.write(f"TBAR 4 2 1 TCAN 2 TSNO 0 TPND 4\n")
                f.write(f"THLQ 0.25 0.15 0.04 THIC 0 0 0 ZPND 0\n")
                f.write(f"RCAN 0 SCAN 0 SNO 0 ALBS 0 RHOS 0 GRO 1\n\n")

        self.logger.info(f"Created CLASS parameters: {class_path}")

    def _create_hydrology_parameters(self) -> None:
        """Create MESH hydrology parameters file."""
        import json

        # Load default hydrology params from meshflow
        try:
            from meshflow.utility import DEFAULT_HYDROLOGY_PARAMS
            with open(DEFAULT_HYDROLOGY_PARAMS, 'r') as f:
                defaults = json.load(f)
        except Exception:
            defaults = {}

        hydro_path = self.forcing_dir / "MESH_parameters_hydrology.ini"

        with open(hydro_path, 'w') as f:
            f.write(";; Hydrology parameter file\n")
            f.write(";; Generated by SYMFLUENCE for MESH 1.5\n\n")

            # Basic hydrology parameters
            f.write("[HYDROLOGY]\n")
            f.write("WF_R1 0.5        ! Overland flow exponent\n")
            f.write("WF_R2 1.0        ! Interflow coefficient\n")
            f.write("WF_KI 0.5        ! Interflow coefficient\n")
            f.write("WF_KC 0.5        ! Channel routing coefficient\n")
            f.write("WF_KD 0.05       ! Deep groundwater coefficient\n")
            f.write("WF_FLZS 0.1      ! Lower zone storage fraction\n")
            f.write("WF_PWR_LZS 2.0   ! Lower zone power\n\n")

            # Manning's roughness for routing
            f.write("[ROUTING]\n")
            f.write("CHNL_MANN 0.035  ! Channel Manning's n\n")
            f.write("CHNL_WD 10.0     ! Channel width (m)\n")
            f.write("CHNL_DP 1.0      ! Channel depth (m)\n")

        self.logger.info(f"Created hydrology parameters: {hydro_path}")

    def _create_streamflow_input(self) -> None:
        """
        Create MESH_input_streamflow.txt with gauge locations and observed data.

        For nc_subbasin format, gauge locations use (IY=1, JX=Rank) to match
        the 1D subbasin indexing. The gauge is placed at the outlet subbasin
        (the one with the largest drainage area that is still "inside" the basin).
        """
        import pandas as pd
        import numpy as np
        import xarray as xr

        streamflow_path = self.forcing_dir / "MESH_input_streamflow.txt"

        # Get simulation time window
        time_window = self.get_simulation_time_window()
        spinup_days = int(self._get_config('MESH_SPINUP_DAYS', 365))

        if time_window:
            from datetime import timedelta
            analysis_start, end_date = time_window
            sim_start = analysis_start - timedelta(days=spinup_days)
            start_year = sim_start.year
            start_month = sim_start.month
            start_day = sim_start.day
        else:
            start_year = 2001
            start_month = 1
            start_day = 1

        # Load drainage database to find outlet subbasin
        ddb_path = self.forcing_dir / "MESH_drainage_database.nc"
        if ddb_path.exists():
            with xr.open_dataset(ddb_path) as ds:
                next_arr = ds['Next'].values
                rank_arr = ds['Rank'].values
                da_arr = ds['DA'].values if 'DA' in ds else ds['GridArea'].values

                # Find the "inside basin" outlet - subbasin that drains to the actual outlet
                # In MESH, cells with Next=0 are treated as "outside basin"
                # So we want the cell that drains TO the Next=0 cell (largest DA with Next>0)
                inside_mask = next_arr > 0
                if inside_mask.any():
                    # Get the subbasin with largest DA that is still "inside"
                    inside_indices = np.where(inside_mask)[0]
                    max_da_idx = inside_indices[np.argmax(da_arr[inside_indices])]
                    outlet_rank = int(rank_arr[max_da_idx])
                    outlet_da = da_arr[max_da_idx] / 1e6  # Convert to km²
                else:
                    # Fallback: use the overall outlet
                    outlet_idx = np.argmax(da_arr)
                    outlet_rank = int(rank_arr[outlet_idx])
                    outlet_da = da_arr[outlet_idx] / 1e6
        else:
            outlet_rank = 1
            outlet_da = 0

        self.logger.info(f"Setting streamflow gauge at Rank {outlet_rank} (DA={outlet_da:.1f} km²)")

        # Get observed streamflow if available
        obs_data = self._load_observed_streamflow()

        # Get gauge info
        gauge_id = self.config_dict.get('STREAMFLOW_STATION_ID', 'gauge1')
        if gauge_id == 'default':
            gauge_id = self.domain_name

        with open(streamflow_path, 'w') as f:
            # Header line 1: comment
            f.write(f"#{self.domain_name} streamflow gauge\n")

            # Header line 2: n_gauges, flag1, flag2, obs_interval_hours, start_year, start_month, start_day
            n_gauges = 1
            obs_interval = 24  # Daily observations
            f.write(f"{n_gauges} 0 0 {obs_interval} {start_year} {start_month} {start_day}\n")

            # Gauge location: IY JX ID
            # For nc_subbasin with 1D array, use IY=1 and JX=Rank
            f.write(f"1 {outlet_rank} {gauge_id}\n")

            # Write observed streamflow data
            # Format: observed_value simulated_flag (-1 = no simulated data to compare)
            if obs_data is not None and len(obs_data) > 0:
                for q in obs_data:
                    if np.isnan(q) or q < 0:
                        f.write("-1\t-1\n")
                    else:
                        f.write(f"{q:.3f}\t-1\n")
            else:
                # No observed data - write placeholder for simulation period
                # Estimate number of days in simulation
                if time_window:
                    from datetime import timedelta
                    n_days = (end_date - sim_start).days + 1
                else:
                    n_days = 365  # Default placeholder

                for _ in range(n_days):
                    f.write("-1\t-1\n")

        self.logger.info(f"Created streamflow input file: {streamflow_path}")

    def _load_observed_streamflow(self):
        """Load observed streamflow data if available."""
        import pandas as pd
        import numpy as np

        # Try to load from observations directory
        obs_dir = self.project_dir / 'observations' / 'streamflow' / 'preprocessed'
        if not obs_dir.exists():
            obs_dir = self.project_dir / 'observations' / 'streamflow' / 'raw_data'

        if not obs_dir.exists():
            self.logger.debug("No observed streamflow directory found")
            return None

        # Look for CSV files with streamflow data
        csv_files = list(obs_dir.glob('*.csv'))
        if not csv_files:
            return None

        try:
            # Try to load the first CSV file
            df = pd.read_csv(csv_files[0])

            # Look for streamflow column
            q_col = None
            for col in ['discharge', 'streamflow', 'Q', 'flow', 'FLOW', 'Value']:
                if col in df.columns:
                    q_col = col
                    break

            if q_col is None:
                return None

            return df[q_col].values
        except Exception as e:
            self.logger.warning(f"Failed to load observed streamflow: {e}")
            return None

    def _copy_settings_to_forcing(self) -> None:
        """Copy MESH settings files from setup_dir to forcing_dir."""
        self.logger.info(f"Copying MESH settings from {self.setup_dir} to {self.forcing_dir}")
        for settings_file in self.setup_dir.glob("*"):
            if settings_file.is_file():
                # Don't overwrite files we just created
                if settings_file.name in ["MESH_input_run_options.ini", "MESH_forcing.nc", "MESH_drainage_database.nc"]:
                    continue
                shutil.copy2(settings_file, self.forcing_dir / settings_file.name)

    def _copy_shapefile(self, src: str, dst: Path) -> None:
        """Copy all files associated with a shapefile."""
        src_path = Path(src)
        for f in src_path.parent.glob(f"{src_path.stem}.*"):
            shutil.copy2(f, dst.parent / f"{dst.stem}{f.suffix}")

    def _sanitize_shapefile(self, shp_path: str) -> None:
        """Remove or rename problematic fields from shapefile."""
        if not shp_path:
            return
        path = Path(shp_path)
        if not path.exists():
            return

        try:
            import geopandas as gpd
            gdf = gpd.read_file(path)

            # 'ID' can cause MergeError in xarray
            if 'ID' in gdf.columns:
                self.logger.info(f"Sanitizing shapefile {path.name}: renaming 'ID' to 'ORIG_ID'")
                gdf = gdf.rename(columns={'ID': 'ORIG_ID'})
                temp_path = path.with_suffix('.tmp.shp')
                gdf.to_file(temp_path)
                shutil.move(temp_path, path)
                # Move associated files
                for ext in ['.shx', '.dbf', '.prj', '.cpg']:
                    temp_ext = temp_path.with_suffix(ext)
                    if temp_ext.exists():
                        shutil.move(temp_ext, path.with_suffix(ext))
        except Exception as e:
            self.logger.warning(f"Failed to sanitize shapefile {path}: {e}")

    def _fix_outlet_segment(self, shp_path: str, outlet_value: int = 0) -> None:
        """Fix outlet segment in river network shapefile."""
        if not shp_path:
            return
        path = Path(shp_path)
        if not path.exists():
            return

        try:
            import geopandas as gpd
            gdf = gpd.read_file(path)

            if 'LINKNO' not in gdf.columns or 'DSLINKNO' not in gdf.columns:
                return

            valid_linknos = set(gdf['LINKNO'].values)
            outlet_mask = ~gdf['DSLINKNO'].isin(valid_linknos) & (gdf['DSLINKNO'] != outlet_value)

            if outlet_mask.any():
                gdf.loc[outlet_mask, 'DSLINKNO'] = outlet_value
                gdf.to_file(path)
                self.logger.info(f"Fixed {outlet_mask.sum()} outlet segment(s) in {path.name}")
        except Exception as e:
            self.logger.warning(f"Failed to fix outlet segment: {e}")

    def _detect_gru_classes(self, landcover_path: Path) -> list:
        """
        Detect which GRU classes exist in the landcover stats file.

        Reads the landcover CSV and extracts GRU class numbers from frac_* columns.

        Args:
            landcover_path: Path to landcover stats CSV

        Returns:
            List of GRU class numbers that exist in the data
        """
        import re

        if not landcover_path or not Path(landcover_path).exists():
            self.logger.warning(f"Landcover file not found: {landcover_path}")
            return []

        try:
            import pandas as pd
            df = pd.read_csv(landcover_path)

            # Find frac_* columns
            frac_cols = [col for col in df.columns if col.startswith('frac_')]

            # Also check for IGBP_* columns (raw counts, will be converted to fractions)
            igbp_cols = [col for col in df.columns if col.startswith('IGBP_')]

            gru_classes = set()

            # Extract class numbers from frac_* columns
            for col in frac_cols:
                match = re.match(r'frac_(\d+)', col)
                if match:
                    gru_classes.add(int(match.group(1)))

            # Extract class numbers from IGBP_* columns
            for col in igbp_cols:
                match = re.match(r'IGBP_(\d+)', col)
                if match:
                    gru_classes.add(int(match.group(1)))

            # Sort and return
            result = sorted(list(gru_classes))
            self.logger.debug(f"Detected GRU classes from {landcover_path.name}: {result}")
            return result

        except Exception as e:
            self.logger.warning(f"Failed to detect GRU classes from {landcover_path}: {e}")
            return []

    def _sanitize_landcover_stats(self, csv_path: str) -> str:
        """Sanitize landcover stats CSV for meshflow compatibility."""
        if not csv_path:
            return csv_path

        path = Path(csv_path)
        if not path.exists():
            return csv_path

        try:
            import pandas as pd
            df = pd.read_csv(path)

            # Remove unnamed index columns
            cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            # Remove duplicate rows
            initial_rows = len(df)
            df = df.drop_duplicates()
            if len(df) < initial_rows:
                self.logger.info(f"Removed {initial_rows - len(df)} duplicate rows from landcover stats")

            # Convert IGBP_* columns to frac_* (meshflow expects fractions)
            igbp_cols = [col for col in df.columns if col.startswith('IGBP_')]
            if igbp_cols:
                count_data = df[igbp_cols].fillna(0)
                row_totals = count_data.sum(axis=1)

                for col in igbp_cols:
                    class_num = col.replace('IGBP_', '')
                    frac_col = f'frac_{class_num}'
                    df[frac_col] = count_data[col] / row_totals.replace(0, 1)
                    df = df.drop(columns=[col])

            # Save to forcing directory
            temp_path = self.forcing_dir / f"temp_{path.name}"
            df.to_csv(temp_path, index=False)
            return str(temp_path)

        except Exception as e:
            self.logger.warning(f"Failed to sanitize landcover stats: {e}")
            return csv_path
