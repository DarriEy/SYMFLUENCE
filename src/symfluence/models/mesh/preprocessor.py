"""
MESH model preprocessor.

Handles data preparation using meshflow library for MESH model setup.
"""

import os

from typing import Dict, Any

from pathlib import Path
import shutil



try:
    from meshflow.core import MESHWorkflow
    MESHFLOW_AVAILABLE = True
    try:
        from meshflow._default_attrs import ddb_local_attrs_default
    except ImportError:
        try:
            from meshflow._default_dicts import ddb_local_attrs_default
        except ImportError:
            ddb_local_attrs_default = {}
except ImportError as e:
    import logging
    logging.debug(f"meshflow import failed: {e}. MESH preprocessing will be limited.")
    MESHFLOW_AVAILABLE = False
    ddb_local_attrs_default = {}

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
from symfluence.core.exceptions import ConfigurationError, ModelExecutionError, symfluence_error_handler


@ModelRegistry.register_preprocessor('MESH')
class MESHPreProcessor(BaseModelPreProcessor, ObservationLoaderMixin):
    """
    Preprocessor for the MESH model.

    Handles data preparation using meshflow library for MESH model setup.
    Inherits common functionality from BaseModelPreProcessor and observation loading from ObservationLoaderMixin.

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
        self._meshflow_config = self.create_json()

    def _prepare_forcing(self) -> None:
        """MESH-specific forcing data preparation (template hook)."""
        # Try to generate landcover stats if missing
        self._ensure_landcover_stats()

        # Branch based on spatial mode
        spatial_mode = self._get_spatial_mode()
        self.logger.info(f"MESH spatial mode: {spatial_mode}")

        if spatial_mode == 'lumped':
            self._prepare_lumped_forcing(self._meshflow_config)
        elif spatial_mode == 'distributed':
            self._prepare_distributed_forcing(self._meshflow_config)
        else:
            raise ConfigurationError(f"Unknown MESH spatial mode: {spatial_mode}")

    def _ensure_landcover_stats(self) -> None:
        """Ensure landcover stats CSV exists, generating it from shapefile if needed."""
        landcover_path = Path(self._meshflow_config.get('landcover', ''))
        if landcover_path and not landcover_path.exists():
            self.logger.info(f"Landcover stats file not found at {landcover_path}. Attempting to generate from shapefile.")
            
            # Ensure directory exists
            landcover_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Look for catchment_with_landclass.shp
            src_shp = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_landclass' / 'catchment_with_landclass.shp'
            if src_shp.exists():
                try:
                    import geopandas as gpd
                    gdf = gpd.read_file(src_shp)
                    # Convert to CSV format expected by meshflow
                    gdf.to_csv(landcover_path, index=False)
                    self.logger.info(f"Generated landcover stats CSV at {landcover_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to generate landcover CSV: {e}")
            else:
                self.logger.warning(f"Source shapefile {src_shp} not found. Cannot generate landcover CSV.")

    def _create_model_configs(self) -> None:
        """Create MESH-specific configuration files (template hook)."""
        self.logger.info("Creating MESH configuration files")
        
        # Always create run options to ensure it has correct flags for this version
        self.create_run_options()
            
        self.copy_settings_to_forcing()

        # Fix for MESH 1.4 compatibility: Create split parameter files from MESH_parameters.txt
        params_txt = self.forcing_dir / "MESH_parameters.txt"
        if params_txt.exists():
            import shutil
            self.logger.info("Creating MESH_parameters_CLASS.ini and MESH_parameters_hydrology.ini for MESH 1.4")
            shutil.copy2(params_txt, self.forcing_dir / "MESH_parameters_CLASS.ini")
            shutil.copy2(params_txt, self.forcing_dir / "MESH_parameters_hydrology.ini")

    def copy_settings_to_forcing(self) -> None:
        """Copy MESH settings files from setup_dir to forcing_dir."""
        self.logger.info(f"Copying MESH settings from {self.setup_dir} to {self.forcing_dir}")
        for settings_file in self.setup_dir.glob("*"):
            if settings_file.is_file():
                import shutil
                # Don't overwrite MESH_input_run_options.ini if we just created it
                if settings_file.name == "MESH_input_run_options.ini":
                    continue
                shutil.copy2(settings_file, self.forcing_dir / settings_file.name)

    def create_run_options(self) -> None:
        """Create MESH_input_run_options.ini file."""
        run_options_path = self.forcing_dir / "MESH_input_run_options.ini"

        # Determine spatial mode for correct SHDFILEFLAG
        spatial_mode = self._get_spatial_mode()

        # Get simulation times
        time_window = self.get_simulation_time_window()
        if time_window:
            start_time, end_time = time_window
        else:
            # Fallback defaults
            import pandas as pd
            start_time = pd.Timestamp("2004-01-01 01:00")
            end_time = pd.Timestamp("2004-01-05 23:00")

        # Set drainage database format based on spatial mode
        # meshflow generates NetCDF output for both modes
        if spatial_mode == 'distributed':
            shd_flag = 'nc_subbasin'  # NetCDF subbasin format (1D) for distributed mode
            basin_flag = 'nc'  # NetCDF forcing
        else:
            shd_flag = 'nc'  # NetCDF drainage database (meshflow generates .nc, not .r2c)
            basin_flag = 'nc'  # NetCDF forcing

        # Basic MESH_input_run_options.ini content
        # Updated for MESH 1.4 compatibility based on source code analysis
        # SHDFILEFLAG set based on spatial mode
        # Using RUNMODE runclass instead of RUNCLASS
        content = f"""MESH input run options file                             # comment line 1                                | *
##### Control Flags #####                               # comment line 2                                | *
----#                                                   # comment line 3                                | *
   13                                                   # Number of control flags                       | I5
SHDFILEFLAG         {shd_flag}                          # Drainage database format (nc=NetCDF, r2c=ASCII)
BASINFORCINGFLAG    {basin_flag}                        # Forcing file format (nc = NetCDF in 1.4)
RUNMODE             runclass                            # Run mode (runclass = CLASS + Routing)
INPUTPARAMSFORMFLAG only txt                            # Parameter file format (txt = MESH_parameters.txt)
RESUMEFLAG          off                                   # Resume from state (0=No)
SAVERESUMEFLAG      off                                   # Save final state (1=Yes)
TIMESTEPFLAG        60                                  # Time step in minutes (default 60)
OUTFIELDSFLAG       all                                 # Output fields (all, none, default)
BASINRUNOFFFLAG     ts                                  # Runoff output format (ts = time series)
LOCATIONFLAG        1                                   # Centroid location
PBSMFLAG            off                                 # Blowing snow (off)
BASEFLOWFLAG        wf_lzs                              # Baseflow formulation
INTERPOLATIONFLAG   0                                   # Interpolation (0=No)
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
        try:
            with open(run_options_path, 'w') as f:
                f.write(content)
            self.logger.info(f"Created {run_options_path} (spatial_mode={spatial_mode}, shd_flag={shd_flag})")
        except Exception as e:
            self.logger.error(f"Failed to create {run_options_path}: {e}")
            raise ModelExecutionError(f"Failed to create MESH run options: {e}")

    def create_json(self):
        """Create configuration dictionary for meshflow."""

        def _get_config_value(key: str, default_value):
            value = self.config_dict.get(key)
            if value is None or value == 'default':
                return default_value
            return value

        # meshflow expects: standard_name -> actual_file_variable_name
        default_forcing_vars = {
            "air_pressure": "airpres",
            "specific_humidity": "spechum",
            "air_temperature": "airtemp",
            "wind_speed": "windspd",
            "precipitation": "pptrate",
            "shortwave_radiation": "SWRadAtm",
            "longwave_radiation": "LWRadAtm",
        }

        # Units keyed by standard names (meshflow expects these keys)
        default_forcing_units = {
            "air_pressure": 'pascal',
            "specific_humidity": 'kg/kg',
            "air_temperature": 'kelvin',
            "wind_speed": 'm/s',
            "precipitation": 'm/s',  # CARRA uses m/s
            "shortwave_radiation": 'W/m^2',
            "longwave_radiation": 'W/m^2',
        }

        default_forcing_to_units = {
            "air_pressure": 'pascal',
            "specific_humidity": 'kg/kg',
            "air_temperature": 'kelvin',
            "wind_speed": 'm/s',
            "precipitation": 'mm/s',
            "shortwave_radiation": 'W/m^2',
            "longwave_radiation": 'W/m^2',
        }

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

        default_ddb_vars = {
            'Slope': 'ChnlSlope',
            'Length': 'ChnlLength',
            'Rank': 'Rank',
            'Next': 'Next',
            'landcover': 'GRU',
            'GRU_area': 'GridArea',
            'river_class': 'strmOrder',  # River class from TauDEM stream order
        }

        default_ddb_units = {
            'ChnlSlope': 'm/m',
            'ChnlLength': 'm',
            'Rank': 'dimensionless',
            'Next': 'dimensionless',
            'GRU': 'dimensionless',
            'GridArea': 'm^2',
            'strmOrder': 'dimensionless',  # River class
        }

        default_ddb_to_units = default_ddb_units.copy()

        default_ddb_min_values = {
            'ChnlSlope': 1e-10,
            'ChnlLength': 1e-3,
            'GridArea': 1e-3,
        }

        forcing_vars = _get_config_value('MESH_FORCING_VARS', default_forcing_vars)
        forcing_units = default_forcing_units.copy()
        forcing_units.update(_get_config_value('MESH_FORCING_UNITS', {}))
        forcing_to_units = default_forcing_to_units.copy()
        forcing_to_units.update(_get_config_value('MESH_FORCING_TO_UNITS', {}))

        missing_units = [var for var in forcing_vars if var not in forcing_units]
        missing_to_units = [var for var in forcing_vars if var not in forcing_to_units]
        if missing_units or missing_to_units:
            raise ConfigurationError(
                "MESH forcing units are incomplete. Missing units for: "
                f"{', '.join(sorted(set(missing_units + missing_to_units)))}"
            )

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

        forcing_files_path = Path(
            _get_config_value(
                'MESH_FORCING_PATH',
                self.project_dir / 'forcing' / 'basin_averaged_data',
            )
        )

        ddb_vars = _get_config_value('MESH_DDB_VARS', default_ddb_vars)
        
        # Filter local attributes to only include mapped variables to avoid KeyError in meshflow
        filtered_ddb_local_attrs = {
            var: attrs for var, attrs in ddb_local_attrs_default.items()
            if var in ddb_vars.values()
        }

        # Use a more specific glob pattern to avoid picking up .csv files (e.g. from easymore)
        forcing_files_glob = os.path.join(str(forcing_files_path), '*.nc')

        # Get simulation dates from symfluence config for meshflow settings
        time_window = self.get_simulation_time_window()
        if time_window:
            start_date, end_date = time_window
            forcing_start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
            sim_start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
            sim_end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            # Fallback defaults
            forcing_start_date = '2001-01-01 00:00:00'
            sim_start_date = '2001-01-01 00:00:00'
            sim_end_date = '2010-12-31 23:00:00'

        # meshflow v0.1.0.dev6 requires settings dict with 'core' and 'class_params' structures
        default_settings = {
            'core': {
                'forcing_files': 'single',  # 'single' or 'multiple'
                'forcing_start_date': forcing_start_date,
                'simulation_start_date': sim_start_date,
                'simulation_end_date': sim_end_date,
                'forcing_time_zone': 'UTC',
            },
            'class_params': {
                'measurement_heights': {
                    'wind_speed': 10.0,  # meters
                    'specific_humidity': 2.0,  # meters
                    'air_temperature': 2.0,  # meters (must equal specific_humidity)
                    'roughness_length': 0.5,  # meters
                },
            },
            'hydrology_params': {
                # Default hydrology parameters - can be customized
            },
        }

        # using meshflow >= v0.1.0.dev5
        # modify the following to match your settings
        config = {
            'riv': os.path.join(str(self.rivers_path / self.rivers_name)),
            'cat': os.path.join(str(self.catchment_path / self.catchment_name)),
            'landcover': os.path.join(str(landcover_path)),
            'forcing_files': forcing_files_glob,
            'forcing_vars': forcing_vars,
            'forcing_units': forcing_units,
            'forcing_to_units': forcing_to_units,
            'main_id': _get_config_value('MESH_MAIN_ID', 'GRU_ID'),
            'ds_main_id': _get_config_value('MESH_DS_MAIN_ID', 'DSLINKNO'),
            'landcover_classes': _get_config_value('MESH_LANDCOVER_CLASSES', default_landcover_classes),
            'ddb_vars': ddb_vars,
            'ddb_units': _get_config_value('MESH_DDB_UNITS', default_ddb_units),
            'ddb_to_units': _get_config_value('MESH_DDB_TO_UNITS', default_ddb_to_units),
            'ddb_min_values': _get_config_value('MESH_DDB_MIN_VALUES', default_ddb_min_values),
            'ddb_local_attrs': filtered_ddb_local_attrs,
            'gru_dim': _get_config_value('MESH_GRU_DIM', 'NGRU'),
            # meshflow v0.1.0.dev6 hardcodes 'subbasin' in several places, so we must use it
            # and rename to 'N' after processing (MESH expects 'N' dimension)
            'hru_dim': 'subbasin',
            'outlet_value': _get_config_value('MESH_OUTLET_VALUE', 0),
            'settings': _get_config_value('MESH_SETTINGS', default_settings),
        }
        return config

    def _create_lumped_forcing_simple(self) -> bool:
        """
        Create lumped MESH forcing from existing distributed forcing files.

        Returns:
            True if successful, False otherwise
        """
        import xarray as xr
        import numpy as np

        # Check if MESH_forcing.nc exists (from previous distributed run)
        forcing_file = self.forcing_dir / "MESH_forcing.nc"

        if not forcing_file.exists():
            self.logger.debug("MESH_forcing.nc not found - cannot create lumped forcing")
            return False

        try:
            # Load distributed forcing
            ds = xr.open_dataset(forcing_file)
            self.logger.info(f"Loaded distributed forcing with {ds.dims.get('N', 0)} GRUs")

            # Average across spatial dimension (N) to create lumped forcing
            ds_lumped = ds.mean(dim='N', keep_attrs=True)

            # Save as MESH_forcing.nc (overwrite distributed version)
            ds_lumped.to_netcdf(forcing_file, format='NETCDF3_CLASSIC')
            self.logger.info(f"Created lumped forcing file: {forcing_file}")

            # Create simple r2c drainage database for lumped mode
            self._create_lumped_drainage_database()

            ds.close()
            return True

        except Exception as e:
            self.logger.warning(f"Failed to create lumped forcing: {e}")
            return False

    def _create_lumped_drainage_database(self):
        """Create a simple r2c (ASCII) drainage database for lumped MESH."""
        ddb_path = self.forcing_dir / "MESH_drainage_database.r2c"

        # R2C format requires proper EnSim header
        # Single grid for lumped mode
        content = """########################################
:FileType r2c ASCII EnSim 1.0
#
# DataType 2D Rect Cell
#
:Application SYMFLUENCE
:Version 1.0
:WrittenBy SYMFLUENCE lumped mode generator
#
#---------------------------------------
#
:Projection LATLONG
:Ellipsoid WGS84
#
:xOrigin -116.0
:yOrigin 51.5
#
:xCount 1
:yCount 1
:xDelta 1.0
:yDelta 1.0
#
#---------------------------------------
#
:AttributeName 1 NEXT
:AttributeName 2 IREACH
:AttributeName 3 RANK
:AttributeName 4 ChnlSlope
:AttributeName 5 GridArea
:AttributeName 6 ChnlLength
:AttributeName 7 ELEV
:AttributeName 8 IAK
#
:EndHeader
0
0
1
0.001
250000000.0
1000.0
1500.0
1
"""

        with open(ddb_path, 'w') as f:
            f.write(content)

        self.logger.info(f"Created lumped drainage database: {ddb_path}")

    def _prepare_lumped_forcing(self, config):
        """
        Prepare forcing data for lumped/basin-averaged mode using meshflow.

        meshflow generates proper drainage database with GRU fractions from landcover.
        """
        # Use meshflow to generate proper MESH inputs with GRU fractions
        if not MESHFLOW_AVAILABLE:
            self.logger.warning("meshflow not available and no distributed forcing found - MESH preprocessing incomplete")
            return

        try:
            # Check if required files exist
            required_files = [
                config.get('riv'),
                config.get('cat'),
                config.get('landcover'),
            ]

            missing_files = [f for f in required_files if f and not Path(f).exists()]

            if missing_files:
                self.logger.warning(f"MESH preprocessing skipped - missing required files: {missing_files}")
                self.logger.info("MESH will run without meshflow preprocessing (may fail or produce limited results)")
                return

            # Sanitize shapefiles to avoid xarray MergeError with 'ID' field
            # Use copies to avoid modifying original shapefiles used in tests/other models
            riv_copy = self.forcing_dir / f"temp_{Path(config.get('riv')).name}"
            cat_copy = self.forcing_dir / f"temp_{Path(config.get('cat')).name}"
            
            import shutil
            # Copy all related shapefile files (.shp, .shx, .dbf, .prj)
            def copy_shapefile(src, dst):
                src_path = Path(src)
                dst_path = Path(dst)
                for f in src_path.parent.glob(f"{src_path.stem}.*"):
                    shutil.copy2(f, dst_path.parent / f"{dst_path.stem}{f.suffix}")
            
            copy_shapefile(config.get('riv'), riv_copy)
            copy_shapefile(config.get('cat'), cat_copy)

            self._sanitize_shapefile(str(riv_copy))
            self._sanitize_shapefile(str(cat_copy))

            # Fix outlet segment in river network (meshflow requires DSLINKNO = outlet_value for outlet)
            outlet_value = config.get('outlet_value', 0)
            self._fix_outlet_segment(str(riv_copy), outlet_value=outlet_value)

            # Sanitize landcover stats
            sanitized_landcover = self._sanitize_landcover_stats(config.get('landcover'))

            # Update config with sanitized copies
            config['riv'] = str(riv_copy)
            config['cat'] = str(cat_copy)
            config['landcover'] = str(sanitized_landcover)

            import meshflow
            self.logger.info(f"meshflow version: {getattr(meshflow, '__version__', 'unknown')}")
            self.logger.info(f"meshflow file: {meshflow.__file__}")

            # Ensure a clean slate for meshflow output files
            output_files_to_clean = [
                self.forcing_dir / "MESH_forcing.nc",
                self.forcing_dir / "MESH_drainage_database.nc",
            ]
            for f in output_files_to_clean:
                if f.exists():
                    self.logger.info(f"Removing existing meshflow output file: {f}")
                    f.unlink()

            # Monkey-patch meshflow to fix bug in v0.1.0.dev1
            try:
                import meshflow.utility.forcing_prep
                import meshflow.utility
                import meshflow.core

                orig_prep = meshflow.utility.forcing_prep.prepare_mesh_forcing

                def patched_prep(*args, **kwargs):
                    self.logger.info("patched_prep called")
                    # Extract variables from args or kwargs
                    variables = None
                    if 'variables' in kwargs:
                        variables = kwargs['variables']
                    elif len(args) > 1:
                        variables = args[1]

                    if variables is not None:
                        if not isinstance(variables, list):
                            # Convert to list if it's dict_values or other iterable
                            new_vars = list(variables)
                            if 'variables' in kwargs:
                                kwargs['variables'] = new_vars
                            elif len(args) > 1:
                                args = list(args)
                                args[1] = new_vars
                                args = tuple(args)

                    # Fix for CDO mergetime: filter out non-nc files from glob
                    from glob import glob
                    if 'path' in kwargs:
                        p = kwargs['path']
                        if isinstance(p, str) and '*' in p:
                            files = [f for f in glob(p) if f.endswith('.nc') and not f.endswith('.nc.csv')]
                            if files:
                                kwargs['path'] = sorted(files)
                    elif len(args) > 0 and isinstance(args[0], str) and '*' in args[0]:
                        p = args[0]
                        files = [f for f in glob(p) if f.endswith('.nc') and not f.endswith('.nc.csv')]
                        if files:
                            # Reconstruct args to replace positional path
                            args = list(args)
                            args[0] = sorted(files)
                            args = tuple(args)

                    return orig_prep(*args, **kwargs)

                def patched_freq(freq_alias):
                    if freq_alias is None: return 'hours'
                    f = str(freq_alias).upper()
                    if f == 'H': return 'hours'
                    if f in ('T', 'MIN'): return 'minutes'
                    if f == 'S': return 'seconds'
                    if f in ('L', 'MS'): return 'milliseconds'
                    if f == 'D': return 'days'
                    return 'hours'

                # Patch everywhere
                meshflow.utility.forcing_prep.prepare_mesh_forcing = patched_prep
                meshflow.utility.prepare_mesh_forcing = patched_prep
                meshflow.core.utility.prepare_mesh_forcing = patched_prep

                meshflow.utility.forcing_prep.freq_long_name = patched_freq
                meshflow.utility.freq_long_name = patched_freq
                meshflow.core.utility.forcing_prep.freq_long_name = patched_freq

                # Patch MESHWorkflow.save to fix NC_UNLIMITED dimension issue
                orig_save = meshflow.core.MESHWorkflow.save

                def patched_save(self, output_dir):
                    """Patched save method that handles unlimited dimensions correctly."""
                    forcing_file = 'MESH_forcing.nc'
                    ddb_file = 'MESH_drainage_database.nc'

                    # Save forcing with only time as unlimited dimension
                    if hasattr(self, 'forcing') and self.forcing is not None:
                        forcing_path = os.path.join(output_dir, forcing_file)
                        # Specify unlimited_dims to avoid "NC_UNLIMITED size already in use" error
                        self.forcing.to_netcdf(forcing_path, unlimited_dims=['time'])

                    # Save drainage database with no unlimited dimensions
                    if hasattr(self, 'ddb') and self.ddb is not None:
                        ddb_path = os.path.join(output_dir, ddb_file)
                        # DDB typically doesn't need unlimited dimensions
                        self.ddb.to_netcdf(ddb_path)

                meshflow.core.MESHWorkflow.save = patched_save

                # Patch _prepare_landcover_mesh to handle duplicate MultiIndex issues
                try:
                    import meshflow.utility.network as network_module
                    orig_prepare_landcover = network_module._prepare_landcover_mesh

                    def patched_prepare_landcover(landcover, *args, **kwargs):
                        """Patched version that handles duplicate rows and index before stacking."""
                        import pandas as pd
                        import logging

                        # If landcover is a DataFrame, ensure no duplicates before stacking
                        if isinstance(landcover, pd.DataFrame):
                            initial_rows = len(landcover)

                            # Drop duplicate rows (same values across all columns)
                            landcover = landcover.drop_duplicates()
                            if len(landcover) < initial_rows:
                                logging.info(f"Removed {initial_rows - len(landcover)} duplicate rows from landcover")

                            # Ensure index is unique - reset if duplicates found
                            if landcover.index.has_duplicates:
                                logging.info("Landcover DataFrame has duplicate index values, resetting index")
                                landcover = landcover.reset_index(drop=True)

                            # Ensure column names are unique
                            if len(landcover.columns) != len(set(landcover.columns)):
                                logging.warning("Landcover DataFrame has duplicate column names")
                                # Keep only first occurrence of each column name
                                landcover = landcover.loc[:, ~landcover.columns.duplicated()]

                        # Call original function with deduplicated data
                        return orig_prepare_landcover(landcover, *args, **kwargs)

                    network_module._prepare_landcover_mesh = patched_prepare_landcover
                    self.logger.info("Successfully patched _prepare_landcover_mesh")
                except Exception as e:
                    self.logger.warning(f"Failed to patch _prepare_landcover_mesh: {e}")

                self.logger.info("Successfully monkey-patched meshflow in multiple locations")
            except Exception as e:
                self.logger.warning(f"Failed to monkey-patch meshflow: {e}")
            self.logger.info(f"MESHWorkflow class origin: {MESHWorkflow.__module__}")
            self.logger.info("Initializing MESHWorkflow with configuration")
            exp = MESHWorkflow(**config)

            self.logger.info(f"Running MESHWorkflow preprocessing, saving to {self.forcing_dir}")
            # Try without arguments if positional failed
            try:
                exp.run(save_path=str(self.forcing_dir))
            except TypeError:
                try:
                    exp.run(str(self.forcing_dir))
                except TypeError:
                    exp.run()

            # Save drainage database and forcing files
            # (forcing_dir already created by base class create_directories())
            self.logger.info("Saving MESH drainage database and forcing files")
            exp.save(str(self.forcing_dir))

            # Rename 'subbasin' dimension/variable to 'N' for MESH compatibility
            # (meshflow v0.1.0.dev6 hardcodes 'subbasin' but MESH expects 'N')
            import xarray as xr
            for nc_file in [self.forcing_dir / "MESH_forcing.nc", self.forcing_dir / "MESH_drainage_database.nc"]:
                if nc_file.exists():
                    try:
                        with xr.open_dataset(nc_file) as ds:
                            # Rename dimension and variable from 'subbasin' to 'N'
                            rename_dict = {}
                            if 'subbasin' in ds.dims:
                                rename_dict['subbasin'] = 'N'
                            if 'subbasin' in ds.variables and 'subbasin' not in ds.dims:
                                rename_dict['subbasin'] = 'N'
                            if rename_dict:
                                ds_renamed = ds.rename(rename_dict)
                                temp_path = nc_file.with_suffix('.tmp.nc')
                                ds_renamed.to_netcdf(temp_path)
                                os.replace(temp_path, nc_file)
                                self.logger.info(f"Renamed 'subbasin' to 'N' in {nc_file.name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to rename subbasin to N in {nc_file.name}: {e}")

            # Reindex N coordinates to sequential integers (1 to NA)
            # MESH expects sequential grid IDs, not arbitrary GRU_IDs from TauDEM
            import numpy as np
            for nc_file in [self.forcing_dir / "MESH_forcing.nc", self.forcing_dir / "MESH_drainage_database.nc"]:
                if nc_file.exists():
                    try:
                        with xr.open_dataset(nc_file) as ds:
                            if 'N' in ds.coords:
                                old_n = ds['N'].values
                                new_n = np.arange(1, len(old_n) + 1)
                                if not np.array_equal(old_n, new_n):
                                    ds_reindexed = ds.assign_coords(N=new_n)
                                    # Also update Rank if present in drainage database
                                    if 'Rank' in ds_reindexed.data_vars:
                                        ds_reindexed['Rank'] = xr.DataArray(
                                            new_n.astype(float), dims=['N'],
                                            attrs=ds_reindexed['Rank'].attrs
                                        )
                                    temp_path = nc_file.with_suffix('.tmp.nc')
                                    ds_reindexed.to_netcdf(temp_path)
                                    os.replace(temp_path, nc_file)
                                    self.logger.info(f"Reindexed N to sequential 1-{len(new_n)} in {nc_file.name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to reindex N in {nc_file.name}: {e}")

            # Add missing variables to drainage database (IREACH, ChnlSlope, ChnlLength)
            # MESH 1.5.6 has a bug checking uninitialized IREACH, so we add it as zeros
            ddb_nc = self.forcing_dir / "MESH_drainage_database.nc"
            if ddb_nc.exists():
                try:
                    import numpy as np
                    with xr.open_dataset(ddb_nc) as ds:
                        n_size = ds.sizes.get('N', len(ds['N']) if 'N' in ds else 13)
                        modified = False

                        # Add IREACH if missing (all zeros = no reservoirs)
                        if 'IREACH' not in ds:
                            ds['IREACH'] = xr.DataArray(
                                np.zeros(n_size, dtype=np.int32),
                                dims=['N'],
                                attrs={'long_name': 'Reservoir reach identifier', '_FillValue': -1}
                            )
                            modified = True
                            self.logger.info("Added IREACH (all zeros) to drainage database")

                        # Add ChnlSlope if missing
                        if 'ChnlSlope' not in ds:
                            ds['ChnlSlope'] = xr.DataArray(
                                np.full(n_size, 0.001, dtype=np.float64),
                                dims=['N'],
                                attrs={'long_name': 'Channel Slope', 'units': 'm/m'}
                            )
                            modified = True

                        # Add ChnlLength if missing
                        if 'ChnlLength' not in ds:
                            ds['ChnlLength'] = xr.DataArray(
                                np.full(n_size, 1000.0, dtype=np.float64),
                                dims=['N'],
                                attrs={'long_name': 'Channel Length', 'units': 'm'}
                            )
                            modified = True

                        if modified:
                            temp_path = ddb_nc.with_suffix('.tmp.nc')
                            ds.to_netcdf(temp_path)
                            os.replace(temp_path, ddb_nc)
                except Exception as e:
                    self.logger.warning(f"Failed to add missing variables to drainage database: {e}")

            # Rename variables in MESH_forcing.nc for MESH 1.4 compatibility
            forcing_nc = self.forcing_dir / "MESH_forcing.nc"
            if forcing_nc.exists():
                try:
                    self.logger.info("Renaming forcing variables for MESH 1.4 compatibility")
                    import xarray as xr
                    with xr.open_dataset(forcing_nc) as ds:
                        rename_map = {
                            'airpres': 'PRES',
                            'spechum': 'QA',
                            'airtemp': 'TA',
                            'windspd': 'UV',
                            'pptrate': 'PRE',
                            'SWRadAtm': 'FSIN',
                            'LWRadAtm': 'FLIN'
                        }
                        # Only rename if they exist
                        existing_rename = {k: v for k, v in rename_map.items() if k in ds.variables}
                        if existing_rename:
                            ds_renamed = ds.rename(existing_rename)
                            # Ensure dimension order is (time, N)
                            # MESH usually prefers time first for NetCDF forcing
                            if 'time' in ds_renamed.dims and 'N' in ds_renamed.dims:
                                ds_renamed = ds_renamed.transpose('time', 'N')
                            
                            temp_path = forcing_nc.with_suffix('.tmp.nc')
                            ds_renamed.to_netcdf(temp_path)
                            ds_renamed.close()
                            os.replace(temp_path, forcing_nc)
                    
                    # Symlink to split names expected by MESH 1.4 Driver
                    split_names = [
                        "basin_shortwave.nc", "basin_longwave.nc", "basin_rain.nc",
                        "basin_temperature.nc", "basin_wind.nc", "basin_pres.nc",
                        "basin_humidity.nc", "WR_runoff.nc"
                    ]
                    for name in split_names:
                        dst = self.forcing_dir / name
                        if dst.exists():
                            dst.unlink()
                        os.symlink(forcing_nc.name, dst)
                    self.logger.info("Created component symlinks for NetCDF forcing")
                except Exception as e:
                    self.logger.warning(f"Failed to rename forcing variables or create symlinks: {e}")

            self.logger.info("MESH preprocessing completed successfully")
        except FileNotFoundError as e:
            self.logger.warning(f"MESH preprocessing skipped - file not found: {str(e)}")
            self.logger.info("MESH will run without meshflow preprocessing (may fail or produce limited results)")
        except Exception as e:
            self.logger.error(f"Error during MESH preprocessing: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _prepare_distributed_forcing(self, config):
        """
        Prepare forcing data for distributed GRU-based mode.

        Creates NetCDF drainage database from delineated catchments.
        Does NOT use meshflow - builds from SYMFLUENCE's existing data.
        """
        self.logger.info("Preparing distributed MESH forcing data")

        try:
            # 1. Load GRU shapefile (from delineation)
            catchment_shp = Path(config.get('cat'))
            if not catchment_shp.exists():
                raise FileNotFoundError(f"Catchment shapefile not found: {catchment_shp}")

            import geopandas as gpd
            gdf = gpd.read_file(catchment_shp)
            self.logger.info(f"Loaded {len(gdf)} GRUs from {catchment_shp}")

            # 2. Load river network shapefile
            river_shp = Path(config.get('riv'))
            if not river_shp.exists():
                raise FileNotFoundError(f"River network not found: {river_shp}")

            riv_gdf = gpd.read_file(river_shp)
            self.logger.info(f"Loaded {len(riv_gdf)} river segments from {river_shp}")

            # 3. Build drainage database NetCDF
            self._create_netcdf_drainage_database(gdf, riv_gdf, config)

            # 4. Prepare forcing NetCDF (per-GRU from basin-averaged data)
            self._create_distributed_forcing(gdf, config)

            self.logger.info("Distributed MESH forcing preparation completed")

        except Exception as e:
            self.logger.error(f"Error during distributed MESH preprocessing: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _create_netcdf_drainage_database(self, gdf, riv_gdf, config):
        """
        Create MESH_drainage_database.nc for distributed mode.

        Structure:
        - Dimension: subbasin (number of GRUs)
        - Variables:
          - Rank: subbasin rank
          - Next: downstream subbasin ID (0 = outlet)
          - GridArea: GRU area (m^2)
          - ChnlLength: channel length (m)
          - ChnlSlope: channel slope (m/m)
          - GRU: landcover class (for each GRU)
        """
        import xarray as xr
        import numpy as np

        n_grus = len(gdf)
        self.logger.info(f"Creating NetCDF drainage database for {n_grus} GRUs")

        # Get GRU IDs (typically 'GRU_ID' or configured main_id)
        gru_id_field = config.get('main_id', 'GRU_ID')
        ds_id_field = config.get('ds_main_id', 'DSLINKNO')

        if gru_id_field not in gdf.columns:
            # Try alternative field names
            for alt_field in ['GRU_ID', 'subbasin', 'LINKNO', 'HRU_ID']:
                if alt_field in gdf.columns:
                    gru_id_field = alt_field
                    break
            else:
                raise ValueError(f"GRU ID field not found in catchment shapefile. Tried: {gru_id_field}")

        gru_ids = gdf[gru_id_field].values
        self.logger.info(f"Using GRU ID field: {gru_id_field}")

        # Initialize variables
        topo_ranks = None
        ds_ids = None

        # Build Next topology from river network
        # Next indicates which GRU (by Rank) this GRU flows to (0 = outlet)
        if riv_gdf is not None and len(riv_gdf) > 0:
            self.logger.info("Building drainage topology from river network")

            # Build mappings from river network
            # LINKNO -> GRU_ID: which GRU each segment belongs to
            # LINKNO -> DSLINKNO: which segment each segment flows to
            link_to_gru = dict(zip(riv_gdf['LINKNO'], riv_gdf['GRU_ID']))
            link_to_dslink = dict(zip(riv_gdf['LINKNO'], riv_gdf['DSLINKNO']))
            gru_to_link = dict(zip(riv_gdf['GRU_ID'], riv_gdf['LINKNO']))

            # Build downstream GRU_IDs first (not ranks yet)
            ds_gru_ids = np.zeros(n_grus, dtype=int)
            for i, gru_id in enumerate(gru_ids):
                if gru_id in gru_to_link:
                    # Get the river segment for this GRU
                    my_link = gru_to_link[gru_id]
                    # Get the downstream segment
                    ds_link = link_to_dslink[my_link]

                    # Find which GRU that downstream segment belongs to
                    if ds_link in link_to_gru:
                        ds_gru_id = link_to_gru[ds_link]
                        if ds_gru_id in gru_ids:
                            ds_gru_ids[i] = ds_gru_id
                            self.logger.debug(f"GRU {gru_id} flows to GRU {ds_gru_id}")
                        else:
                            self.logger.warning(f"Downstream GRU {ds_gru_id} not found in GDF, setting to outlet")
                            ds_gru_ids[i] = 0
                    else:
                        # Downstream segment not in network = outlet
                        ds_gru_ids[i] = 0
                        self.logger.debug(f"GRU {gru_id} is outlet (DSLINKNO={ds_link} not in network)")
                else:
                    self.logger.warning(f"GRU {gru_id} not found in river network, setting to outlet")
                    ds_gru_ids[i] = 0

            n_outlets = np.sum(ds_gru_ids == 0)
            self.logger.info(f"Built topology: {n_grus} GRUs, {n_outlets} outlet(s)")

            # Calculate topological hierarchy (outlets have highest level, headwaters have level 1)
            topo_levels = self._calculate_subbasin_rank(gru_ids, ds_gru_ids)
            self.logger.info(f"Calculated topological levels: min={topo_levels.min()}, max={topo_levels.max()}")

            # Sort GRUs by topological level to get ordering
            # Within same level, maintain original order for stability
            sort_indices = np.argsort(topo_levels, kind='stable')

            # Create mapping from original GRU_ID to new sequential Rank (1..N)
            # GRUs are sorted by topology, then assigned sequential ranks
            gru_id_to_rank = {}
            for new_rank, orig_idx in enumerate(sort_indices, start=1):
                gru_id_to_rank[gru_ids[orig_idx]] = new_rank

            self.logger.info(f"Assigned sequential Ranks 1-{n_grus} in topological order")

            # Build Next array using the new sequential ranks
            ds_ids = np.zeros(n_grus, dtype=int)
            for i, ds_gru_id in enumerate(ds_gru_ids):
                if ds_gru_id == 0:
                    ds_ids[i] = 0  # Outlet
                elif ds_gru_id in gru_id_to_rank:
                    ds_ids[i] = gru_id_to_rank[ds_gru_id]
                else:
                    ds_ids[i] = 0  # Shouldn't happen, but safe fallback

            # CRITICAL: Reorder ALL arrays by topological order
            # This ensures Rank array is sequential [1,2,3,...,N] in file
            self.logger.info(f"Reordering all arrays by topological order (indices: {sort_indices[:5]}...)")

            # Reorder the GeoDataFrame by topological order
            gdf = gdf.iloc[sort_indices].reset_index(drop=True)
            gru_ids = gdf[gru_id_field].values
            ds_gru_ids = ds_gru_ids[sort_indices]

            self.logger.debug(f"After reordering: gru_ids = {gru_ids[:5]}...")

            # Rebuild GRU_ID to Rank mapping for sorted order (now just sequential)
            gru_id_to_rank = {gru_id: i+1 for i, gru_id in enumerate(gru_ids)}

            # Rebuild Next array using new sequential indices
            ds_ids = np.zeros(n_grus, dtype=int)
            for i, ds_gru_id in enumerate(ds_gru_ids):
                if ds_gru_id == 0:
                    ds_ids[i] = 0  # Outlet
                elif ds_gru_id in gru_id_to_rank:
                    ds_ids[i] = gru_id_to_rank[ds_gru_id]
                else:
                    self.logger.warning(f"Downstream GRU {ds_gru_id} not found after reordering, setting to outlet")
                    ds_ids[i] = 0

            # After reordering, Rank is simply sequential 1..N
            topo_ranks = np.arange(1, n_grus + 1, dtype=int)

            self.logger.info(f"After reordering: Rank = {topo_ranks}, Next = {ds_ids}")
        else:
            # No river network provided, use fallback
            if ds_id_field in gdf.columns:
                ds_ids = gdf[ds_id_field].values
                self.logger.warning(f"Using downstream IDs from catchment shapefile field '{ds_id_field}'")
            else:
                self.logger.warning(f"No river network and no '{ds_id_field}' field. Using zeros (all outlets)")
                ds_ids = np.zeros(n_grus, dtype=int)

        # Calculate areas (m^2) - now from reordered gdf
        areas = gdf.geometry.area.values

        # Get channel properties from river network or attributes
        # Priority: catchment shapefile > river network > defaults
        if 'ChnlLength' in gdf.columns or 'Length' in gdf.columns:
            # Use channel properties from catchment shapefile
            if 'ChnlLength' in gdf.columns:
                chnl_length = gdf['ChnlLength'].values
            else:
                chnl_length = gdf['Length'].values
            self.logger.info("Using channel length from catchment shapefile")
        elif riv_gdf is not None and len(riv_gdf) > 0 and 'Length' in riv_gdf.columns:
            # Get channel properties from river network
            self.logger.info("Using channel properties from river network")
            gru_id_to_link = dict(zip(riv_gdf['GRU_ID'], riv_gdf['LINKNO']))
            link_to_length = dict(zip(riv_gdf['LINKNO'], riv_gdf['Length']))

            chnl_length = np.zeros(n_grus)
            for i, gru_id in enumerate(gru_ids):
                if gru_id in gru_id_to_link:
                    link_no = gru_id_to_link[gru_id]
                    if link_no in link_to_length:
                        chnl_length[i] = link_to_length[link_no]
                    else:
                        chnl_length[i] = np.sqrt(areas[i])
                else:
                    chnl_length[i] = np.sqrt(areas[i])
        else:
            # Default: use sqrt of area as approximation
            chnl_length = np.sqrt(areas)
            self.logger.warning("Channel length not found, using sqrt(area) as approximation")

        # Get channel slope
        if 'ChnlSlope' in gdf.columns or 'Slope' in gdf.columns:
            if 'ChnlSlope' in gdf.columns:
                chnl_slope = gdf['ChnlSlope'].values
            else:
                chnl_slope = gdf['Slope'].values
            self.logger.info("Using channel slope from catchment shapefile")
        elif riv_gdf is not None and len(riv_gdf) > 0 and 'Slope' in riv_gdf.columns:
            # Get slope from river network
            self.logger.info("Using channel slope from river network")
            gru_id_to_link = dict(zip(riv_gdf['GRU_ID'], riv_gdf['LINKNO']))
            link_to_slope = dict(zip(riv_gdf['LINKNO'], riv_gdf['Slope']))

            chnl_slope = np.zeros(n_grus)
            for i, gru_id in enumerate(gru_ids):
                if gru_id in gru_id_to_link:
                    link_no = gru_id_to_link[gru_id]
                    if link_no in link_to_slope:
                        chnl_slope[i] = link_to_slope[link_no]
                    else:
                        chnl_slope[i] = 0.001
                else:
                    chnl_slope[i] = 0.001
        else:
            chnl_slope = np.ones(n_grus) * 0.001  # Default 0.001 m/m
            self.logger.warning("Channel slope not found, using default 0.001 m/m")

        # Rank field in MESH must follow topological order
        # If we have topology info, use topological ranks; otherwise sequential
        if topo_ranks is not None:
            rank = topo_ranks
            self.logger.info(f"Using topological Ranks: min={rank.min()}, max={rank.max()}")
        else:
            rank = np.arange(1, n_grus + 1, dtype=int)
            self.logger.debug(f"Using sequential Rank IDs: 1 to {n_grus}")

        # GRU landcover class (default to 1 if not specified)
        if 'GRU' in gdf.columns:
            gru_class = gdf['GRU'].values.astype(int)
        else:
            gru_class = np.ones(n_grus, dtype=int)
            self.logger.warning("GRU landcover class not found, using default (1)")

        # Extract lat/lon coordinates from GRU centroids
        # MESH requires X/Y location for each GRU
        centroids = gdf.geometry.centroid
        lons = centroids.x.values
        lats = centroids.y.values
        self.logger.debug(f"Extracted centroids: lat range [{lats.min():.4f}, {lats.max():.4f}], lon range [{lons.min():.4f}, {lons.max():.4f}]")

        # Create xarray Dataset
        # MESH expects lat/lon or x/y for coordinate locations
        # Include IREACH (reservoir ID) - set to 0 for no reservoirs
        ireach = np.zeros(n_grus, dtype=int)  # 0 = no reservoir

        # Create dataset without explicit coordinate variable for subbasin
        # MESH expects pure dimension, not coordinate variable
        ddb = xr.Dataset(
            data_vars={
                'Rank': (['subbasin'], rank.astype(int)),
                'Next': (['subbasin'], ds_ids.astype(int)),
                'GridArea': (['subbasin'], areas.astype(float)),
                'ChnlLength': (['subbasin'], chnl_length.astype(float)),
                'ChnlSlope': (['subbasin'], chnl_slope.astype(float)),
                'GRU': (['subbasin'], gru_class.astype(int)),
                'IREACH': (['subbasin'], ireach.astype(int)),  # Reservoir ID (0 = none)
                'lon': (['subbasin'], lons.astype(float)),  # Use short names for MESH
                'lat': (['subbasin'], lats.astype(float)),
            }
        )

        # Add attributes
        ddb['Rank'].attrs = {'long_name': 'Subbasin rank', 'units': 'dimensionless'}
        ddb['Next'].attrs = {'long_name': 'Downstream subbasin ID', 'units': 'dimensionless'}
        ddb['GridArea'].attrs = {'long_name': 'GRU area', 'units': 'm^2'}
        ddb['ChnlLength'].attrs = {'long_name': 'Channel length', 'units': 'm'}
        ddb['ChnlSlope'].attrs = {'long_name': 'Channel slope', 'units': 'm/m'}
        ddb['GRU'].attrs = {'long_name': 'Landcover class', 'units': 'dimensionless'}
        ddb['IREACH'].attrs = {'long_name': 'Reservoir ID', 'units': 'dimensionless', 'flag_values': '0', 'flag_meanings': 'no_reservoir'}
        ddb['lat'].attrs = {'long_name': 'Latitude', 'units': 'degrees_north', 'standard_name': 'latitude'}
        ddb['lon'].attrs = {'long_name': 'Longitude', 'units': 'degrees_east', 'standard_name': 'longitude'}

        # Add global attributes for MESH compatibility
        from datetime import datetime
        ddb.attrs['Conventions'] = 'CF-1.6'
        ddb.attrs['Projection'] = 'LATLONG'
        ddb.attrs['grid_mapping_name'] = 'latitude_longitude'
        ddb.attrs['Ellipsoid'] = 'WGS84'
        ddb.attrs['title'] = 'MESH drainage database'
        ddb.attrs['history'] = f'Created by SYMFLUENCE on {datetime.now().isoformat()}'

        # Save to forcing directory using netCDF4-python for precise control
        # CRITICAL: MESH expects exact NetCDF structure for nc_subbasin format
        ddb_path = self.forcing_dir / 'MESH_drainage_database.nc'

        try:
            from netCDF4 import Dataset as NC4Dataset

            # Create NetCDF file manually for MESH compatibility
            # CRITICAL: Use NETCDF3_CLASSIC for Fortran compatibility
            with NC4Dataset(ddb_path, 'w', format='NETCDF3_CLASSIC') as ncfile:
                # Create dimensions
                ncfile.createDimension('subbasin', n_grus)
                # NOTE: No NGRU dimension needed - GRU is a 1D array of class IDs, not 2D fractions

                # Global attributes
                ncfile.Conventions = 'CF-1.6'
                ncfile.Projection = 'LATLONG'
                ncfile.grid_mapping_name = 'latitude_longitude'
                ncfile.Ellipsoid = 'WGS84'
                ncfile.title = 'MESH drainage database'
                ncfile.history = f'Created by SYMFLUENCE on {datetime.now().isoformat()}'

                # Create coordinate reference system variable (required by MESH)
                var_crs = ncfile.createVariable('crs', 'i4')
                var_crs.grid_mapping_name = 'latitude_longitude'
                var_crs.longitude_of_prime_meridian = 0.0
                var_crs.semi_major_axis = 6378137.0
                var_crs.inverse_flattening = 298.257223563
                var_crs[:] = 1

                # Create variables (order matters for MESH!)
                # Integer variables
                # CRITICAL: ID variable is required by MESH (same as Rank)
                var_id = ncfile.createVariable('ID', 'i4', ('subbasin',))
                var_id.long_name = 'Subbasin ID'
                var_id.units = 'dimensionless'
                var_id[:] = rank

                var_rank = ncfile.createVariable('Rank', 'i4', ('subbasin',))
                var_rank.long_name = 'Subbasin rank'
                var_rank.units = 'dimensionless'
                # REMOVED: var_rank.grid_mapping = 'crs'
                # REMOVED: var_rank.coordinates = 'lon lat'
                var_rank[:] = rank

                var_next = ncfile.createVariable('Next', 'i4', ('subbasin',))
                var_next.long_name = 'Downstream subbasin ID'
                var_next.units = 'dimensionless'
                # REMOVED: var_next.grid_mapping = 'crs'
                # REMOVED: var_next.coordinates = 'lon lat'
                var_next[:] = ds_ids

                # CRITICAL: GRU must be 1D array of landcover class IDs (integers)
                # MESH distributed mode expects a map of class IDs, NOT fractions
                # Each subbasin gets assigned a single landcover class ID
                var_gru = ncfile.createVariable('GRU', 'i4', ('subbasin',))
                var_gru.long_name = 'Landcover class'
                var_gru.units = 'dimensionless'
                # REMOVED: var_gru.grid_mapping = 'crs'
                # REMOVED: var_gru.coordinates = 'lon lat'
                # For distributed mode: assign landcover class ID to each subbasin
                # Use gru_class array (already extracted from shapefile or defaulted to 1)
                var_gru[:] = gru_class

                var_ireach = ncfile.createVariable('IREACH', 'i4', ('subbasin',))
                var_ireach.long_name = 'Reservoir ID'
                var_ireach.units = 'dimensionless'
                var_ireach[:] = ireach

                # Float variables
                var_area = ncfile.createVariable('GridArea', 'f8', ('subbasin',))
                var_area.long_name = 'GRU area'
                var_area.units = 'm^2'
                # REMOVED: var_area.grid_mapping = 'crs'
                # REMOVED: var_area.coordinates = 'lon lat'
                var_area[:] = areas

                var_length = ncfile.createVariable('ChnlLength', 'f8', ('subbasin',))
                var_length.long_name = 'Channel length'
                var_length.units = 'm'
                # REMOVED: var_length.grid_mapping = 'crs'
                # REMOVED: var_length.coordinates = 'lon lat'
                var_length[:] = chnl_length

                var_slope = ncfile.createVariable('ChnlSlope', 'f8', ('subbasin',))
                var_slope.long_name = 'Channel slope'
                var_slope.units = 'm/m'
                # REMOVED: var_slope.grid_mapping = 'crs'
                # REMOVED: var_slope.coordinates = 'lon lat'
                var_slope[:] = chnl_slope

                # Coordinate variables
                var_lon = ncfile.createVariable('lon', 'f8', ('subbasin',))
                var_lon.long_name = 'Longitude'
                var_lon.units = 'degrees_east'
                var_lon.standard_name = 'longitude'
                var_lon.axis = 'X'
                var_lon[:] = lons

                var_lat = ncfile.createVariable('lat', 'f8', ('subbasin',))
                var_lat.long_name = 'Latitude'
                var_lat.units = 'degrees_north'
                var_lat.standard_name = 'latitude'
                var_lat.axis = 'Y'
                var_lat[:] = lats

            self.logger.info(f"Created drainage database: {ddb_path} ({n_grus} GRUs)")

        except ImportError:
            # Fall back to xarray if netCDF4 not available
            self.logger.warning("netCDF4-python not available, using xarray (may have compatibility issues)")

            # Disable fill values for cleaner NetCDF
            encoding = {}
            for var in ddb.data_vars:
                if var in ['Rank', 'Next', 'GRU', 'IREACH']:
                    encoding[var] = {'_FillValue': None, 'dtype': 'int32'}
                else:
                    encoding[var] = {'_FillValue': None, 'dtype': 'float64'}

            ddb.to_netcdf(ddb_path, format='NETCDF4', encoding=encoding)
            self.logger.info(f"Created drainage database: {ddb_path} ({n_grus} GRUs) [xarray]")

    def _calculate_subbasin_rank(self, gru_ids, ds_ids):
        """
        Calculate subbasin rank based on flow topology.

        Rank 1 = headwater (no upstream basins)
        Rank N = outlet or downstream basin with rank N-1 upstream
        """
        import numpy as np

        n = len(gru_ids)
        rank = np.ones(n, dtype=int)

        # Build lookup dict
        id_to_idx = {int(gid): i for i, gid in enumerate(gru_ids)}

        # Iteratively assign ranks
        max_iterations = n  # Prevent infinite loops
        for iteration in range(max_iterations):
            changed = False
            for i, (gid, ds_id) in enumerate(zip(gru_ids, ds_ids)):
                if ds_id == 0 or int(ds_id) not in id_to_idx:
                    continue  # Outlet or external

                ds_idx = id_to_idx[int(ds_id)]
                if rank[ds_idx] <= rank[i]:
                    rank[ds_idx] = rank[i] + 1
                    changed = True

            if not changed:
                break

        return rank

    def _create_distributed_forcing(self, gdf, config):
        """
        Create MESH_forcing.nc for distributed mode.

        Reads from forcing/basin_averaged_data/*.nc and maps to GRUs.
        """
        import xarray as xr
        import glob

        self.logger.info("Creating distributed forcing NetCDF")

        # Look for basin-averaged forcing data
        forcing_pattern = str(self.project_dir / 'forcing' / 'basin_averaged_data' / '*.nc')
        forcing_files = sorted(glob.glob(forcing_pattern))

        if not forcing_files:
            # Try alternative location
            forcing_pattern = str(self.forcing_dir / '*.nc')
            forcing_files = sorted(glob.glob(forcing_pattern))
            # Exclude drainage database if it exists
            forcing_files = [f for f in forcing_files if 'drainage_database' not in f.lower()]

        if not forcing_files:
            self.logger.warning(f"No forcing files found. MESH may fail without forcing data.")
            return

        self.logger.info(f"Loading {len(forcing_files)} forcing files")

        # Load and concatenate forcing files
        try:
            ds_list = [xr.open_dataset(f) for f in forcing_files]
            forcing_ds = xr.concat(ds_list, dim='time', data_vars='minimal', coords='minimal', compat='override')
        except Exception as e:
            self.logger.warning(f"Could not concatenate forcing files: {e}")
            # Try loading just the first file
            forcing_ds = xr.open_dataset(forcing_files[0])

        # Get number of GRUs
        n_grus = len(gdf)

        # Map forcing variables to MESH 1.4 names
        # Try multiple common naming conventions
        rename_map = {}

        # Direct mapping from common forcing variable names to MESH names
        common_mappings = {
            # Standard names
            'air_pressure': 'PRES',
            'specific_humidity': 'QA',
            'air_temperature': 'TA',
            'wind_speed': 'UV',
            'precipitation': 'PRE',
            'shortwave_radiation': 'FSIN',
            'longwave_radiation': 'FLIN',
            # Short names (used in some forcing files)
            'airpres': 'PRES',
            'spechum': 'QA',
            'airtemp': 'TA',
            'windspd': 'UV',
            'pptrate': 'PRE',
            'SWRadAtm': 'FSIN',
            'LWRadAtm': 'FLIN',
        }

        # Build rename map for variables that exist in the dataset
        for old_name, new_name in common_mappings.items():
            if old_name in forcing_ds:
                rename_map[old_name] = new_name

        # Apply renaming
        forcing_renamed = forcing_ds.rename(rename_map) if rename_map else forcing_ds
        self.logger.debug(f"Renamed forcing variables: {list(rename_map.keys())} -> {list(rename_map.values())}")

        # Ensure correct spatial dimension
        # Check if 'N' dimension already exists
        if 'N' in forcing_renamed.dims:
            forcing_expanded = forcing_renamed
            self.logger.info(f"Using existing 'N' dimension with {forcing_renamed.dims['N']} elements")
        elif any(dim in forcing_renamed.dims for dim in ['subbasin', 'hru', 'gru', 'latitude', 'longitude']):
            # Rename other spatial dimensions to 'N'
            for old_dim in ['subbasin', 'hru', 'gru', 'latitude', 'longitude']:
                if old_dim in forcing_renamed.dims:
                    forcing_expanded = forcing_renamed.rename({old_dim: 'N'})
                    self.logger.info(f"Renamed '{old_dim}' dimension to 'N' ({forcing_renamed.dims[old_dim]} elements)")
                    break
        else:
            # No spatial dimension - broadcast single point to all GRUs
            self.logger.info(f"Broadcasting lumped forcing to {n_grus} GRUs")
            forcing_expanded = forcing_renamed.expand_dims({'N': n_grus})

        # Ensure time is the first dimension
        if 'time' in forcing_expanded.dims:
            dim_order = ['time', 'N'] + [d for d in forcing_expanded.dims if d not in ['time', 'N']]
            forcing_final = forcing_expanded.transpose(*dim_order)
        else:
            forcing_final = forcing_expanded

        # Save forcing as separate files per variable for MESH distributed mode
        # MESH expects basin_<variable>.nc for distributed mode
        mesh_var_mapping = {
            'FSIN': 'basin_shortwave.nc',
            'FLIN': 'basin_longwave.nc',
            'PRES': 'basin_pres.nc',  # MESH expects basin_pres.nc (not basin_pressure.nc)
            'TA': 'basin_temperature.nc',
            'QA': 'basin_humidity.nc',
            'UV': 'basin_wind.nc',
            'PRE': 'basin_rain.nc'  # MESH expects basin_rain.nc (total precipitation as rainfall)
        }

        files_created = []

        # Use netCDF4-python for precise control and MESH compatibility
        try:
            from netCDF4 import Dataset as NC4Dataset

            for mesh_var, filename in mesh_var_mapping.items():
                if mesh_var in forcing_final:
                    var_path = self.forcing_dir / filename
                    var_data = forcing_final[mesh_var].values
                    time_data = forcing_final['time'].values if 'time' in forcing_final else None
                    n_time = var_data.shape[0] if var_data.ndim > 0 else 1
                    n_gru = var_data.shape[1] if var_data.ndim > 1 else 1

                    # Create NetCDF file with NETCDF3_CLASSIC for Fortran compatibility
                    with NC4Dataset(var_path, 'w', format='NETCDF3_CLASSIC') as ncfile:
                        # Create dimensions
                        ncfile.createDimension('time', None)  # unlimited
                        ncfile.createDimension('N', n_gru)

                        # Create CRS variable (required by MESH)
                        var_crs = ncfile.createVariable('crs', 'i4')
                        var_crs.grid_mapping_name = 'latitude_longitude'
                        var_crs.longitude_of_prime_meridian = 0.0
                        var_crs.semi_major_axis = 6378137.0
                        var_crs.inverse_flattening = 298.257223563
                        var_crs.units = ''  # Empty string for dimensionless
                        var_crs[:] = 1

                        # Create data variable
                        var = ncfile.createVariable(mesh_var, 'f4', ('time', 'N'), fill_value=-9999.0)

                        # Set variable attributes based on variable type
                        if mesh_var == 'FSIN':
                            var.long_name = 'downward shortwave radiation at the surface'
                            var.units = 'W m-2'
                        elif mesh_var == 'FLIN':
                            var.long_name = 'downward longwave radiation at the surface'
                            var.units = 'W m-2'
                        elif mesh_var == 'PRES':
                            var.long_name = 'air pressure'
                            var.units = 'Pa'
                        elif mesh_var == 'TA':
                            var.long_name = 'air temperature'
                            var.units = 'K'
                        elif mesh_var == 'QA':
                            var.long_name = 'specific humidity'
                            var.units = 'kg kg-1'
                        elif mesh_var == 'UV':
                            var.long_name = 'wind speed'
                            var.units = 'm s-1'
                        elif mesh_var == 'PRE':
                            var.long_name = 'precipitation rate'
                            var.units = 'm s-1'

                        # Write data
                        var[:] = var_data

                        # Create time variable
                        if time_data is not None:
                            var_time = ncfile.createVariable('time', 'f8', ('time',))
                            var_time.long_name = 'time'
                            var_time.standard_name = 'time'
                            var_time.axis = 'T'
                            var_time.units = 'hours since 1900-01-01'
                            var_time.calendar = 'gregorian'

                            # Convert datetime64 to hours since 1900-01-01
                            import numpy as np
                            import pandas as pd
                            reference = pd.Timestamp('1900-01-01')
                            time_hours = [(pd.Timestamp(t) - reference).total_seconds() / 3600.0 for t in time_data]
                            var_time[:] = time_hours

                        # Create N coordinate variable (MESH expects this)
                        var_n = ncfile.createVariable('N', 'i4', ('N',))
                        var_n.units = ''  # Empty string for dimensionless
                        var_n[:] = np.arange(1, n_gru + 1)  # 1-indexed like MESH expects

                    files_created.append(filename)
                    self.logger.debug(f"Created {filename}")

            self.logger.info(f"Created {len(files_created)} distributed forcing files: {', '.join(files_created)}")

        except ImportError:
            # Fallback to xarray if netCDF4 not available
            self.logger.warning("netCDF4-python not available, using xarray for forcing files (may have compatibility issues)")

            for mesh_var, filename in mesh_var_mapping.items():
                if mesh_var in forcing_final:
                    var_ds = forcing_final[[mesh_var]]  # Extract single variable
                    var_path = self.forcing_dir / filename

                    # Create dataset with correct structure
                    var_ds_out = xr.Dataset({
                        mesh_var: var_ds[mesh_var]
                    })

                    # Add time coordinate if it exists
                    if 'time' in forcing_final:
                        var_ds_out = var_ds_out.assign_coords({'time': forcing_final['time']})

                    # Save with unlimited time dimension in NETCDF3_CLASSIC format for Fortran compatibility
                    var_ds_out.to_netcdf(var_path, unlimited_dims=['time'], format='NETCDF3_CLASSIC')
                    files_created.append(filename)
                    self.logger.debug(f"Created {filename}")

            self.logger.info(f"Created {len(files_created)} distributed forcing files: {', '.join(files_created)}")

    def _sanitize_shapefile(self, shp_path: str) -> None:
        """Remove or rename problematic fields like 'ID' from shapefile."""
        if not shp_path:
            return
        path = Path(shp_path)
        if not path.exists():
            return
        
        try:
            import geopandas as gpd
            gdf = gpd.read_file(path)
            self.logger.debug(f"Shapefile {path.name} columns before sanitization: {gdf.columns.tolist()}")

            # 'ID' can cause MergeError in xarray when meshflow combines datasets
            if 'ID' in gdf.columns:
                self.logger.info(f"Sanitizing shapefile {path.name}: renaming 'ID' to 'ORIG_ID'")
                gdf = gdf.rename(columns={'ID': 'ORIG_ID'})
                # Save to a temporary file, then replace to avoid partial writes
                temp_path = path.with_suffix('.tmp.shp')
                gdf.to_file(temp_path)
                shutil.move(temp_path, path)
            self.logger.debug(f"Shapefile {path.name} columns after sanitization: {gdf.columns.tolist()}")
        except Exception as e:
            self.logger.warning(f"Failed to sanitize shapefile {path}: {e}")

    def _fix_outlet_segment(self, shp_path: str, outlet_value: int = 0) -> None:
        """
        Fix outlet segment in river network shapefile for meshflow compatibility.

        meshflow requires the outlet segment to have DSLINKNO = outlet_value (default 0).
        TauDEM-generated networks have DSLINKNO pointing to non-existent downstream segments.
        This method detects the outlet (where DSLINKNO is not a valid LINKNO) and fixes it.

        Args:
            shp_path: Path to the river network shapefile
            outlet_value: Value to set for outlet segment's DSLINKNO (default 0)
        """
        if not shp_path:
            return
        path = Path(shp_path)
        if not path.exists():
            return

        try:
            import geopandas as gpd
            gdf = gpd.read_file(path)

            # Check if required columns exist
            if 'LINKNO' not in gdf.columns or 'DSLINKNO' not in gdf.columns:
                self.logger.warning(f"Shapefile {path.name} missing LINKNO or DSLINKNO columns")
                return

            # Find valid LINKNOs
            valid_linknos = set(gdf['LINKNO'].values)

            # Find outlets: segments where DSLINKNO is not a valid LINKNO and not already outlet_value
            outlet_mask = ~gdf['DSLINKNO'].isin(valid_linknos) & (gdf['DSLINKNO'] != outlet_value)

            if outlet_mask.any():
                num_outlets = outlet_mask.sum()
                original_values = gdf.loc[outlet_mask, 'DSLINKNO'].tolist()
                gdf.loc[outlet_mask, 'DSLINKNO'] = outlet_value
                self.logger.info(
                    f"Fixed {num_outlets} outlet segment(s) in {path.name}: "
                    f"DSLINKNO {original_values} -> {outlet_value}"
                )

                # Save updated shapefile
                temp_path = path.with_suffix('.tmp.shp')
                gdf.to_file(temp_path)
                # Move all shapefile components
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    src = temp_path.with_suffix(ext)
                    dst = path.with_suffix(ext)
                    if src.exists():
                        shutil.move(src, dst)
            else:
                # Check if outlet already has correct value
                already_correct = (gdf['DSLINKNO'] == outlet_value).any()
                if already_correct:
                    self.logger.debug(f"Outlet segment in {path.name} already has DSLINKNO={outlet_value}")
                else:
                    self.logger.warning(f"Could not identify outlet segment in {path.name}")

        except Exception as e:
            self.logger.warning(f"Failed to fix outlet segment in {path}: {e}")

    def _sanitize_landcover_stats(self, csv_path: str) -> str:
        """
        Sanitize landcover stats CSV for meshflow compatibility.
        Converts IGBP_* columns (pixel counts) to frac_* columns (fractions).

        meshflow expects columns like 'frac_1', 'frac_7', etc. with values 0-1.
        Our input has 'IGBP_1', 'IGBP_7', etc. with raw pixel counts.

        Args:
            csv_path: Path to the landcover stats CSV

        Returns:
            Path to the sanitized CSV file
        """
        if not csv_path:
            return csv_path

        path = Path(csv_path)
        if not path.exists():
            return csv_path

        try:
            import pandas as pd
            df = pd.read_csv(path)

            # Remove Unnamed: 0 or similar index columns
            cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            # Check for duplicate rows (which cause MultiIndex issues in xarray)
            initial_rows = len(df)
            df = df.drop_duplicates()
            if len(df) < initial_rows:
                self.logger.info(f"Sanitizing landcover stats {path.name}: removed {initial_rows - len(df)} duplicate rows")

            # Find IGBP_* columns (pixel counts)
            igbp_cols = [col for col in df.columns if col.startswith('IGBP_')]

            if igbp_cols:
                self.logger.info(f"Converting {len(igbp_cols)} IGBP columns to fractional landcover")

                # Calculate total count per row
                count_data = df[igbp_cols].fillna(0)
                row_totals = count_data.sum(axis=1)

                # Convert counts to fractions and rename to frac_*
                for col in igbp_cols:
                    class_num = col.replace('IGBP_', '')
                    frac_col = f'frac_{class_num}'
                    # Avoid division by zero
                    df[frac_col] = count_data[col] / row_totals.replace(0, 1)
                    # Drop the original IGBP column
                    df = df.drop(columns=[col])

                self.logger.info(f"Created fractional landcover columns: {[c for c in df.columns if c.startswith('frac_')]}")

            # Save to a temp file in forcing directory to avoid modifying source
            temp_path = self.forcing_dir / f"temp_{path.name}"
            df.to_csv(temp_path, index=False)
            return str(temp_path)

        except Exception as e:
            self.logger.warning(f"Failed to sanitize landcover stats {path}: {e}")
            return csv_path
