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
        self.prepare_forcing_data(self._meshflow_config)

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
        
        # Get simulation times
        time_window = self.get_simulation_time_window()
        if time_window:
            start_time, end_time = time_window
        else:
            # Fallback defaults
            import pandas as pd
            start_time = pd.Timestamp("2004-01-01 01:00")
            end_time = pd.Timestamp("2004-01-05 23:00")

        # Basic MESH_input_run_options.ini content
        # Updated for MESH 1.4 compatibility based on source code analysis
        # Using SHDFILEFLAG nc to trigger reading MESH_drainage_database.nc
        # Using RUNMODE runclass instead of RUNCLASS
        content = f"""MESH input run options file                             # comment line 1                                | * 
##### Control Flags #####                               # comment line 2                                | * 
----#                                                   # comment line 3                                | * 
   13                                                   # Number of control flags                       | I5
SHDFILEFLAG         nc                                  # Drainage database format (nc = NetCDF)
BASINFORCINGFLAG    nc                                  # Forcing file format (nc = NetCDF in 1.4)
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
            self.logger.info(f"Created {run_options_path}")
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

        default_forcing_vars = {
            "airpres": "air_pressure",
            "spechum": "specific_humidity",
            "airtemp": "air_temperature",
            "windspd": "wind_speed",
            "pptrate": "precipitation",
            "SWRadAtm": "shortwave_radiation",
            "LWRadAtm": "longwave_radiation",
        }

        default_forcing_units = {
            "airpres": 'pascal',
            "spechum": 'kg/kg',
            "airtemp": 'kelvin',
            "windspd": 'm/s',
            "pptrate": 'mm/s',
            "SWRadAtm": 'W/m^2',
            "LWRadAtm": 'W/m^2',
        }

        default_forcing_to_units = {
            "airpres": 'pascal',
            "spechum": 'kg/kg',
            "airtemp": 'kelvin',
            "windspd": 'm/s',
            "pptrate": 'mm/s',
            "SWRadAtm": 'W/m^2',
            "LWRadAtm": 'W/m^2',
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
        }

        default_ddb_units = {
            'ChnlSlope': 'm/m',
            'ChnlLength': 'm',
            'Rank': 'dimensionless',
            'Next': 'dimensionless',
            'GRU': 'dimensionless',
            'GridArea': 'm^2',
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
            'hru_dim': _get_config_value('MESH_HRU_DIM', 'N'),
            'outlet_value': _get_config_value('MESH_OUTLET_VALUE', 0),
        }
        return config

    def prepare_forcing_data(self, config):
        """Prepare forcing data using meshflow."""
        if not MESHFLOW_AVAILABLE:
            self.logger.warning("meshflow not available - skipping MESH preprocessing")
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

    def _sanitize_landcover_stats(self, csv_path: str) -> str:
        """
        Sanitize landcover stats CSV for meshflow compatibility.
        Removes IGBP_ prefix from column names and duplicate rows.

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

            # Check if columns have IGBP_ prefix
            has_prefix = any(col.startswith('IGBP_') for col in df.columns)

            # Check for duplicate rows (which cause MultiIndex issues in xarray)
            initial_rows = len(df)
            df = df.drop_duplicates()
            has_duplicates = len(df) < initial_rows

            if has_prefix or cols_to_drop or has_duplicates:
                if has_duplicates:
                    self.logger.info(f"Sanitizing landcover stats {path.name}: removed {initial_rows - len(df)} duplicate rows")
                if has_prefix or cols_to_drop:
                    self.logger.info(f"Sanitizing landcover stats {path.name}: removing IGBP_ prefix and/or index")

                def rename_col(col):
                    if col.startswith('IGBP_'):
                        return col.replace('IGBP_', '')
                    return col

                df = df.rename(columns=rename_col)

                # Save to a temp file in forcing directory to avoid modifying source
                temp_path = self.forcing_dir / f"temp_{path.name}"
                df.to_csv(temp_path, index=False)
                return str(temp_path)

            return csv_path

        except Exception as e:
            self.logger.warning(f"Failed to sanitize landcover stats {path}: {e}")
            return csv_path
