"""
MESH model preprocessor.

Handles data preparation using meshflow library for MESH model setup.
Uses meshflow exclusively for all preprocessing - both lumped and distributed modes.
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

from ..base import BaseModelPreProcessor

from ..registry import ModelRegistry

from .preprocessing import (
    MESHConfigDefaults,
    MESHConfigGenerator,
    MESHDataPreprocessor,
    MESHDrainageDatabase,
    MESHFlowManager,
    MESHForcingProcessor,
    MESHParameterFixer,
    is_elevation_band_mode,
)


@ModelRegistry.register_preprocessor('MESH')
class MESHPreProcessor(BaseModelPreProcessor):  # type: ignore[misc]
    """
    Preprocessor for the MESH model.

    Handles data preparation using meshflow library for MESH model setup.
    All preprocessing is done through meshflow for both lumped and distributed modes.

    Delegates specialized operations to:
    - MESHFlowManager: Meshflow execution with fallback strategies
    - MESHDrainageDatabase: DDB topology fixes and completeness
    - MESHParameterFixer: Parameter file fixes for stability
    - MESHConfigGenerator: INI file generation
    - MESHForcingProcessor: Forcing file processing
    - MESHDataPreprocessor: Shapefile and data preparation
    """

    MODEL_NAME = "MESH"

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the MESH preprocessor.

        Sets up paths for catchment and river network shapefiles, and prepares
        lazy-initialized components for meshflow-based preprocessing.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                MESH model settings, paths, and domain parameters.
            logger: Logger instance for status messages and debugging.
        """
        super().__init__(config, logger)

        # MESH-specific catchment path
        self.catchment_path = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')

        if self.config:
            self.catchment_name = self.config.paths.river_basins_name
            if self.catchment_name == 'default':
                self.catchment_name = f"{self.domain_name}_riverBasins_{self.config.domain.definition_method}.shp"
        else:
            self.catchment_name = self._get_config_value(lambda: self.config.paths.river_basins_name)
            if self.catchment_name == 'default':
                self.catchment_name = f"{self.domain_name}_riverBasins_{self.domain_definition_method}.shp"

        self.rivers_path = self.get_river_network_path().parent
        self.rivers_name = self.get_river_network_path().name

        # Lazy-initialized components
        self._meshflow_config = None
        self._meshflow_manager = None
        self._drainage_database = None
        self._parameter_fixer = None
        self._config_generator = None
        self._forcing_processor = None
        self._data_preprocessor = None

    # Lazy initialization properties for MESH preprocessing components
    @property
    def meshflow_manager(self) -> MESHFlowManager:
        """
        Meshflow execution manager with fallback strategies.

        Returns:
            MESHFlowManager: Configured manager for meshflow library interaction.
        """
        if self._meshflow_manager is None:
            self._meshflow_manager = MESHFlowManager(
                self.forcing_dir,
                self._meshflow_config,
                self.logger
            )
        assert self._meshflow_manager is not None
        return self._meshflow_manager

    @property
    def drainage_database(self) -> MESHDrainageDatabase:
        """
        Drainage database manager for topology fixes and completeness.

        Returns:
            MESHDrainageDatabase: Manager for DDB NetCDF file operations.
        """
        if self._drainage_database is None:
            self._drainage_database = MESHDrainageDatabase(
                self.forcing_dir,
                self.rivers_path,
                self.rivers_name,
                self.catchment_path,
                self.catchment_name,
                self.config_dict,
                self.logger
            )
        assert self._drainage_database is not None
        return self._drainage_database

    @property
    def parameter_fixer(self) -> MESHParameterFixer:
        """
        Parameter file fixer for MESH stability and compatibility.

        Returns:
            MESHParameterFixer: Manager for INI parameter file corrections.
        """
        if self._parameter_fixer is None:
            self._parameter_fixer = MESHParameterFixer(
                self.forcing_dir,
                self.setup_dir,
                self.config_dict,
                self.logger,
                self.get_simulation_time_window
            )
        assert self._parameter_fixer is not None
        return self._parameter_fixer

    @property
    def config_generator(self) -> MESHConfigGenerator:
        """
        Configuration file generator for MESH INI files.

        Returns:
            MESHConfigGenerator: Generator for run options, CLASS/hydrology parameters.
        """
        if self._config_generator is None:
            self._config_generator = MESHConfigGenerator(
                self.forcing_dir,
                self.project_dir,
                self.config_dict,
                self.logger,
                self.get_simulation_time_window
            )
        assert self._config_generator is not None
        return self._config_generator

    @property
    def forcing_processor(self) -> MESHForcingProcessor:
        """
        Forcing file processor for MESH-compatible format.

        Returns:
            MESHForcingProcessor: Processor for forcing data conversion and alignment.
        """
        if self._forcing_processor is None:
            self._forcing_processor = MESHForcingProcessor(
                self.forcing_dir,
                self._meshflow_config,
                self.logger
            )
        assert self._forcing_processor is not None
        return self._forcing_processor

    @property
    def data_preprocessor(self) -> MESHDataPreprocessor:
        """
        Data preprocessor for shapefile and landcover preparation.

        Returns:
            MESHDataPreprocessor: Preprocessor for spatial data sanitization.
        """
        if self._data_preprocessor is None:
            self._data_preprocessor = MESHDataPreprocessor(
                self.forcing_dir,
                self.setup_dir,
                self.config_dict,
                self.logger
            )
        assert self._data_preprocessor is not None
        return self._data_preprocessor

    def _get_spatial_mode(self) -> str:
        """Determine MESH spatial mode from configuration."""
        spatial_mode = self._get_config_value(lambda: self.config.model.mesh.spatial_mode, default='auto')

        if spatial_mode != 'auto':
            return spatial_mode

        domain_method = self.domain_definition_method

        if domain_method in ['point', 'lumped']:
            return 'lumped'
        elif domain_method in ['delineate', 'semidistributed', 'semi_distributed', 'distributed']:
            # Note: 'delineate' is auto-mapped to 'semidistributed' for backward compatibility
            return 'distributed'

        return 'lumped'

    def run_preprocessing(self):
        """Run the complete MESH preprocessing workflow."""
        self.logger.info("Starting MESH preprocessing")
        return self.run_preprocessing_template()

    def _pre_setup(self) -> None:
        """MESH-specific pre-setup: create meshflow config (template hook)."""
        self._meshflow_config = self._create_meshflow_config()

    def _prepare_forcing(self) -> None:
        """MESH-specific forcing data preparation using meshflow (template hook)."""
        spatial_mode = self._get_spatial_mode()
        self.logger.info(f"MESH spatial mode: {spatial_mode}")

        self._run_meshflow()

    def _create_model_configs(self) -> None:
        """Create MESH-specific configuration files (template hook).

        Meshflow generates all required config files. This method only:
        1. Creates streamflow input (observation-dependent)
        2. Copies settings files
        3. Applies fixes for MESH compatibility
        """
        self.logger.info("Finalizing MESH configuration files")

        # Streamflow input requires observation data - always create
        self.config_generator.create_streamflow_input()

        # Copy settings files
        self.data_preprocessor.copy_settings_to_forcing()

        # Apply fixes for MESH compatibility
        self.parameter_fixer.fix_run_options_var_names()
        self.parameter_fixer.fix_run_options_snow_params()
        self.parameter_fixer.fix_run_options_output_dirs()

        # For elevation band discretization, GRU handling was done in _postprocess_meshflow_output
        # Skip fix_gru_count_mismatch to avoid overriding elevation band setup
        if not is_elevation_band_mode(self.config_dict):
            self.parameter_fixer.fix_gru_count_mismatch()

        self.parameter_fixer.fix_hydrology_wf_r2()
        self.parameter_fixer.fix_missing_hydrology_params()
        self.parameter_fixer.fix_class_initial_conditions()
        self.parameter_fixer.fix_class_vegetation_parameters()
        self.parameter_fixer.fix_reservoir_file()
        self.parameter_fixer.configure_lumped_outputs()
        self.parameter_fixer.create_safe_forcing()

    def _create_meshflow_config(self) -> Dict[str, Any]:
        """
        Create configuration dictionary for meshflow library.

        Builds a comprehensive configuration including paths to shapefiles,
        forcing files, landcover data, variable mappings, and MESH settings.
        Handles spinup period calculation and GRU class detection.

        Returns:

            Dictionary containing all meshflow configuration parameters.
        """

        def _get_mesh_config_value(key: str, default_value):
            value = self._get_config_value(lambda: None, dict_key=key)
            if value is None or value == 'default':
                return default_value
            return value

        # Build forcing files path
        forcing_files_path = Path(
            _get_mesh_config_value(
                'MESH_FORCING_PATH',
                self.project_forcing_dir / 'basin_averaged_data',
            )
        )
        forcing_files_glob = str(forcing_files_path / '*.nc')

        # Landcover stats file
        landcover_path = self._get_landcover_path(_get_mesh_config_value)

        # Detect GRU classes
        detected_gru_classes = self.data_preprocessor.detect_gru_classes(Path(landcover_path))
        self.logger.info(f"Detected GRU classes in landcover: {detected_gru_classes}")

        # Get simulation dates with spinup
        time_window = self.get_simulation_time_window()
        spinup_days = int(_get_mesh_config_value('MESH_SPINUP_DAYS', 730))

        if time_window:
            analysis_start, end_date = time_window
            sim_start = analysis_start - timedelta(days=spinup_days)
            forcing_start_date = sim_start.strftime('%Y-%m-%d %H:%M:%S')
            sim_start_date = sim_start.strftime('%Y-%m-%d %H:%M:%S')
            sim_end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
            self.logger.info(
                f"MESH simulation: {sim_start_date} to {sim_end_date} "
                f"(spinup: {spinup_days} days before {analysis_start.strftime('%Y-%m-%d')})"
            )
        else:
            forcing_start_date = '2001-01-01 00:00:00'
            sim_start_date = '2001-01-01 00:00:00'
            sim_end_date = '2010-12-31 23:00:00'

        # Get GRU mapping
        gru_mapping = MESHConfigDefaults.get_gru_mapping_for_classes(detected_gru_classes)

        # Filter landcover_classes to only include detected classes
        # This prevents meshflow NGRU dimension mismatch errors
        all_landcover_classes = _get_mesh_config_value('MESH_LANDCOVER_CLASSES', MESHConfigDefaults.LANDCOVER_CLASSES)
        if detected_gru_classes:
            landcover_classes = {k: v for k, v in all_landcover_classes.items() if k in detected_gru_classes}
            self.logger.info(f"Filtered landcover_classes to detected classes: {list(landcover_classes.keys())}")
        else:
            landcover_classes = all_landcover_classes

        # Build settings
        default_settings = MESHConfigDefaults.get_default_settings(
            forcing_start_date, sim_start_date, sim_end_date, gru_mapping
        )
        default_settings['class_params']['grus'] = _get_mesh_config_value('MESH_GRU_MAPPING', gru_mapping)

        config = {
            'riv': str(self.rivers_path / self.rivers_name),
            'cat': str(self.catchment_path / self.catchment_name),
            'landcover': str(landcover_path),
            'forcing_files': forcing_files_glob,
            'forcing_vars': _get_mesh_config_value('MESH_FORCING_VARS', MESHConfigDefaults.FORCING_VARS),
            'forcing_units': _get_mesh_config_value('MESH_FORCING_UNITS', MESHConfigDefaults.FORCING_UNITS),
            'forcing_to_units': _get_mesh_config_value('MESH_FORCING_TO_UNITS', MESHConfigDefaults.FORCING_TO_UNITS),
            'main_id': _get_mesh_config_value('MESH_MAIN_ID', 'GRU_ID'),
            'ds_main_id': _get_mesh_config_value('MESH_DS_MAIN_ID', 'DSLINKNO'),
            'landcover_classes': landcover_classes,
            'ddb_vars': _get_mesh_config_value('MESH_DDB_VARS', MESHConfigDefaults.DDB_VARS),
            'ddb_units': _get_mesh_config_value('MESH_DDB_UNITS', MESHConfigDefaults.DDB_UNITS),
            'ddb_to_units': _get_mesh_config_value('MESH_DDB_TO_UNITS', MESHConfigDefaults.DDB_UNITS),
            'ddb_min_values': _get_mesh_config_value('MESH_DDB_MIN_VALUES', MESHConfigDefaults.DDB_MIN_VALUES),
            'gru_dim': _get_mesh_config_value('MESH_GRU_DIM', 'NGRU'),
            'hru_dim': _get_mesh_config_value('MESH_HRU_DIM', 'subbasin'),
            'outlet_value': _get_mesh_config_value('MESH_OUTLET_VALUE', -9999),
            'settings': _get_mesh_config_value('MESH_SETTINGS', default_settings),
        }
        return config

    def _get_landcover_path(self, _get_config_value) -> str:
        """
        Get landcover statistics file path, creating minimal file if needed.

        Searches for landcover CSV files in configured locations. If no file
        is found, creates a minimal landcover file with a default grassland class.

        Args:
            _get_config_value: Helper function to retrieve config values with defaults.

        Returns:
            Path to the landcover statistics CSV file.
        """
        landcover_stats_path = _get_config_value('MESH_LANDCOVER_STATS_PATH', None)

        if landcover_stats_path:
            path = Path(landcover_stats_path)
            if path.exists():
                return str(path)

        landcover_file = _get_config_value(
            'MESH_LANDCOVER_STATS_FILE',
            'modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv',
        )
        landcover_dir = Path(
            _get_config_value(
                'MESH_LANDCOVER_STATS_DIR',
                self.project_attributes_dir / 'gistool-outputs',
            )
        )
        landcover_path = landcover_dir / landcover_file

        # If the default file doesn't exist, look for any landcover CSV in the directory
        if not landcover_path.exists() and landcover_dir.exists():
            landcover_csvs = list(landcover_dir.glob('*landcover*.csv'))
            if landcover_csvs:
                self.logger.info(f"Using alternative landcover file: {landcover_csvs[0].name}")
                return str(landcover_csvs[0])

        # If still not found, create a minimal landcover CSV file
        if not landcover_path.exists():
            self.logger.warning("Landcover file not found. Creating minimal landcover file.")
            landcover_dir.mkdir(parents=True, exist_ok=True)
            self._create_minimal_landcover_csv(landcover_path)

        return str(landcover_path)

    def _create_minimal_landcover_csv(self, csv_path: Path) -> None:
        """
        Create a minimal landcover CSV file for domains without landcover data.

        Creates a simple CSV with a single dominant landcover class (IGBP_10 = Grassland)
        which is commonly used as a fallback in hydrological modeling.
        """
        import csv

        # Create minimal landcover data with a single grassland class (IGBP_10)
        # This is a reasonable default for catchments without detailed landcover data
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header: ID column, single IGBP class, count column
            writer.writerow(['GRU_ID', 'IGBP_10', 'count'])
            # Data: single row with ID 1, 100% grassland coverage
            writer.writerow(['1', '100000', '100000'])

        self.logger.info(f"Created minimal landcover file at {csv_path}")

    def _run_meshflow(self) -> None:
        """
        Run meshflow to generate MESH input files.

        Executes the complete meshflow workflow including:
        1. Copying and sanitizing shapefiles
        2. Ensuring required ID columns exist
        3. Fixing missing columns and outlet segments
        4. Preparing landcover statistics
        5. Running meshflow with forcing and post-processing callbacks

        Raises:
            ModelExecutionError: If meshflow library is not available.
        """
        if not MESHFlowManager.is_available():
            from symfluence.core.exceptions import ModelExecutionError
            from .preprocessing.meshflow_manager import _meshflow_import_error
            detail = f" ({_meshflow_import_error})" if _meshflow_import_error else ""
            raise ModelExecutionError(
                f"meshflow is not available{detail}. Install with:\n"
                "  pip install git+https://github.com/kasra-keshavarz/hydrant.git\n"
                "  pip install git+https://github.com/CH-Earth/meshflow.git@main\n"
                "Note: hydrant must be installed BEFORE meshflow to avoid pulling "
                "the wrong 'hydrant' package from PyPI."
            )

        assert self._meshflow_config is not None, "meshflow config must be set before running"

        # Prepare shapefiles
        riv_copy = self.forcing_dir / f"temp_{Path(self._meshflow_config['riv']).name}"
        cat_copy = self.forcing_dir / f"temp_{Path(self._meshflow_config['cat']).name}"

        self.data_preprocessor.copy_shapefile(self._meshflow_config['riv'], riv_copy)
        self.data_preprocessor.copy_shapefile(self._meshflow_config['cat'], cat_copy)

        self.data_preprocessor.sanitize_shapefile(str(riv_copy))
        self.data_preprocessor.sanitize_shapefile(str(cat_copy))

        # Fix network topology fields to ensure scalar values (required by meshflow)
        self.data_preprocessor.fix_network_topology_fields(str(riv_copy))
        self.data_preprocessor.fix_network_topology_fields(str(cat_copy))

        # Ensure GRU_ID and hru_dim exist for meshflow joining and indexing
        main_id = self._meshflow_config.get('main_id', 'GRU_ID')
        hru_dim = self._meshflow_config.get('hru_dim', 'subbasin')

        self.data_preprocessor.ensure_gru_id(str(cat_copy))
        self.data_preprocessor.ensure_hru_id(str(cat_copy), hru_dim, main_id)
        self.data_preprocessor.ensure_hru_id(str(riv_copy), hru_dim, main_id)

        # Fix missing columns required by meshflow (e.g. strmOrder)
        ddb_vars = self._meshflow_config.get('ddb_vars', {})
        self.data_preprocessor.fix_missing_columns(str(riv_copy), ddb_vars)

        outlet_value = self._meshflow_config.get('outlet_value', -9999)
        self.data_preprocessor.fix_outlet_segment(str(riv_copy), outlet_value=outlet_value)

        # Prepare landcover
        landcover_path = self._meshflow_config.get('landcover', '')
        if landcover_path and Path(landcover_path).exists():
            # Pass catchment path to expand landcover to match all catchment GRU_IDs
            sanitized = self.data_preprocessor.sanitize_landcover_stats(
                landcover_path, catchment_path=str(cat_copy)
            )
            self.data_preprocessor.convert_landcover_to_maf(sanitized)
            self._meshflow_config['landcover'] = sanitized

        # Update config with copies
        self._meshflow_config['riv'] = str(riv_copy)
        self._meshflow_config['cat'] = str(cat_copy)

        # Run meshflow
        self.meshflow_manager.run(
            prepare_forcing_callback=self.forcing_processor.prepare_forcing_direct,
            postprocess_callback=self._postprocess_meshflow_output
        )

    def _postprocess_meshflow_output(self) -> None:
        """Post-process meshflow output for MESH compatibility."""
        self.forcing_processor.postprocess_meshflow_output()
        self.drainage_database.fix_drainage_topology()
        self.drainage_database.ensure_completeness()

        # Check if using elevation band discretization
        if is_elevation_band_mode(self.config_dict):
            self._apply_elevation_band_grus()
        else:
            # Fix GRU count mismatch for landcover-based GRUs
            self.parameter_fixer.fix_gru_count_mismatch()

        self.drainage_database.reorder_by_rank_and_normalize()
        # Do NOT call fix_gru_count_mismatch again - it would double-trim
        self.parameter_fixer.fix_run_options_output_dirs()
        self.parameter_fixer.fix_hydrology_wf_r2()
        self.parameter_fixer.fix_missing_hydrology_params()
        self.parameter_fixer.fix_class_initial_conditions()
        self.parameter_fixer.fix_class_vegetation_parameters()
        # Fix reservoir file after drainage database is finalized
        self.parameter_fixer.fix_reservoir_file()
        # Configure output for lumped mode calibration
        self.parameter_fixer.configure_lumped_outputs()

    def _apply_elevation_band_grus(self) -> None:
        """Apply elevation band discretization as GRUs for MESH.

        This method:
        1. Finds the elevation band HRU shapefile
        2. Converts the DDB to use elevation bands as GRUs (multi-subbasin if lapsing)
        3. Creates CLASS parameter blocks for each elevation band
        4. Applies temperature/pressure lapsing to forcing (if enabled)
        """
        self.logger.info("Applying elevation band GRU discretization")

        # Determine lapse rate settings
        apply_lapse = self._get_config_value(lambda: self.config.forcing.apply_lapse_rate, default=True)
        if isinstance(apply_lapse, str):
            apply_lapse = apply_lapse.lower() in ('true', '1', 'yes')
        lapse_rate = float(self._get_config_value(lambda: self.config.forcing.lapse_rate, default=0.0065))

        # Find elevation band HRU shapefile
        experiment_id = self.experiment_id
        domain_method = self.domain_definition_method

        # Look for elevation band shapefile
        hru_shp_candidates = [
            self.project_dir / 'shapefiles' / 'catchment' / domain_method / experiment_id / f'{self.domain_name}_HRUs_elevation.shp',
            self.project_dir / 'shapefiles' / 'catchment' / domain_method / experiment_id / f'{self.domain_name}_HRUs_elevation_aspect.shp',
            self.project_dir / 'shapefiles' / 'catchment' / f'{self.domain_name}_HRUs_elevation.shp',
        ]

        hru_shapefile = None
        for candidate in hru_shp_candidates:
            if candidate.exists():
                hru_shapefile = candidate
                break

        if hru_shapefile is None:
            self.logger.warning(
                f"Could not find elevation band HRU shapefile. Tried: {hru_shp_candidates}"
            )
            # Fall back to landcover-based GRUs
            self.parameter_fixer.fix_gru_count_mismatch()
            return

        self.logger.info(f"Using elevation band shapefile: {hru_shapefile}")

        if apply_lapse:
            # Multi-subbasin approach: each elevation band gets its own subbasin
            # with elevation-lapsed forcing
            self.logger.info(
                f"Using multi-subbasin elevation bands with lapse rate={lapse_rate} K/m"
            )
            n_bands, elevation_info = self.drainage_database.convert_to_multi_subbasin_elevation_bands(
                hru_shapefile
            )
        else:
            # Legacy: single subbasin, all bands share forcing
            self.logger.info("Using single-subbasin elevation bands (no lapsing)")
            n_bands, elevation_info = self.drainage_database.convert_to_elevation_band_grus(
                hru_shapefile
            )

        if n_bands == 0:
            self.logger.warning("Failed to convert to elevation band GRUs, falling back to landcover GRUs")
            self.parameter_fixer.fix_gru_count_mismatch()
            return

        # Create CLASS parameter blocks for elevation bands
        success = self.parameter_fixer.create_elevation_band_class_blocks(elevation_info)

        if not success:
            self.logger.warning("Failed to create elevation band CLASS blocks")
            return

        # Apply forcing lapsing (only for multi-subbasin approach)
        if apply_lapse:
            self.forcing_processor.apply_elevation_lapsing(elevation_info, lapse_rate)

        self.logger.info(
            f"Successfully configured {n_bands} elevation band GRUs for MESH"
            + (f" with temperature lapsing (rate={lapse_rate} K/m)" if apply_lapse else "")
        )
