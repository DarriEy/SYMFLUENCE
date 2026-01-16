"""
Configuration transformation utilities for SYMFLUENCE.

This module handles conversion between flat and hierarchical configuration formats:
- Flat format: Uppercase keys like {'DOMAIN_NAME': 'test', 'FORCING_DATASET': 'ERA5'}
- Nested format: Hierarchical structure like {'domain': {'name': 'test'}, 'forcing': {'dataset': 'ERA5'}}

Key functions:
- transform_flat_to_nested(): Convert flat dict to nested structure for Pydantic models
- flatten_nested_config(): Convert SymfluenceConfig instance back to flat dict for backward compatibility
"""

from typing import Dict, Any, Tuple, TYPE_CHECKING, Optional
from pathlib import Path
import threading
import logging

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig

logger = logging.getLogger(__name__)

# Global cache for auto-generated mapping (thread-safe)
_AUTO_GENERATED_MAP: Optional[Dict[str, Tuple[str, ...]]] = None
_GENERATION_LOCK = threading.Lock()


def get_flat_to_nested_map() -> Dict[str, Tuple[str, ...]]:
    """
    Get flat-to-nested mapping via lazy auto-generation.

    Thread-safe with caching for performance.
    First call generates mapping, subsequent calls return cached version.

    Returns:
        Dictionary mapping flat keys to nested paths
    """
    global _AUTO_GENERATED_MAP

    # Fast path: return cached mapping
    if _AUTO_GENERATED_MAP is not None:
        return _AUTO_GENERATED_MAP

    # Slow path: generate mapping (thread-safe)
    with _GENERATION_LOCK:
        # Double-check after acquiring lock
        if _AUTO_GENERATED_MAP is not None:
            return _AUTO_GENERATED_MAP

        try:
            from symfluence.core.config.introspection import generate_flat_to_nested_map
            from symfluence.core.config.models import SymfluenceConfig

            _AUTO_GENERATED_MAP = generate_flat_to_nested_map(
                SymfluenceConfig,
                include_model_overrides=True
            )

            logger.info(f"Auto-generated {len(_AUTO_GENERATED_MAP)} configuration mappings")

        except Exception as e:
            logger.error(f"Auto-generation failed: {e}, falling back to manual mapping")
            # Fallback to manual mapping (Phase 1 only)
            _AUTO_GENERATED_MAP = FLAT_TO_NESTED_MAP.copy()

    return _AUTO_GENERATED_MAP


# ========================================
# FLAT-TO-NESTED MAPPING
# ========================================

# Comprehensive mapping from flat uppercase keys to nested paths
# Format: 'FLAT_KEY': ('section', 'subsection', 'field') or ('section', 'field')
#
# NOTE: This mapping can be auto-generated from Pydantic model aliases using:
#   from symfluence.core.config.introspection import generate_flat_to_nested_map
#   from symfluence.core.config.models import SymfluenceConfig
#   auto_mapping = generate_flat_to_nested_map(SymfluenceConfig)
#
# The manual mapping is kept for backward compatibility and explicit control.
# Use validate_mapping_against_pydantic() to verify sync with Pydantic models.
FLAT_TO_NESTED_MAP: Dict[str, Tuple[str, ...]] = {
    # ========== SYSTEM CONFIGURATION ==========
    'SYMFLUENCE_DATA_DIR': ('system', 'data_dir'),
    'SYMFLUENCE_CODE_DIR': ('system', 'code_dir'),
    'MPI_PROCESSES': ('system', 'mpi_processes'),
    'DEBUG_MODE': ('system', 'debug_mode'),
    'LOG_LEVEL': ('system', 'log_level'),
    'LOG_TO_FILE': ('system', 'log_to_file'),
    'LOG_FORMAT': ('system', 'log_format'),
    'FORCE_RUN_ALL_STEPS': ('system', 'force_run_all_steps'),
    # Note: FORCE_DOWNLOAD mapped to ('data', 'force_download') in Data section
    'USE_LOCAL_SCRATCH': ('system', 'use_local_scratch'),
    'RANDOM_SEED': ('system', 'random_seed'),
    'STOP_ON_ERROR': ('system', 'stop_on_error'),

    # ========== DOMAIN CONFIGURATION ==========
    'DOMAIN_NAME': ('domain', 'name'),
    'EXPERIMENT_ID': ('domain', 'experiment_id'),
    'EXPERIMENT_TIME_START': ('domain', 'time_start'),
    'EXPERIMENT_TIME_END': ('domain', 'time_end'),
    'CALIBRATION_PERIOD': ('domain', 'calibration_period'),
    'CALIBRATION_START_DATE': ('domain', 'calibration_start_date'),
    'CALIBRATION_END_DATE': ('domain', 'calibration_end_date'),
    'EVALUATION_PERIOD': ('domain', 'evaluation_period'),
    'SPINUP_PERIOD': ('domain', 'spinup_period'),
    'DOMAIN_DEFINITION_METHOD': ('domain', 'definition_method'),
    'DOMAIN_DISCRETIZATION': ('domain', 'discretization'),
    'POUR_POINT_COORDS': ('domain', 'pour_point_coords'),
    'BOUNDING_BOX_COORDS': ('domain', 'bounding_box_coords'),
    'MIN_GRU_SIZE': ('domain', 'min_gru_size'),
    'MIN_HRU_SIZE': ('domain', 'min_hru_size'),
    'ELEVATION_BAND_SIZE': ('domain', 'elevation_band_size'),
    'RADIATION_CLASS_NUMBER': ('domain', 'radiation_class_number'),
    'ASPECT_CLASS_NUMBER': ('domain', 'aspect_class_number'),
    'ASPECT_PATH': ('domain', 'aspect_path'),
    'GRID_CELL_SIZE': ('domain', 'grid_cell_size'),
    'CLIP_GRID_TO_WATERSHED': ('domain', 'clip_grid_to_watershed'),
    'DATA_ACCESS': ('domain', 'data_access'),
    'DOWNLOAD_DEM': ('domain', 'download_dem'),
    'DOWNLOAD_SOIL': ('domain', 'download_soil'),
    'DOWNLOAD_LAND_COVER': ('domain', 'download_landcover'),
    'DEM_SOURCE': ('domain', 'dem_source'),
    'LAND_CLASS_SOURCE': ('domain', 'land_class_source'),
    'LAND_CLASS_NAME': ('domain', 'land_class_name'),

    # ========== DATA CONFIGURATION ==========
    'ADDITIONAL_OBSERVATIONS': ('data', 'additional_observations'),
    # Note: SUPPLEMENT_FORCING mapped to ('forcing', 'supplement') in Forcing section
    'FORCE_DOWNLOAD': ('data', 'force_download'),
    'STREAMFLOW_DATA_PROVIDER': ('data', 'streamflow_data_provider'),
    'USGS_SITE_CODE': ('data', 'usgs_site_code'),
    'DOWNLOAD_USGS_DATA': ('data', 'download_usgs_data'),
    # Note: DOWNLOAD_USGS_GW mapped to ('evaluation', 'usgs_gw', 'download') in Evaluation section
    # Note: DOWNLOAD_MODIS_SNOW mapped to ('evaluation', 'modis_snow', 'download') in Evaluation section
    # Note: DOWNLOAD_SNOTEL mapped to ('evaluation', 'snotel', 'download') in Evaluation section
    'DOWNLOAD_SMHI_DATA': ('data', 'download_smhi_data'),
    'DOWNLOAD_LAMAH_ICE_DATA': ('data', 'download_lamah_ice_data'),
    'DOWNLOAD_ISMN': ('data', 'download_ismn'),
    'DOWNLOAD_GLACIER_DATA': ('data', 'download_glacier_data'),
    'LAMAH_ICE_PATH': ('data', 'lamah_ice_path'),
    'STREAMFLOW_STATION_ID': ('data', 'streamflow_station_id'),
    'ELEV_CHUNK_SIZE': ('data', 'elev_chunk_size'),
    'ELEV_TILE_TARGET': ('data', 'elev_tile_target'),

    # Data > Geospatial > SoilGrids
    'SOILGRIDS_LAYER': ('data', 'geospatial', 'soilgrids', 'layer'),
    'SOILGRIDS_WCS_MAP': ('data', 'geospatial', 'soilgrids', 'wcs_map'),
    'SOILGRIDS_COVERAGE_ID': ('data', 'geospatial', 'soilgrids', 'coverage_id'),
    'SOILGRIDS_HS_CACHE_DIR': ('data', 'geospatial', 'soilgrids', 'hs_cache_dir'),
    'SOILGRIDS_HS_RESOURCE_ID': ('data', 'geospatial', 'soilgrids', 'hs_resource_id'),
    'SOILGRIDS_HS_API_URL': ('data', 'geospatial', 'soilgrids', 'hs_api_url'),

    # Data > Geospatial > MODIS Landcover
    'MODIS_LANDCOVER_YEARS': ('data', 'geospatial', 'modis_landcover', 'years'),
    'MODIS_LANDCOVER_START_YEAR': ('data', 'geospatial', 'modis_landcover', 'start_year'),
    'MODIS_LANDCOVER_END_YEAR': ('data', 'geospatial', 'modis_landcover', 'end_year'),
    'LANDCOVER_YEAR': ('data', 'geospatial', 'modis_landcover', 'year'),
    'MODIS_LANDCOVER_BASE_URL': ('data', 'geospatial', 'modis_landcover', 'base_url'),
    'MODIS_LANDCOVER_CACHE_DIR': ('data', 'geospatial', 'modis_landcover', 'cache_dir'),
    'LANDCOVER_LOCAL_FILE': ('data', 'geospatial', 'modis_landcover', 'local_file'),

    # Data > Geospatial > NLCD
    'NLCD_COVERAGE_ID': ('data', 'geospatial', 'nlcd', 'coverage_id'),

    # Data > Geospatial > NASADEM
    'NASADEM_LOCAL_DIR': ('data', 'geospatial', 'nasadem', 'local_dir'),

    # Domain > Delineation
    'ROUTING_DELINEATION': ('domain', 'delineation', 'routing'),
    'GEOFABRIC_TYPE': ('domain', 'delineation', 'geofabric_type'),
    'DELINEATION_METHOD': ('domain', 'delineation', 'method'),
    'CURVATURE_THRESHOLD': ('domain', 'delineation', 'curvature_threshold'),
    'MIN_SOURCE_THRESHOLD': ('domain', 'delineation', 'min_source_threshold'),
    'STREAM_THRESHOLD': ('domain', 'delineation', 'stream_threshold'),
    'SLOPE_AREA_THRESHOLD': ('domain', 'delineation', 'slope_area_threshold'),
    'SLOPE_AREA_EXPONENT': ('domain', 'delineation', 'slope_area_exponent'),
    'AREA_EXPONENT': ('domain', 'delineation', 'area_exponent'),
    'MULTI_SCALE_THRESHOLDS': ('domain', 'delineation', 'multi_scale_thresholds'),
    'USE_DROP_ANALYSIS': ('domain', 'delineation', 'use_drop_analysis'),
    'DROP_ANALYSIS_MIN_THRESHOLD': ('domain', 'delineation', 'drop_analysis_min_threshold'),
    'DROP_ANALYSIS_MAX_THRESHOLD': ('domain', 'delineation', 'drop_analysis_max_threshold'),
    'DROP_ANALYSIS_NUM_THRESHOLDS': ('domain', 'delineation', 'drop_analysis_num_thresholds'),
    'DROP_ANALYSIS_LOG_SPACING': ('domain', 'delineation', 'drop_analysis_log_spacing'),
    'LUMPED_WATERSHED_METHOD': ('domain', 'delineation', 'lumped_watershed_method'),
    'CLEANUP_INTERMEDIATE_FILES': ('domain', 'delineation', 'cleanup_intermediate_files'),
    'DELINEATE_COASTAL_WATERSHEDS': ('domain', 'delineation', 'delineate_coastal_watersheds'),
    'DELINEATE_BY_POURPOINT': ('domain', 'delineation', 'delineate_by_pourpoint'),
    'MOVE_OUTLETS_MAX_DISTANCE': ('domain', 'delineation', 'move_outlets_max_distance'),

    # ========== FORCING CONFIGURATION ==========
    'FORCING_DATASET': ('forcing', 'dataset'),
    'FORCING_TIME_STEP_SIZE': ('forcing', 'time_step_size'),
    'FORCING_VARIABLES': ('forcing', 'variables'),
    'FORCING_MEASUREMENT_HEIGHT': ('forcing', 'measurement_height'),
    'APPLY_LAPSE_RATE': ('forcing', 'apply_lapse_rate'),
    'LAPSE_RATE': ('forcing', 'lapse_rate'),
    'FORCING_SHAPE_LAT_NAME': ('forcing', 'shape_lat_name'),
    'FORCING_SHAPE_LON_NAME': ('forcing', 'shape_lon_name'),
    'PET_METHOD': ('forcing', 'pet_method'),
    'SUPPLEMENT_FORCING': ('forcing', 'supplement'),

    # Forcing > ERA5
    'ERA5_USE_CDS': ('forcing', 'era5_use_cds'),
    'ERA5_ZARR_PATH': ('forcing', 'era5', 'zarr_path'),
    'ERA5_TIME_STEP_HOURS': ('forcing', 'era5', 'time_step_hours'),
    'ERA5_VARS': ('forcing', 'era5', 'variables'),

    # Forcing > NEX
    'NEX_MODELS': ('forcing', 'nex', 'models'),
    'NEX_SCENARIOS': ('forcing', 'nex', 'scenarios'),
    'NEX_ENSEMBLES': ('forcing', 'nex', 'ensembles'),
    'NEX_VARIABLES': ('forcing', 'nex', 'variables'),

    # Forcing > EM-Earth
    'EM_EARTH_PRCP_DIR': ('forcing', 'em_earth', 'prcp_dir'),
    'EM_EARTH_TMEAN_DIR': ('forcing', 'em_earth', 'tmean_dir'),
    'EM_EARTH_MIN_BBOX_SIZE': ('forcing', 'em_earth', 'min_bbox_size'),
    'EM_EARTH_MAX_EXPANSION': ('forcing', 'em_earth', 'max_expansion'),
    'EM_PRCP': ('forcing', 'em_earth', 'prcp_var'),
    'EM_EARTH_DATA_TYPE': ('forcing', 'em_earth', 'data_type'),

    # ========== MODEL CONFIGURATION ==========
    'HYDROLOGICAL_MODEL': ('model', 'hydrological_model'),
    'ROUTING_MODEL': ('model', 'routing_model'),

    # Model > SUMMA
    'SUMMA_INSTALL_PATH': ('model', 'summa', 'install_path'),
    'SUMMA_EXE': ('model', 'summa', 'exe'),
    'SETTINGS_SUMMA_PATH': ('model', 'summa', 'settings_path'),
    'SETTINGS_SUMMA_FILEMANAGER': ('model', 'summa', 'filemanager'),
    'SETTINGS_SUMMA_FORCING_LIST': ('model', 'summa', 'forcing_list'),
    'SETTINGS_SUMMA_COLDSTATE': ('model', 'summa', 'coldstate'),
    'SETTINGS_SUMMA_TRIALPARAMS': ('model', 'summa', 'trialparams'),
    'SETTINGS_SUMMA_ATTRIBUTES': ('model', 'summa', 'attributes'),
    'SETTINGS_SUMMA_OUTPUT': ('model', 'summa', 'output'),
    'SETTINGS_SUMMA_BASIN_PARAMS_FILE': ('model', 'summa', 'basin_params_file'),
    'SETTINGS_SUMMA_LOCAL_PARAMS_FILE': ('model', 'summa', 'local_params_file'),
    'SETTINGS_SUMMA_CONNECT_HRUS': ('model', 'summa', 'connect_hrus'),
    'SETTINGS_SUMMA_TRIALPARAM_N': ('model', 'summa', 'trialparam_n'),
    'SETTINGS_SUMMA_TRIALPARAM_1': ('model', 'summa', 'trialparam_1'),
    'SETTINGS_SUMMA_USE_PARALLEL_SUMMA': ('model', 'summa', 'use_parallel'),
    'SETTINGS_SUMMA_CPUS_PER_TASK': ('model', 'summa', 'cpus_per_task'),
    'SETTINGS_SUMMA_TIME_LIMIT': ('model', 'summa', 'time_limit'),
    'SETTINGS_SUMMA_MEM': ('model', 'summa', 'mem'),
    'SETTINGS_SUMMA_GRU_COUNT': ('model', 'summa', 'gru_count'),
    'SETTINGS_SUMMA_GRU_PER_JOB': ('model', 'summa', 'gru_per_job'),
    'SETTINGS_SUMMA_PARALLEL_PATH': ('model', 'summa', 'parallel_path'),
    'SETTINGS_SUMMA_PARALLEL_EXE': ('model', 'summa', 'parallel_exe'),
    'EXPERIMENT_OUTPUT_SUMMA': ('model', 'summa', 'experiment_output'),
    'EXPERIMENT_LOG_SUMMA': ('model', 'summa', 'experiment_log'),
    'PARAMS_TO_CALIBRATE': ('model', 'summa', 'params_to_calibrate'),
    'BASIN_PARAMS_TO_CALIBRATE': ('model', 'summa', 'basin_params_to_calibrate'),
    'SUMMA_DECISION_OPTIONS': ('model', 'summa', 'decision_options'),
    'CALIBRATE_DEPTH': ('model', 'summa', 'calibrate_depth'),
    'DEPTH_TOTAL_MULT_BOUNDS': ('model', 'summa', 'depth_total_mult_bounds'),
    'DEPTH_SHAPE_FACTOR_BOUNDS': ('model', 'summa', 'depth_shape_factor_bounds'),
    'SETTINGS_SUMMA_GLACIER_MODE': ('model', 'summa', 'glacier_mode'),
    'SETTINGS_SUMMA_GLACIER_ATTRIBUTES': ('model', 'summa', 'glacier_attributes'),
    'SETTINGS_SUMMA_GLACIER_COLDSTATE': ('model', 'summa', 'glacier_coldstate'),
    'SUMMA_TIMEOUT': ('model', 'summa', 'timeout'),

    # Model > FUSE
    'FUSE_INSTALL_PATH': ('model', 'fuse', 'install_path'),
    'FUSE_EXE': ('model', 'fuse', 'exe'),
    'FUSE_ROUTING_INTEGRATION': ('model', 'fuse', 'routing_integration'),
    'SETTINGS_FUSE_PATH': ('model', 'fuse', 'settings_path'),
    'SETTINGS_FUSE_FILEMANAGER': ('model', 'fuse', 'filemanager'),
    'FUSE_SPATIAL_MODE': ('model', 'fuse', 'spatial_mode'),
    'FUSE_SUBCATCHMENT_DIM': ('model', 'fuse', 'subcatchment_dim'),
    'EXPERIMENT_OUTPUT_FUSE': ('model', 'fuse', 'experiment_output'),
    'SETTINGS_FUSE_PARAMS_TO_CALIBRATE': ('model', 'fuse', 'params_to_calibrate'),
    'FUSE_DECISION_OPTIONS': ('model', 'fuse', 'decision_options'),
    'FUSE_FILE_ID': ('model', 'fuse', 'file_id'),
    'FUSE_N_ELEVATION_BANDS': ('model', 'fuse', 'n_elevation_bands'),
    'FUSE_TIMEOUT': ('model', 'fuse', 'timeout'),

    # Model > GR
    'GR_INSTALL_PATH': ('model', 'gr', 'install_path'),
    'GR_EXE': ('model', 'gr', 'exe'),
    'GR_SPATIAL_MODE': ('model', 'gr', 'spatial_mode'),
    'GR_ROUTING_INTEGRATION': ('model', 'gr', 'routing_integration'),
    'SETTINGS_GR_PATH': ('model', 'gr', 'settings_path'),
    'SETTINGS_GR_CONTROL': ('model', 'gr', 'control'),
    'GR_PARAMS_TO_CALIBRATE': ('model', 'gr', 'params_to_calibrate'),

    # Model > HYPE
    'HYPE_INSTALL_PATH': ('model', 'hype', 'install_path'),
    'HYPE_EXE': ('model', 'hype', 'exe'),
    'SETTINGS_HYPE_PATH': ('model', 'hype', 'settings_path'),
    'SETTINGS_HYPE_INFO': ('model', 'hype', 'info_file'),
    'HYPE_PARAMS_TO_CALIBRATE': ('model', 'hype', 'params_to_calibrate'),
    'HYPE_SPINUP_DAYS': ('model', 'hype', 'spinup_days'),

    # Model > NGEN
    'NGEN_INSTALL_PATH': ('model', 'ngen', 'install_path'),
    'NGEN_EXE': ('model', 'ngen', 'exe'),
    'NGEN_MODULES_TO_CALIBRATE': ('model', 'ngen', 'modules_to_calibrate'),
    'NGEN_CFE_PARAMS_TO_CALIBRATE': ('model', 'ngen', 'cfe_params_to_calibrate'),
    'NGEN_NOAH_PARAMS_TO_CALIBRATE': ('model', 'ngen', 'noah_params_to_calibrate'),
    'NGEN_PET_PARAMS_TO_CALIBRATE': ('model', 'ngen', 'pet_params_to_calibrate'),
    'NGEN_ACTIVE_CATCHMENT_ID': ('model', 'ngen', 'active_catchment_id'),

    # Model > MESH
    'MESH_INSTALL_PATH': ('model', 'mesh', 'install_path'),
    'MESH_EXE': ('model', 'mesh', 'exe'),
    'MESH_SPATIAL_MODE': ('model', 'mesh', 'spatial_mode'),
    'SETTINGS_MESH_PATH': ('model', 'mesh', 'settings_path'),
    'EXPERIMENT_OUTPUT_MESH': ('model', 'mesh', 'experiment_output'),
    'MESH_FORCING_PATH': ('model', 'mesh', 'forcing_path'),
    'MESH_FORCING_VARS': ('model', 'mesh', 'forcing_vars'),
    'MESH_FORCING_UNITS': ('model', 'mesh', 'forcing_units'),
    'MESH_FORCING_TO_UNITS': ('model', 'mesh', 'forcing_to_units'),
    'MESH_LANDCOVER_STATS_PATH': ('model', 'mesh', 'landcover_stats_path'),
    'MESH_LANDCOVER_STATS_DIR': ('model', 'mesh', 'landcover_stats_dir'),
    'MESH_LANDCOVER_STATS_FILE': ('model', 'mesh', 'landcover_stats_file'),
    'MESH_MAIN_ID': ('model', 'mesh', 'main_id'),
    'MESH_DS_MAIN_ID': ('model', 'mesh', 'ds_main_id'),
    'MESH_LANDCOVER_CLASSES': ('model', 'mesh', 'landcover_classes'),
    'MESH_DDB_VARS': ('model', 'mesh', 'ddb_vars'),
    'MESH_DDB_UNITS': ('model', 'mesh', 'ddb_units'),
    'MESH_DDB_TO_UNITS': ('model', 'mesh', 'ddb_to_units'),
    'MESH_DDB_MIN_VALUES': ('model', 'mesh', 'ddb_min_values'),
    'MESH_GRU_DIM': ('model', 'mesh', 'gru_dim'),
    'MESH_HRU_DIM': ('model', 'mesh', 'hru_dim'),
    'MESH_OUTLET_VALUE': ('model', 'mesh', 'outlet_value'),
    'SETTINGS_MESH_INPUT': ('model', 'mesh', 'input_file'),
    'MESH_PARAMS_TO_CALIBRATE': ('model', 'mesh', 'params_to_calibrate'),
    'MESH_SPINUP_DAYS': ('model', 'mesh', 'spinup_days'),
    'MESH_GRU_MIN_TOTAL': ('model', 'mesh', 'gru_min_total'),

    # Model > mizuRoute
    'INSTALL_PATH_MIZUROUTE': ('model', 'mizuroute', 'install_path'),
    'EXE_NAME_MIZUROUTE': ('model', 'mizuroute', 'exe'),
    'SETTINGS_MIZU_PATH': ('model', 'mizuroute', 'settings_path'),
    'SETTINGS_MIZU_WITHIN_BASIN': ('model', 'mizuroute', 'within_basin'),
    'SETTINGS_MIZU_ROUTING_DT': ('model', 'mizuroute', 'routing_dt'),
    'SETTINGS_MIZU_ROUTING_UNITS': ('model', 'mizuroute', 'routing_units'),
    'SETTINGS_MIZU_ROUTING_VAR': ('model', 'mizuroute', 'routing_var'),
    'SETTINGS_MIZU_OUTPUT_FREQ': ('model', 'mizuroute', 'output_freq'),
    'SETTINGS_MIZU_OUTPUT_VARS': ('model', 'mizuroute', 'output_vars'),
    'SETTINGS_MIZU_MAKE_OUTLET': ('model', 'mizuroute', 'make_outlet'),
    'SETTINGS_MIZU_NEEDS_REMAP': ('model', 'mizuroute', 'needs_remap'),
    'SETTINGS_MIZU_TOPOLOGY': ('model', 'mizuroute', 'topology'),
    'SETTINGS_MIZU_PARAMETERS': ('model', 'mizuroute', 'parameters'),
    'SETTINGS_MIZU_CONTROL_FILE': ('model', 'mizuroute', 'control_file'),
    'SETTINGS_MIZU_REMAP': ('model', 'mizuroute', 'remap'),
    'MIZU_FROM_MODEL': ('model', 'mizuroute', 'from_model'),
    'EXPERIMENT_LOG_MIZUROUTE': ('model', 'mizuroute', 'experiment_log'),
    'EXPERIMENT_OUTPUT_MIZUROUTE': ('model', 'mizuroute', 'experiment_output'),
    'SETTINGS_MIZU_OUTPUT_VAR': ('model', 'mizuroute', 'output_var'),
    'SETTINGS_MIZU_PARAMETER_FILE': ('model', 'mizuroute', 'parameter_file'),
    'SETTINGS_MIZU_REMAP_FILE': ('model', 'mizuroute', 'remap_file'),
    'SETTINGS_MIZU_TOPOLOGY_FILE': ('model', 'mizuroute', 'topology_file'),
    'MIZUROUTE_PARAMS_TO_CALIBRATE': ('model', 'mizuroute', 'params_to_calibrate'),
    'CALIBRATE_MIZUROUTE': ('model', 'mizuroute', 'calibrate'),
    'MIZUROUTE_TIMEOUT': ('model', 'mizuroute', 'timeout'),

    # Model > LSTM
    'LSTM_LOAD': ('model', 'lstm', 'load'),
    'LSTM_HIDDEN_SIZE': ('model', 'lstm', 'hidden_size'),
    'LSTM_NUM_LAYERS': ('model', 'lstm', 'num_layers'),
    'LSTM_EPOCHS': ('model', 'lstm', 'epochs'),
    'LSTM_BATCH_SIZE': ('model', 'lstm', 'batch_size'),
    'LSTM_LEARNING_RATE': ('model', 'lstm', 'learning_rate'),
    'LSTM_LEARNING_PATIENCE': ('model', 'lstm', 'learning_patience'),
    'LSTM_LOOKBACK': ('model', 'lstm', 'lookback'),
    'LSTM_DROPOUT': ('model', 'lstm', 'dropout'),
    'LSTM_L2_REGULARIZATION': ('model', 'lstm', 'l2_regularization'),
    'LSTM_USE_ATTENTION': ('model', 'lstm', 'use_attention'),
    'LSTM_USE_SNOW': ('model', 'lstm', 'use_snow'),
    'LSTM_TRAIN_THROUGH_ROUTING': ('model', 'lstm', 'train_through_routing'),

    # Model > RHESSys
    'RHESSYS_INSTALL_PATH': ('model', 'rhessys', 'install_path'),
    'RHESSYS_EXE': ('model', 'rhessys', 'exe'),
    'SETTINGS_RHESSYS_PATH': ('model', 'rhessys', 'settings_path'),
    'EXPERIMENT_OUTPUT_RHESSYS': ('model', 'rhessys', 'experiment_output'),
    'FORCING_RHESSYS_PATH': ('model', 'rhessys', 'forcing_path'),
    'RHESSYS_WORLD_TEMPLATE': ('model', 'rhessys', 'world_template'),
    'RHESSYS_FLOW_TEMPLATE': ('model', 'rhessys', 'flow_template'),
    'RHESSYS_SKIP_CALIBRATION': ('model', 'rhessys', 'skip_calibration'),
    'RHESSYS_USE_WMFIRE': ('model', 'rhessys', 'use_wmfire'),
    'WMFIRE_INSTALL_PATH': ('model', 'rhessys', 'wmfire_install_path'),
    'WMFIRE_LIB': ('model', 'rhessys', 'wmfire_lib'),
    'RHESSYS_USE_VMFIRE': ('model', 'rhessys', 'use_vmfire'),
    'VMFIRE_INSTALL_PATH': ('model', 'rhessys', 'vmfire_install_path'),
    'RHESSYS_TIMEOUT': ('model', 'rhessys', 'timeout'),

    # Model > GNN
    'GNN_LOAD': ('model', 'gnn', 'load'),
    'GNN_HIDDEN_SIZE': ('model', 'gnn', 'hidden_size'),
    'GNN_NUM_LAYERS': ('model', 'gnn', 'num_layers'),
    'GNN_EPOCHS': ('model', 'gnn', 'epochs'),
    'GNN_BATCH_SIZE': ('model', 'gnn', 'batch_size'),
    'GNN_LEARNING_RATE': ('model', 'gnn', 'learning_rate'),
    'GNN_LEARNING_PATIENCE': ('model', 'gnn', 'learning_patience'),
    'GNN_DROPOUT': ('model', 'gnn', 'dropout'),
    'GNN_L2_REGULARIZATION': ('model', 'gnn', 'l2_regularization'),
    'GNN_PARAMS_TO_CALIBRATE': ('model', 'gnn', 'params_to_calibrate'),
    'GNN_PARAMETER_BOUNDS': ('model', 'gnn', 'parameter_bounds'),

    # ========== OPTIMIZATION CONFIGURATION ==========
    'OPTIMIZATION_METHODS': ('optimization', 'methods'),
    'OPTIMIZATION_TARGET': ('optimization', 'target'),
    'CALIBRATION_VARIABLE': ('optimization', 'calibration_variable'),
    'CALIBRATION_TIMESTEP': ('optimization', 'calibration_timestep'),
    'ITERATIVE_OPTIMIZATION_ALGORITHM': ('optimization', 'algorithm'),
    'OPTIMIZATION_METRIC': ('optimization', 'metric'),
    'NUMBER_OF_ITERATIONS': ('optimization', 'iterations'),
    'POPULATION_SIZE': ('optimization', 'population_size'),
    'FINAL_EVALUATION_NUMERICAL_METHOD': ('optimization', 'final_evaluation_numerical_method'),
    'CLEANUP_PARALLEL_DIRS': ('optimization', 'cleanup_parallel_dirs'),
    'PARAMS_KEEP_TRIALS': ('optimization', 'params_keep_trials'),

    # Optimization > PSO
    'SWRMSIZE': ('optimization', 'pso', 'swrmsize'),
    'PSO_COGNITIVE_PARAM': ('optimization', 'pso', 'cognitive_param'),
    'PSO_SOCIAL_PARAM': ('optimization', 'pso', 'social_param'),
    'PSO_INERTIA_WEIGHT': ('optimization', 'pso', 'inertia_weight'),
    'PSO_INERTIA_REDUCTION_RATE': ('optimization', 'pso', 'inertia_reduction_rate'),
    'INERTIA_SCHEDULE': ('optimization', 'pso', 'inertia_schedule'),

    # Optimization > DE
    'DE_SCALING_FACTOR': ('optimization', 'de', 'scaling_factor'),
    'DE_CROSSOVER_RATE': ('optimization', 'de', 'crossover_rate'),

    # Optimization > DDS
    'DDS_R': ('optimization', 'dds', 'r'),
    'ASYNC_DDS_POOL_SIZE': ('optimization', 'dds', 'async_pool_size'),
    'ASYNC_DDS_BATCH_SIZE': ('optimization', 'dds', 'async_batch_size'),
    'MAX_STAGNATION_BATCHES': ('optimization', 'dds', 'max_stagnation_batches'),

    # Optimization > SCE-UA
    'NUMBER_OF_COMPLEXES': ('optimization', 'sce_ua', 'number_of_complexes'),
    'POINTS_PER_SUBCOMPLEX': ('optimization', 'sce_ua', 'points_per_subcomplex'),
    'NUMBER_OF_EVOLUTION_STEPS': ('optimization', 'sce_ua', 'number_of_evolution_steps'),
    'EVOLUTION_STAGNATION': ('optimization', 'sce_ua', 'evolution_stagnation'),
    'PERCENT_CHANGE_THRESHOLD': ('optimization', 'sce_ua', 'percent_change_threshold'),

    # Optimization > NSGA2
    'NSGA2_MULTI_TARGET': ('optimization', 'nsga2', 'multi_target'),
    'NSGA2_PRIMARY_TARGET': ('optimization', 'nsga2', 'primary_target'),
    'NSGA2_SECONDARY_TARGET': ('optimization', 'nsga2', 'secondary_target'),
    'NSGA2_PRIMARY_METRIC': ('optimization', 'nsga2', 'primary_metric'),
    'NSGA2_SECONDARY_METRIC': ('optimization', 'nsga2', 'secondary_metric'),
    'NSGA2_CROSSOVER_RATE': ('optimization', 'nsga2', 'crossover_rate'),
    'NSGA2_MUTATION_RATE': ('optimization', 'nsga2', 'mutation_rate'),
    'NSGA2_ETA_C': ('optimization', 'nsga2', 'eta_c'),
    'NSGA2_ETA_M': ('optimization', 'nsga2', 'eta_m'),
    # Legacy aliases for NSGA2 multi-target settings
    'OPTIMIZATION_TARGET2': ('optimization', 'nsga2', 'secondary_target'),
    'OPTIMIZATION_METRIC2': ('optimization', 'nsga2', 'secondary_metric'),

    # Optimization > DPE
    'DPE_TRAINING_CACHE': ('optimization', 'dpe', 'training_cache'),
    'DPE_HIDDEN_DIMS': ('optimization', 'dpe', 'hidden_dims'),
    'DPE_TRAINING_SAMPLES': ('optimization', 'dpe', 'training_samples'),
    'DPE_VALIDATION_SAMPLES': ('optimization', 'dpe', 'validation_samples'),
    'DPE_EPOCHS': ('optimization', 'dpe', 'epochs'),
    'DPE_LEARNING_RATE': ('optimization', 'dpe', 'learning_rate'),
    'DPE_OPTIMIZATION_LR': ('optimization', 'dpe', 'optimization_lr'),
    'DPE_OPTIMIZATION_STEPS': ('optimization', 'dpe', 'optimization_steps'),
    'DPE_OPTIMIZER': ('optimization', 'dpe', 'optimizer'),
    'DPE_OBJECTIVE_WEIGHTS': ('optimization', 'dpe', 'objective_weights'),
    'DPE_EMULATOR_ITERATE': ('optimization', 'dpe', 'emulator_iterate'),
    'DPE_ITERATE_MAX_ITERATIONS': ('optimization', 'dpe', 'iterate_max_iterations'),
    'DPE_ITERATE_SAMPLES_PER_CYCLE': ('optimization', 'dpe', 'iterate_samples_per_cycle'),
    'DPE_ITERATE_SAMPLING_RADIUS': ('optimization', 'dpe', 'iterate_sampling_radius'),
    'DPE_ITERATE_CONVERGENCE_TOL': ('optimization', 'dpe', 'iterate_convergence_tol'),
    'DPE_ITERATE_MIN_IMPROVEMENT': ('optimization', 'dpe', 'iterate_min_improvement'),
    'DPE_ITERATE_SAMPLING_METHOD': ('optimization', 'dpe', 'iterate_sampling_method'),
    'DPE_USE_NN_HEAD': ('optimization', 'dpe', 'use_nn_head'),
    'DPE_PRETRAIN_NN_HEAD': ('optimization', 'dpe', 'pretrain_nn_head'),
    'DPE_USE_SUNDIALS': ('optimization', 'dpe', 'use_sundials'),
    'DPE_AUTODIFF_STEPS': ('optimization', 'dpe', 'autodiff_steps'),
    'DPE_AUTODIFF_LR': ('optimization', 'dpe', 'autodiff_lr'),
    'DPE_FD_STEP': ('optimization', 'dpe', 'fd_step'),
    'DPE_GD_STEP_SIZE': ('optimization', 'dpe', 'gd_step_size'),

    # Optimization > Emulation
    'EMULATION_NUM_SAMPLES': ('optimization', 'emulation', 'num_samples'),
    'EMULATION_SEED': ('optimization', 'emulation', 'seed'),
    'EMULATION_SAMPLING_METHOD': ('optimization', 'emulation', 'sampling_method'),
    'EMULATION_PARALLEL_ENSEMBLE': ('optimization', 'emulation', 'parallel_ensemble'),
    'EMULATION_MAX_PARALLEL_JOBS': ('optimization', 'emulation', 'max_parallel_jobs'),
    'EMULATION_SKIP_MIZUROUTE': ('optimization', 'emulation', 'skip_mizuroute'),
    'EMULATION_USE_ATTRIBUTES': ('optimization', 'emulation', 'use_attributes'),
    'EMULATION_MAX_ITERATIONS': ('optimization', 'emulation', 'max_iterations'),

    # ========== EVALUATION CONFIGURATION ==========
    'EVALUATION_DATA': ('evaluation', 'evaluation_data'),
    'ANALYSES': ('evaluation', 'analyses'),
    'SIM_REACH_ID': ('evaluation', 'sim_reach_id'),
    'HRU_GAUGE_MAPPING': ('evaluation', 'hru_gauge_mapping'),

    # Evaluation > Streamflow
    # NOTE: STREAMFLOW_DATA_PROVIDER and DOWNLOAD_USGS_DATA are mapped to 'data' section
    # for observation handlers. Evaluation uses the same values from data section.
    'DOWNLOAD_WSC_DATA': ('evaluation', 'streamflow', 'download_wsc'),
    'STATION_ID': ('evaluation', 'streamflow', 'station_id'),
    'STREAMFLOW_RAW_PATH': ('evaluation', 'streamflow', 'raw_path'),
    'STREAMFLOW_RAW_NAME': ('evaluation', 'streamflow', 'raw_name'),
    'STREAMFLOW_PROCESSED_PATH': ('evaluation', 'streamflow', 'processed_path'),
    'HYDAT_PATH': ('evaluation', 'streamflow', 'hydat_path'),

    # Evaluation > SNOTEL
    'DOWNLOAD_SNOTEL': ('evaluation', 'snotel', 'download'),
    'SNOTEL_STATION': ('evaluation', 'snotel', 'station'),
    'SNOTEL_PATH': ('evaluation', 'snotel', 'path'),

    # Evaluation > FluxNet
    'DOWNLOAD_FLUXNET': ('evaluation', 'fluxnet', 'download'),
    'FLUXNET_STATION': ('evaluation', 'fluxnet', 'station'),
    'FLUXNET_PATH': ('evaluation', 'fluxnet', 'path'),

    # Evaluation > USGS GW
    'DOWNLOAD_USGS_GW': ('evaluation', 'usgs_gw', 'download'),
    'USGS_STATION': ('evaluation', 'usgs_gw', 'station'),

    # Evaluation > SMAP
    'DOWNLOAD_SMAP': ('evaluation', 'smap', 'download'),
    'SMAP_PRODUCT': ('evaluation', 'smap', 'product'),
    'SMAP_PATH': ('evaluation', 'smap', 'path'),
    'SMAP_MAX_GRANULES': ('evaluation', 'smap', 'max_granules'),
    'SMAP_USE_OPENDAP': ('evaluation', 'smap', 'use_opendap'),
    'SMAP_SURFACE_DEPTH_M': ('evaluation', 'smap', 'surface_depth_m'),
    'SMAP_ROOTZONE_DEPTH_M': ('evaluation', 'smap', 'rootzone_depth_m'),
    'ISMN_PATH': ('evaluation', 'ismn', 'path'),
    'ISMN_API_BASE': ('evaluation', 'ismn', 'api_base'),
    'ISMN_METADATA_URL': ('evaluation', 'ismn', 'metadata_url'),
    'ISMN_VARIABLE_LIST_URL': ('evaluation', 'ismn', 'variable_list_url'),
    'ISMN_DATA_URL_TEMPLATE': ('evaluation', 'ismn', 'data_url_template'),
    'ISMN_MAX_STATIONS': ('evaluation', 'ismn', 'max_stations'),
    'ISMN_SEARCH_RADIUS_KM': ('evaluation', 'ismn', 'search_radius_km'),
    'ISMN_TARGET_DEPTH_M': ('evaluation', 'ismn', 'target_depth_m'),
    'ISMN_TEMPORAL_AGGREGATION': ('evaluation', 'ismn', 'temporal_aggregation'),

    # Evaluation > GRACE
    'DOWNLOAD_GRACE': ('evaluation', 'grace', 'download'),
    'GRACE_PRODUCT': ('evaluation', 'grace', 'product'),
    'GRACE_PATH': ('evaluation', 'grace', 'path'),
    'GRACE_DATA_DIR': ('evaluation', 'grace', 'data_dir'),

    # Evaluation > MODIS Snow
    'DOWNLOAD_MODIS_SNOW': ('evaluation', 'modis_snow', 'download'),
    'MODIS_SNOW_PRODUCT': ('evaluation', 'modis_snow', 'product'),
    'MODIS_SNOW_PATH': ('evaluation', 'modis_snow', 'path'),
    'MODIS_SNOW_DIR': ('evaluation', 'modis_snow', 'data_dir'),
    'MODIS_MIN_PIXELS': ('evaluation', 'modis_snow', 'min_pixels'),
    'MODIS_SCA_MERGE': ('evaluation', 'modis_snow', 'merge'),
    'MODIS_SCA_PRODUCTS': ('evaluation', 'modis_snow', 'products'),
    'MODIS_SCA_MERGE_STRATEGY': ('evaluation', 'modis_snow', 'merge_strategy'),
    'MODIS_SCA_CLOUD_FILTER': ('evaluation', 'modis_snow', 'cloud_filter'),
    'MODIS_SCA_MIN_VALID_RATIO': ('evaluation', 'modis_snow', 'min_valid_ratio'),
    'MODIS_SCA_NORMALIZE': ('evaluation', 'modis_snow', 'normalize'),
    'MODIS_SCA_USE_CATCHMENT_MASK': ('evaluation', 'modis_snow', 'use_catchment_mask'),

    # Evaluation > MODIS ET
    'DOWNLOAD_MODIS_ET': ('evaluation', 'modis_et', 'download'),
    'MODIS_ET_PRODUCT': ('evaluation', 'modis_et', 'product'),
    'MODIS_ET_PATH': ('evaluation', 'modis_et', 'path'),
    'MOD16_ET_DIR': ('evaluation', 'modis_et', 'data_dir'),

    # Evaluation > Attributes
    'ATTRIBUTES_DATA_DIR': ('evaluation', 'attributes', 'data_dir'),
    'ATTRIBUTES_SOILGRIDS_PATH': ('evaluation', 'attributes', 'soilgrids_path'),
    'ATTRIBUTES_PELLETIER_PATH': ('evaluation', 'attributes', 'pelletier_path'),
    'ATTRIBUTES_MERIT_PATH': ('evaluation', 'attributes', 'merit_path'),
    'ATTRIBUTES_MODIS_PATH': ('evaluation', 'attributes', 'modis_path'),
    'ATTRIBUTES_GLCLU_PATH': ('evaluation', 'attributes', 'glclu_path'),
    'ATTRIBUTES_FOREST_HEIGHT_PATH': ('evaluation', 'attributes', 'forest_height_path'),
    'ATTRIBUTES_WORLDCLIM_PATH': ('evaluation', 'attributes', 'worldclim_path'),
    'ATTRIBUTES_GLIM_PATH': ('evaluation', 'attributes', 'glim_path'),
    'ATTRIBUTES_GROUNDWATER_PATH': ('evaluation', 'attributes', 'groundwater_path'),
    'ATTRIBUTES_STREAMFLOW_PATH': ('evaluation', 'attributes', 'streamflow_path'),
    'ATTRIBUTES_GLWD_PATH': ('evaluation', 'attributes', 'glwd_path'),
    'ATTRIBUTES_HYDROLAKES_PATH': ('evaluation', 'attributes', 'hydrolakes_path'),
    'ATTRIBUTES_OUTPUT_DIR': ('evaluation', 'attributes', 'output_dir'),

    # ========== PATHS CONFIGURATION ==========
    # Catchment shapefile
    'CATCHMENT_PATH': ('paths', 'catchment_path'),
    'CATCHMENT_SHP_NAME': ('paths', 'catchment_name'),
    'CATCHMENT_SHP_LAT': ('paths', 'catchment_lat'),
    'CATCHMENT_SHP_LON': ('paths', 'catchment_lon'),
    'CATCHMENT_SHP_AREA': ('paths', 'catchment_area'),
    'CATCHMENT_SHP_HRUID': ('paths', 'catchment_hruid'),
    'CATCHMENT_SHP_GRUID': ('paths', 'catchment_gruid'),

    # River basins shapefile
    'RIVER_BASINS_PATH': ('paths', 'river_basins_path'),
    'RIVER_BASINS_NAME': ('paths', 'river_basins_name'),
    'RIVER_BASIN_SHP_RM_GRUID': ('paths', 'river_basin_rm_gruid'),
    'RIVER_BASIN_SHP_HRU_TO_SEG': ('paths', 'river_basin_hru_to_seg'),
    'RIVER_BASIN_SHP_AREA': ('paths', 'river_basin_area'),

    # River network shapefile
    'RIVER_NETWORK_SHP_PATH': ('paths', 'river_network_path'),
    'RIVER_NETWORK_SHP_NAME': ('paths', 'river_network_name'),
    'RIVER_NETWORK_SHP_LENGTH': ('paths', 'river_network_length'),
    'RIVER_NETWORK_SHP_SEGID': ('paths', 'river_network_segid'),
    'RIVER_NETWORK_SHP_DOWNSEGID': ('paths', 'river_network_downsegid'),
    'RIVER_NETWORK_SHP_SLOPE': ('paths', 'river_network_slope'),

    # Pour point shapefile
    'POUR_POINT_SHP_PATH': ('paths', 'pour_point_path'),
    'POUR_POINT_SHP_NAME': ('paths', 'pour_point_name'),

    # Common paths
    'FORCING_PATH': ('paths', 'forcing_path'),
    'OBSERVATIONS_PATH': ('paths', 'observations_path'),
    'SIMULATIONS_PATH': ('paths', 'simulations_path'),
    'INTERSECT_SOIL_PATH': ('paths', 'intersect_soil_path'),
    'INTERSECT_SOIL_NAME': ('paths', 'intersect_soil_name'),
    'INTERSECT_ROUTING_PATH': ('paths', 'intersect_routing_path'),
    'INTERSECT_ROUTING_NAME': ('paths', 'intersect_routing_name'),
    'INTERSECT_DEM_PATH': ('paths', 'intersect_dem_path'),
    'INTERSECT_DEM_NAME': ('paths', 'intersect_dem_name'),
    'INTERSECT_LAND_PATH': ('paths', 'intersect_land_path'),
    'INTERSECT_LAND_NAME': ('paths', 'intersect_land_name'),
    'OUTPUT_BASINS_PATH': ('paths', 'output_basins_path'),
    'OUTPUT_RIVERS_PATH': ('paths', 'output_rivers_path'),
    'DEM_PATH': ('paths', 'dem_path'),
    'DEM_NAME': ('paths', 'dem_name'),
    'SOURCE_GEOFABRIC_BASINS_PATH': ('paths', 'source_geofabric_basins_path'),
    'SOURCE_GEOFABRIC_RIVERS_PATH': ('paths', 'source_geofabric_rivers_path'),
    'TAUDEM_DIR': ('paths', 'taudem_dir'),
    'OUTPUT_DIR': ('paths', 'output_dir'),
    'CATCHMENT_PLOT_DIR': ('paths', 'catchment_plot_dir'),
    'SOIL_CLASS_PATH': ('paths', 'soil_class_path'),
    'SOIL_CLASS_NAME': ('paths', 'soil_class_name'),
    'LAND_CLASS_PATH': ('paths', 'land_class_path'),
    'RADIATION_PATH': ('paths', 'radiation_path'),

    # Tool paths
    'DATATOOL_PATH': ('paths', 'datatool_path'),
    'GISTOOL_PATH': ('paths', 'gistool_path'),
    'EASYMORE_CLIENT': ('paths', 'easymore_client'),
    'DATATOOL_DATASET_ROOT': ('paths', 'datatool_dataset_root'),
    'GISTOOL_DATASET_ROOT': ('paths', 'gistool_dataset_root'),
    'TOOL_CACHE': ('paths', 'tool_cache'),
    'EASYMORE_CACHE': ('paths', 'easymore_cache'),
    'EASYMORE_JOB_CONF': ('paths', 'easymore_job_conf'),
    'CLUSTER_JSON': ('paths', 'cluster_json'),
    'GISTOOL_LIB_PATH': ('paths', 'gistool_lib_path'),
}


# ========================================
# TRANSFORMATION FUNCTIONS
# ========================================

def _set_nested_value(d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    """
    Helper to set value at nested path in dict.

    Args:
        d: Dictionary to modify
        path: Tuple of keys representing nested path
        value: Value to set

    Example:
        >>> d = {}
        >>> _set_nested_value(d, ('domain', 'name'), 'test')
        >>> d
        {'domain': {'name': 'test'}}
    """
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def transform_flat_to_nested(flat_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform flat configuration dict to nested structure.

    Maps uppercase keys like 'DOMAIN_NAME' to nested paths like
    {'domain': {'name': ...}}.

    PHASE 1: Currently uses manual mapping for backward compatibility.
    Auto-generated mapping is available via get_flat_to_nested_map() for validation.
    In Phase 4, this will switch to use auto-generated mapping exclusively.

    Args:
        flat_config: Flat configuration dictionary with uppercase keys

    Returns:
        Nested configuration dictionary

    Example:
        >>> flat = {'DOMAIN_NAME': 'test', 'FORCING_DATASET': 'ERA5'}
        >>> nested = transform_flat_to_nested(flat)
        >>> nested
        {
            'domain': {'name': 'test'},
            'forcing': {'dataset': 'ERA5'}
        }
    """
    nested: Dict[str, Any] = {
        'system': {},
        'domain': {},
        'data': {},
        'forcing': {},
        'model': {},
        'optimization': {},
        'evaluation': {},
        'paths': {}
    }

    # Build combined mapping: base + model-specific transformers
    combined_mapping = FLAT_TO_NESTED_MAP.copy()

    # Try to get model-specific transformers from ModelRegistry
    hydrological_model = flat_config.get('HYDROLOGICAL_MODEL')
    if hydrological_model:
        try:
            from symfluence.models.registry import ModelRegistry
            model_transformers = ModelRegistry.get_config_transformers(hydrological_model)
            if model_transformers:
                # Model-specific transformers override base mapping
                combined_mapping.update(model_transformers)
        except Exception:
            # If ModelRegistry not available or model not registered, just use base mapping
            pass

    # Apply mapping
    for flat_key, value in flat_config.items():
        if flat_key in combined_mapping:
            path = combined_mapping[flat_key]
            _set_nested_value(nested, path, value)
        else:
            # Unknown keys stored in _extra (Pydantic extra='allow' will handle)
            nested.setdefault('_extra', {})[flat_key] = value

    return nested


def flatten_nested_config(config: 'SymfluenceConfig') -> Dict[str, Any]:
    """
    Convert SymfluenceConfig instance to flat dict with uppercase keys.

    This is the inverse operation of transform_flat_to_nested, used for
    backward compatibility with legacy code expecting flat configs.

    Args:
        config: SymfluenceConfig instance

    Returns:
        Flat configuration dictionary with uppercase keys

    Example:
        >>> from symfluence.core.config.models import SymfluenceConfig
        >>> config = SymfluenceConfig.from_preset('fuse-basic')
        >>> flat = flatten_nested_config(config)
        >>> flat['DOMAIN_NAME']
        'test_basin'
    """
    flat = {}

    # Create reverse mapping (nested path -> flat key)
    nested_to_flat = {v: k for k, v in FLAT_TO_NESTED_MAP.items()}

    def _flatten_section(section_name: str, section_obj: Any, prefix: Tuple[str, ...] = ()) -> None:
        """Recursively flatten a config section"""
        if section_obj is None:
            return

        # Get the dict representation
        # Use exclude_none=True so that .get() falls back to defaults for unset values
        if hasattr(section_obj, 'model_dump'):
            section_dict = section_obj.model_dump(by_alias=False, exclude_none=True)
        else:
            section_dict = section_obj if isinstance(section_obj, dict) else {}

        for key, value in section_dict.items():
            current_path = prefix + (key,)

            # Check if this path maps to a flat key
            if current_path in nested_to_flat:
                flat_key = nested_to_flat[current_path]
                # Convert Path to string for compatibility
                if isinstance(value, Path):
                    flat[flat_key] = str(value)
                else:
                    flat[flat_key] = value
            elif isinstance(value, dict) or hasattr(value, 'model_dump'):
                # Recurse into nested objects
                _flatten_section(key, value, current_path)

    # Flatten each section
    _flatten_section('system', config.system, ('system',))
    _flatten_section('domain', config.domain, ('domain',))
    _flatten_section('data', config.data, ('data',))
    _flatten_section('forcing', config.forcing, ('forcing',))
    _flatten_section('model', config.model, ('model',))
    _flatten_section('optimization', config.optimization, ('optimization',))
    _flatten_section('evaluation', config.evaluation, ('evaluation',))
    _flatten_section('paths', config.paths, ('paths',))

    # Include extra fields from root config (e.g. CUSTOM_PATH in tests)
    # Extra fields can be at top-level or nested inside '_extra' dict
    if hasattr(config, 'model_extra') and config.model_extra:
        for key, value in config.model_extra.items():
            if key == '_extra' and isinstance(value, dict):
                # Handle nested _extra dict (from transform_flat_to_nested)
                for extra_key, extra_value in value.items():
                    if isinstance(extra_value, Path):
                        flat[extra_key] = str(extra_value)
                    else:
                        flat[extra_key] = extra_value
            elif isinstance(value, Path):
                flat[key] = str(value)
            else:
                flat[key] = value

    return flat


# ========================================
# MAPPING VALIDATION (for testing)
# ========================================

def validate_mapping_against_pydantic() -> Dict[str, Any]:
    """
    Validate that FLAT_TO_NESTED_MAP matches auto-generated mapping from Pydantic models.

    This function can be used in tests to ensure the manual mapping stays in sync
    with Pydantic model aliases. In the future, this manual mapping can be replaced
    entirely with auto-generation.

    Returns:
        Dictionary with validation results:
        - 'equivalent': bool - True if mappings match
        - 'missing_in_manual': list - Keys in Pydantic but not in manual
        - 'extra_in_manual': list - Keys in manual but not in Pydantic
        - 'mismatched': dict - Keys with different paths
    """
    from symfluence.core.config.introspection import (
        generate_flat_to_nested_map,
        validate_mapping_equivalence
    )
    from symfluence.core.config.models import SymfluenceConfig

    auto_mapping = generate_flat_to_nested_map(SymfluenceConfig)
    result = validate_mapping_equivalence(auto_mapping, FLAT_TO_NESTED_MAP)

    # Rename keys for clarity from manual mapping perspective
    return {
        'equivalent': result['equivalent'],
        'missing_in_manual': result['extra_in_auto'],  # In Pydantic but not manual
        'extra_in_manual': result['missing_in_auto'],  # In manual but not Pydantic
        'mismatched': result['mismatched'],
        'manual_count': result['manual_count'],
        'pydantic_count': result['auto_count']
    }
