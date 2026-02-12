"""
SUMMA initialization presets.

This module registers SUMMA-specific presets with the PresetRegistry,
keeping model-specific configuration within the model directory.
"""

from symfluence.cli.preset_registry import PresetRegistry


@PresetRegistry.register_preset('summa-basic')
def summa_basic_preset():
    """Generic SUMMA distributed setup with ERA5 forcing."""
    return {
        'description': 'Generic SUMMA distributed setup with ERA5 forcing',
        'base_template': 'config_template_comprehensive.yaml',
        'settings': {
            # Global settings
            'EXPERIMENT_ID': 'run_1',
            'NUM_PROCESSES': 1,
            'FORCE_RUN_ALL_STEPS': False,

            # Geospatial settings
            'DOMAIN_DEFINITION_METHOD': 'delineate',
            'SUB_GRID_DISCRETIZATION': 'GRUs',
            'ROUTING_DELINEATION': 'distributed',
            'GEOFABRIC_TYPE': 'TDX',
            'STREAM_THRESHOLD': 1000,
            'ELEVATION_BAND_SIZE': 200,
            'MIN_HRU_SIZE': 4,

            # Forcing settings
            'FORCING_DATASET': 'ERA5',
            'FORCING_TIME_STEP_SIZE': 3600,
            'DATA_ACCESS': 'cloud',

            # Model settings
            'HYDROLOGICAL_MODEL': 'SUMMA',
            'ROUTING_MODEL': 'mizuRoute',
            'SUMMA_INSTALL_PATH': 'default',
            'SUMMA_EXE': 'summa_sundials.exe',
            'SETTINGS_SUMMA_PATH': 'default',
            'SETTINGS_SUMMA_FILEMANAGER': 'fileManager.txt',
            'SETTINGS_SUMMA_CONNECT_HRUS': 'yes',
            'SETTINGS_SUMMA_USE_PARALLEL_SUMMA': False,
            'EXPERIMENT_OUTPUT_SUMMA': 'default',

            # mizuRoute settings
            'INSTALL_PATH_MIZUROUTE': 'default',
            'EXE_NAME_MIZUROUTE': 'mizuRoute.exe',
            'SETTINGS_MIZU_PATH': 'default',
            'SETTINGS_MIZU_WITHIN_BASIN': 0,
            'SETTINGS_MIZU_ROUTING_DT': 3600,
            'SETTINGS_MIZU_NEEDS_REMAP': 'no',
            'EXPERIMENT_OUTPUT_MIZUROUTE': 'default',

            # Calibration parameters
            'PARAMS_TO_CALIBRATE': 'k_soil,theta_sat,aquiferBaseflowExp,aquiferBaseflowRate',
            'BASIN_PARAMS_TO_CALIBRATE': 'routingGammaScale,routingGammaShape',

            # Evaluation settings
            'DOWNLOAD_USGS_DATA': True,

            # Optimization settings
            'OPTIMIZATION_METHODS': ['iteration'],
            'ITERATIVE_OPTIMIZATION_ALGORITHM': 'DDS',
            'NUMBER_OF_ITERATIONS': 200,
            'OPTIMIZATION_METRIC': 'KGE',
            'DDS_R': 0.2,
            'RANDOM_SEED': 42,
        },
        'summa_decisions': {
            'snowIncept': ['lightSnow'],
            'compaction': ['consettl'],
            'snowLayers': ['CLM_2010'],
            'alb_method': ['conDecay'],
            'thCondSnow': ['tyen1965']
        }
    }
