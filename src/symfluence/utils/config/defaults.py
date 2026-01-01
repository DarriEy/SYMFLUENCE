"""
Centralized default configuration values for SYMFLUENCE.

This module provides default values for optional configuration parameters.
Required parameters (defined in config_loader.py) must be set in config files.
"""

from typing import Dict, Any


class ConfigDefaults:
    """Default configuration values for SYMFLUENCE"""

    # === System Settings ===
    MPI_PROCESSES = 1
    DEBUG_MODE = False
    LOG_LEVEL = 'INFO'
    LOG_TO_FILE = True
    FORCE_RUN_ALL_STEPS = False
    STOP_ON_ERROR = True

    # === Data Access ===
    DATA_ACCESS = 'MAF'  # Options: 'MAF', 'cloud'
    DEM_SOURCE = 'merit_hydro'  # Options: 'copernicus', 'fabdem', 'nasadem', 'merit_hydro'
    LAND_CLASS_NAME = 'default'

    # === Forcing Data ===
    FORCING_TIME_STEP_SIZE = 3600  # seconds
    FORCING_VARIABLES = 'default'
    SUPPLEMENT_FORCING = False

    # === EM-Earth Settings ===
    EM_EARTH_REGION = 'NorthAmerica'
    EM_EARTH_MIN_BBOX_SIZE = 0.1  # degrees

    # === Domain Definition ===
    STREAM_THRESHOLD = 1000  # km²
    MIN_HRU_SIZE = 0.0
    MIN_GRU_SIZE = 0.0
    ELEVATION_BAND_SIZE = 400  # meters
    RADIATION_CLASS_NUMBER = 1
    ASPECT_CLASS_NUMBER = 1
    MOVE_OUTLETS_MAX_DISTANCE = 1000  # meters

    # === Shapefile Settings ===
    RIVER_BASIN_SHP_AREA = 'GRU_area'
    RIVER_BASIN_SHP_RM_GRUID = 'GRU_ID'
    CATCHMENT_SHP_GRUID = 'GRU_ID'
    RIVER_BASINS_NAME = 'default'

    # === Model Settings ===
    FUSE_SPATIAL_MODE = 'lumped'  # Options: 'lumped', 'semi_distributed', 'distributed'
    GR_SPATIAL_MODE = 'lumped'

    # === Optimization ===
    NUMBER_OF_ITERATIONS = 1000
    RANDOM_SEED = 42
    LAPSE_RATE = -0.0065  # °C/m (standard atmospheric lapse rate)

    # === Large Domain Emulator ===
    LARGE_DOMAIN_EMULATOR_MODE = 'EMULATOR'
    LARGE_DOMAIN_EMULATOR_HIDDEN_DIM = 512
    LARGE_DOMAIN_EMULATOR_N_HEADS = 8
    LARGE_DOMAIN_EMULATOR_N_LAYERS = 6
    LARGE_DOMAIN_EMULATOR_DROPOUT = 0.1
    LARGE_DOMAIN_EMULATOR_PRETRAIN_NN_HEAD = False
    LARGE_DOMAIN_EMULATOR_BATCH_SIZE = 16
    LARGE_DOMAIN_EMULATOR_LEARNING_RATE = 1e-4
    LARGE_DOMAIN_EMULATOR_EPOCHS = 50
    LARGE_DOMAIN_EMULATOR_VALIDATION_SPLIT = 0.2
    LARGE_DOMAIN_EMULATOR_WINDOW_DAYS = 30
    LARGE_DOMAIN_EMULATOR_TRAINING_SAMPLES = 500
    LARGE_DOMAIN_EMULATOR_OPTIMIZATION_STEPS = 200
    LARGE_DOMAIN_EMULATOR_OPTIMIZATION_LR = 1e-2
    LARGE_DOMAIN_EMULATOR_FD_STEP = 1e-3
    LARGE_DOMAIN_EMULATOR_FD_STEPS = 100
    LARGE_DOMAIN_EMULATOR_FD_STEP_SIZE = 1e-1
    LARGE_DOMAIN_EMULATOR_AUTODIFF_STEPS = 200
    LARGE_DOMAIN_EMULATOR_AUTODIFF_LR = 1e-2
    LARGE_DOMAIN_EMULATOR_USE_NN_HEAD = True

    # === Drop Analysis ===
    DROP_ANALYSIS_MIN_THRESHOLD = 100
    DROP_ANALYSIS_MAX_THRESHOLD = 10000
    DROP_ANALYSIS_NUM_THRESHOLDS = 20

    @classmethod
    def get_defaults(cls) -> Dict[str, Any]:
        """
        Get all default values as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary of configuration defaults
        """
        return {
            k: v for k, v in vars(cls).items()
            if not k.startswith('_') and k.isupper()
        }

    @classmethod
    def get_default(cls, key: str, fallback: Any = None) -> Any:
        """
        Get a single default value.

        Args:
            key: Configuration key name
            fallback: Value to return if key not found in defaults

        Returns:
            Default value for the key, or fallback if not found
        """
        return getattr(cls, key, fallback)


class ModelDefaults:
    """Model-specific default configuration values."""

    FUSE = {
        'FUSE_SPATIAL_MODE': 'lumped',
        'ROUTING_MODEL': 'none',
        'FUSE_INSTALL_PATH': 'default',
        'SETTINGS_FUSE_PATH': 'default',
        'SETTINGS_FUSE_FILEMANAGER': 'fm_catch.txt',
        'FUSE_EXE': 'fuse.exe',
        'EXPERIMENT_OUTPUT_FUSE': 'default',
    }

    SUMMA = {
        'ROUTING_MODEL': 'mizuRoute',
        'SUMMA_INSTALL_PATH': 'default',
        'SETTINGS_SUMMA_PATH': 'default',
        'SETTINGS_SUMMA_FILEMANAGER': 'fileManager.txt',
        'SUMMA_EXE': 'summa_sundials.exe',
        'SETTINGS_SUMMA_CONNECT_HRUS': 'yes',
        'SETTINGS_SUMMA_USE_PARALLEL_SUMMA': False,
        'EXPERIMENT_OUTPUT_SUMMA': 'default',
        'INSTALL_PATH_MIZUROUTE': 'default',
        'EXE_NAME_MIZUROUTE': 'mizuroute.exe',
        'SETTINGS_MIZU_PATH': 'default',
        'EXPERIMENT_OUTPUT_MIZUROUTE': 'default',
    }

    GR = {
        'GR_MODEL_TYPE': 'GR4J',
        'GR_SPATIAL_MODE': 'lumped',
        'GR_EXE': 'GR.r',
    }

    HYPE = {
        'SETTINGS_HYPE_PATH': 'default',
        'HYPE_INSTALL_PATH': 'default',
        'HYPE_EXE': 'hype',
        'SETTINGS_HYPE_CONTROL_FILE': 'info.txt',
    }

    @classmethod
    def get_defaults_for_model(cls, model: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific model.

        Args:
            model: Model name (FUSE, SUMMA, GR, HYPE, etc.)

        Returns:
            Dict[str, Any]: Model-specific default configuration
        """
        return getattr(cls, model.upper(), {}).copy()


class ForcingDefaults:
    """Forcing dataset-specific default configuration values."""

    ERA5 = {
        'FORCING_TIME_STEP_SIZE': 3600,
        'DATA_ACCESS': 'cloud',
    }

    CONUS404 = {
        'FORCING_TIME_STEP_SIZE': 3600,
        'DATA_ACCESS': 'cloud',
    }

    RDRS = {
        'FORCING_TIME_STEP_SIZE': 3600,
        'DATA_ACCESS': 'cloud',
    }

    NLDAS = {
        'FORCING_TIME_STEP_SIZE': 3600,
        'DATA_ACCESS': 'cloud',
    }

    @classmethod
    def get_defaults_for_forcing(cls, forcing: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific forcing dataset.

        Args:
            forcing: Forcing dataset name (ERA5, CONUS404, etc.)

        Returns:
            Dict[str, Any]: Forcing-specific default configuration
        """
        return getattr(cls, forcing.upper(), {}).copy()
