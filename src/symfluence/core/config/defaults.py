"""
Centralized default configuration values for SYMFLUENCE.

This module provides default values for optional configuration parameters.
Required parameters (defined in config_loader.py) must be set in config files.

Model-specific defaults are now registered via the DefaultsRegistry pattern.
Each model registers its own defaults in its directory (models/{model}/defaults.py).
The ModelDefaults class below is maintained for backward compatibility.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigDefaults:
    """Default configuration values for SYMFLUENCE"""

    # === System Settings ===
    MPI_PROCESSES = 1
    DEBUG_MODE = False
    LOG_LEVEL = 'INFO'
    LOG_TO_FILE = True
    FORCE_RUN_ALL_STEPS = False
    STOP_ON_ERROR = True
    
    # === Resource Paths ===
    SETTINGS_BASE_DIR = 'src/symfluence/resources/base_settings'

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
    """
    Model-specific default configuration values.

    Note: Model defaults are now primarily registered via the DefaultsRegistry
    pattern. Each model registers its own defaults in models/{model}/defaults.py.
    This class is maintained for backward compatibility.
    """

    # Legacy hardcoded defaults - kept for backward compatibility
    # New code should use DefaultsRegistry.get_model_defaults()
    _LEGACY_FUSE = {
        'FUSE_SPATIAL_MODE': 'lumped',
        'ROUTING_MODEL': 'none',
        'FUSE_INSTALL_PATH': 'default',
        'SETTINGS_FUSE_PATH': 'default',
        'SETTINGS_FUSE_FILEMANAGER': 'fm_catch.txt',
        'FUSE_EXE': 'fuse.exe',
        'EXPERIMENT_OUTPUT_FUSE': 'default',
        'SETTINGS_FUSE_PARAMS_TO_CALIBRATE': 'MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE',
    }

    _LEGACY_SUMMA = {
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
        'PARAMS_TO_CALIBRATE': 'albedo_max,albedo_min,canopy_capacity,slow_drainage',
        'BASIN_PARAMS_TO_CALIBRATE': 'routingGammaShape,routingGammaScale',
    }

    _LEGACY_GR = {
        'GR_MODEL_TYPE': 'GR4J',
        'GR_SPATIAL_MODE': 'lumped',
        'GR_EXE': 'GR.r',
        'GR_PARAMS_TO_CALIBRATE': 'X1,X2,X3,X4,CTG,Kf,Gratio,Albedo_diff',
    }

    _LEGACY_HYPE = {
        'SETTINGS_HYPE_PATH': 'default',
        'HYPE_INSTALL_PATH': 'default',
        'HYPE_EXE': 'hype',
        'SETTINGS_HYPE_CONTROL_FILE': 'info.txt',
        'HYPE_PARAMS_TO_CALIBRATE': 'ttmp,cmlt,cevp,lp,epotdist,rrcs1,rrcs2,rcgrw,rivvel,damp',
    }

    _LEGACY_NGEN = {
        'NGEN_MODULES_TO_CALIBRATE': 'CFE',
        'NGEN_CFE_PARAMS_TO_CALIBRATE': 'maxsmc,satdk,bb,slop',
        'NGEN_NOAH_PARAMS_TO_CALIBRATE': 'refkdt,slope,smcmax,dksat',
        'NGEN_PET_PARAMS_TO_CALIBRATE': 'wind_speed_measurement_height_m',
    }

    _LEGACY_MESH = {
        'MESH_PARAMS_TO_CALIBRATE': 'ZSNL,MANN,RCHARG,BASEFLW,DTMINUSR',
    }

    _LEGACY_MIZUROUTE = {
        'MIZUROUTE_PARAMS_TO_CALIBRATE': 'velo,diff',
    }

    _LEGACY_RHESSYS = {
        'RHESSYS_PARAMS_TO_CALIBRATE': 'sat_to_gw_coeff,gw_loss_coeff,m,Ksat_0,porosity_0,soil_depth,snow_melt_Tcoef',
    }

    _LEGACY_GNN = {
        'GNN_PARAMS_TO_CALIBRATE': 'precip_mult,temp_offset,routing_velocity',
    }

    # Mapping for backward compatibility attribute access
    FUSE = _LEGACY_FUSE
    SUMMA = _LEGACY_SUMMA
    GR = _LEGACY_GR
    HYPE = _LEGACY_HYPE
    NGEN = _LEGACY_NGEN
    MESH = _LEGACY_MESH
    MIZUROUTE = _LEGACY_MIZUROUTE
    RHESSYS = _LEGACY_RHESSYS
    GNN = _LEGACY_GNN

    @classmethod
    def _import_model_defaults(cls) -> None:
        """Import model defaults modules to trigger registration."""
        model_names = ['summa', 'fuse', 'gr', 'hype', 'ngen', 'mesh', 'mizuroute', 'rhessys', 'gnn']
        for model_name in model_names:
            try:
                __import__(f'symfluence.models.{model_name}', fromlist=['defaults'])
            except ImportError:
                pass  # Model not installed or defaults not available

    @classmethod
    def get_defaults_for_model(cls, model: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific model.

        This method checks the ModelRegistry for config adapters (new pattern),
        then DefaultsRegistry (legacy pattern), then hardcoded defaults (backward compat).

        Args:
            model: Model name (FUSE, SUMMA, GR, HYPE, etc.)

        Returns:
            Dict[str, Any]: Model-specific default configuration
        """
        # Import registries lazily to avoid circular imports
        from symfluence.core.config.defaults_registry import DefaultsRegistry
        from symfluence.models.registry import ModelRegistry

        # Ensure model defaults are imported
        cls._import_model_defaults()

        # 1. Try ModelRegistry config adapter first (NEW PATTERN)
        try:
            adapter_defaults = ModelRegistry.get_config_defaults(model)
            if adapter_defaults:
                logger.debug(f"Using ModelRegistry adapter defaults for model: {model}")
                return adapter_defaults
        except Exception as e:
            logger.debug(f"Could not get defaults from ModelRegistry for {model}: {e}")

        # 2. Try DefaultsRegistry (LEGACY PATTERN - still supported)
        registry_defaults = DefaultsRegistry.get_model_defaults(model)
        if registry_defaults:
            logger.debug(f"Using DefaultsRegistry defaults for model: {model}")
            return registry_defaults

        # 3. Fall back to legacy hardcoded defaults (BACKWARD COMPATIBILITY)
        legacy_attr = f'_LEGACY_{model.upper()}'
        legacy_defaults = getattr(cls, legacy_attr, None)
        if legacy_defaults:
            logger.debug(f"Using legacy hardcoded defaults for model: {model}")
            return legacy_defaults.copy()

        # 4. Final fallback to direct attribute (for backward compat)
        logger.debug(f"Using direct attribute defaults for model: {model}")
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
