"""
SUMMA Model Configuration.

Provides configuration schema, defaults, transformers, and validation
for the Structure for Unifying Multiple Modeling Alternatives (SUMMA).

This module registers SUMMA-specific configuration components with the
ModelRegistry, enabling the core config system to remain model-agnostic.
"""

from typing import Dict, Any, Tuple
from symfluence.models.base import ModelConfigAdapter, ConfigValidationError
from symfluence.core.config.models.model_configs import SUMMAConfig
from symfluence.core.config.defaults_registry import DefaultsRegistry


# ============================================================================
# SUMMA Default Configuration Values
# ============================================================================

@DefaultsRegistry.register_defaults('SUMMA')
class SUMMADefaults:
    """Default configuration values for SUMMA model."""

    # Core settings
    ROUTING_MODEL = 'mizuRoute'
    SUMMA_INSTALL_PATH = 'default'
    SETTINGS_SUMMA_PATH = 'default'
    SUMMA_EXE = 'summa_sundials.exe'

    # File names
    SETTINGS_SUMMA_FILEMANAGER = 'fileManager.txt'
    SETTINGS_SUMMA_FORCING_LIST = 'forcingFileList.txt'
    SETTINGS_SUMMA_COLDSTATE = 'coldState.nc'
    SETTINGS_SUMMA_TRIALPARAMS = 'trialParams.nc'
    SETTINGS_SUMMA_ATTRIBUTES = 'attributes.nc'
    SETTINGS_SUMMA_OUTPUT = 'outputControl.txt'
    SETTINGS_SUMMA_BASIN_PARAMS_FILE = 'basinParamInfo.txt'
    SETTINGS_SUMMA_LOCAL_PARAMS_FILE = 'localParamInfo.txt'

    # HRU connectivity
    SETTINGS_SUMMA_CONNECT_HRUS = True

    # Trial parameters
    SETTINGS_SUMMA_TRIALPARAM_N = 0
    SETTINGS_SUMMA_TRIALPARAM_1 = None

    # Parallel execution
    SETTINGS_SUMMA_USE_PARALLEL_SUMMA = False
    SETTINGS_SUMMA_CPUS_PER_TASK = 32
    SETTINGS_SUMMA_TIME_LIMIT = '01:00:00'
    SETTINGS_SUMMA_MEM = '5G'
    SETTINGS_SUMMA_GRU_COUNT = 85
    SETTINGS_SUMMA_GRU_PER_JOB = 5
    SETTINGS_SUMMA_PARALLEL_PATH = 'default'
    SETTINGS_SUMMA_PARALLEL_EXE = 'summa_actors.exe'

    # Outputs
    EXPERIMENT_OUTPUT_SUMMA = 'default'
    EXPERIMENT_LOG_SUMMA = 'default'

    # Calibration parameters
    PARAMS_TO_CALIBRATE = 'albedo_max,albedo_min,canopy_capacity,slow_drainage'
    BASIN_PARAMS_TO_CALIBRATE = 'routingGammaShape,routingGammaScale'

    # Depth calibration
    CALIBRATE_DEPTH = False
    DEPTH_TOTAL_MULT_BOUNDS = None
    DEPTH_SHAPE_FACTOR_BOUNDS = None

    # Glacier mode
    SETTINGS_SUMMA_GLACIER_MODE = False
    SETTINGS_SUMMA_GLACIER_ATTRIBUTES = 'attributes_glac.nc'
    SETTINGS_SUMMA_GLACIER_COLDSTATE = 'coldState_glac.nc'

    # Execution timeout
    SUMMA_TIMEOUT = 7200  # seconds

    # mizuRoute defaults (when using SUMMA + mizuRoute)
    INSTALL_PATH_MIZUROUTE = 'default'
    EXE_NAME_MIZUROUTE = 'mizuroute.exe'
    SETTINGS_MIZU_PATH = 'default'
    EXPERIMENT_OUTPUT_MIZUROUTE = 'default'


# ============================================================================
# SUMMA Field Transformers (Flat to Nested Mapping)
# ============================================================================

SUMMA_FIELD_TRANSFORMERS: Dict[str, Tuple[str, ...]] = {
    # Core SUMMA settings
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

    # Parallel execution
    'SETTINGS_SUMMA_USE_PARALLEL_SUMMA': ('model', 'summa', 'use_parallel'),
    'SETTINGS_SUMMA_CPUS_PER_TASK': ('model', 'summa', 'cpus_per_task'),
    'SETTINGS_SUMMA_TIME_LIMIT': ('model', 'summa', 'time_limit'),
    'SETTINGS_SUMMA_MEM': ('model', 'summa', 'mem'),
    'SETTINGS_SUMMA_GRU_COUNT': ('model', 'summa', 'gru_count'),
    'SETTINGS_SUMMA_GRU_PER_JOB': ('model', 'summa', 'gru_per_job'),
    'SETTINGS_SUMMA_PARALLEL_PATH': ('model', 'summa', 'parallel_path'),
    'SETTINGS_SUMMA_PARALLEL_EXE': ('model', 'summa', 'parallel_exe'),

    # Experiment outputs
    'EXPERIMENT_OUTPUT_SUMMA': ('model', 'summa', 'experiment_output'),
    'EXPERIMENT_LOG_SUMMA': ('model', 'summa', 'experiment_log'),

    # Calibration
    'PARAMS_TO_CALIBRATE': ('model', 'summa', 'params_to_calibrate'),
    'BASIN_PARAMS_TO_CALIBRATE': ('model', 'summa', 'basin_params_to_calibrate'),
    'SUMMA_DECISION_OPTIONS': ('model', 'summa', 'decision_options'),
    'CALIBRATE_DEPTH': ('model', 'summa', 'calibrate_depth'),
    'DEPTH_TOTAL_MULT_BOUNDS': ('model', 'summa', 'depth_total_mult_bounds'),
    'DEPTH_SHAPE_FACTOR_BOUNDS': ('model', 'summa', 'depth_shape_factor_bounds'),

    # Glacier mode
    'SETTINGS_SUMMA_GLACIER_MODE': ('model', 'summa', 'glacier_mode'),
    'SETTINGS_SUMMA_GLACIER_ATTRIBUTES': ('model', 'summa', 'glacier_attributes'),
    'SETTINGS_SUMMA_GLACIER_COLDSTATE': ('model', 'summa', 'glacier_coldstate'),

    # Execution
    'SUMMA_TIMEOUT': ('model', 'summa', 'timeout'),
}


# ============================================================================
# SUMMA Config Adapter
# ============================================================================

class SUMMAConfigAdapter(ModelConfigAdapter):
    """
    Configuration adapter for SUMMA model.

    Provides schema, defaults, transformers, and validation for SUMMA-specific
    configuration. Registered with ModelRegistry to enable core config system
    to be model-agnostic.

    Example:
        >>> from symfluence.models.registry import ModelRegistry
        >>> adapter = ModelRegistry.get_config_adapter('SUMMA')
        >>> defaults = adapter.get_defaults()
        >>> schema = adapter.get_config_schema()
    """

    def __init__(self, model_name: str = 'SUMMA'):
        super().__init__(model_name)

    def get_config_schema(self):
        """Get Pydantic model class for SUMMA configuration."""
        return SUMMAConfig

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values for SUMMA."""
        return {
            k: v for k, v in vars(SUMMADefaults).items()
            if not k.startswith('_') and k.isupper()
        }

    def get_field_transformers(self) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested field transformers for SUMMA."""
        return SUMMA_FIELD_TRANSFORMERS

    def validate(self, config: Dict[str, Any]) -> None:
        """
        Validate SUMMA-specific configuration.

        Args:
            config: Configuration dictionary (flat format with uppercase keys)

        Raises:
            ConfigValidationError: If configuration is invalid
        """
        # Check required fields
        required_fields = self.get_required_keys()
        missing_fields = []

        for field in required_fields:
            value = config.get(field)
            if value is None or value == '' or value == 'None':
                missing_fields.append(field)

        if missing_fields:
            raise ConfigValidationError(
                f"SUMMA configuration incomplete. Missing required fields:\n"
                + "\n".join(f"  • {field}" for field in missing_fields)
            )

        # Validate mizuRoute requirements if routing is enabled
        routing_model = config.get('ROUTING_MODEL', '').upper()
        if routing_model == 'MIZUROUTE':
            mizu_required = ['INSTALL_PATH_MIZUROUTE', 'EXE_NAME_MIZUROUTE']
            mizu_missing = [
                f for f in mizu_required
                if not config.get(f) or config.get(f) in ['', 'None']
            ]
            if mizu_missing:
                raise ConfigValidationError(
                    f"mizuRoute routing enabled but configuration incomplete:\n"
                    + "\n".join(f"  • {field}" for field in mizu_missing)
                )

        # Validate parallel SUMMA requirements
        if config.get('SETTINGS_SUMMA_USE_PARALLEL_SUMMA'):
            parallel_required = {
                'SETTINGS_SUMMA_GRU_COUNT': config.get('SETTINGS_SUMMA_GRU_COUNT'),
                'SETTINGS_SUMMA_GRU_PER_JOB': config.get('SETTINGS_SUMMA_GRU_PER_JOB'),
            }
            parallel_missing = [
                field for field, value in parallel_required.items()
                if value is None or value == 0
            ]
            if parallel_missing:
                raise ConfigValidationError(
                    f"Parallel SUMMA enabled but configuration incomplete:\n"
                    + "\n".join(f"  • {field}" for field in parallel_missing)
                )

        # Validate glacier mode requirements
        if config.get('SETTINGS_SUMMA_GLACIER_MODE'):
            glacier_files = {
                'SETTINGS_SUMMA_GLACIER_ATTRIBUTES': config.get('SETTINGS_SUMMA_GLACIER_ATTRIBUTES'),
                'SETTINGS_SUMMA_GLACIER_COLDSTATE': config.get('SETTINGS_SUMMA_GLACIER_COLDSTATE'),
            }
            glacier_missing = [
                field for field, value in glacier_files.items()
                if not value or value in ['', 'None', 'default']
            ]
            if glacier_missing:
                raise ConfigValidationError(
                    f"Glacier mode enabled but configuration incomplete:\n"
                    + "\n".join(f"  • {field}" for field in glacier_missing)
                )

    def get_required_keys(self) -> list:
        """Get list of required configuration keys for SUMMA."""
        return [
            'SUMMA_EXE',
            'SETTINGS_SUMMA_PATH',
        ]

    def get_conditional_requirements(self) -> Dict[str, Dict[str, list]]:
        """Get conditional requirements based on other config values."""
        return {
            'ROUTING_MODEL': {
                'mizuRoute': ['INSTALL_PATH_MIZUROUTE', 'EXE_NAME_MIZUROUTE'],
                'MIZUROUTE': ['INSTALL_PATH_MIZUROUTE', 'EXE_NAME_MIZUROUTE'],
                'none': [],
            },
            'SETTINGS_SUMMA_USE_PARALLEL_SUMMA': {
                True: ['SETTINGS_SUMMA_GRU_COUNT', 'SETTINGS_SUMMA_GRU_PER_JOB'],
                False: [],
            },
            'SETTINGS_SUMMA_GLACIER_MODE': {
                True: ['SETTINGS_SUMMA_GLACIER_ATTRIBUTES', 'SETTINGS_SUMMA_GLACIER_COLDSTATE'],
                False: [],
            },
        }
