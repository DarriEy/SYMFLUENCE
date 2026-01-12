"""
Default configuration values for SUMMA model.

This module registers SUMMA-specific defaults with the DefaultsRegistry,
keeping model configuration within the model directory.
"""

from symfluence.core.config.defaults_registry import DefaultsRegistry


@DefaultsRegistry.register_defaults('SUMMA')
class SUMMADefaults:
    """SUMMA model default configuration values."""

    ROUTING_MODEL = 'mizuRoute'
    SUMMA_INSTALL_PATH = 'default'
    SETTINGS_SUMMA_PATH = 'default'
    SETTINGS_SUMMA_FILEMANAGER = 'fileManager.txt'
    SUMMA_EXE = 'summa_sundials.exe'
    SETTINGS_SUMMA_CONNECT_HRUS = 'yes'
    SETTINGS_SUMMA_USE_PARALLEL_SUMMA = False
    EXPERIMENT_OUTPUT_SUMMA = 'default'
    INSTALL_PATH_MIZUROUTE = 'default'
    EXE_NAME_MIZUROUTE = 'mizuroute.exe'
    SETTINGS_MIZU_PATH = 'default'
    EXPERIMENT_OUTPUT_MIZUROUTE = 'default'
    PARAMS_TO_CALIBRATE = 'albedo_max,albedo_min,canopy_capacity,slow_drainage'
    BASIN_PARAMS_TO_CALIBRATE = 'routingGammaShape,routingGammaScale'
