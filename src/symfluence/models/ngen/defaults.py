"""
Default configuration values for NGen model.

This module registers NGen-specific defaults with the DefaultsRegistry,
keeping model configuration within the model directory.
"""

from symfluence.core.config.defaults_registry import DefaultsRegistry


@DefaultsRegistry.register_defaults('NGEN')
class NGENDefaults:
    """NGen model default configuration values."""

    NGEN_MODULES_TO_CALIBRATE = 'CFE'
    NGEN_CFE_PARAMS_TO_CALIBRATE = 'maxsmc,satdk,bb,slop'
    NGEN_NOAH_PARAMS_TO_CALIBRATE = 'refkdt,slope,smcmax,dksat'
    NGEN_PET_PARAMS_TO_CALIBRATE = 'wind_speed_measurement_height_m'
