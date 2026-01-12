"""
Default configuration values for RHESSys model.

This module registers RHESSys-specific defaults with the DefaultsRegistry,
keeping model configuration within the model directory.
"""

from symfluence.core.config.defaults_registry import DefaultsRegistry


@DefaultsRegistry.register_defaults('RHESSYS')
class RHESSysDefaults:
    """RHESSys model default configuration values."""

    RHESSYS_PARAMS_TO_CALIBRATE = 'sat_to_gw_coeff,gw_loss_coeff,m,Ksat_0,porosity_0,soil_depth,snow_melt_Tcoef'
