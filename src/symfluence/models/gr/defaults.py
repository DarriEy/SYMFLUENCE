"""
Default configuration values for GR model.

This module registers GR-specific defaults with the DefaultsRegistry,
keeping model configuration within the model directory.
"""

from symfluence.core.config.defaults_registry import DefaultsRegistry


@DefaultsRegistry.register_defaults('GR')
class GRDefaults:
    """GR model default configuration values."""

    GR_MODEL_TYPE = 'GR4J'
    GR_SPATIAL_MODE = 'lumped'
    GR_EXE = 'GR.r'
    GR_PARAMS_TO_CALIBRATE = 'X1,X2,X3,X4,CTG,Kf,Gratio,Albedo_diff'
