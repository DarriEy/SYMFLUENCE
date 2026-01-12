"""
Default configuration values for MESH model.

This module registers MESH-specific defaults with the DefaultsRegistry,
keeping model configuration within the model directory.
"""

from symfluence.core.config.defaults_registry import DefaultsRegistry


@DefaultsRegistry.register_defaults('MESH')
class MESHDefaults:
    """MESH model default configuration values."""

    MESH_PARAMS_TO_CALIBRATE = 'ZSNL,MANN,RCHARG,BASEFLW,DTMINUSR'
