"""
Default configuration values for mizuRoute model.

This module registers mizuRoute-specific defaults with the DefaultsRegistry,
keeping model configuration within the model directory.
"""

from symfluence.core.config.defaults_registry import DefaultsRegistry


@DefaultsRegistry.register_defaults('MIZUROUTE')
class MizuRouteDefaults:
    """mizuRoute model default configuration values."""

    MIZUROUTE_PARAMS_TO_CALIBRATE = 'velo,diff'
