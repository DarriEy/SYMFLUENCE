"""
Default configuration values for GNN model.

This module registers GNN-specific defaults with the DefaultsRegistry,
keeping model configuration within the model directory.
"""

from symfluence.core.config.defaults_registry import DefaultsRegistry


@DefaultsRegistry.register_defaults('GNN')
class GNNDefaults:
    """GNN model default configuration values."""

    GNN_PARAMS_TO_CALIBRATE = 'precip_mult,temp_offset,routing_velocity'
