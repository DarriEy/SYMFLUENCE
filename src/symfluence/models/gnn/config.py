"""
GNN Model Configuration.

Provides configuration schema, defaults, transformers, and validation
for the Graph Neural Network (GNN) model.

This module registers GNN-specific configuration components with the
ModelRegistry, enabling the core config system to remain model-agnostic.
"""

from typing import Dict, Any, Tuple
from symfluence.models.base import ModelConfigAdapter, ConfigValidationError
from symfluence.core.config.defaults_registry import DefaultsRegistry


# ============================================================================
# GNN Default Configuration Values
# ============================================================================

@DefaultsRegistry.register_defaults('GNN')
class GNNDefaults:
    """Default configuration values for GNN model."""

    # Calibration parameters
    GNN_PARAMS_TO_CALIBRATE = 'precip_mult,temp_offset,routing_velocity'


# ============================================================================
# GNN Field Transformers (Flat to Nested Mapping)
# ============================================================================

GNN_FIELD_TRANSFORMERS: Dict[str, Tuple[str, ...]] = {
    'GNN_PARAMS_TO_CALIBRATE': ('model', 'gnn', 'params_to_calibrate'),
}


# ============================================================================
# GNN Config Adapter
# ============================================================================

class GNNConfigAdapter(ModelConfigAdapter):
    """Configuration adapter for GNN model."""

    def __init__(self, model_name: str = 'GNN'):
        super().__init__(model_name)

    def get_config_schema(self):
        """Get Pydantic model class for GNN configuration."""
        # GNN doesn't have a config schema yet in model_configs.py
        # Return None for now, can be added later
        return None

    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values for GNN."""
        return {
            k: v for k, v in vars(GNNDefaults).items()
            if not k.startswith('_') and k.isupper()
        }

    def get_field_transformers(self) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested field transformers for GNN."""
        return GNN_FIELD_TRANSFORMERS

    def validate(self, config: Dict[str, Any]) -> None:
        """Validate GNN-specific configuration."""
        # Minimal validation for GNN
        pass

    def get_required_keys(self) -> list:
        """Get list of required configuration keys for GNN."""
        return []
