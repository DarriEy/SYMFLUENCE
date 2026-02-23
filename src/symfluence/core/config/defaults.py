"""
Default configuration values for SYMFLUENCE.

All optional-field defaults are defined as Pydantic ``Field(default=...)``
declarations in the nested config models (``core.config.models.*``).  Flat
and nested config formats both resolve to the same Pydantic defaults — there
is a single source of truth.

This module retains two helper classes used during config construction:
- ``ModelDefaults``: forwards to ModelRegistry for model-specific defaults
- ``ForcingDefaults``: dataset-conditional overrides (e.g. ERA5 → cloud)
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ModelDefaults:
    """Compatibility shim that forwards to ModelRegistry for model defaults."""

    # Legacy attributes for backward compatibility
    SUMMA: Dict[str, Any] = {}
    FUSE: Dict[str, Any] = {}

    @classmethod
    def get_defaults_for_model(cls, model: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific model.

        Uses ModelRegistry as the single source of truth. All defaults are
        auto-generated from Pydantic Field declarations in model config schemas.

        Args:
            model: Model name (FUSE, SUMMA, GR, HYPE, etc.)

        Returns:
            Dict[str, Any]: Model-specific default configuration

        Example:
            >>> defaults = ModelDefaults.get_defaults_for_model('SUMMA')
            >>> defaults['SUMMA_EXE']
            'summa_sundials.exe'
        """
        from symfluence.models.registries.config_registry import ConfigRegistry

        # Ensure model modules are imported so config adapters are registered.
        # Importing symfluence.models triggers all model __init__.py files
        # which register their adapters with the unified registry via decorators.
        try:
            import symfluence.models  # noqa: F401
        except (ImportError, OSError):
            pass

        # Get defaults via ConfigRegistry which checks adapters first, then R.*
        defaults = ConfigRegistry.get_config_defaults(model.upper()) or {}

        if not defaults:
            logger.warning(
                f"No defaults found for model '{model}'. "
                f"Ensure {model}ConfigAdapter is registered with ModelRegistry."
            )
            return {}

        logger.debug(f"Retrieved {len(defaults)} defaults for {model} from ModelRegistry")
        return defaults


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
