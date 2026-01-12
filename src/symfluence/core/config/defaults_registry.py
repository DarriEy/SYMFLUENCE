"""
Registry for model-specific default configurations.

This module provides a centralized registry for model defaults,
allowing each model to register its own defaults without hardcoding
in the core configuration module.
"""

from typing import Dict, Any, Type, Callable
import logging

logger = logging.getLogger(__name__)


class DefaultsRegistry:
    """
    Registry for model-specific default configurations.

    Models register their defaults using the @register_defaults decorator,
    enabling dynamic discovery without hardcoded model names in core code.

    Example:
        @DefaultsRegistry.register_defaults('SUMMA')
        class SUMMADefaults:
            ROUTING_MODEL = 'mizuRoute'
            SUMMA_INSTALL_PATH = 'default'
            ...
    """

    _model_defaults: Dict[str, Type] = {}
    _forcing_defaults: Dict[str, Type] = {}

    @classmethod
    def register_defaults(cls, model_name: str) -> Callable[[Type], Type]:
        """
        Decorator to register model defaults.

        Args:
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE')

        Returns:
            Decorator function

        Example:
            @DefaultsRegistry.register_defaults('SUMMA')
            class SUMMADefaults:
                ROUTING_MODEL = 'mizuRoute'
        """
        def decorator(defaults_cls: Type) -> Type:
            cls._model_defaults[model_name.upper()] = defaults_cls
            logger.debug(f"Registered defaults for model: {model_name}")
            return defaults_cls
        return decorator

    @classmethod
    def register_forcing_defaults(cls, forcing_name: str) -> Callable[[Type], Type]:
        """
        Decorator to register forcing dataset defaults.

        Args:
            forcing_name: Name of the forcing dataset (e.g., 'ERA5', 'CONUS404')

        Returns:
            Decorator function
        """
        def decorator(defaults_cls: Type) -> Type:
            cls._forcing_defaults[forcing_name.upper()] = defaults_cls
            logger.debug(f"Registered defaults for forcing: {forcing_name}")
            return defaults_cls
        return decorator

    @classmethod
    def get_model_defaults(cls, model_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific model.

        Args:
            model_name: Model name (e.g., 'SUMMA', 'FUSE')

        Returns:
            Dictionary of model-specific default configuration values
        """
        model_key = model_name.upper()
        defaults_cls = cls._model_defaults.get(model_key)

        if defaults_cls is None:
            logger.debug(f"No registered defaults for model: {model_name}")
            return {}

        # Extract class attributes as defaults
        return {
            k: v for k, v in vars(defaults_cls).items()
            if not k.startswith('_') and k.isupper()
        }

    @classmethod
    def get_forcing_defaults(cls, forcing_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific forcing dataset.

        Args:
            forcing_name: Forcing dataset name (e.g., 'ERA5', 'CONUS404')

        Returns:
            Dictionary of forcing-specific default configuration values
        """
        forcing_key = forcing_name.upper()
        defaults_cls = cls._forcing_defaults.get(forcing_key)

        if defaults_cls is None:
            logger.debug(f"No registered defaults for forcing: {forcing_name}")
            return {}

        return {
            k: v for k, v in vars(defaults_cls).items()
            if not k.startswith('_') and k.isupper()
        }

    @classmethod
    def get_registered_models(cls) -> list:
        """Get list of models with registered defaults."""
        return list(cls._model_defaults.keys())

    @classmethod
    def get_registered_forcings(cls) -> list:
        """Get list of forcing datasets with registered defaults."""
        return list(cls._forcing_defaults.keys())

    @classmethod
    def get_default(cls, model_name: str, key: str, fallback: Any = None) -> Any:
        """
        Get a single default value for a model.

        Args:
            model_name: Model name
            key: Configuration key name
            fallback: Value to return if key not found

        Returns:
            Default value for the key, or fallback if not found
        """
        defaults = cls.get_model_defaults(model_name)
        return defaults.get(key, fallback)
