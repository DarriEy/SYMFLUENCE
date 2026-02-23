"""
Forcing Adapter Registry for SYMFLUENCE.

This module provides a registry for model forcing adapters,
enabling dynamic discovery and instantiation without hardcoded model names.
"""

import logging
import warnings
from typing import Any, Callable, Dict, Optional, Type

from symfluence.core.registries import R

from .base_adapter import ForcingAdapter

logger = logging.getLogger(__name__)


class ForcingAdapterRegistry:
    """
    Registry for model forcing adapters.

    Adapters register themselves using the @register_adapter decorator,
    enabling dynamic discovery without hardcoded model names.

    Example:
        >>> @ForcingAdapterRegistry.register_adapter('SUMMA')
        >>> class SUMMAForcingAdapter(ForcingAdapter):
        ...     pass
        ...
        >>> adapter = ForcingAdapterRegistry.get_adapter('SUMMA', config)
    """

    @classmethod
    def register_adapter(cls, model_name: str) -> Callable[[Type[ForcingAdapter]], Type[ForcingAdapter]]:
        """
        Decorator to register a forcing adapter.

        Args:
            model_name: Model name (e.g., 'SUMMA', 'HYPE')

        Returns:
            Decorator function

        Example:
            >>> @ForcingAdapterRegistry.register_adapter('SUMMA')
            >>> class SUMMAForcingAdapter(ForcingAdapter):
            ...     pass
        """
        def decorator(adapter_cls: Type[ForcingAdapter]) -> Type[ForcingAdapter]:
            warnings.warn(
                "ForcingAdapterRegistry.register_adapter() is deprecated; "
                "use R.forcing_adapters.add() or model_manifest() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            R.forcing_adapters.add(model_name, adapter_cls)
            logger.debug(f"Registered forcing adapter for model: {model_name}")
            return adapter_cls
        return decorator

    @classmethod
    def get_adapter(
        cls,
        model_name: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> ForcingAdapter:
        """
        Get an adapter instance for a model.

        Args:
            model_name: Model name (e.g., 'SUMMA', 'HYPE')
            config: Configuration dictionary
            logger: Optional logger instance

        Returns:
            Instantiated adapter for the model

        Raises:
            ValueError: If no adapter is registered for the model
        """
        # Ensure adapters are imported
        cls._import_adapters()

        key = model_name.upper()
        adapter_cls = R.forcing_adapters.get(key)

        if adapter_cls is None:
            available = R.forcing_adapters.keys()
            raise ValueError(
                f"No forcing adapter registered for model '{model_name}'. "
                f"Available adapters: {available}"
            )

        return adapter_cls(config, logger)

    @classmethod
    def get_adapter_class(cls, model_name: str) -> Optional[Type[ForcingAdapter]]:
        """
        Get the adapter class for a model (without instantiating).

        Args:
            model_name: Model name

        Returns:
            Adapter class, or None if not registered
        """
        cls._import_adapters()
        return R.forcing_adapters.get(model_name.upper())

    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """
        Check if an adapter is registered for a model.

        Args:
            model_name: Model name

        Returns:
            True if adapter is registered
        """
        cls._import_adapters()
        return model_name.upper() in R.forcing_adapters

    @classmethod
    def get_registered_models(cls) -> list:
        """
        Get list of models with registered adapters.

        Returns:
            List of model names
        """
        cls._import_adapters()
        return R.forcing_adapters.keys()

    @classmethod
    def _import_adapters(cls) -> None:
        """
        Import model adapter modules to trigger registration.

        This method attempts to import the forcing_adapter module
        from each known model package.
        """
        from symfluence.core.constants import SupportedModels

        for model_name in SupportedModels.WITH_FORCING_ADAPTER:
            try:
                __import__(
                    f'symfluence.models.{model_name}.forcing_adapter',
                    fromlist=['forcing_adapter']
                )
            except ImportError:
                logging.getLogger(__name__).debug(
                    f"Forcing adapter for '{model_name}' not available"
                )


def transform_cfif_to_model(
    cfif_data,
    model_name: str,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
):
    """
    Convenience function to transform CFIF data to model format.

    Args:
        cfif_data: xarray Dataset in CFIF format
        model_name: Target model name
        config: Configuration dictionary
        logger: Optional logger

    Returns:
        xarray Dataset in model-specific format

    Example:
        >>> from symfluence.models.adapters import transform_cfif_to_model
        >>> summa_forcing = transform_cfif_to_model(cfif_data, 'SUMMA', config)
    """
    adapter = ForcingAdapterRegistry.get_adapter(model_name, config, logger)
    return adapter.transform(cfif_data)
