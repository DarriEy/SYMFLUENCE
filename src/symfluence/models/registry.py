"""Central registry for model components (preprocessors, runners, postprocessors).

Facade over ComponentRegistry, ConfigRegistry, and ResultExtractorRegistry.
Models self-register via decorators at import time; the workflow layer
queries by model name to discover and instantiate components.
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Type

from symfluence.core.registries import R
from symfluence.models.registries.component_registry import ComponentRegistry
from symfluence.models.registries.config_registry import ConfigRegistry
from symfluence.models.registries.result_registry import ResultExtractorRegistry

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Facade over ComponentRegistry, ConfigRegistry, and ResultExtractorRegistry.

    Register components via decorators (``@ModelRegistry.register_runner('SUMMA')``)
    and look them up by model name (``ModelRegistry.get_runner('SUMMA')``).
    All delegation methods are one-liner pass-throughs to the sub-registries.
    """

    # =========================================================================
    # Class-level attributes for backward compatibility
    # These are aliases to the underlying registry dictionaries
    # =========================================================================

    # Component registry attributes
    _preprocessors = ComponentRegistry._preprocessors
    _runners = ComponentRegistry._runners
    _postprocessors = ComponentRegistry._postprocessors
    _visualizers = ComponentRegistry._visualizers
    _runner_methods = ComponentRegistry._runner_methods

    # Config registry attributes
    _config_adapters = ConfigRegistry._config_adapters
    _config_schemas = ConfigRegistry._config_schemas
    _config_defaults = ConfigRegistry._config_defaults
    _config_transformers = ConfigRegistry._config_transformers
    _config_validators = ConfigRegistry._config_validators

    # Result extractor registry: state now lives in R.result_extractors

    # =========================================================================
    # Component Registration (Delegates to ComponentRegistry)
    # =========================================================================

    @classmethod
    def register_preprocessor(cls, model_name: str) -> Callable[[Type], Type]:
        """Register a preprocessor class for a model.
        """
        return ComponentRegistry.register_preprocessor(model_name)

    @classmethod
    def register_runner(
        cls, model_name: str, method_name: str = "run"
    ) -> Callable[[Type], Type]:
        """Register a runner class for a model.
        """
        return ComponentRegistry.register_runner(model_name, method_name)

    @classmethod
    def register_postprocessor(cls, model_name: str) -> Callable[[Type], Type]:
        """Register a postprocessor class for a model.
        """
        return ComponentRegistry.register_postprocessor(model_name)

    @classmethod
    def register_visualizer(cls, model_name: str) -> Callable[[Callable], Callable]:
        """Register a visualization function for a model.
        """
        return ComponentRegistry.register_visualizer(model_name)

    # =========================================================================
    # Component Retrieval (Delegates to ComponentRegistry)
    # =========================================================================

    @classmethod
    def get_preprocessor(cls, model_name: str) -> Optional[Type]:
        """Get preprocessor class for a model.
        """
        return R.preprocessors.get(model_name.upper())

    @classmethod
    def get_runner(cls, model_name: str) -> Optional[Type]:
        """Get runner class for a model.
        """
        return R.runners.get(model_name.upper())

    @classmethod
    def get_postprocessor(cls, model_name: str) -> Optional[Type]:
        """Get postprocessor class for a model.
        """
        return R.postprocessors.get(model_name.upper())

    @classmethod
    def get_visualizer(cls, model_name: str) -> Optional[Callable]:
        """Get visualizer function for a model.
        """
        return R.visualizers.get(model_name.upper())

    @classmethod
    def get_runner_method(cls, model_name: str) -> str:
        """Get the runner method name for a model.
        """
        return R.runners.meta(model_name.upper()).get("runner_method", "run")

    @classmethod
    def list_models(cls) -> list[str]:
        """List all models with registered components.
        """
        return R.registered_models()

    @classmethod
    def get_model_components(cls, model_name: str) -> Dict[str, Any]:
        """Get all registered component classes for a model.
        """
        return R.for_model(model_name)

    @classmethod
    def validate_model_registration(
        cls,
        model_name: str,
        require_all: bool = False
    ) -> Dict[str, Any]:
        """Validate that a model has all required components registered.
        """
        return R.validate_model(model_name)

    @classmethod
    def validate_all_models(
        cls,
        require_all: bool = False,
        logger: logging.Logger = None
    ) -> Dict[str, Dict[str, Any]]:
        """Validate registration status of all registered models.
        """
        return ComponentRegistry.validate_all_models(require_all, logger)

    # =========================================================================
    # Config Management Registration (Delegates to ConfigRegistry)
    # =========================================================================

    @classmethod
    def register_config_adapter(cls, model_name: str) -> Callable[[Type], Type]:
        """Register a complete config adapter for a model.
        """
        return ConfigRegistry.register_config_adapter(model_name)

    @classmethod
    def register_config_schema(cls, model_name: str, schema: Type) -> Type:
        """Register Pydantic config schema for a model.
        """
        return ConfigRegistry.register_config_schema(model_name, schema)

    @classmethod
    def register_config_defaults(
        cls, model_name: str, defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Register default configuration values for a model.
        """
        return ConfigRegistry.register_config_defaults(model_name, defaults)

    @classmethod
    def register_config_transformers(
        cls, model_name: str, transformers: Dict[str, Tuple[str, ...]]
    ) -> Dict[str, Tuple[str, ...]]:
        """Register flat-to-nested field transformers for a model.
        """
        return ConfigRegistry.register_config_transformers(model_name, transformers)

    @classmethod
    def register_config_validator(cls, model_name: str, validator: Callable) -> Callable:
        """Register custom validation function for a model.
        """
        return ConfigRegistry.register_config_validator(model_name, validator)

    # =========================================================================
    # Config Management Retrieval (Delegates to ConfigRegistry)
    # =========================================================================

    @classmethod
    def get_config_adapter(cls, model_name: str) -> Optional[Any]:
        """Get config adapter instance for a model.
        """
        adapter_cls = R.config_adapters.get(model_name.upper())
        return adapter_cls(model_name) if adapter_cls else None

    @classmethod
    def get_config_schema(cls, model_name: str) -> Optional[Type]:
        """Get Pydantic config schema for a model.
        """
        return ConfigRegistry.get_config_schema(model_name)

    @classmethod
    def get_config_defaults(cls, model_name: str) -> Dict[str, Any]:
        """Get default configuration for a model.
        """
        return ConfigRegistry.get_config_defaults(model_name)

    @classmethod
    def get_config_transformers(
        cls, model_name: str
    ) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested transformers for a model.
        """
        return ConfigRegistry.get_config_transformers(model_name)

    @classmethod
    def get_config_validator(cls, model_name: str) -> Optional[Callable]:
        """Get config validator function for a model.
        """
        return ConfigRegistry.get_config_validator(model_name)

    @classmethod
    def validate_model_config(cls, model_name: str, config: Dict[str, Any]) -> None:
        """Validate model configuration using registered validator.
        """
        ConfigRegistry.validate_model_config(model_name, config)

    # =========================================================================
    # Result Extraction Registry Methods (Delegates to ResultExtractorRegistry)
    # =========================================================================

    @classmethod
    def register_result_extractor(cls, model_name: str) -> Callable[[Type], Type]:
        """Register a result extractor for a model.
        """
        return ResultExtractorRegistry.register_result_extractor(model_name)

    @classmethod
    def get_result_extractor(cls, model_name: str) -> Optional[Any]:
        """Get result extractor instance for a model.
        """
        extractor_cls = R.result_extractors.get(model_name.upper())
        return extractor_cls(model_name) if extractor_cls else None

    @classmethod
    def has_result_extractor(cls, model_name: str) -> bool:
        """Check if a model has a registered result extractor.
        """
        return model_name.upper() in R.result_extractors

    @classmethod
    def list_result_extractors(cls) -> list[str]:
        """List all models with registered result extractors.
        """
        return R.result_extractors.keys()

    # =========================================================================
    # Forcing Adapter Registry Methods (Delegates to ForcingAdapterRegistry)
    # =========================================================================

    @classmethod
    def _ensure_forcing_adapters_loaded(cls) -> None:
        """Trigger lazy import of forcing adapter modules."""
        from symfluence.models.adapters.adapter_registry import ForcingAdapterRegistry
        ForcingAdapterRegistry._import_adapters()

    @classmethod
    def get_forcing_adapter(cls, model_name: str, config: Dict, logger=None) -> Optional[Any]:
        """Get forcing adapter instance for a model.

        Args:
            model_name: Model name
            config: Configuration dictionary
            logger: Optional logger instance

        Returns:
            ForcingAdapter instance or None if not registered
        """
        cls._ensure_forcing_adapters_loaded()
        adapter_cls = R.forcing_adapters.get(model_name.upper())
        return adapter_cls(config, logger) if adapter_cls else None

    @classmethod
    def has_forcing_adapter(cls, model_name: str) -> bool:
        """Check if a model has a registered forcing adapter.

        Args:
            model_name: Model name

        Returns:
            bool: True if adapter is registered
        """
        cls._ensure_forcing_adapters_loaded()
        return model_name.upper() in R.forcing_adapters

    @classmethod
    def list_forcing_adapters(cls) -> list[str]:
        """List all models with registered forcing adapters.

        Returns:
            List of model names with forcing adapters
        """
        cls._ensure_forcing_adapters_loaded()
        return R.forcing_adapters.keys()
