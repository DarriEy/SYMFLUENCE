"""Config Registry  (Phase 4 delegation shim)

Registry for model configuration management including config adapters, schemas,
defaults, transformers, and validators. Enables model-specific configuration
handling while maintaining a unified interface.

Configuration Types:
    - Config Adapters: Complete config handler with schema, defaults, transformers
    - Config Schemas: Pydantic models for validation
    - Config Defaults: Default values for model parameters
    - Config Transformers: Flat-to-nested field path mappings
    - Config Validators: Custom validation functions

Registration Pattern:
    Models can register complete adapters or individual components:

    >>> @ConfigRegistry.register_config_adapter('SUMMA')
    ... class SUMMAConfigAdapter(ModelConfigAdapter):
    ...     def get_config_schema(self):
    ...         return SUMMAConfig

    Or register components directly:

    >>> ConfigRegistry.register_config_schema('SUMMA', SUMMAConfig)
    >>> ConfigRegistry.register_config_defaults('SUMMA', {...})

.. deprecated::
    This registry is a thin delegation shim around
    :pydata:`symfluence.core.registries.R`.  Prefer ``R.config_adapters``,
    ``R.config_schemas``, etc. directly.
"""

import logging
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type

from symfluence.core.registries import R
from symfluence.core.registry import _RegistryProxy

logger = logging.getLogger(__name__)


class ConfigRegistry:
    """Registry for model configuration management.

    Provides centralized storage and retrieval of configuration-related
    components for hydrological models. Supports both complete config adapters
    (which bundle schema, defaults, transformers, and validation) and
    individual component registration.

    Attributes:
        _config_adapters: Dict[model_name] -> adapter_class
        _config_schemas: Dict[model_name] -> Pydantic schema class
        _config_defaults: Dict[model_name] -> defaults dict
        _config_transformers: Dict[model_name] -> transformers dict
        _config_validators: Dict[model_name] -> validator function

    Example Adapter Registration::

        @ConfigRegistry.register_config_adapter('SUMMA')
        class SUMMAConfigAdapter(ModelConfigAdapter):
            def get_config_schema(self):
                return SUMMAConfig

            def get_defaults(self):
                return {'timestep': 3600}

    Example Direct Registration::

        ConfigRegistry.register_config_schema('SUMMA', SUMMAConfig)
        ConfigRegistry.register_config_defaults('SUMMA', {'timestep': 3600})

    .. deprecated::
        Use ``R.config_adapters``, ``R.config_schemas``,
        ``R.config_defaults``, ``R.config_transformers``,
        ``R.config_validators`` from :mod:`symfluence.core.registries` instead.
    """

    # Backward-compat proxies: read-only views into R.* so that code
    # accessing e.g. ``ConfigRegistry._config_adapters`` still works.
    _config_adapters: Dict[str, Type] = _RegistryProxy(R.config_adapters)
    _config_schemas: Dict[str, Type] = _RegistryProxy(R.config_schemas)
    _config_defaults: Dict[str, Dict[str, Any]] = _RegistryProxy(R.config_defaults)
    _config_transformers: Dict[str, Dict[str, Tuple[str, ...]]] = _RegistryProxy(R.config_transformers)
    _config_validators: Dict[str, Callable] = _RegistryProxy(R.config_validators)

    @classmethod
    def register_config_adapter(cls, model_name: str) -> Callable[[Type], Type]:
        """Register a complete config adapter for a model.

        The adapter provides schema, defaults, transformers, and validation.

        Args:
            model_name: Model name (e.g., 'SUMMA', 'FUSE')

        Returns:
            Decorator function that registers the adapter class

        Example:
            >>> @ConfigRegistry.register_config_adapter('SUMMA')
            ... class SUMMAConfigAdapter(ModelConfigAdapter):
            ...     def get_config_schema(self):
            ...         return SUMMAConfig

        .. deprecated::
            Use ``R.config_adapters.add()`` or ``model_manifest()`` instead.
        """
        def decorator(adapter_cls: Type) -> Type:
            warnings.warn(
                "ConfigRegistry.register_config_adapter() is deprecated; "
                "use R.config_adapters.add() or model_manifest() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            R.config_adapters.add(model_name, adapter_cls)
            return adapter_cls
        return decorator

    @classmethod
    def register_config_schema(cls, model_name: str, schema: Type) -> Type:
        """Register Pydantic config schema for a model.

        Args:
            model_name: Model name
            schema: Pydantic BaseModel class

        Returns:
            The registered schema class

        .. deprecated::
            Use ``R.config_schemas.add()`` or ``model_manifest()`` instead.
        """
        warnings.warn(
            "ConfigRegistry.register_config_schema() is deprecated; "
            "use R.config_schemas.add() or model_manifest() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        R.config_schemas.add(model_name, schema)
        return schema

    @classmethod
    def register_config_defaults(
        cls, model_name: str, defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Register default configuration values for a model.

        Args:
            model_name: Model name
            defaults: Dictionary of default values

        Returns:
            The registered defaults dictionary

        .. deprecated::
            Use ``R.config_defaults.add()`` or ``model_manifest()`` instead.
        """
        warnings.warn(
            "ConfigRegistry.register_config_defaults() is deprecated; "
            "use R.config_defaults.add() or model_manifest() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        R.config_defaults.add(model_name, defaults)
        return defaults

    @classmethod
    def register_config_transformers(
        cls, model_name: str, transformers: Dict[str, Tuple[str, ...]]
    ) -> Dict[str, Tuple[str, ...]]:
        """Register flat-to-nested field transformers for a model.

        Args:
            model_name: Model name
            transformers: Dictionary mapping flat keys to nested paths

        Returns:
            The registered transformers dictionary

        .. deprecated::
            Use ``R.config_transformers.add()`` or ``model_manifest()`` instead.
        """
        warnings.warn(
            "ConfigRegistry.register_config_transformers() is deprecated; "
            "use R.config_transformers.add() or model_manifest() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        R.config_transformers.add(model_name, transformers)
        return transformers

    @classmethod
    def register_config_validator(
        cls, model_name: str, validator: Callable
    ) -> Callable:
        """Register custom validation function for a model.

        Args:
            model_name: Model name
            validator: Callable that takes config dict and raises on validation error

        Returns:
            The registered validator function

        .. deprecated::
            Use ``R.config_validators.add()`` or ``model_manifest()`` instead.
        """
        warnings.warn(
            "ConfigRegistry.register_config_validator() is deprecated; "
            "use R.config_validators.add() or model_manifest() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        R.config_validators.add(model_name, validator)
        return validator

    @classmethod
    def get_config_adapter(cls, model_name: str) -> Optional[Any]:
        """Get config adapter instance for a model.

        Args:
            model_name: Model name (case-insensitive via uppercase)

        Returns:
            Config adapter instance or None if not registered
        """
        adapter_cls = R.config_adapters.get(model_name.upper())
        return adapter_cls(model_name) if adapter_cls else None

    @classmethod
    def get_config_schema(cls, model_name: str) -> Optional[Type]:
        """Get Pydantic config schema for a model.

        First tries to get schema from registered adapter, then falls back
        to directly registered schema.

        Args:
            model_name: Model name (case-insensitive via uppercase)

        Returns:
            Pydantic schema class or None if not registered
        """
        adapter = cls.get_config_adapter(model_name)
        if adapter:
            return adapter.get_config_schema()
        return R.config_schemas.get(model_name.upper())

    @classmethod
    def get_config_defaults(cls, model_name: str) -> Dict[str, Any]:
        """Get default configuration for a model.

        First tries to get defaults from registered adapter, then falls back
        to directly registered defaults.

        Args:
            model_name: Model name (case-insensitive via uppercase)

        Returns:
            Dictionary of default values (empty dict if not registered)
        """
        adapter = cls.get_config_adapter(model_name)
        if adapter:
            return adapter.get_defaults()
        return R.config_defaults.get(model_name.upper()) or {}

    @classmethod
    def get_config_transformers(
        cls, model_name: str
    ) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested transformers for a model.

        First tries to get transformers from registered adapter, then falls back
        to directly registered transformers.

        Args:
            model_name: Model name (case-insensitive via uppercase)

        Returns:
            Dictionary of field transformers (empty dict if not registered)
        """
        adapter = cls.get_config_adapter(model_name)
        if adapter:
            return adapter.get_field_transformers()
        return R.config_transformers.get(model_name.upper()) or {}

    @classmethod
    def get_config_validator(cls, model_name: str) -> Optional[Callable]:
        """Get config validator function for a model.

        First tries to get validator from registered adapter, then falls back
        to directly registered validator.

        Args:
            model_name: Model name (case-insensitive via uppercase)

        Returns:
            Validator function or None if not registered
        """
        adapter = cls.get_config_adapter(model_name)
        if adapter:
            return adapter.validate
        return R.config_validators.get(model_name.upper())

    @classmethod
    def validate_model_config(cls, model_name: str, config: Dict[str, Any]) -> None:
        """Validate model configuration using registered validator.

        Args:
            model_name: Model name
            config: Configuration dictionary

        Raises:
            ConfigValidationError: If validation fails
        """
        validator = cls.get_config_validator(model_name)
        if validator:
            validator(config)

    @classmethod
    def list_config_adapters(cls) -> list[str]:
        """List all models with registered config adapters.

        Returns:
            Sorted list of model names with config adapters
        """
        return sorted(R.config_adapters.keys())

    @classmethod
    def has_config_adapter(cls, model_name: str) -> bool:
        """Check if a model has a registered config adapter.

        Args:
            model_name: Model name

        Returns:
            True if adapter is registered
        """
        return model_name.upper() in R.config_adapters
