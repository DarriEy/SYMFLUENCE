"""
Base configuration adapters for model-specific configuration management.

This module provides abstract interfaces that each model implements to register
its configuration schema, defaults, transformers, and validators with the framework.

Architecture:
    - ModelConfigAdapter: Interface for model-specific config handling
    - Each model implements this interface in models/{model}/config.py
    - Components register with ModelRegistry for centralized lookup
    - Core config system uses registry instead of hardcoded model logic

Example:
    >>> # In models/summa/config.py
    >>> class SUMMAConfigAdapter(ModelConfigAdapter):
    ...     def get_config_schema(self):
    ...         return SUMMAConfig
    ...
    ...     def get_defaults(self):
    ...         return {'ROUTING_MODEL': 'mizuRoute', ...}
    ...
    ...     def get_field_transformers(self):
    ...         return {'SUMMA_INSTALL_PATH': ('model', 'summa', 'install_path'), ...}
    ...
    ...     def validate(self, config):
    ...         if not config.get('SUMMA_EXE'):
    ...             raise ConfigurationError("SUMMA_EXE required")
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Type, Optional, Callable, List
from pydantic import BaseModel
from pydantic_core import PydanticUndefined


class ModelConfigAdapter(ABC):
    """
    Abstract interface for model-specific configuration handling.

    Each model implements this interface to provide:
    1. Pydantic schema for type validation
    2. Default configuration values
    3. Field transformers (flat-to-nested mapping)
    4. Custom validation logic

    This enables the core config system to be model-agnostic while
    supporting model-specific requirements through a plugin pattern.

    Attributes:
        model_name: Name of the model (e.g., 'SUMMA', 'FUSE')
    """

    def __init__(self, model_name: str):
        """
        Initialize config adapter.

        Args:
            model_name: Model name (e.g., 'SUMMA', 'FUSE')
        """
        self.model_name = model_name.upper()

    @abstractmethod
    def get_config_schema(self) -> Type[BaseModel]:
        """
        Get Pydantic model class for this model's configuration.

        Returns:
            Pydantic BaseModel subclass (e.g., SUMMAConfig, FUSEConfig)

        Example:
            >>> def get_config_schema(self):
            ...     return SUMMAConfig
        """
        pass

    @abstractmethod
    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default configuration values for this model.

        Returns:
            Dictionary of default config values with uppercase keys

        Example:
            >>> def get_defaults(self):
            ...     return {
            ...         'ROUTING_MODEL': 'mizuRoute',
            ...         'SUMMA_INSTALL_PATH': 'default',
            ...         'SUMMA_EXE': 'summa_sundials.exe',
            ...     }
        """
        pass

    @abstractmethod
    def get_field_transformers(self) -> Dict[str, Tuple[str, ...]]:
        """
        Get flat-to-nested field transformers for this model.

        Maps uppercase flat keys to nested path tuples for Pydantic models.

        Returns:
            Dictionary mapping 'FLAT_KEY' -> ('section', 'subsection', 'field')

        Example:
            >>> def get_field_transformers(self):
            ...     return {
            ...         'SUMMA_INSTALL_PATH': ('model', 'summa', 'install_path'),
            ...         'SUMMA_EXE': ('model', 'summa', 'exe'),
            ...         'SETTINGS_SUMMA_PATH': ('model', 'summa', 'settings_path'),
            ...     }
        """
        pass

    def validate(self, config: Dict[str, Any]) -> None:
        """
        Perform model-specific configuration validation.

        Override to add custom validation beyond Pydantic schema validation.
        Raise ConfigurationError for validation failures.

        Args:
            config: Configuration dictionary (flat format with uppercase keys)

        Raises:
            ConfigurationError: If configuration is invalid

        Example:
            >>> def validate(self, config):
            ...     if config.get('SUMMA_SPATIAL_MODE') == 'distributed':
            ...         if not config.get('SETTINGS_SUMMA_ATTRIBUTES'):
            ...             raise ConfigurationError("attributes.nc required for distributed SUMMA")
        """
        pass

    def get_required_keys(self) -> list:
        """
        Get list of required configuration keys for this model.

        Returns:
            List of uppercase config keys that must be present

        Example:
            >>> def get_required_keys(self):
            ...     return ['SUMMA_EXE', 'SETTINGS_SUMMA_PATH', 'SETTINGS_SUMMA_FILEMANAGER']
        """
        return []

    def get_conditional_requirements(self) -> Dict[str, Dict[Any, List[str]]]:
        """
        Get conditional requirements based on other config values.

        Returns:
            Dictionary mapping condition key to {value: [required_keys]} mappings

        Example:
            >>> def get_conditional_requirements(self):
            ...     return {
            ...         'ROUTING_MODEL': {
            ...             'mizuRoute': ['INSTALL_PATH_MIZUROUTE', 'EXE_NAME_MIZUROUTE'],
            ...             'none': []
            ...         }
            ...     }
        """
        return {}


class AutoGeneratedConfigAdapter(ModelConfigAdapter):
    """
    Config adapter with auto-generated transformers and defaults.

    Eliminates boilerplate by extracting configuration from Pydantic models:
    - Defaults: Auto-extracted from Field(default=...) declarations
    - Transformers: Auto-generated from Field(alias='...') declarations

    Subclasses only need to implement:
    - get_config_schema() → Returns Pydantic model class
    - validate() → Optional custom validation logic

    This reduces model adapters from ~140 lines to ~60 lines on average,
    with all defaults and transformers maintained in a single source of truth
    (the Pydantic model Field declarations).

    Example:
        >>> class FUSEConfigAdapter(AutoGeneratedConfigAdapter):
        ...     def get_config_schema(self):
        ...         return FUSEConfig
        ...
        ...     def validate(self, config: Dict[str, Any]):
        ...         # Custom FUSE validation only
        ...         if config.get('FUSE_SPATIAL_MODE') == 'distributed':
        ...             if not config.get('FUSE_ROUTING_INTEGRATION'):
        ...                 raise ConfigValidationError("Distributed mode requires routing")
    """

    def get_defaults(self) -> Dict[str, Any]:
        """
        Auto-generate defaults from Pydantic Field defaults.

        Extracts all fields with uppercase aliases and non-None defaults.

        Returns:
            Dictionary of {FLAT_KEY: default_value}
        """
        schema = self.get_config_schema()
        defaults = {}

        if hasattr(schema, 'model_fields'):
            for field_name, field_info in schema.model_fields.items():
                # Skip fields without defaults
                if field_info.default in (None, PydanticUndefined):
                    continue

                # Use alias if present and uppercase (flat config key)
                alias = field_info.alias
                if alias and alias.isupper():
                    defaults[alias] = field_info.default

        return defaults

    def get_field_transformers(self) -> Dict[str, Tuple[str, ...]]:
        """
        Auto-generate field transformers from Pydantic model structure.

        Uses introspection to build flat→nested mapping for this model's fields.

        Returns:
            Dictionary of {FLAT_KEY: ('section', 'subsection', 'field')}
        """
        schema = self.get_config_schema()
        mapping: Dict[str, Tuple[str, ...]] = {}

        # Determine prefix for this model (e.g., 'model.summa')
        model_name_lower = self.model_name.lower()

        # Map fields to their nested path
        if hasattr(schema, 'model_fields'):
            for field_name, field_info in schema.model_fields.items():
                alias = field_info.alias
                if alias and alias.isupper():
                    mapping[alias] = ('model', model_name_lower, field_name)

        return mapping


from symfluence.core.exceptions import ConfigValidationError  # noqa: F401, E402


def create_config_adapter(model_name: str,
                         schema: Type[BaseModel],
                         defaults: Dict[str, Any],
                         transformers: Dict[str, Tuple[str, ...]],
                         validator: Optional[Callable] = None,
                         required_keys: Optional[list] = None) -> ModelConfigAdapter:
    """
    Factory function to create a ModelConfigAdapter from components.

    Useful for simple models that don't need custom adapter classes.

    Args:
        model_name: Model name (e.g., 'SUMMA', 'FUSE')
        schema: Pydantic model class
        defaults: Default configuration dictionary
        transformers: Flat-to-nested field mapping
        validator: Optional custom validation function
        required_keys: Optional list of required keys

    Returns:
        ModelConfigAdapter instance

    Example:
        >>> summa_adapter = create_config_adapter(
        ...     'SUMMA',
        ...     SUMMAConfig,
        ...     SUMMA_DEFAULTS,
        ...     SUMMA_TRANSFORMERS,
        ...     validator=validate_summa_config,
        ...     required_keys=['SUMMA_EXE', 'SETTINGS_SUMMA_PATH']
        ... )
    """
    class DynamicConfigAdapter(ModelConfigAdapter):
        def get_config_schema(self):
            return schema

        def get_defaults(self):
            return defaults

        def get_field_transformers(self):
            return transformers

        def validate(self, config):
            if validator:
                validator(config)

        def get_required_keys(self):
            return required_keys or []

    return DynamicConfigAdapter(model_name)
