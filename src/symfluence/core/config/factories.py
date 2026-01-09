"""
Factory methods for creating SYMFLUENCE configurations.

This module provides factory functions for creating SymfluenceConfig instances
from various sources:
- from_file_factory: Load from YAML file with 5-layer hierarchy
- from_preset_factory: Load from named preset
- from_minimal_factory: Create minimal configuration with smart defaults

Each factory handles the complexity of merging defaults, loading from sources,
and transforming to the hierarchical structure required by Pydantic models.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
import os
import yaml

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


def _is_nested_config(config: Dict[str, Any]) -> bool:
    """
    Detect if a configuration dictionary is in nested format.

    Nested format has lowercase section keys like 'system', 'domain', 'forcing', 'model'.
    Flat format has uppercase keys like 'DOMAIN_NAME', 'FORCING_DATASET'.

    Args:
        config: Configuration dictionary loaded from YAML

    Returns:
        True if config appears to be in nested format
    """
    nested_section_keys = {'system', 'domain', 'forcing', 'model', 'optimization', 'evaluation', 'paths', 'data'}
    config_keys_lower = {k.lower() for k in config.keys()}
    # If any nested section keys are present, treat as nested config
    return bool(nested_section_keys & config_keys_lower)


def _normalize_nested_config(nested_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a nested configuration to ensure section keys are lowercase.

    This handles cases where section keys might be uppercase (SYSTEM vs system).

    Args:
        nested_config: Nested configuration dictionary

    Returns:
        Normalized nested configuration with lowercase section keys
    """
    section_keys = {'system', 'domain', 'forcing', 'model', 'optimization', 'evaluation', 'paths', 'data'}
    normalized = {}

    for key, value in nested_config.items():
        key_lower = key.lower()
        if key_lower in section_keys:
            # Ensure section key is lowercase
            normalized[key_lower] = value
        else:
            # Keep other keys as-is
            normalized[key] = value

    return normalized


def from_file_factory(
    cls: type,
    path: Path,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    use_env: bool = True,
    validate: bool = True
) -> 'SymfluenceConfig':
    """
    Load configuration from YAML file with full 5-layer hierarchy.

    Loading precedence (highest to lowest):
    1. CLI overrides (programmatic)
    2. Environment variables (SYMFLUENCE_*)
    3. Config file (YAML)
    4. Defaults from nested Pydantic models

    Supports both flat format (uppercase keys like DOMAIN_NAME) and
    nested format (hierarchical structure like domain.name).

    Args:
        cls: SymfluenceConfig class
        path: Path to configuration YAML file
        overrides: Dictionary of CLI/programmatic overrides
        use_env: Whether to load environment variables (default: True)
        validate: Whether to validate using Pydantic (default: True)

    Returns:
        Validated SymfluenceConfig instance

    Raises:
        ConfigurationError: If configuration is invalid
        FileNotFoundError: If config file is missing
    """
    from symfluence.core.config.config_loader import (
        _load_env_overrides,
        _normalize_key,
        _format_validation_error
    )
    from symfluence.core.config.defaults import ConfigDefaults
    from symfluence.core.config.transformers import transform_flat_to_nested
    from symfluence.core.exceptions import ConfigurationError
    from pydantic import ValidationError

    # 1. Load from file first to detect format
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        file_config = yaml.safe_load(f) or {}

    # 2. Detect if config is in nested or flat format
    is_nested = _is_nested_config(file_config)

    if is_nested:
        # Handle nested config format - don't transform, just normalize section keys
        nested_config = _normalize_nested_config(file_config)

        # Apply environment variable overrides (converted to nested format)
        if use_env:
            env_overrides = _load_env_overrides()
            if env_overrides:
                # Transform env overrides to nested and merge
                env_nested = transform_flat_to_nested(env_overrides)
                nested_config = _deep_merge(nested_config, env_nested)

        # Apply CLI overrides (can be flat or nested)
        if overrides:
            if _is_nested_config(overrides):
                nested_config = _deep_merge(nested_config, _normalize_nested_config(overrides))
            else:
                # Flat overrides - transform and merge
                normalized_overrides = {_normalize_key(k): v for k, v in overrides.items()}
                override_nested = transform_flat_to_nested(normalized_overrides)
                nested_config = _deep_merge(nested_config, override_nested)
    else:
        # Handle flat config format (original behavior)
        # Start with defaults
        config_dict = ConfigDefaults.get_defaults().copy()

        # Add sensible defaults for required system paths if not in env
        if 'SYMFLUENCE_DATA_DIR' not in config_dict:
            config_dict['SYMFLUENCE_DATA_DIR'] = os.getenv('SYMFLUENCE_DATA_DIR', str(Path.cwd() / 'data'))
        if 'SYMFLUENCE_CODE_DIR' not in config_dict:
            config_dict['SYMFLUENCE_CODE_DIR'] = os.getenv('SYMFLUENCE_CODE_DIR', str(Path.cwd()))

        # Normalize keys from file config
        file_config = {_normalize_key(k): v for k, v in file_config.items()}
        config_dict.update(file_config)

        # Override with environment variables
        if use_env:
            env_overrides = _load_env_overrides()
            config_dict.update(env_overrides)

        # Apply CLI overrides (highest priority)
        if overrides:
            normalized_overrides = {_normalize_key(k): v for k, v in overrides.items()}
            config_dict.update(normalized_overrides)

        # Transform flat dict to nested structure
        nested_config = transform_flat_to_nested(config_dict)

    # Validate and create
    if validate:
        try:
            instance = cls(**nested_config)
            # Store source file path as internal attribute
            object.__setattr__(instance, '_source_file', path)
            return instance
        except ValidationError as e:
            error_msg = _format_validation_error(e, file_config if is_nested else config_dict)
            raise ConfigurationError(error_msg) from e
    else:
        instance = cls.model_construct(**nested_config)
        object.__setattr__(instance, '_source_file', path)
        return instance


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    For nested dictionaries, recursively merge. For other values, override wins.

    Args:
        base: Base dictionary
        override: Override dictionary (values take precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def from_preset_factory(
    cls: type,
    preset_name: str,
    **overrides
) -> 'SymfluenceConfig':
    """
    Create configuration from a named preset.

    Args:
        cls: SymfluenceConfig class
        preset_name: Name of preset ('fuse-provo', 'summa-basic', etc.)
        **overrides: Additional overrides to apply on top of preset

    Returns:
        Fully validated SymfluenceConfig instance

    Raises:
        ConfigurationError: If preset not found or configuration invalid
    """
    from symfluence.cli.init_presets import get_preset
    from symfluence.core.config.transformers import transform_flat_to_nested
    from symfluence.core.exceptions import ConfigurationError
    from pydantic import ValidationError

    # 1. Load preset definition
    try:
        preset = get_preset(preset_name)
    except (KeyError, ValueError) as e:
        raise ConfigurationError(
            f"Preset '{preset_name}' not found. "
            f"Use 'symfluence project list-presets' to see available presets."
        )

    preset_settings = preset['settings'].copy()

    # 2. Apply model-specific decisions
    if 'fuse_decisions' in preset:
        preset_settings['FUSE_DECISION_OPTIONS'] = preset['fuse_decisions']
    if 'summa_decisions' in preset:
        preset_settings['SUMMA_DECISION_OPTIONS'] = preset['summa_decisions']

    # 2.5. Add sensible defaults for required system paths if not provided
    if 'SYMFLUENCE_DATA_DIR' not in preset_settings:
        preset_settings['SYMFLUENCE_DATA_DIR'] = os.getenv('SYMFLUENCE_DATA_DIR', str(Path.cwd() / 'data'))
    if 'SYMFLUENCE_CODE_DIR' not in preset_settings:
        preset_settings['SYMFLUENCE_CODE_DIR'] = os.getenv('SYMFLUENCE_CODE_DIR', str(Path.cwd()))

    # 3. Apply user overrides (highest priority)
    preset_settings.update(overrides)

    # 4. Transform flat dict to nested structure
    nested_config = transform_flat_to_nested(preset_settings)

    # 5. Create and validate
    try:
        return cls(**nested_config)
    except ValidationError as e:
        from symfluence.core.config.config_loader import _format_validation_error
        error_msg = _format_validation_error(e, preset_settings)
        raise ConfigurationError(
            f"Failed to create config from preset '{preset_name}':\n{error_msg}"
        ) from e


def from_minimal_factory(
    cls: type,
    domain_name: str,
    model: str,
    forcing_dataset: str = 'ERA5',
    **overrides
) -> 'SymfluenceConfig':
    """
    Create minimal viable configuration for quick setup.

    Automatically applies sensible defaults based on model choice.

    Args:
        cls: SymfluenceConfig class
        domain_name: Name for the domain/basin
        model: Hydrological model ('SUMMA', 'FUSE', 'GR', etc.)
        forcing_dataset: Forcing data source (default: 'ERA5')
        **overrides: Additional configuration overrides

    Returns:
        Validated SymfluenceConfig with minimal required fields

    Raises:
        ConfigurationError: If required fields missing or configuration invalid
    """
    from symfluence.core.config.defaults import ModelDefaults, ForcingDefaults
    from symfluence.core.config.transformers import transform_flat_to_nested
    from symfluence.core.exceptions import ConfigurationError
    from pydantic import ValidationError

    # 1. Start with absolute minimal required fields
    minimal = {
        'DOMAIN_NAME': domain_name,
        'EXPERIMENT_ID': 'run_1',
        'HYDROLOGICAL_MODEL': model,
        'FORCING_DATASET': forcing_dataset,

        # Required paths (from environment or defaults)
        'SYMFLUENCE_DATA_DIR': overrides.get(
            'SYMFLUENCE_DATA_DIR',
            os.getenv('SYMFLUENCE_DATA_DIR', str(Path.cwd() / 'data'))
        ),
        'SYMFLUENCE_CODE_DIR': overrides.get(
            'SYMFLUENCE_CODE_DIR',
            os.getenv('SYMFLUENCE_CODE_DIR', str(Path.cwd()))
        ),

        # Required domain settings (user should override, but provide safe defaults)
        'DOMAIN_DEFINITION_METHOD': overrides.get('DOMAIN_DEFINITION_METHOD', 'lumped'),
        'DOMAIN_DISCRETIZATION': overrides.get('DOMAIN_DISCRETIZATION', 'lumped'),

        # Required time settings (user MUST override these)
        'EXPERIMENT_TIME_START': overrides.get('EXPERIMENT_TIME_START', '2010-01-01 00:00'),
        'EXPERIMENT_TIME_END': overrides.get('EXPERIMENT_TIME_END', '2020-12-31 23:00'),
    }

    # 2. Apply model-specific defaults
    model_defaults = ModelDefaults.get_defaults_for_model(model.upper())
    if model_defaults:
        minimal.update(model_defaults)

    # 3. Apply forcing-specific defaults
    forcing_defaults = ForcingDefaults.get_defaults_for_forcing(forcing_dataset.upper())
    if forcing_defaults:
        minimal.update(forcing_defaults)

    # 4. Apply user overrides (highest priority)
    minimal.update(overrides)

    # 5. Validate required overrides
    required_overrides = ['EXPERIMENT_TIME_START', 'EXPERIMENT_TIME_END']
    missing = []
    for field in required_overrides:
        if field not in overrides:
            # Check if they provided placeholder values
            if minimal[field] in ['2010-01-01 00:00', '2020-12-31 23:00']:
                missing.append(field)

    if missing:
        raise ConfigurationError(
            f"Missing required fields for minimal config: {', '.join(missing)}\n\n"
            f"Example:\n"
            f"  config = SymfluenceConfig.from_minimal(\n"
            f"      domain_name='{domain_name}',\n"
            f"      model='{model}',\n"
            f"      EXPERIMENT_TIME_START='2020-01-01 00:00',\n"
            f"      EXPERIMENT_TIME_END='2020-12-31 23:00',\n"
            f"      POUR_POINT_COORDS='51.17/-115.57'  # Optional but recommended\n"
            f"  )"
        )

    # 6. Transform and create
    nested_config = transform_flat_to_nested(minimal)

    try:
        return cls(**nested_config)
    except ValidationError as e:
        from symfluence.core.config.config_loader import _format_validation_error
        error_msg = _format_validation_error(e, minimal)
        raise ConfigurationError(
            f"Failed to create minimal config:\n{error_msg}"
        ) from e
