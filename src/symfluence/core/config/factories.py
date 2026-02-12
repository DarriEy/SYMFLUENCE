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


def _resolve_default_code_dir() -> str:
    """Resolve the default SYMFLUENCE_CODE_DIR.

    Priority:
    1. SYMFLUENCE_CODE_DIR environment variable
    2. Repository root (detected from package location)
    3. Current working directory
    """
    env_val = os.getenv('SYMFLUENCE_CODE_DIR')
    if env_val:
        return env_val

    # Try to find repo root from package location
    try:
        import symfluence
        pkg_dir = Path(symfluence.__file__).parent  # src/symfluence/
        # Walk up to find repo root (contains pyproject.toml or .git)
        candidate = pkg_dir
        for _ in range(5):
            candidate = candidate.parent
            if (candidate / 'pyproject.toml').exists() or (candidate / '.git').exists():
                return str(candidate)
    except Exception:
        pass

    return str(Path.cwd())


def _resolve_default_data_dir(code_dir: Optional[str] = None) -> str:
    """Resolve the default SYMFLUENCE_DATA_DIR.

    Priority:
    1. SYMFLUENCE_DATA_DIR environment variable
    2. Sibling directory to CODE_DIR named SYMFLUENCE_data
    3. Current working directory / data
    """
    env_val = os.getenv('SYMFLUENCE_DATA_DIR')
    if env_val:
        return env_val

    # Use sibling directory to code_dir
    if code_dir is None:
        code_dir = _resolve_default_code_dir()
    code_path = Path(code_dir)
    return str(code_path.parent / 'SYMFLUENCE_data')


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

        # Add sensible defaults for required system paths
        if 'SYMFLUENCE_CODE_DIR' not in config_dict:
            config_dict['SYMFLUENCE_CODE_DIR'] = _resolve_default_code_dir()
        if 'SYMFLUENCE_DATA_DIR' not in config_dict:
            config_dict['SYMFLUENCE_DATA_DIR'] = _resolve_default_data_dir(config_dict.get('SYMFLUENCE_CODE_DIR'))

        # Normalize keys from file config
        file_config = {_normalize_key(k): v for k, v in file_config.items()}

        # Treat 'default' as sentinel: use computed defaults instead
        for path_key in ('SYMFLUENCE_DATA_DIR', 'SYMFLUENCE_CODE_DIR'):
            if file_config.get(path_key) == 'default':
                del file_config[path_key]

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

    # Filter out None values so Pydantic can use field defaults
    nested_config = _filter_none_values(nested_config)

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


def _filter_none_values(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively filter out None values from a nested dictionary.

    This allows Pydantic to use field defaults when config explicitly sets null.

    Args:
        d: Dictionary potentially containing None values

    Returns:
        New dictionary with None values removed at all levels
    """
    result = {}
    for key, value in d.items():
        if value is None:
            continue
        if isinstance(value, dict):
            filtered = _filter_none_values(value)
            if filtered:  # Only include non-empty dicts
                result[key] = filtered
        else:
            result[key] = value
    return result


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
    except (KeyError, ValueError):
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
    if 'SYMFLUENCE_CODE_DIR' not in preset_settings:
        preset_settings['SYMFLUENCE_CODE_DIR'] = _resolve_default_code_dir()
    if 'SYMFLUENCE_DATA_DIR' not in preset_settings:
        preset_settings['SYMFLUENCE_DATA_DIR'] = _resolve_default_data_dir(preset_settings.get('SYMFLUENCE_CODE_DIR'))

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


def _normalize_config_key(key: str) -> str:
    """
    Normalize configuration key to uppercase format.

    Accepts both modern lowercase/snake_case and legacy UPPERCASE formats.

    Args:
        key: Configuration key in any format (time_start, EXPERIMENT_TIME_START, etc.)

    Returns:
        Normalized uppercase key (EXPERIMENT_TIME_START)

    Examples:
        >>> _normalize_config_key('time_start')
        'EXPERIMENT_TIME_START'
        >>> _normalize_config_key('EXPERIMENT_TIME_START')
        'EXPERIMENT_TIME_START'
        >>> _normalize_config_key('routing_model')
        'ROUTING_MODEL'
    """
    from symfluence.core.config.transformers import get_flat_to_nested_map

    # If already uppercase, return as-is (legacy format)
    if key.isupper() or key == key.upper():
        return key

    # Build reverse mapping: nested path -> uppercase key
    # This maps ('domain', 'time_start') -> 'EXPERIMENT_TIME_START'
    flat_to_nested = get_flat_to_nested_map()
    nested_to_flat = {path: flat_key for flat_key, path in flat_to_nested.items()}

    # Common patterns for new lowercase keys
    # Try to find matching nested path based on the key name
    lowercase_key = key.lower()

    # Direct field name matches (e.g., 'time_start' in ('domain', 'time_start'))
    for nested_path, flat_key in nested_to_flat.items():
        if nested_path[-1] == lowercase_key:
            return flat_key

    # Special handling for common abbreviated forms
    key_mappings = {
        'time_start': 'EXPERIMENT_TIME_START',
        'time_end': 'EXPERIMENT_TIME_END',
        'routing_model': 'ROUTING_MODEL',
        'forcing_dataset': 'FORCING_DATASET',
        'definition_method': 'DOMAIN_DEFINITION_METHOD',
        'discretization': 'SUB_GRID_DISCRETIZATION',
        'data_access': 'DATA_ACCESS',
        'forcing_measurement_height': 'FORCING_MEASUREMENT_HEIGHT',
        'spinup_period': 'SPINUP_PERIOD',
        'calibration_period': 'CALIBRATION_PERIOD',
        'evaluation_period': 'EVALUATION_PERIOD',
        'pour_point_coords': 'POUR_POINT_COORDS',
        'bounding_box_coords': 'BOUNDING_BOX_COORDS',
        'lumped_watershed_method': 'LUMPED_WATERSHED_METHOD',
        'dem_source': 'DEM_SOURCE',
        'download_dem': 'DOWNLOAD_DEM',
        'station_id': 'STATION_ID',
        'streamflow_data_provider': 'STREAMFLOW_DATA_PROVIDER',
        'download_usgs_data': 'DOWNLOAD_USGS_DATA',
        'params_to_calibrate': 'PARAMS_TO_CALIBRATE',
        'basin_params_to_calibrate': 'BASIN_PARAMS_TO_CALIBRATE',
        'optimization_target': 'OPTIMIZATION_TARGET',
        'optimization_algorithm': 'ITERATIVE_OPTIMIZATION_ALGORITHM',
        'optimization_metric': 'OPTIMIZATION_METRIC',
        'calibration_timestep': 'CALIBRATION_TIMESTEP',
        'iterations': 'NUMBER_OF_ITERATIONS',
        'max_iterations': 'NUMBER_OF_ITERATIONS',  # Alias for iterations
    }

    if lowercase_key in key_mappings:
        return key_mappings[lowercase_key]

    # Fallback: return uppercase version of the key
    # This handles simple cases like 'DEBUG_MODE' or 'debug_mode' -> 'DEBUG_MODE'
    return key.upper()


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

    Supports both modern lowercase/snake_case keys and legacy UPPERCASE keys:
    - Modern: time_start='2020-01-01 00:00', routing_model='mizuRoute'
    - Legacy: EXPERIMENT_TIME_START='2020-01-01 00:00', ROUTING_MODEL='mizuRoute'

    Args:
        cls: SymfluenceConfig class
        domain_name: Name for the domain/basin
        model: Hydrological model ('SUMMA', 'FUSE', 'GR', etc.)
        forcing_dataset: Forcing data source (default: 'ERA5')
        **overrides: Additional configuration overrides (accepts both formats)

    Returns:
        Validated SymfluenceConfig with minimal required fields

    Raises:
        ConfigurationError: If required fields missing or configuration invalid

    Examples:
        >>> # Modern syntax
        >>> config = SymfluenceConfig.from_minimal(
        ...     domain_name='test',
        ...     model='SUMMA',
        ...     time_start='2020-01-01 00:00',
        ...     time_end='2020-12-31 23:00',
        ...     routing_model='mizuRoute'
        ... )
        >>>
        >>> # Legacy syntax (still supported)
        >>> config = SymfluenceConfig.from_minimal(
        ...     domain_name='test',
        ...     model='SUMMA',
        ...     EXPERIMENT_TIME_START='2020-01-01 00:00',
        ...     EXPERIMENT_TIME_END='2020-12-31 23:00',
        ...     ROUTING_MODEL='mizuRoute'
        ... )
    """
    from symfluence.core.config.defaults import ModelDefaults, ForcingDefaults
    from symfluence.core.config.transformers import transform_flat_to_nested
    from symfluence.core.exceptions import ConfigurationError
    from pydantic import ValidationError

    # 0. Normalize all override keys to uppercase format
    normalized_overrides = {_normalize_config_key(k): v for k, v in overrides.items()}

    # 1. Start with absolute minimal required fields
    minimal = {
        'DOMAIN_NAME': domain_name,
        'EXPERIMENT_ID': 'run_1',
        'HYDROLOGICAL_MODEL': model,
        'FORCING_DATASET': forcing_dataset,

        # Required paths (from environment or defaults)
        'SYMFLUENCE_CODE_DIR': normalized_overrides.get(
            'SYMFLUENCE_CODE_DIR',
            _resolve_default_code_dir()
        ),
        'SYMFLUENCE_DATA_DIR': normalized_overrides.get(
            'SYMFLUENCE_DATA_DIR',
            _resolve_default_data_dir(normalized_overrides.get('SYMFLUENCE_CODE_DIR'))
        ),

        # Required domain settings (user should override, but provide safe defaults)
        'DOMAIN_DEFINITION_METHOD': normalized_overrides.get('DOMAIN_DEFINITION_METHOD', 'lumped'),
        'SUB_GRID_DISCRETIZATION': normalized_overrides.get('SUB_GRID_DISCRETIZATION', 'lumped'),

        # Required time settings (user MUST override these)
        'EXPERIMENT_TIME_START': normalized_overrides.get('EXPERIMENT_TIME_START', '2010-01-01 00:00'),
        'EXPERIMENT_TIME_END': normalized_overrides.get('EXPERIMENT_TIME_END', '2020-12-31 23:00'),
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
    minimal.update(normalized_overrides)

    # 5. Validate required overrides (check both original and normalized keys)
    required_overrides = ['EXPERIMENT_TIME_START', 'EXPERIMENT_TIME_END']
    missing = []
    for field in required_overrides:
        if field not in normalized_overrides:
            # Check if they provided placeholder values
            if minimal[field] in ['2010-01-01 00:00', '2020-12-31 23:00']:
                missing.append(field)

    if missing:
        raise ConfigurationError(
            f"Missing required fields for minimal config: {', '.join(missing)}\n\n"
            f"Example (modern syntax):\n"
            f"  config = SymfluenceConfig.from_minimal(\n"
            f"      domain_name='{domain_name}',\n"
            f"      model='{model}',\n"
            f"      time_start='2020-01-01 00:00',\n"
            f"      time_end='2020-12-31 23:00',\n"
            f"      pour_point_coords='51.17/-115.57'  # Optional but recommended\n"
            f"  )\n\n"
            f"Example (legacy syntax):\n"
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
