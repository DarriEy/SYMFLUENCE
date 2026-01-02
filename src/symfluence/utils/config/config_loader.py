from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from difflib import get_close_matches

import yaml
from pydantic import ValidationError

from symfluence.utils.config.defaults import ConfigDefaults
from symfluence.utils.config.models import SymfluenceConfig
from symfluence.utils.exceptions import ConfigurationError


ALIAS_MAP = {
    "GR_SPATIAL": "GR_SPATIAL_MODE",
    "OPTIMISATION_METHODS": "OPTIMIZATION_METHODS",
    "OPTIMISATION_TARGET": "OPTIMIZATION_TARGET",
    "OPTIMIZATION_ALGORITHM": "ITERATIVE_OPTIMIZATION_ALGORITHM",
}


def load_config(
    path: Path,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    validate: bool = True,
    use_env: bool = True,
) -> Dict[str, Any]:
    """
    Load configuration with precedence: CLI overrides > ENV vars > Config file > Defaults

    Args:
        path: Path to configuration YAML file
        overrides: Dictionary of CLI/programmatic overrides
        validate: Whether to validate using Pydantic (default: True)
        use_env: Whether to load environment variables (default: True)

    Returns:
        Validated configuration dictionary
    
    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If config file is missing
    """
    # 1. Start with defaults from the class
    config = ConfigDefaults.get_defaults().copy()

    # 2. Load from file
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, "r") as f:
        file_config = yaml.safe_load(f) or {}
    
    # Normalize keys in file config
    file_config = {_normalize_key(k): v for k, v in file_config.items()}
    config.update(file_config)

    # 3. Override with environment variables
    if use_env:
        env_overrides = _load_env_overrides()
        config.update(env_overrides)

    # 4. Apply CLI overrides (highest priority)
    if overrides:
        # Normalize keys in overrides
        normalized_overrides = {_normalize_key(k): v for k, v in overrides.items()}
        config.update(normalized_overrides)

    # 5. Validate with Pydantic
    if validate:
        try:
            # We filter out None values to let Pydantic defaults/validators handle them
            # or raise errors for required fields
            clean_config = {k: v for k, v in config.items() if v is not None}
            
            model = SymfluenceConfig(**clean_config)
            
            # Return as dict, preserving types converted by Pydantic
            # by_alias=True not strictly needed as we don't use aliases yet, but good practice
            validated_config = model.model_dump()
            
            # Pydantic strips extra fields if configured to 'ignore', but we set 'allow'.
            # model_dump() returns them.
            return validated_config
            
        except ValidationError as e:
            # Format error with actionable suggestions
            error_msg = _format_validation_error(e, config)
            raise ConfigurationError(error_msg) from e
            
    return config


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize configuration keys using aliases and perform type coercion.
    
    Args:
        config: Dictionary of configuration settings
        
    Returns:
        New dictionary with normalized keys and coerced values
    """
    normalized = {}
    for k, v in config.items():
        norm_key = _normalize_key(k)
        normalized[norm_key] = _coerce_value(v)
    return normalized


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration using Pydantic model.
    
    Args:
        config: Dictionary of configuration settings
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = [
        'SYMFLUENCE_DATA_DIR',
        'SYMFLUENCE_CODE_DIR',
        'DOMAIN_NAME',
        'EXPERIMENT_ID',
        'EXPERIMENT_TIME_START',
        'EXPERIMENT_TIME_END',
        'DOMAIN_DEFINITION_METHOD',
        'DOMAIN_DISCRETIZATION',
        'HYDROLOGICAL_MODEL',
        'FORCING_DATASET',
    ]

    missing = [key for key in required_fields if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")

    try:
        # We filter out None values to let Pydantic defaults/validators handle them
        # or raise errors for required fields
        clean_config = {k: v for k, v in config.items() if v is not None}

        model = SymfluenceConfig(**clean_config)
        return model.model_dump()

    except ValidationError as e:
        # Format error with actionable suggestions
        error_msg = _format_validation_error(e, config)
        raise ValueError(error_msg) from e


def _load_env_overrides() -> Dict[str, Any]:
    """
    Load configuration overrides from environment variables.
    """
    env_overrides = {}
    prefix = "SYMFLUENCE_"

    for env_key, env_value in os.environ.items():
        if env_key.startswith(prefix):
            config_key = env_key[len(prefix):]
            norm_key = _normalize_key(config_key)
            env_overrides[norm_key] = _coerce_value(env_value)

    return env_overrides


def _normalize_key(key: str) -> str:
    key_upper = key.upper()
    return ALIAS_MAP.get(key_upper, key_upper)


def _coerce_value(value: Any) -> Any:
    """Helper to attempt basic coercion for values."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    lower = stripped.lower()

    if lower in ('true', 'yes', '1'):
        return True
    if lower in ('false', 'no', '0'):
        return False
    if lower in ('none', 'null', ''):
        return None

    # Try number
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except ValueError:
        pass

    # Handle comma-separated lists
    if "," in stripped:
        return [item.strip() for item in stripped.split(",")]

    return stripped


def _format_validation_error(error: ValidationError, config: Dict[str, Any]) -> str:
    """
    Format Pydantic ValidationError with helpful suggestions.

    Args:
        error: Pydantic ValidationError
        config: Configuration dict that failed validation

    Returns:
        Formatted error message with suggestions
    """
    error_lines = ["=" * 70]
    error_lines.append("Configuration Validation Failed")
    error_lines.append("=" * 70)

    missing_fields = []
    invalid_values = []
    other_errors = []

    # Get all valid field names from the model
    valid_fields = set(SymfluenceConfig.model_fields.keys())

    for err in error.errors():
        field_name = str(err['loc'][0]) if err['loc'] else 'unknown'
        error_type = err['type']
        error_msg = err['msg']

        if error_type == 'missing':
            missing_fields.append(field_name)
        elif 'literal' in error_type.lower() or 'type' in error_type.lower():
            invalid_values.append((field_name, error_msg, err.get('ctx', {})))
        else:
            other_errors.append((field_name, error_msg))

    # Format missing fields
    if missing_fields:
        error_lines.append("\nMissing Required Fields:")
        error_lines.append("-" * 70)
        for field in missing_fields:
            error_lines.append(f"  ✗ {field}")
        error_lines.append("")
        error_lines.append("  Tip: See 0_config_files/config_template.yaml for all required fields")

    # Format invalid values with suggestions
    if invalid_values:
        error_lines.append("\nInvalid Field Values:")
        error_lines.append("-" * 70)
        for field, msg, ctx in invalid_values:
            error_lines.append(f"  ✗ {field}: {msg}")

            # Add expected values if available in context
            if 'expected' in ctx:
                error_lines.append(f"    Expected: {ctx['expected']}")

            # Add actual value if provided in config
            if field in config:
                error_lines.append(f"    Got: {config[field]}")

    # Format other validation errors
    if other_errors:
        error_lines.append("\nValidation Errors:")
        error_lines.append("-" * 70)
        for field, msg in other_errors:
            error_lines.append(f"  ✗ {field}: {msg}")
            if field in config:
                error_lines.append(f"    Current value: {config[field]}")

    # Check for potential typos in config keys
    config_keys = set(k.upper() for k in config.keys())
    unknown_keys = config_keys - valid_fields

    if unknown_keys:
        suggestions = {}
        for unknown in unknown_keys:
            matches = get_close_matches(unknown, valid_fields, n=3, cutoff=0.6)
            if matches:
                suggestions[unknown] = matches

        if suggestions:
            error_lines.append("\nPossible Typos (Did you mean?):")
            error_lines.append("-" * 70)
            for wrong_key, correct_options in suggestions.items():
                options_display = ", ".join([f"'{opt}'" for opt in correct_options])
                error_lines.append(f"  '{wrong_key}' → {options_display}")

    # Add helpful footer
    error_lines.append("")
    error_lines.append("=" * 70)
    error_lines.append("For configuration help:")
    error_lines.append("  • Template: 0_config_files/config_template.yaml")
    error_lines.append("  • Examples: 0_config_files/config_*_tutorial.yaml")
    error_lines.append("  • Docs: https://github.com/CH-Earth/SUMMA")
    error_lines.append("=" * 70)

    return "\n".join(error_lines)
