"""
Configuration loading, normalization, and validation for SYMFLUENCE.

This module provides the core configuration loading pipeline that transforms raw YAML
configuration files into validated, type-safe configuration dictionaries. It handles
key normalization, backwards compatibility via aliases, type coercion, environment
variable overrides, and user-friendly validation error messages.

Configuration Flow:
    1. **Load**: Read YAML configuration file from disk
    2. **Normalize**: Apply key aliases and type coercion (normalize_config)
    3. **Override**: Apply environment variable overrides (_load_env_overrides)
    4. **Validate**: Validate against Pydantic schema (validate_config)
    5. **Use**: Configuration ready for SYMFLUENCE components

Key Functionality:
    Normalization (normalize_config):
        - Converts all keys to uppercase for consistency
        - Applies alias mappings for backwards compatibility
        - Coerces string values to appropriate types (bool, int, float, list)
        - Handles legacy CONFLUENCE naming → SYMFLUENCE

    Validation (validate_config):
        - Checks for required fields (8 mandatory keys)
        - Validates field types using Pydantic models
        - Validates enum values (literal choices)
        - Provides detailed error messages with suggestions

    Type Coercion (_coerce_value):
        - Booleans: 'true', 'yes', '1' → True; 'false', 'no', '0' → False
        - None: 'none', 'null', '' → None
        - Numbers: Automatic int/float detection
        - Lists: Comma-separated strings → list of items
        - Pass-through: Other types unchanged

    Environment Overrides (_load_env_overrides):
        - Reads SYMFLUENCE_* environment variables
        - Strips prefix and normalizes keys
        - Applies type coercion
        - Overrides file-based configuration

    Error Formatting (_format_validation_error):
        - Groups errors by type (missing, invalid, other)
        - Suggests similar field names for typos (fuzzy matching)
        - Shows expected vs actual values
        - Links to documentation and templates

Alias Mapping:
    Normalization aliases are defined in ``legacy_aliases.NORMALIZATION_ALIASES``
    and provide backwards compatibility::

        GR_SPATIAL → GR_SPATIAL_MODE
        OPTIMISATION_METHODS → OPTIMIZATION_METHODS (UK → US spelling)
        OPTIMISATION_TARGET → OPTIMIZATION_TARGET
        OPTIMIZATION_ALGORITHM → ITERATIVE_OPTIMIZATION_ALGORITHM
        CONFLUENCE_DATA_DIR → SYMFLUENCE_DATA_DIR (legacy name)
        CONFLUENCE_CODE_DIR → SYMFLUENCE_CODE_DIR
        DOMAIN_DISCRETIZATION → SUB_GRID_DISCRETIZATION

Required Configuration Fields:
    Core:
        - SYMFLUENCE_DATA_DIR: Data directory path
        - SYMFLUENCE_CODE_DIR: Code directory path
        - DOMAIN_NAME: Basin/domain identifier
        - EXPERIMENT_ID: Experiment/run identifier

    Temporal:
        - EXPERIMENT_TIME_START: Simulation start time (YYYY-MM-DD HH:MM)
        - EXPERIMENT_TIME_END: Simulation end time (YYYY-MM-DD HH:MM)

    Spatial:
        - DOMAIN_DEFINITION_METHOD: Delineation method (lumped/TBL/distribute)
        - SUB_GRID_DISCRETIZATION: Discretization approach (lumped/elevation/...)

    Model:
        - HYDROLOGICAL_MODEL: Model name (SUMMA/FUSE/GR/...)
        - FORCING_DATASET: Forcing dataset name (ERA5/CONUS404/...)

Environment Variable Support:
    All configuration keys can be overridden via environment variables using
    the SYMFLUENCE_ prefix::

        export SYMFLUENCE_DOMAIN_NAME="test_basin"
        export SYMFLUENCE_EXPERIMENT_ID="run_001"
        export SYMFLUENCE_HYDROLOGICAL_MODEL="SUMMA"

    Environment variables are:
    - Normalized using same rules as file-based config
    - Type-coerced automatically
    - Applied after file loading (highest precedence)

Validation Error Handling:
    When validation fails, the module provides structured error messages::

        ======================================================================
        Configuration Validation Failed
        ======================================================================

        Missing Required Fields:
        ----------------------------------------------------------------------
          ✗ DOMAIN_NAME

          Tip: Use 'symfluence config list' to see available templates

        Invalid Field Values:
        ----------------------------------------------------------------------
          ✗ HYDROLOGICAL_MODEL: Input should be 'SUMMA', 'FUSE', ...
            Expected: One of ['SUMMA', 'FUSE', 'GR', 'HYPE', 'NGEN', ...]
            Got: summa

        Possible Typos (Did you mean?):
        ----------------------------------------------------------------------
          'DOMAINNAME' → 'DOMAIN_NAME'
          'FORCINGDATASET' → 'FORCING_DATASET'

        ======================================================================

Usage Example:
    Basic configuration loading::

        >>> from symfluence.core.config.config_loader import (
        ...     normalize_config, validate_config
        ... )
        >>> import yaml
        >>>
        >>> # Load raw config from YAML
        >>> with open('config.yaml') as f:
        ...     raw_config = yaml.safe_load(f)
        >>>
        >>> # Normalize keys and values
        >>> normalized = normalize_config(raw_config)
        >>>
        >>> # Validate and get type-safe config
        >>> config = validate_config(normalized)
        >>>
        >>> # Use in SYMFLUENCE
        >>> domain_name = config['DOMAIN_NAME']
        >>> model = config['HYDROLOGICAL_MODEL']

    Type coercion examples::

        >>> from symfluence.core.config.config_loader import _coerce_value
        >>>
        >>> _coerce_value('true')
        True
        >>> _coerce_value('3.14')
        3.14
        >>> _coerce_value('1,2,3,4')
        ['1', '2', '3', '4']
        >>> _coerce_value('none')
        None

    Alias normalization::

        >>> from symfluence.core.config.config_loader import _normalize_key
        >>>
        >>> _normalize_key('gr_spatial')
        'GR_SPATIAL_MODE'
        >>> _normalize_key('confluence_data_dir')
        'SYMFLUENCE_DATA_DIR'

Integration:
    This module is used by:
    - CLI commands: config validation before workflow execution
    - Project initialization: Template-based configuration setup
    - Configuration manager: Runtime config access and modification
    - All preprocessors: Require validated configuration

    The module integrates with:
    - core.config.models.SymfluenceConfig: Pydantic schema definition
    - core.config.defaults.ModelDefaults/ForcingDefaults: Model/forcing-specific overrides
    - core.exceptions.ConfigurationError: Custom exception types

Error Recovery:
    Common configuration errors and fixes:

    1. Missing field:
       Error: "Missing required configuration keys: DOMAIN_NAME"
       Fix: Add DOMAIN_NAME: "my_basin" to config.yaml

    2. Invalid enum value:
       Error: "HYDROLOGICAL_MODEL: Input should be 'SUMMA', 'FUSE', ..."
       Fix: Use exact case-sensitive value (e.g., SUMMA not summa)

    3. Type mismatch:
       Error: "EXPERIMENT_TIME_START: Input should be a valid datetime"
       Fix: Use format YYYY-MM-DD HH:MM (e.g., 2015-01-01 00:00)

    4. Typo in key:
       Suggestion: 'DOMAINNAME' → 'DOMAIN_NAME'
       Fix: Use suggested key with underscore

Notes:
    - All keys are normalized to UPPERCASE for consistency
    - Type coercion is best-effort; validation catches type errors
    - Environment variables override file-based configuration
    - Alias mapping ensures backwards compatibility with legacy configs
    - Validation uses Pydantic for type safety and schema enforcement
    - Error messages include actionable suggestions and documentation links

See Also:
    - core.config.models.SymfluenceConfig: Pydantic configuration schema
    - core.config.defaults: ModelDefaults/ForcingDefaults helpers
    - core.config.config_manager.ConfigManager: Configuration access interface
    - core.exceptions.ConfigurationError: Configuration-related exceptions
    - cli.commands.config_commands: CLI configuration management commands
"""
from __future__ import annotations

import os
from difflib import get_close_matches
from typing import Any, Dict

from pydantic import ValidationError

from symfluence.core.config.legacy_aliases import NORMALIZATION_ALIASES
from symfluence.core.config.models import SymfluenceConfig


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
        'SUB_GRID_DISCRETIZATION',
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
    return NORMALIZATION_ALIASES.get(key_upper, key_upper)


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
        error_lines.append("  Tip: Use 'symfluence config list' to see available templates")

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
    error_lines.append("  • List templates: symfluence config list")
    error_lines.append(
        "  • Example configs: src/symfluence/resources/config_templates/examples/*_tutorial.yaml"
    )
    error_lines.append("  • Docs: https://github.com/CH-Earth/SUMMA")
    error_lines.append("=" * 70)

    return "\n".join(error_lines)
