# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ModelConfigSchema - Declarative configuration contracts for hydrological models.

This module provides a schema-based approach to model configuration that:
1. Declares required vs optional configuration keys for each model
2. Provides sensible defaults
3. Validates configuration at runtime
4. Documents the configuration contract for each model

Usage:
    from symfluence.models.config import validate_model_config

    # Validate before model run
    errors = validate_model_config('SUMMA', config_dict)
    if errors:
        raise ConfigurationError(f"Invalid config: {errors}")

    # Or use schema directly
    schema = get_model_schema('SUMMA')
    config = schema.apply_defaults(config_dict)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


class ConfigKeyType(Enum):
    """Type classification for configuration keys."""
    PATH = "path"
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    LIST = "list"
    DICT = "dict"


@dataclass
class ConfigKey:
    """Definition of a single configuration key.

    Attributes:
        name: Configuration key name (e.g., 'SUMMA_INSTALL_PATH')
        key_type: Type of the value
        required: Whether this key must be present
        default: Default value if not provided
        description: Human-readable description
        valid_values: For enum types, list of valid values
        validator: Optional custom validation function
        legacy_names: Alternative key names for backward compatibility
    """
    name: str
    key_type: ConfigKeyType
    required: bool = False
    default: Any = None
    description: str = ""
    valid_values: Optional[List[str]] = None
    validator: Optional[Callable[[Any], bool]] = None
    legacy_names: List[str] = field(default_factory=list)


@dataclass
class InstallationConfig:
    """Configuration for model installation paths.

    Standardizes the pattern:
        install_path_key -> default_subpath
        exe_name_key -> default_exe
    """
    install_path_key: str
    default_install_subpath: str
    exe_name_key: Optional[str] = None
    default_exe_name: Optional[str] = None
    version_key: Optional[str] = None


@dataclass
class ExecutionConfig:
    """Configuration for model execution settings."""
    method: str = "subprocess"  # subprocess, slurm, slurm_array
    supports_parallel: bool = False
    parallel_key: Optional[str] = None
    default_timeout: int = 3600
    default_memory: str = "4G"
    default_cpus: int = 1
    env_vars: Dict[str, str] = field(default_factory=dict)


@dataclass
class InputConfig:
    """Configuration for model input requirements."""
    forcing_dir_key: str
    default_forcing_subpath: str
    forcing_file_pattern: str = "{domain}_input.nc"
    required_variables: List[str] = field(default_factory=list)
    optional_variables: List[str] = field(default_factory=list)


@dataclass
class OutputConfig:
    """Configuration for model output settings."""
    output_dir_key: str
    default_output_subpath: str
    output_file_pattern: str = "{experiment_id}_{model}_output.nc"
    primary_output_var: str = "streamflow"
    expected_dimensions: List[str] = field(default_factory=lambda: ["time", "gru"])


@dataclass
class ModelConfigSchema:
    """Complete configuration schema for a hydrological model.

    This class defines the full configuration contract for a model,
    including all required keys, defaults, and validation rules.

    Example:
        SUMMA_SCHEMA = ModelConfigSchema(
            model_name='SUMMA',
            installation=InstallationConfig(
                install_path_key='SUMMA_INSTALL_PATH',
                default_install_subpath='installs/summa/bin',
                exe_name_key='SUMMA_EXE',
                default_exe_name='summa.exe'
            ),
            ...
        )
    """
    model_name: str
    installation: InstallationConfig
    execution: ExecutionConfig
    input: InputConfig
    output: OutputConfig
    config_keys: List[ConfigKey] = field(default_factory=list)
    spatial_mode_key: Optional[str] = None
    routing_key: Optional[str] = None
    description: str = ""

    def get_required_keys(self) -> Set[str]:
        """Return set of required configuration keys."""
        required = set()
        for key in self.config_keys:
            if key.required:
                required.add(key.name)
        # Add installation keys
        required.add(self.installation.install_path_key)
        if self.installation.exe_name_key:
            required.add(self.installation.exe_name_key)
        return required

    def get_all_keys(self) -> Set[str]:
        """Return set of all recognized configuration keys."""
        keys = {k.name for k in self.config_keys}
        keys.add(self.installation.install_path_key)
        if self.installation.exe_name_key:
            keys.add(self.installation.exe_name_key)
        keys.add(self.input.forcing_dir_key)
        keys.add(self.output.output_dir_key)
        if self.spatial_mode_key:
            keys.add(self.spatial_mode_key)
        if self.routing_key:
            keys.add(self.routing_key)
        return keys

    def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to configuration dict."""
        result = config.copy()

        for key in self.config_keys:
            if key.name not in result and key.default is not None:
                result[key.name] = key.default
            # Check legacy names
            if key.name not in result:
                for legacy in key.legacy_names:
                    if legacy in result:
                        result[key.name] = result[legacy]
                        break

        return result

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration against schema.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required keys
        for key in self.config_keys:
            if key.required and key.name not in config:
                # Check legacy names
                has_legacy = any(leg in config for leg in key.legacy_names)
                if not has_legacy:
                    errors.append(f"Missing required key: {key.name}")

        # Check types and values
        for key in self.config_keys:
            value = config.get(key.name)
            if value is None:
                continue

            # Type validation
            if key.key_type == ConfigKeyType.INTEGER:
                if not isinstance(value, int):
                    try:
                        int(value)
                    except (ValueError, TypeError):
                        errors.append(f"{key.name}: expected integer, got {type(value).__name__}")

            elif key.key_type == ConfigKeyType.FLOAT:
                if not isinstance(value, (int, float)):
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        errors.append(f"{key.name}: expected float, got {type(value).__name__}")

            elif key.key_type == ConfigKeyType.BOOLEAN:
                if not isinstance(value, bool):
                    if str(value).lower() not in ('true', 'false', 'yes', 'no', '1', '0'):
                        errors.append(f"{key.name}: expected boolean, got {value}")

            elif key.key_type == ConfigKeyType.ENUM:
                if key.valid_values and value not in key.valid_values:
                    errors.append(
                        f"{key.name}: invalid value '{value}'. Valid: {key.valid_values}"
                    )

            elif key.key_type == ConfigKeyType.PATH:
                if value and value != 'default':
                    Path(value)
                    # Note: we don't check existence here, just format

            # Custom validator
            if key.validator and not key.validator(value):
                errors.append(f"{key.name}: failed custom validation")

        return errors


# =============================================================================
# Registered Model Schemas
# =============================================================================


# =============================================================================
# Schema Registry (machinery only — per-model schema definitions live with
# their model packages and register here at import time)
# =============================================================================

REGISTERED_SCHEMAS: Dict[str, ModelConfigSchema] = {}


def register_model_schema(name: str, schema: ModelConfigSchema):
    """
    Register a model schema (in-tree definitions and external packages alike).

    Args:
        name: Model name (will be uppercased)
        schema: ModelConfigSchema instance
    """
    REGISTERED_SCHEMAS[name.upper()] = schema


def get_model_schema(model_name: str) -> ModelConfigSchema:
    """
    Get configuration schema for a model.

    Args:
        model_name: Name of the model (case-insensitive)

    Returns:
        ModelConfigSchema for the requested model

    Raises:
        KeyError: If model is not registered
    """
    key = model_name.upper()
    if key not in REGISTERED_SCHEMAS:
        available = list(REGISTERED_SCHEMAS.keys())
        raise KeyError(f"Unknown model: {model_name}. Available: {available}")
    return REGISTERED_SCHEMAS[key]


def validate_model_config(
    model_name: str,
    config: Dict[str, Any],
    apply_defaults: bool = True
) -> List[str]:
    """
    Validate configuration for a model.

    Args:
        model_name: Name of the model
        config: Configuration dictionary to validate
        apply_defaults: Whether to apply defaults before validation

    Returns:
        List of validation error messages (empty if valid)
    """
    schema = get_model_schema(model_name)

    if apply_defaults:
        config = schema.apply_defaults(config)

    return schema.validate(config)
