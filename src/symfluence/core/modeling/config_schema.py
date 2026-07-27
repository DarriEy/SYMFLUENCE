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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


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
    """Configuration for a model's own output settings.

    Describes the artifact the model itself writes (``timeOUT.txt`` for HYPE,
    ``nex-*_output.csv`` for NGEN, ...). This is *not* necessarily the file a
    routing model consumes — see :class:`RunoffConfig`.
    """
    output_dir_key: str
    default_output_subpath: str
    output_file_pattern: str = "{experiment_id}_{model}_output.nc"
    primary_output_var: str = "streamflow"
    expected_dimensions: List[str] = field(default_factory=lambda: ["time", "gru"])


@dataclass
class RunoffConfig:
    """Declaration of the runoff artifact a routing model consumes.

    Deliberately a sibling of :class:`OutputConfig` rather than extra fields on
    it: the two describe different files for most models. HYPE's own output is
    ``timeOUT.txt`` while the runoff mizuRoute/tRoute read is
    ``{experiment_id}_timestep.nc``; NGEN writes ``nex-*_output.csv`` but is
    routed from ``{experiment_id}_runoff.nc``; GR's primary output variable is
    ``Qsim`` while the routed variable is ``q_routed``. Folding one into the
    other would force a model to declare a single value for two distinct
    artifacts, so both declarations stay, each with its own meaning.

    Only models that can feed a routing model declare this; everything else
    leaves ``ModelConfigSchema.runoff`` as ``None``.

    Attributes:
        output_dir_key: Config key holding the runoff output directory.
        output_dir_name: Directory name under ``simulations/{experiment_id}/``
            and the canonical source-model key routing components pass around.
        default_var: Runoff variable name to look for first.
        default_units: Units string written into routing control files.
        default_dt: Runoff timestep in seconds, as a string.
        output_file_pattern: Filename pattern; ``{experiment_id}`` and
            ``{domain_name}`` are substituted at resolution time.
        hru_dim: Name of the spatial dimension in the runoff file.
        hru_var: Name of the spatial-id variable in the runoff file.
        comment_name: Model label written into generated control files.
        aliases: Alternate source-model spellings resolving to this
            declaration (e.g. GR's ``GR4J``/``GR5J``/``GR6J`` variants).
    """
    output_dir_key: str
    output_dir_name: str
    default_var: str
    default_units: str
    default_dt: str
    output_file_pattern: str
    hru_dim: str = 'gru'
    hru_var: str = 'gruId'
    comment_name: str = 'model'
    aliases: Tuple[str, ...] = ()


@dataclass
class ParallelCalibrationConfig:
    """How PARALLEL calibration rewrites this model's per-process config files.

    A sibling of :class:`RunoffConfig`, deliberately **not** a reuse of it. The
    two describe different files for the same model, because parallel
    calibration does not route the model's standard output: it routes a
    per-process artifact written into ``process_<n>/simulations/<exp>/<MODEL>/``
    under a process-scoped name. FUSE is the clearest case — its ``runoff``
    declaration names ``{domain_name}_{experiment_id}_runs_def.nc`` while its
    parallel-calibration control file names
    ``proc_{proc_id:02d}_{experiment_id}_timestep.nc``. GR is the mirror image:
    the same ``_runs_def.nc`` name as its ``runoff`` declaration and *no*
    ``proc_`` prefix, which is correct rather than an oversight because the
    per-process directory already disambiguates. Folding the two declarations
    together would force one of those two files to be misnamed.

    Every field's default is the value core used to apply to a model it had no
    branch for, so a model that declares nothing keeps exactly the behaviour it
    had, and ``ModelConfigSchema.parallel_calibration`` stays ``None`` for it.

    mizuRoute control-file fields:
        fname_pattern: ``<fname_qsim>`` pattern. Formatted with ``proc_id``
            (int), ``experiment_id`` and ``domain_name``.
        runoff_var: ``<vname_qsim>`` value.
        runoff_var_from_config: When True, ``SETTINGS_MIZU_ROUTING_VAR``
            overrides *runoff_var* (``'default'``/empty falling back to it).
            When False the declared name is used unconditionally — SUMMA's
            ``averageRoutedRunoff`` is not a user-selectable variable.
        dt_qsim: ``<dt_qsim>`` in seconds, as a string, when the model's output
            cadence is fixed regardless of configuration. ``None`` means
            ``SETTINGS_MIZU_ROUTING_DT`` decides (default ``'3600'``).
        sim_start_time: Time-of-day appended to ``<sim_start>``.
        sim_end_time: Time-of-day appended to ``<sim_end>``.
        hru_dim: ``<dname_hruid>`` value.
        hru_var: ``<vname_hruid>`` value.

    File-manager / settings-file dialect:
        settings_values_quoted: True when the model's file manager quotes its
            values (``outputPath  '/path/'``). False for models whose settings
            file is bare key/value — a line with no quoted value is then still
            eligible for rewriting instead of being passed through untouched.
        output_dir_directive: Name of an additional directive that carries the
            run's output directory and must be repointed at the per-process
            ``output/`` directory (HYPE's ``resultdir``). ``None`` for models
            whose output path is covered by ``outputPath``.
    """
    fname_pattern: str = 'proc_{proc_id:02d}_{experiment_id}_timestep.nc'
    runoff_var: str = 'q_routed'
    runoff_var_from_config: bool = False
    dt_qsim: Optional[str] = None
    sim_start_time: str = '01:00'
    sim_end_time: str = '23:00'
    hru_dim: str = 'hru'
    hru_var: str = 'hruId'
    settings_values_quoted: bool = True
    output_dir_directive: Optional[str] = None


#: Shared instance served for every model that declares nothing. Its field
#: defaults *are* the historical else-branch of core's per-model table.
DEFAULT_PARALLEL_CALIBRATION = ParallelCalibrationConfig()


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
    #: Config key core's ``RoutingDecider`` consults to see whether the model
    #: asks for routing integration (``'mizuRoute'``). Distinct from
    #: ``routing_key``, which is the model's own descriptive routing key and
    #: is not consulted by the routing decision: SUMMA declares
    #: ``ROUTING_DELINEATION`` there, which is a different question entirely.
    #: Only models whose routing-integration key core actually reads set this.
    routing_integration_key: Optional[str] = None
    #: Runoff artifact a routing model consumes, when the model can feed one.
    runoff: Optional[RunoffConfig] = None
    #: How parallel calibration rewrites this model's per-process config files.
    #: ``None`` means "the shared defaults", which are exactly what core used to
    #: apply to a model it had no branch for.
    parallel_calibration: Optional[ParallelCalibrationConfig] = None
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
        if self.routing_integration_key:
            keys.add(self.routing_integration_key)
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


def schema_key_table(attribute: str) -> Dict[str, str]:
    """Model -> config key, for every schema declaring *attribute*.

    The read side of the per-model config-key declarations (currently
    ``spatial_mode_key`` and ``routing_integration_key``). Core consumers call
    this instead of carrying their own hardcoded table, so a model package —
    in-tree or external — contributes its keys purely by registering a schema.

    Keys are the registration names (uppercase); schemas leaving *attribute*
    ``None`` are absent, which is what makes "core does not consult this key
    for this model" an explicit, per-model declaration.
    """
    table: Dict[str, str] = {}
    for name, schema in REGISTERED_SCHEMAS.items():
        value = getattr(schema, attribute, None)
        if value is not None:
            table[name] = value
    return table


def registered_runoff_configs() -> Dict[str, RunoffConfig]:
    """Model -> :class:`RunoffConfig`, for every schema declaring one."""
    return {
        name: schema.runoff
        for name, schema in REGISTERED_SCHEMAS.items()
        if schema.runoff is not None
    }


def parallel_calibration_config(model_name: str) -> ParallelCalibrationConfig:
    """The model's parallel-calibration declaration, or the shared default.

    Read side of :attr:`ModelConfigSchema.parallel_calibration`. Unlike
    :func:`~symfluence.core.modeling.utilities.runoff_loader.get_model_config`
    this never raises: parallel calibration must keep working for a model that
    declares nothing (and for the coupled optimizer, which drives file managers
    for arbitrary sub-model names), so an unregistered or undeclaring model
    gets :data:`DEFAULT_PARALLEL_CALIBRATION` — the historical else-branch.
    """
    schema = REGISTERED_SCHEMAS.get((model_name or '').strip().upper())
    declared = getattr(schema, 'parallel_calibration', None)
    return declared if declared is not None else DEFAULT_PARALLEL_CALIBRATION


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
