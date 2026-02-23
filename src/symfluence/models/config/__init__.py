"""
Model Configuration Schema Module.

Provides type-safe configuration contracts for all hydrological models,
with validation and sensible defaults.
"""

from .model_config_schema import (
    REGISTERED_SCHEMAS,
    ExecutionConfig,
    InputConfig,
    InstallationConfig,
    ModelConfigSchema,
    OutputConfig,
    get_model_schema,
    validate_model_config,
)

__all__ = [
    'ModelConfigSchema',
    'InstallationConfig',
    'ExecutionConfig',
    'InputConfig',
    'OutputConfig',
    'get_model_schema',
    'validate_model_config',
    'REGISTERED_SCHEMAS',
]
