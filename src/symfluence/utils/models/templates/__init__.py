"""
Model Templates Module.

Provides base classes and templates for implementing new hydrological models
using the unified execution framework.
"""

from .model_template import (
    UnifiedModelRunner,
    ModelRunResult,
    create_model_runner,
)

__all__ = [
    'UnifiedModelRunner',
    'ModelRunResult',
    'create_model_runner',
]
