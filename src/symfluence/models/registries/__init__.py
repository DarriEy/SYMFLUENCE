# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model Registries Package

Provides focused registry classes for different aspects of model management:

- ComponentRegistry: Preprocessors, runners, postprocessors, visualizers
- ConfigRegistry: Config adapters, schemas, defaults, transformers, validators
- ResultExtractorRegistry: Model-specific result extraction

All registry classes are re-exported from this module for convenient access.
The main ModelRegistry facade in the parent module combines all registries
for backward compatibility.

Example:
    >>> from symfluence.models.registries import ComponentRegistry
    >>> @ComponentRegistry.register_preprocessor('MYMODEL')
    ... class MyPreprocessor: ...

    >>> from symfluence.models.registries import ConfigRegistry
    >>> ConfigRegistry.register_config_defaults('MYMODEL', {'timestep': 3600})

    >>> from symfluence.models.registries import ResultExtractorRegistry
    >>> @ResultExtractorRegistry.register_result_extractor('MYMODEL')
    ... class MyResultExtractor: ...
"""

from symfluence.models.registries.component_registry import ComponentRegistry
from symfluence.models.registries.config_registry import ConfigRegistry
from symfluence.models.registries.result_registry import ResultExtractorRegistry

__all__ = [
    "ComponentRegistry",
    "ConfigRegistry",
    "ResultExtractorRegistry",
]
