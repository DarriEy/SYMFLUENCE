# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Mixins for model preprocessors and runners."""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy re-exports (PEP 562): several mixins pull the geospatial/observation
# stacks, and this package is imported (via model base classes) during plugin
# discovery at `import symfluence`.
_LAZY_IMPORTS = {
    'DatasetBuilderMixin': ('.dataset_builder', 'DatasetBuilderMixin'),
    'ModelComponentMixin': ('.model_component', 'ModelComponentMixin'),
    'ObservationLoaderMixin': ('.observation_loader', 'ObservationLoaderMixin'),
    'OutputConverterMixin': ('.output_converter', 'OutputConverterMixin'),
    'PETCalculatorMixin': ('.pet_calculator', 'PETCalculatorMixin'),
    'SlurmExecutionMixin': ('.slurm_execution', 'SlurmExecutionMixin'),
    'SpatialModeDetectionMixin': ('.spatial_mode_mixin', 'SpatialModeDetectionMixin'),
    'SubprocessExecutionMixin': ('.subprocess_execution', 'SubprocessExecutionMixin'),
}


def __getattr__(name: str):
    """Lazy import handler for mixin re-exports."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        value = getattr(import_module(module_path, package=__name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(_LAZY_IMPORTS.keys())


if TYPE_CHECKING:
    from .dataset_builder import DatasetBuilderMixin
    from .model_component import ModelComponentMixin
    from .observation_loader import ObservationLoaderMixin
    from .output_converter import OutputConverterMixin
    from .pet_calculator import PETCalculatorMixin
    from .slurm_execution import SlurmExecutionMixin
    from .spatial_mode_mixin import SpatialModeDetectionMixin
    from .subprocess_execution import SubprocessExecutionMixin

__all__ = [
    'PETCalculatorMixin',
    'ObservationLoaderMixin',
    'DatasetBuilderMixin',
    'OutputConverterMixin',
    'ModelComponentMixin',
    'SpatialModeDetectionMixin',
    'SubprocessExecutionMixin',
    'SlurmExecutionMixin',
]
