# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""LISFLOOD Distributed Hydrological Model (JRC/Deltares)."""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'LisfloodPreProcessor': ('.preprocessor', 'LisfloodPreProcessor'),
    'LisfloodRunner': ('.runner', 'LisfloodRunner'),
    'LisfloodResultExtractor': ('.extractor', 'LisfloodResultExtractor'),
    'LisfloodPostProcessor': ('.postprocessor', 'LisfloodPostProcessor'),
    'LisfloodModelOptimizer': ('.calibration', 'LisfloodModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for LISFLOOD module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['LisfloodConfigAdapter'])


__all__ = [
    "LisfloodPreProcessor",
    "LisfloodRunner",
    "LisfloodResultExtractor",
    "LisfloodPostProcessor",
    "LisfloodConfigAdapter",
]

from symfluence.core.registry import model_manifest

from .config import LisfloodConfigAdapter


def register() -> None:
    """Register LISFLOOD components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "LISFLOOD",
        config_adapter=LisfloodConfigAdapter,
        build_instructions_module="symfluence.models.lisflood.build_instructions",
    )
    base = 'symfluence.models.lisflood'
    R.preprocessors.add_lazy("LISFLOOD", f"{base}.preprocessor.LisfloodPreProcessor")
    R.runners.add_lazy("LISFLOOD", f"{base}.runner.LisfloodRunner")
    R.postprocessors.add_lazy("LISFLOOD", f"{base}.postprocessor.LisfloodPostProcessor")
    R.result_extractors.add_lazy("LISFLOOD", f"{base}.extractor.LisfloodResultExtractor")
    R.optimizers.add_lazy("LISFLOOD", f"{base}.calibration.optimizer.LisfloodModelOptimizer")
    R.workers.add_lazy("LISFLOOD", f"{base}.calibration.worker.LisfloodWorker")
    R.parameter_managers.add_lazy("LISFLOOD", f"{base}.calibration.parameter_manager.LisfloodParameterManager")


if TYPE_CHECKING:
    from .calibration import LisfloodModelOptimizer
    from .extractor import LisfloodResultExtractor
    from .postprocessor import LisfloodPostProcessor
    from .preprocessor import LisfloodPreProcessor
    from .runner import LisfloodRunner
