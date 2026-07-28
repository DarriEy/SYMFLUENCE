# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Noah-MP (noah-owp-modular) — NOAA-OWP Standalone Land Surface Model."""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'NoahMPPreProcessor': ('.preprocessor', 'NoahMPPreProcessor'),
    'NoahMPRunner': ('.runner', 'NoahMPRunner'),
    'NoahMPResultExtractor': ('.extractor', 'NoahMPResultExtractor'),
    'NoahMPPostProcessor': ('.postprocessor', 'NoahMPPostProcessor'),
    'NoahMPModelOptimizer': ('.calibration', 'NoahMPModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for Noah-MP module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['NoahMPConfigAdapter'])


__all__ = [
    "NoahMPRunner",
    "NoahMPResultExtractor",
    "NoahMPPostProcessor",
    "NoahMPPreProcessor",
    "NoahMPConfigAdapter",
]

# Register Noah-MP config adapter via unified registry
from symfluence.core.registry import model_manifest

from .config import NoahMPConfigAdapter


def register() -> None:
    """Register NOAHMP components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "NOAHMP",
        config_adapter=NoahMPConfigAdapter,
        build_instructions_module="symfluence.models.noahmp.build_instructions",
    )
    base = 'symfluence.models.noahmp'
    R.preprocessors.add_lazy("NOAHMP", f"{base}.preprocessor.NoahMPPreProcessor")
    R.runners.add_lazy("NOAHMP", f"{base}.runner.NoahMPRunner")
    R.postprocessors.add_lazy("NOAHMP", f"{base}.postprocessor.NoahMPPostProcessor")
    R.result_extractors.add_lazy("NOAHMP", f"{base}.extractor.NoahMPResultExtractor")
    R.optimizers.add_lazy("NOAHMP", f"{base}.calibration.optimizer.NoahMPModelOptimizer")
    R.workers.add_lazy("NOAHMP", f"{base}.calibration.worker.NoahMPWorker")
    R.base_settings.add("NOAH", "symfluence.models.noahmp")
    R.parameter_managers.add_lazy("NOAHMP", f"{base}.calibration.parameter_manager.NoahMPParameterManager")
    from .parameter_bounds import register_bounds
    register_bounds()


if TYPE_CHECKING:
    from .calibration import NoahMPModelOptimizer
    from .extractor import NoahMPResultExtractor
    from .postprocessor import NoahMPPostProcessor
    from .preprocessor import NoahMPPreProcessor
    from .runner import NoahMPRunner
