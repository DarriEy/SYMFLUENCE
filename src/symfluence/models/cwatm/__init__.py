# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CWatM (Community Water Model) — IIASA Global Hydrological Model."""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution classes pull the model stack and must not
# load at plugin-discovery time.
_LAZY_IMPORTS = {
    'CWatMPreProcessor': ('.preprocessor', 'CWatMPreProcessor'),
    'CWatMRunner': ('.runner', 'CWatMRunner'),
    'CWatMResultExtractor': ('.extractor', 'CWatMResultExtractor'),
    'CWatMPostProcessor': ('.postprocessor', 'CWatMPostProcessor'),
}


def __getattr__(name: str):
    """Lazy import handler for CWatM module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['CWatMConfigAdapter'])


__all__ = [
    "CWatMPreProcessor",
    "CWatMRunner",
    "CWatMResultExtractor",
    "CWatMPostProcessor",
    "CWatMConfigAdapter",
]

from symfluence.core.registry import model_manifest

from .config import CWatMConfigAdapter


def register() -> None:
    """Register CWATM components with the unified registry.

    Execution classes are registered lazily — imported on first registry
    access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "CWATM",
        config_adapter=CWatMConfigAdapter,
        build_instructions_module="symfluence.models.cwatm.build_instructions",
    )
    base = 'symfluence.models.cwatm'
    R.preprocessors.add_lazy("CWATM", f"{base}.preprocessor.CWatMPreProcessor")
    R.runners.add_lazy("CWATM", f"{base}.runner.CWatMRunner")
    R.postprocessors.add_lazy("CWATM", f"{base}.postprocessor.CWatMPostProcessor")
    R.result_extractors.add_lazy("CWATM", f"{base}.extractor.CWatMResultExtractor")


if TYPE_CHECKING:
    from .extractor import CWatMResultExtractor
    from .postprocessor import CWatMPostProcessor
    from .preprocessor import CWatMPreProcessor
    from .runner import CWatMRunner
