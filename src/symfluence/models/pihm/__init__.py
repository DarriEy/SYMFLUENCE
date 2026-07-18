# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PIHM (Penn State Integrated Hydrologic Model) Integration.

This module implements PIHM support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install pihm`
- Preprocessing (generates PIHM input files for lumped single-element mesh)
- Model execution (standalone via mm-pihm)
- Result extraction (river flux, groundwater head)
- SUMMA → PIHM coupling (recharge → baseflow)
- Postprocessing (combined surface + subsurface flow)

PIHM is a finite-volume, unstructured-mesh, fully-coupled
surface-subsurface model solving Richards equation + diffusion wave
overland flow + 1D channel routing. Uses SUNDIALS CVODE solver.

Configuration Parameters:
    PIHM_INSTALL_PATH: Path to PIHM installation
    PIHM_EXE: Executable name (default: pihm)
    PIHM_K_SAT: Saturated hydraulic conductivity (m/s)
    PIHM_POROSITY: Total porosity
    PIHM_VG_ALPHA: van Genuchten alpha (1/m)
    PIHM_VG_N: van Genuchten n
    PIHM_COUPLING_SOURCE: Land surface model for coupling (default: SUMMA)

References:
    Qu, Y. & Duffy, C.J. (2007): A semidiscrete finite volume
    formulation for multiprocess watershed simulation.
    Water Resources Research 43(8).

    https://github.com/PSUmodeling/MM-PIHM
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and plotting classes pull the heavy
# model/plotting stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'PIHMPreProcessor': ('.preprocessor', 'PIHMPreProcessor'),
    'PIHMRunner': ('.runner', 'PIHMRunner'),
    'PIHMResultExtractor': ('.extractor', 'PIHMResultExtractor'),
    'PIHMPostProcessor': ('.postprocessor', 'PIHMPostProcessor'),
    'PIHMPlotter': ('.plotter', 'PIHMPlotter'),
}


def __getattr__(name: str):
    """Lazy import handler for PIHM module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['PIHMConfigAdapter'])


__all__ = [
    "PIHMPreProcessor",
    "PIHMRunner",
    "PIHMResultExtractor",
    "PIHMPostProcessor",
    "PIHMConfigAdapter",
    "PIHMPlotter",
]

from symfluence.core.registry import model_manifest

from .config import PIHMConfigAdapter


def register() -> None:
    """Register PIHM components with the unified registry.

    Execution and plotting classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "PIHM",
        config_adapter=PIHMConfigAdapter,
        build_instructions_module="symfluence.models.pihm.build_instructions",
    )
    base = 'symfluence.models.pihm'
    R.preprocessors.add_lazy("PIHM", f"{base}.preprocessor.PIHMPreProcessor")
    R.runners.add_lazy("PIHM", f"{base}.runner.PIHMRunner")
    R.postprocessors.add_lazy("PIHM", f"{base}.postprocessor.PIHMPostProcessor")
    R.result_extractors.add_lazy("PIHM", f"{base}.extractor.PIHMResultExtractor")
    R.plotters.add_lazy("PIHM", f"{base}.plotter.PIHMPlotter")


if TYPE_CHECKING:
    from .extractor import PIHMResultExtractor
    from .plotter import PIHMPlotter
    from .postprocessor import PIHMPostProcessor
    from .preprocessor import PIHMPreProcessor
    from .runner import PIHMRunner
