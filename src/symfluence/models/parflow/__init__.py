# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ParFlow Integrated Hydrologic Model Integration.

This module implements ParFlow support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install parflow`
- Preprocessing (generates ParFlow .pfidb run files via pftools API)
- Model execution (standalone variably-saturated + overland flow)
- Result extraction (pressure head, saturation, overland flow from .pfb)
- SUMMA -> ParFlow coupling (recharge -> subsurface flow)
- Postprocessing (combined surface + subsurface flow)

ParFlow is a parallel integrated hydrologic model that solves
variably-saturated flow (Richards equation) and overland flow.
In SYMFLUENCE it is used as an alternative to MODFLOW for coupled
land surface + groundwater simulations with full vadose zone support.

Configuration Parameters:
    PARFLOW_INSTALL_PATH: Path to ParFlow installation
    PARFLOW_EXE: Executable name (default: parflow)
    PARFLOW_DIR: ParFlow install root (sets PARFLOW_DIR env var)
    PARFLOW_K_SAT: Saturated hydraulic conductivity (m/hr)
    PARFLOW_POROSITY: Porosity (dimensionless)
    PARFLOW_VG_ALPHA: van Genuchten alpha (1/m)
    PARFLOW_VG_N: van Genuchten n (dimensionless, > 1)
    PARFLOW_TOP/BOT: Domain top/bottom elevation (m)
    PARFLOW_MANNINGS_N: Manning's roughness for overland flow
    PARFLOW_COUPLING_SOURCE: Land surface model for coupling (default: SUMMA)

References:
    Kollet, S.J. & Maxwell, R.M. (2006): Integrated surface-groundwater
    flow modeling: A free-surface overland flow boundary condition in a
    parallel groundwater flow model. Advances in Water Resources 29(7).

    https://github.com/parflow/parflow
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'ParFlowPreProcessor': ('.preprocessor', 'ParFlowPreProcessor'),
    'ParFlowRunner': ('.runner', 'ParFlowRunner'),
    'ParFlowResultExtractor': ('.extractor', 'ParFlowResultExtractor'),
    'ParFlowPostProcessor': ('.postprocessor', 'ParFlowPostProcessor'),
    'ParFlowPlotter': ('.plotter', 'ParFlowPlotter'),
}


def __getattr__(name: str):
    """Lazy import handler for ParFlow module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['ParFlowConfigAdapter'])


__all__ = [
    "ParFlowPreProcessor",
    "ParFlowRunner",
    "ParFlowResultExtractor",
    "ParFlowPostProcessor",
    "ParFlowConfigAdapter",
    "ParFlowPlotter",
]

from symfluence.core.registry import model_manifest

from .config import ParFlowConfigAdapter


def register() -> None:
    """Register PARFLOW components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "PARFLOW",
        config_adapter=ParFlowConfigAdapter,
        build_instructions_module="symfluence.models.parflow.build_instructions",
    )
    base = 'symfluence.models.parflow'
    R.preprocessors.add_lazy("PARFLOW", f"{base}.preprocessor.ParFlowPreProcessor")
    R.runners.add_lazy("PARFLOW", f"{base}.runner.ParFlowRunner", runner_method="run_parflow")
    R.postprocessors.add_lazy("PARFLOW", f"{base}.postprocessor.ParFlowPostProcessor")
    R.result_extractors.add_lazy("PARFLOW", f"{base}.extractor.ParFlowResultExtractor")
    R.plotters.add_lazy("PARFLOW", f"{base}.plotter.ParFlowPlotter")
    R.optimizers.add_lazy("PARFLOW", f"{base}.calibration.optimizer.ParFlowModelOptimizer")
    R.workers.add_lazy("PARFLOW", f"{base}.calibration.worker.ParFlowWorker")
    R.parameter_managers.add_lazy("PARFLOW", f"{base}.calibration.parameter_manager.ParFlowParameterManager")


if TYPE_CHECKING:
    from .extractor import ParFlowResultExtractor
    from .plotter import ParFlowPlotter
    from .postprocessor import ParFlowPostProcessor
    from .preprocessor import ParFlowPreProcessor
    from .runner import ParFlowRunner
