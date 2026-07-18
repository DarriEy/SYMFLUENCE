# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""MODFLOW 6 (USGS Modular Groundwater Flow Model) Integration.

This module implements MODFLOW 6 support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install modflow`
- Preprocessing (generates MODFLOW 6 text input files)
- Model execution (standalone single-cell lumped model)
- Result extraction (head, drain discharge)
- SUMMA → MODFLOW coupling (recharge → baseflow)
- Postprocessing (combined surface + subsurface flow)

MODFLOW 6 is the USGS modular groundwater flow model. In SYMFLUENCE
it is used as a lumped single-cell groundwater box coupled with land
surface models to produce physically-based baseflow separation.

Configuration Parameters:
    MODFLOW_INSTALL_PATH: Path to MODFLOW 6 installation
    MODFLOW_EXE: Executable name (default: mf6)
    MODFLOW_K: Hydraulic conductivity (m/d)
    MODFLOW_SY: Specific yield
    MODFLOW_TOP/BOT: Aquifer top/bottom elevation (m)
    MODFLOW_DRAIN_ELEVATION: Drain outlet elevation (m)
    MODFLOW_DRAIN_CONDUCTANCE: Drain conductance (m2/d)
    MODFLOW_COUPLING_SOURCE: Land surface model for coupling (default: SUMMA)

References:
    Langevin, C.D., et al. (2017): Documentation for the MODFLOW 6
    Groundwater Flow Model. USGS Techniques and Methods 6-A55.

    https://github.com/MODFLOW-ORG/modflow6
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'MODFLOWPreProcessor': ('.preprocessor', 'MODFLOWPreProcessor'),
    'MODFLOWRunner': ('.runner', 'MODFLOWRunner'),
    'MODFLOWResultExtractor': ('.extractor', 'MODFLOWResultExtractor'),
    'MODFLOWPostProcessor': ('.postprocessor', 'MODFLOWPostProcessor'),
    'MODFLOWPlotter': ('.plotter', 'MODFLOWPlotter'),
}


def __getattr__(name: str):
    """Lazy import handler for MODFLOW module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['MODFLOWConfigAdapter'])


__all__ = [
    "MODFLOWPreProcessor",
    "MODFLOWRunner",
    "MODFLOWResultExtractor",
    "MODFLOWPostProcessor",
    "MODFLOWConfigAdapter",
    "MODFLOWPlotter",
]

from symfluence.core.registry import model_manifest

from .config import MODFLOWConfigAdapter


def register() -> None:
    """Register MODFLOW components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "MODFLOW",
        config_adapter=MODFLOWConfigAdapter,
        build_instructions_module="symfluence.models.modflow.build_instructions",
    )
    base = 'symfluence.models.modflow'
    R.preprocessors.add_lazy("MODFLOW", f"{base}.preprocessor.MODFLOWPreProcessor")
    R.runners.add_lazy("MODFLOW", f"{base}.runner.MODFLOWRunner")
    R.postprocessors.add_lazy("MODFLOW", f"{base}.postprocessor.MODFLOWPostProcessor")
    R.result_extractors.add_lazy("MODFLOW", f"{base}.extractor.MODFLOWResultExtractor")
    R.plotters.add_lazy("MODFLOW", f"{base}.plotter.MODFLOWPlotter")
    R.optimizers.add_lazy("COUPLED_GW", f"{base}.calibration.optimizer.CoupledGWModelOptimizer")
    R.workers.add_lazy("COUPLED_GW", f"{base}.calibration.worker.CoupledGWWorker")
    R.parameter_managers.add_lazy("COUPLED_GW", f"{base}.calibration.parameter_manager.CoupledGWParameterManager")


if TYPE_CHECKING:
    from .extractor import MODFLOWResultExtractor
    from .plotter import MODFLOWPlotter
    from .postprocessor import MODFLOWPostProcessor
    from .preprocessor import MODFLOWPreProcessor
    from .runner import MODFLOWRunner
