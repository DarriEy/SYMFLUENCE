# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CLM (Community Land Model / CTSM 5.x) Integration.

This module implements CLM5 support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install clm`
- Preprocessing (domain, surface data, parameters, forcing)
- Model execution (standalone single-point)
- Result extraction
- Calibration support (26 parameters)

CLM5 is the land component of CESM (Community Earth System Model).
It is the most physics-heavy LSM in the SYMFLUENCE ensemble, covering
biogeophysics, biogeochemistry, hydrology, snow, and vegetation dynamics.

Key design: CIME is used only for the one-time build. At calibration
runtime, the compiled cesm.exe is invoked directly with modified
parameter NetCDF + namelists to avoid per-iteration rebuild overhead.

Model Architecture:
    CLM uses a single-point structure with:

    1. **Domain File**: NetCDF defining model grid (xc, yc, mask, frac, area)
    2. **Surface Data**: NetCDF with soil, PFT, topography properties
    3. **Parameter File**: clm5_params.nc with global CLM parameters
    4. **Forcing Files**: DATM stream format (one NetCDF/year)
    5. **Namelists**: user_nl_clm, drv_in, datm_in

Configuration Parameters:
    CLM_INSTALL_PATH: Path to CTSM installation
    CLM_EXE: Executable name (default: cesm.exe)
    CLM_PARAMS_TO_CALIBRATE: Calibration parameters
    CLM_TIMEOUT: Execution timeout in seconds

References:
    Lawrence, D. M., et al. (2019): The Community Land Model version 5:
    Description of new features, benchmarking, and impact of forcing
    uncertainty. JAMES, 11, 4245-4287.

    https://github.com/ESCOMP/CTSM
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'CLMPreProcessor': ('.preprocessor', 'CLMPreProcessor'),
    'CLMRunner': ('.runner', 'CLMRunner'),
    'CLMResultExtractor': ('.extractor', 'CLMResultExtractor'),
    'CLMPostProcessor': ('.postprocessor', 'CLMPostProcessor'),
    'CLMModelOptimizer': ('.calibration', 'CLMModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for CLM module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['CLMConfigAdapter'])


__all__ = [
    "CLMPreProcessor",
    "CLMRunner",
    "CLMResultExtractor",
    "CLMPostProcessor",
    "CLMConfigAdapter",
]

from symfluence.core.registry import model_manifest

from .config import CLMConfigAdapter


def register() -> None:
    """Register CLM components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "CLM",
        config_adapter=CLMConfigAdapter,
        build_instructions_module="symfluence.models.clm.build_instructions",
    )
    base = 'symfluence.models.clm'
    R.preprocessors.add_lazy("CLM", f"{base}.preprocessor.CLMPreProcessor")
    R.runners.add_lazy("CLM", f"{base}.runner.CLMRunner")
    R.postprocessors.add_lazy("CLM", f"{base}.postprocessor.CLMPostProcessor")
    R.result_extractors.add_lazy("CLM", f"{base}.extractor.CLMResultExtractor")
    R.optimizers.add_lazy("CLM", f"{base}.calibration.optimizer.CLMModelOptimizer")
    R.workers.add_lazy("CLM", f"{base}.calibration.worker.CLMWorker")
    R.base_settings.add("CLM", "symfluence.models.clm")
    R.parameter_managers.add_lazy("CLM", f"{base}.calibration.parameter_manager.CLMParameterManager")


if TYPE_CHECKING:
    from .calibration import CLMModelOptimizer
    from .extractor import CLMResultExtractor
    from .postprocessor import CLMPostProcessor
    from .preprocessor import CLMPreProcessor
    from .runner import CLMRunner
