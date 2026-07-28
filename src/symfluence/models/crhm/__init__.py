# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CRHM (Cold Regions Hydrological Model).

This module implements CRHM support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install crhm`
- Preprocessing (project file, observation/forcing file)
- Model execution (crhm binary)
- Result extraction
- Calibration support

CRHM is a physically-based, object-oriented hydrological model designed
specifically for cold-region processes. It includes modules for blowing
snow (PBSM), energy-balance snowmelt (EBSM), frozen soil infiltration,
and prairie/alpine hydrology.

Model Architecture:
    CRHM uses a module-based structure with:

    1. **Project File** (.prj): Text file defining model structure
       - Module selection and connectivity
       - Parameter values (key-value format)
       - Basin and HRU definitions

    2. **Observation File** (.obs): Text file with forcing data
       - Header with variable names and metadata
       - Space-separated meteorological data
       - Variables: temperature, precipitation, humidity, wind, radiation

    3. **Output**: CSV files with simulated variables
       - Date, flow, SWE, soil moisture, evapotranspiration

Design Rationale:
    CRHM is well-suited for:
    - Cold-region hydrology (Arctic, subarctic, alpine)
    - Blowing snow redistribution and sublimation
    - Energy-balance snowmelt processes
    - Frozen soil infiltration dynamics
    - Prairie and wetland hydrology

Key Components:
    CRHMPreProcessor: Observation file generation from ERA5 forcing
    CRHMRunner: Model execution with crhm binary
    CRHMResultExtractor: Output extraction from CSV results

Configuration Parameters:
    CRHM_INSTALL_PATH: Path to CRHM installation
    CRHM_EXE: Executable name (default: crhm)
    CRHM_PROJECT_FILE: Project file name (default: model.prj)
    CRHM_OBSERVATION_FILE: Observation file name (default: forcing.obs)
    CRHM_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'basin_area,Ht,Asnow,inhibit_evap,Ksat,soil_rechr_max,
                   soil_moist_max,soil_gw_K,Sdmax,fetch')

Typical Workflow:
    1. Prepare .obs observation/forcing file from ERA5 data
    2. Configure .prj project file with modules and parameters
    3. Run CRHM binary
    4. Extract and analyze CSV outputs

References:
    Pomeroy, J.W., et al. (2007): The Cold Regions Hydrological Model:
    a platform for basing process representation and model structure on
    physical evidence. Hydrological Processes, 21(19), 2650-2667.

    https://github.com/CentreForHydrology/CRHM
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'CRHMPreProcessor': ('.preprocessor', 'CRHMPreProcessor'),
    'CRHMRunner': ('.runner', 'CRHMRunner'),
    'CRHMResultExtractor': ('.extractor', 'CRHMResultExtractor'),
    'CRHMPostProcessor': ('.postprocessor', 'CRHMPostProcessor'),
    'CRHMModelOptimizer': ('.calibration', 'CRHMModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for CRHM module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['CRHMConfigAdapter'])


__all__ = [
    "CRHMPreProcessor",
    "CRHMRunner",
    "CRHMResultExtractor",
    "CRHMPostProcessor",
    "CRHMConfigAdapter",
]

from symfluence.core.registry import model_manifest

from .config import CRHMConfigAdapter


def register() -> None:
    """Register CRHM components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "CRHM",
        config_adapter=CRHMConfigAdapter,
        build_instructions_module="symfluence.models.crhm.build_instructions",
    )
    base = 'symfluence.models.crhm'
    R.preprocessors.add_lazy("CRHM", f"{base}.preprocessor.CRHMPreProcessor")
    R.runners.add_lazy("CRHM", f"{base}.runner.CRHMRunner")
    R.postprocessors.add_lazy("CRHM", f"{base}.postprocessor.CRHMPostProcessor")
    R.result_extractors.add_lazy("CRHM", f"{base}.extractor.CRHMResultExtractor")
    R.optimizers.add_lazy("CRHM", f"{base}.calibration.optimizer.CRHMModelOptimizer")
    R.workers.add_lazy("CRHM", f"{base}.calibration.worker.CRHMWorker")
    R.parameter_managers.add_lazy("CRHM", f"{base}.calibration.parameter_manager.CRHMParameterManager")

    from symfluence.core.modeling.spatial_modes import (
        ModelSpatialCapability,
        SpatialMode,
        register_model_spatial_capability,
    )
    register_model_spatial_capability(
        "CRHM",
        ModelSpatialCapability(
            supported_modes={SpatialMode.LUMPED},
            default_mode=SpatialMode.LUMPED,
            requires_routing={SpatialMode.LUMPED: False},
            warning_message=(
                "CRHM is a cold-region hydrological model. Lumped mode uses "
                "a single-HRU configuration with blowing snow and frozen soil."
            ),
        ),
    )


if TYPE_CHECKING:
    from .calibration import CRHMModelOptimizer
    from .extractor import CRHMResultExtractor
    from .postprocessor import CRHMPostProcessor
    from .preprocessor import CRHMPreProcessor
    from .runner import CRHMRunner
