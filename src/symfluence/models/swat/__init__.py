# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SWAT (Soil and Water Assessment Tool) Hydrological Model.

This module implements SWAT support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install swat`
- Preprocessing (TxtInOut directory with forcing, soil, management files)
- Model execution (swat_rel.exe from TxtInOut directory)
- Result extraction (output.rch fixed-width text)
- Calibration support (14 parameters across .bsn, .gw, .hru, .sol, .mgt files)

SWAT is a river basin scale model developed by the USDA Agricultural
Research Service (ARS) to predict the impact of land management practices
on water, sediment, and agricultural chemical yields in large complex
watersheds with varying soils, land use, and management conditions.

Model Architecture:
    SWAT uses a semi-distributed, HRU-based structure with:

    1. **TxtInOut Directory**: All input and output files in one directory
       - file.cio: Master control file
       - .pcp/.tmp: Precipitation and temperature forcing
       - .sub/.hru/.gw/.mgt/.sol: Sub-basin and HRU parameters
       - .bsn: Basin-level parameters

    2. **Output Files**: Fixed-width text format
       - output.rch: Reach-level results (FLOW_OUTcms, etc.)
       - output.sub: Sub-basin results
       - output.hru: HRU-level results

    3. **Calibration**: Parameters modified via text file editing
       - r__ (relative): new = original * (1 + change)
       - v__ (value replacement): new = change
       - a__ (absolute): new = original + change

Design Rationale:
    SWAT is well-suited for:
    - Agricultural watershed management
    - Water quality and sediment yield assessment
    - Land use change impact studies
    - Long-term continuous simulation

Key Components:
    SWATPreProcessor: TxtInOut directory and forcing file generation
    SWATRunner: Model execution with swat_rel.exe
    SWATResultExtractor: Output extraction from output.rch
    SWATPostProcessor: Streamflow extraction and unit handling
    SWATConfigAdapter: Configuration schema and validation

Configuration Parameters:
    SWAT_INSTALL_PATH: Path to SWAT installation
    SWAT_EXE: Executable name (default: swat_rel.exe)
    SWAT_TXTINOUT_DIR: TxtInOut directory name (default: TxtInOut)
    SWAT_SPATIAL_MODE: 'lumped' or 'semi_distributed'
    SWAT_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'CN2,ALPHA_BF,GW_DELAY,GWQMN,GW_REVAP,ESCO,SOL_AWC,SOL_K,SURLAG,SFTMP,SMTMP,SMFMX,SMFMN,TIMP')

Typical Workflow:
    1. Prepare TxtInOut directory with catchment data
    2. Generate forcing files (.pcp, .tmp) from ERA5 data
    3. Configure basin, sub-basin, HRU, groundwater, management, and soil files
    4. Create file.cio master control file
    5. Run swat_rel.exe from within TxtInOut
    6. Extract streamflow from output.rch

References:
    Arnold, J.G., Srinivasan, R., Muttiah, R.S., and Williams, J.R. (1998):
    Large area hydrologic modeling and assessment Part I: Model development.
    Journal of the American Water Resources Association, 34(1), 73-89.

    Neitsch, S.L., Arnold, J.G., Kiniry, J.R., and Williams, J.R. (2011):
    Soil and Water Assessment Tool Theoretical Documentation Version 2009.
    Texas Water Resources Institute Technical Report No. 406.

    https://github.com/WatershedModels/SWAT
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution and calibration classes pull the model/
# optimization stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'SWATPreProcessor': ('.preprocessor', 'SWATPreProcessor'),
    'SWATRunner': ('.runner', 'SWATRunner'),
    'SWATResultExtractor': ('.extractor', 'SWATResultExtractor'),
    'SWATPostProcessor': ('.postprocessor', 'SWATPostProcessor'),
    'SWATModelOptimizer': ('.calibration', 'SWATModelOptimizer'),
}


def __getattr__(name: str):
    """Lazy import handler for SWAT module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['SWATConfigAdapter'])


__all__ = [
    "SWATPreProcessor",
    "SWATRunner",
    "SWATResultExtractor",
    "SWATPostProcessor",
    "SWATConfigAdapter",
]

from symfluence.core.registry import model_manifest

from .config import SWATConfigAdapter


def register() -> None:
    """Register SWAT components with the unified registry.

    Execution and calibration classes are registered lazily — imported on
    first registry access rather than at plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "SWAT",
        config_adapter=SWATConfigAdapter,
        build_instructions_module="symfluence.models.swat.build_instructions",
    )
    base = 'symfluence.models.swat'
    R.preprocessors.add_lazy("SWAT", f"{base}.preprocessor.SWATPreProcessor")
    R.runners.add_lazy("SWAT", f"{base}.runner.SWATRunner")
    R.postprocessors.add_lazy("SWAT", f"{base}.postprocessor.SWATPostProcessor")
    R.result_extractors.add_lazy("SWAT", f"{base}.extractor.SWATResultExtractor")
    R.optimizers.add_lazy("SWAT", f"{base}.calibration.optimizer.SWATModelOptimizer")
    R.workers.add_lazy("SWAT", f"{base}.calibration.worker.SWATWorker")
    R.parameter_managers.add_lazy("SWAT", f"{base}.calibration.parameter_manager.SWATParameterManager")

    # Spatial capabilities are owned by this package (service-decomposition
    # item 2): declared at plugin-discovery time so core carries no per-model
    # spatial knowledge and a capability change never needs a core release.
    from symfluence.core.modeling.spatial_modes import (
        ModelSpatialCapability,
        SpatialMode,
        register_model_spatial_capability,
    )
    register_model_spatial_capability(
        "SWAT",
        ModelSpatialCapability(
            supported_modes={SpatialMode.LUMPED},
            default_mode=SpatialMode.LUMPED,
            requires_routing={SpatialMode.LUMPED: False},
            warning_message=(
                "SWAT is a semi-distributed model. Lumped mode uses "
                "a single-HRU/subbasin configuration."
            ),
        ),
    )


if TYPE_CHECKING:
    from .calibration import SWATModelOptimizer
    from .extractor import SWATResultExtractor
    from .postprocessor import SWATPostProcessor
    from .preprocessor import SWATPreProcessor
    from .runner import SWATRunner
