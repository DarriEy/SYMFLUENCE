# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""FUSE (Framework for Understanding Structural Errors) Hydrological Model.

This module implements the FUSE modular modeling framework, which enables systematic
exploration of model structural uncertainty by combining different representations
of hydrological processes. FUSE can generate up to 1,248 unique model structures
by mixing and matching components for upper/lower soil zones, percolation, routing,
evaporation, and baseflow.

Model Architecture:
    1. **Upper Zone**: Tension storage, free storage with configurable overflow/drainage
    2. **Lower Zone**: Single or dual baseflow reservoirs with linear/power-law release
    3. **Percolation**: Saturation excess, field capacity, or demand-based drainage
    4. **Surface Runoff**: Infiltration excess (Horton) or saturation excess (Dunne)
    5. **Snow Module**: Temperature-index snowmelt with optional elevation bands

Design Rationale:
    FUSE addresses the challenge of model structural uncertainty:
    - Most calibration focuses only on parameter uncertainty
    - Different process representations can yield equally good fits but different predictions
    - FUSE enables ensemble runs across multiple structures for robust uncertainty estimates
    - Structure selection can be automated via structure ensemble calibration

Spatial Modes:
    lumped: Single spatial unit representing entire catchment
    semi-distributed: Multiple HRUs with elevation bands for snow processes
    distributed: Grid-based or subcatchment-based with optional mizuRoute routing

Key Components:
    FUSEPreProcessor: Forcing preparation, spatial setup, file manager generation
    FUSERunner: Model execution with structure selection and parameter mapping
    FUSEPostProcessor: Output extraction and result formatting
    FuseStructureAnalyzer: Ensemble analysis comparing different model structures

Configuration Parameters:
    FUSE_SPATIAL_MODE: Spatial discretization (default: 'lumped')
    FUSE_N_ELEVATION_BANDS: Number of elevation bands for snow (default: 1)
    FUSE_ROUTING_INTEGRATION: Routing model (default: 'default', options: 'none', 'mizuroute')
    FUSE_DECISION_OPTIONS: Structure decision dictionary for ensemble runs
    SETTINGS_FUSE_PARAMS_TO_CALIBRATE: Calibration parameters
        (default: 'MAXWATR_1,MAXWATR_2,BASERTE,QB_POWR,TIMEDELAY,PERCRTE,FRACTEN,RTFRAC1,MBASE,MFMAX,MFMIN,PXTEMP,LAPSE')

Typical Workflow:
    1. Initialize FUSEPreProcessor with configuration
    2. Process forcing data (precipitation, temperature, PET)
    3. Generate elevation bands if semi-distributed mode
    4. Create file manager and control files
    5. Execute FUSE via FUSERunner for one or more structures
    6. Postprocess outputs, optionally analyze structure ensemble

Limitations and Considerations:
    - Elevation band mode requires DEM and careful band delineation
    - Structure ensemble runs increase computational cost significantly
    - Some structure combinations may be physically inconsistent
    - Requires FUSE executable built from source (Fortran)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — execution, analysis, and plotting classes pull the
# geospatial/matplotlib stacks and must not load at plugin-discovery time.
_LAZY_IMPORTS = {
    'FuseElevationBandManager': ('.elevation_band_manager', 'FuseElevationBandManager'),
    'FuseForcingProcessor': ('.forcing_processor', 'FuseForcingProcessor'),
    'FUSEPostProcessor': ('.postprocessor', 'FUSEPostProcessor'),
    'FUSEPreProcessor': ('.preprocessor', 'FUSEPreProcessor'),
    'FUSERunner': ('.runner', 'FUSERunner'),
    'FuseStructureAnalyzer': ('.structure_analyzer', 'FuseStructureAnalyzer'),
    'FuseSyntheticDataGenerator': ('.synthetic_data_generator', 'FuseSyntheticDataGenerator'),
    'visualize_fuse': ('.visualizer', 'visualize_fuse'),
    'FUSEResultExtractor': ('.extractor', 'FUSEResultExtractor'),
    'FUSEPlotter': ('.plotter', 'FUSEPlotter'),
}


def __getattr__(name: str):
    """Lazy import handler for FUSE module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_IMPORTS.keys()) + ['FUSEConfigAdapter'])


__all__ = [
    # Main classes (public API)
    'FUSEPreProcessor',
    'FUSERunner',
    'FUSEPostProcessor',
    'FuseStructureAnalyzer',
    # Manager classes (advanced usage)
    'FuseForcingProcessor',
    'FuseElevationBandManager',
    'FuseSyntheticDataGenerator',
]

# Register all FUSE components via unified registry
from symfluence.core.registry import model_manifest

from .config import FUSEConfigAdapter


def register() -> None:
    """Register FUSE components with the unified registry.

    Execution, extraction, analysis, and plotting classes are registered
    lazily — imported on first registry access rather than at
    plugin-discovery time.
    """
    from symfluence.core.registries import Registries as R
    model_manifest(
        "FUSE",
        config_adapter=FUSEConfigAdapter,
        build_instructions_module="symfluence.models.fuse.build_instructions",
    )
    base = 'symfluence.models.fuse'
    R.preprocessors.add_lazy("FUSE", f"{base}.preprocessor.FUSEPreProcessor")
    R.base_settings.add("FUSE", "symfluence.models.fuse")
    R.runners.add_lazy("FUSE", f"{base}.runner.FUSERunner", runner_method='run_fuse')
    R.postprocessors.add_lazy("FUSE", f"{base}.postprocessor.FUSEPostProcessor")
    R.visualizers.add_lazy("FUSE", f"{base}.visualizer.visualize_fuse")
    R.result_extractors.add_lazy("FUSE", f"{base}.extractor.FUSEResultExtractor")
    R.decision_analyzers.add_lazy("FUSE", f"{base}.structure_analyzer.FuseStructureAnalyzer")
    R.plotters.add_lazy("FUSE", f"{base}.plotter.FUSEPlotter")
    # Calibration bounds are owned by this package (service-decomposition
    # item 2): registering here means plugin discovery is what makes them
    # servable, so a bound change never needs a core release.
    from .parameter_bounds import register_bounds
    register_bounds()


if TYPE_CHECKING:
    from .elevation_band_manager import FuseElevationBandManager
    from .extractor import FUSEResultExtractor
    from .forcing_processor import FuseForcingProcessor
    from .plotter import FUSEPlotter
    from .postprocessor import FUSEPostProcessor
    from .preprocessor import FUSEPreProcessor
    from .runner import FUSERunner
    from .structure_analyzer import FuseStructureAnalyzer
    from .synthetic_data_generator import FuseSyntheticDataGenerator
    from .visualizer import visualize_fuse
