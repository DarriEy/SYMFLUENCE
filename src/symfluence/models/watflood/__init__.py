# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""WATFLOOD (Kouwen) Distributed Flood Forecasting Model.

WATFLOOD is a physically-based, distributed hydrological model using
Grouped Response Units (GRUs) on a regular grid with internal channel
routing. It requires only precipitation and temperature forcing
(simplified energy balance).

Input Files:
    .shd: Watershed definition (GRU grid)
    .par: Parameters (per-land-class blocks)
    .evt: Event control
    .met/.rag: Meteorological forcing

Output Files:
    .tb0: Time-bin streamflow/state output
    .csv: Summary output

References:
    Kouwen, N. (2018): WATFLOOD/WATROUTE Hydrological Model Routing
    & Flood Forecasting System. University of Waterloo.
"""
from .config import WATFLOODConfigAdapter
from .extractor import WATFLOODResultExtractor
from .postprocessor import WATFLOODPostProcessor
from .preprocessor import WATFLOODPreProcessor
from .runner import WATFLOODRunner

__all__ = [
    "WATFLOODPreProcessor",
    "WATFLOODRunner",
    "WATFLOODResultExtractor",
    "WATFLOODPostProcessor",
    "WATFLOODConfigAdapter",
]

# Register all WATFLOOD components via unified registry
from symfluence.core.registry import model_manifest

model_manifest(
    "WATFLOOD",
    preprocessor=WATFLOODPreProcessor,
    runner=WATFLOODRunner,
    result_extractor=WATFLOODResultExtractor,
    config_adapter=WATFLOODConfigAdapter,
    build_instructions_module="symfluence.models.watflood.build_instructions",
)

# Register calibration components
try:
    from .calibration import WATFLOODModelOptimizer  # noqa: F401
except ImportError:
    pass
