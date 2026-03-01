# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GSFLOW (coupled PRMS + MODFLOW-NWT) Hydrological Model.

GSFLOW is a USGS coupled groundwater-surface-water model that integrates
PRMS (surface/soil processes) with MODFLOW-NWT (saturated zone) via SFR
and UZF packages for bidirectional exchange.

Supports three operation modes:
- PRMS: Surface processes only
- MODFLOW: Groundwater only
- COUPLED: Full bidirectional PRMS↔MODFLOW-NWT exchange (default)

References:
    Markstrom, S.L., et al. (2008): GSFLOW---Coupled Ground-Water and
    Surface-Water Flow Model. USGS Techniques and Methods 6-D1.
"""
from .config import GSFLOWConfigAdapter
from .extractor import GSFLOWResultExtractor
from .postprocessor import GSFLOWPostProcessor
from .preprocessor import GSFLOWPreProcessor
from .runner import GSFLOWRunner

__all__ = [
    "GSFLOWPreProcessor",
    "GSFLOWRunner",
    "GSFLOWResultExtractor",
    "GSFLOWPostProcessor",
    "GSFLOWConfigAdapter",
]

# Register all GSFLOW components via unified registry
from symfluence.core.registry import model_manifest

model_manifest(
    "GSFLOW",
    preprocessor=GSFLOWPreProcessor,
    runner=GSFLOWRunner,
    result_extractor=GSFLOWResultExtractor,
    config_adapter=GSFLOWConfigAdapter,
    build_instructions_module="symfluence.models.gsflow.build_instructions",
)

# Register calibration components
try:
    from .calibration import GSFLOWModelOptimizer  # noqa: F401
except ImportError:
    pass
