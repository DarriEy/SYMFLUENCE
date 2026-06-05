# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PCR-GLOBWB 2.0 Global Distributed Hydrological Model."""
from __future__ import annotations

from .config import PCRGLOBWBConfigAdapter
from .extractor import PCRGLOBWBResultExtractor
from .postprocessor import PCRGLOBWBPostProcessor
from .preprocessor import PCRGLOBWBPreProcessor
from .runner import PCRGLOBWBRunner

__all__ = [
    "PCRGLOBWBPreProcessor",
    "PCRGLOBWBRunner",
    "PCRGLOBWBResultExtractor",
    "PCRGLOBWBPostProcessor",
    "PCRGLOBWBConfigAdapter",
]

from symfluence.core.registry import model_manifest


def register() -> None:
    """Register PCRGLOBWB components with the unified registry."""
    model_manifest(
        "PCRGLOBWB",
        preprocessor=PCRGLOBWBPreProcessor,
        runner=PCRGLOBWBRunner,
        result_extractor=PCRGLOBWBResultExtractor,
        config_adapter=PCRGLOBWBConfigAdapter,
        build_instructions_module="symfluence.models.pcrglobwb.build_instructions",
    )

try:
    from .calibration import PCRGLOBWBModelOptimizer  # noqa: F401
except ImportError:
    pass
