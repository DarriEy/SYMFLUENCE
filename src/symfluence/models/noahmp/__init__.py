# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Noah-MP (noah-owp-modular) — NOAA-OWP Standalone Land Surface Model."""
from __future__ import annotations

from .config import NoahMPConfigAdapter
from .extractor import NoahMPResultExtractor
from .postprocessor import NoahMPPostProcessor
from .preprocessor import NoahMPPreProcessor
from .runner import NoahMPRunner

__all__ = [
    "NoahMPRunner",
    "NoahMPResultExtractor",
    "NoahMPPostProcessor",
    "NoahMPPreProcessor",
    "NoahMPConfigAdapter",
]

# Register Noah-MP config adapter via unified registry
from symfluence.core.registry import model_manifest


def register() -> None:
    """Register NOAHMP components with the unified registry."""
    model_manifest(
        "NOAHMP",
        preprocessor=NoahMPPreProcessor,
        runner=NoahMPRunner,
        result_extractor=NoahMPResultExtractor,
        config_adapter=NoahMPConfigAdapter,
        postprocessor=NoahMPPostProcessor,
        build_instructions_module="symfluence.models.noahmp.build_instructions",
    )

# Register calibration components with OptimizerRegistry
try:
    from .calibration import NoahMPModelOptimizer  # noqa: F401
except ImportError:
    pass  # Calibration dependencies optional
