# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CWatM (Community Water Model) — IIASA Global Hydrological Model."""
from __future__ import annotations

from .config import CWatMConfigAdapter
from .extractor import CWatMResultExtractor
from .postprocessor import CWatMPostProcessor
from .preprocessor import CWatMPreProcessor
from .runner import CWatMRunner

__all__ = [
    "CWatMPreProcessor",
    "CWatMRunner",
    "CWatMResultExtractor",
    "CWatMPostProcessor",
    "CWatMConfigAdapter",
]

from symfluence.core.registry import model_manifest


def register() -> None:
    """Register CWATM components with the unified registry."""
    model_manifest(
        "CWATM",
        preprocessor=CWatMPreProcessor,
        runner=CWatMRunner,
        result_extractor=CWatMResultExtractor,
        config_adapter=CWatMConfigAdapter,
        build_instructions_module="symfluence.models.cwatm.build_instructions",
    )
