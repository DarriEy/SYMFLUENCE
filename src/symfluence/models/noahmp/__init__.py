# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Noah-MP (noah-owp-modular) — NOAA-OWP Standalone Land Surface Model."""
from .config import NoahMPConfigAdapter
from .extractor import NoahMPResultExtractor
from .postprocessor import NoahMPPostProcessor
from .runner import NoahMPRunner

__all__ = [
    "NoahMPRunner",
    "NoahMPResultExtractor",
    "NoahMPPostProcessor",
    "NoahMPConfigAdapter",
]

from symfluence.core.registry import model_manifest

model_manifest(
    "NOAHMP",
    runner=NoahMPRunner,
    result_extractor=NoahMPResultExtractor,
    config_adapter=NoahMPConfigAdapter,
    build_instructions_module="symfluence.models.noahmp.build_instructions",
)
