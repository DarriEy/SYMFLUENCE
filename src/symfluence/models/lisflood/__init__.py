# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""LISFLOOD Distributed Hydrological Model (JRC/Deltares)."""

from .config import LisfloodConfigAdapter
from .extractor import LisfloodResultExtractor
from .postprocessor import LisfloodPostProcessor
from .preprocessor import LisfloodPreProcessor
from .runner import LisfloodRunner

__all__ = [
    "LisfloodPreProcessor",
    "LisfloodRunner",
    "LisfloodResultExtractor",
    "LisfloodPostProcessor",
    "LisfloodConfigAdapter",
]

from symfluence.core.registry import model_manifest

model_manifest(
    "LISFLOOD",
    preprocessor=LisfloodPreProcessor,
    runner=LisfloodRunner,
    result_extractor=LisfloodResultExtractor,
    config_adapter=LisfloodConfigAdapter,
    build_instructions_module="symfluence.models.lisflood.build_instructions",
)

try:
    from .calibration import LisfloodModelOptimizer  # noqa: F401
except ImportError:
    pass
