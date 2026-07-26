# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SYMFLUENCE coupling integration layer.

Provides adapters that wrap SYMFLUENCE models as dCoupler components,
a config-driven graph builder, and a BaseWorker implementation for
calibration through the dCoupler CouplingGraph.

dCoupler is an optional dependency. When not installed, SYMFLUENCE falls
back to its built-in sequential coupling implementations.
"""
from __future__ import annotations

import logging

from symfluence.core.modeling.coupling import INSTALL_SUGGESTION, is_dcoupler_available

logger = logging.getLogger(__name__)


__all__ = ["is_dcoupler_available", "INSTALL_SUGGESTION"]

# Conditionally export dCoupler-dependent classes
try:
    from symfluence.coupling.bmi_registry import BMIRegistry
    from symfluence.coupling.graph_builder import CouplingGraphBuilder
    from symfluence.coupling.worker import DCouplerWorker

    __all__.extend([
        "CouplingGraphBuilder",
        "DCouplerWorker",
        "BMIRegistry",
    ])
except ImportError:
    pass
