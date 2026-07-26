# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model-facing capability facade for the optional coupling engine."""
from __future__ import annotations

from typing import Any

INSTALL_SUGGESTION = (
    "dCoupler not installed. For unified graph-based model coupling with "
    "conservation checking and differentiable connections, install with: "
    "pip install dcoupler"
)


def is_dcoupler_available() -> bool:
    """Return whether the optional dCoupler dependency is importable."""
    try:
        import dcoupler  # noqa: F401
    except ImportError:
        return False
    return True


def build_coupling_graph(config: Any):
    """Build a coupling graph through the installed coupling capability."""
    if not is_dcoupler_available():
        raise ImportError(INSTALL_SUGGESTION)

    from symfluence.coupling.graph_builder import CouplingGraphBuilder

    return CouplingGraphBuilder().build(config)


__all__ = ["INSTALL_SUGGESTION", "build_coupling_graph", "is_dcoupler_available"]
