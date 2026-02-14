"""SYMFLUENCE coupling integration layer.

Provides adapters that wrap SYMFLUENCE models as dCoupler components,
a config-driven graph builder, and a BaseWorker implementation for
calibration through the dCoupler CouplingGraph.

dCoupler is an optional dependency. When not installed, SYMFLUENCE falls
back to its built-in sequential coupling implementations.
"""

import logging

logger = logging.getLogger(__name__)

INSTALL_SUGGESTION = (
    "dCoupler not installed. For unified graph-based model coupling with "
    "conservation checking and differentiable connections, install with: "
    "pip install dcoupler"
)


def is_dcoupler_available() -> bool:
    """Check if dCoupler is installed and importable."""
    try:
        import dcoupler  # noqa: F401
        return True
    except ImportError:
        return False


__all__ = ["is_dcoupler_available", "INSTALL_SUGGESTION"]

# Conditionally export dCoupler-dependent classes
try:
    from symfluence.coupling.graph_builder import CouplingGraphBuilder
    from symfluence.coupling.worker import DCouplerWorker
    from symfluence.coupling.bmi_registry import BMIRegistry

    __all__.extend([
        "CouplingGraphBuilder",
        "DCouplerWorker",
        "BMIRegistry",
    ])
except ImportError:
    pass
