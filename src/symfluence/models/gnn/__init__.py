# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Graph Neural Network model for spatio-temporal hydrological prediction.

Combines LSTM temporal processing with directed-graph spatial propagation
along the river network DAG for distributed streamflow forecasting.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping — avoids importing PyTorch at module level
_LAZY_IMPORTS = {
    'GNNRunner': ('.runner', 'GNNRunner'),
    'GNNPreProcessor': ('.preprocessor', 'GNNPreProcessor'),
    'GNNPostProcessor': ('.postprocessor', 'GNNPostProcessor'),
}


def __getattr__(name: str):
    """Lazy import handler for GNN module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys())


# Register all GNN components via unified registry
from symfluence.core.registry import model_manifest

from .config import GNNConfigAdapter
from .extractor import GNNResultExtractor


def register() -> None:
    """Register GNN components with the unified registry."""
    model_manifest(
        "GNN",
        config_adapter=GNNConfigAdapter,
        result_extractor=GNNResultExtractor,
        # Trained by gradient descent during the run step, not by an external
        # DDS/PSO parameter search, so calibration and sensitivity analysis
        # skip it rather than reporting a failure.
        self_training=True,
    )

    # Spatial capabilities are owned by this package (service-decomposition
    # item 2): declared at plugin-discovery time so core carries no per-model
    # spatial knowledge and a capability change never needs a core release.
    from symfluence.core.modeling.spatial_modes import (
        ModelSpatialCapability,
        SpatialMode,
        register_model_spatial_capability,
    )
    register_model_spatial_capability(
        "GNN",
        ModelSpatialCapability(
            supported_modes={SpatialMode.DISTRIBUTED},
            default_mode=SpatialMode.DISTRIBUTED,
            # GNN has internal graph-based routing.
            requires_routing={SpatialMode.DISTRIBUTED: False},
            warning_message=(
                "GNN requires distributed domain with graph structure. "
                "Use LSTM for lumped modeling."
            ),
        ),
    )


if TYPE_CHECKING:
    from .postprocessor import GNNPostProcessor
    from .preprocessor import GNNPreProcessor
    from .runner import GNNRunner


__all__ = ['GNNRunner', 'GNNPreProcessor', 'GNNPostProcessor']
