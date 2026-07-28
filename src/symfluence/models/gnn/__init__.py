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
    from symfluence.core.registries import Registries as R

    model_manifest(
        "GNN",
        config_adapter=GNNConfigAdapter,
        result_extractor=GNNResultExtractor,
        # Trained by gradient descent during the run step, not by an external
        # DDS/PSO parameter search, so calibration and sensitivity analysis
        # skip it rather than reporting a failure.
        self_training=True,
    )
    # GNN had never registered these, so R.runners.get('GNN') was None and its
    # runner/preprocessor/postprocessor (~1275 lines, all importable, with
    # GNNRunner subclassing BaseModelRunner) were unreachable — the model could
    # not be executed through the registry at all. Not a self-training
    # exemption: LSTM is equally self_training=True and registers all three.
    base = 'symfluence.models.gnn'
    R.preprocessors.add_lazy("GNN", f"{base}.preprocessor.GNNPreProcessor")
    R.runners.add_lazy("GNN", f"{base}.runner.GNNRunner")
    R.postprocessors.add_lazy("GNN", f"{base}.postprocessor.GNNPostProcessor")

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
