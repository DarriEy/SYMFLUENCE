# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Graph Neural Network model for spatio-temporal hydrological prediction.

Combines LSTM temporal processing with directed-graph spatial propagation
along the river network DAG for distributed streamflow forecasting.
"""

from typing import TYPE_CHECKING

# Lazy import mapping — avoids importing PyTorch at module level
_LAZY_IMPORTS = {
    'GNNRunner': ('.runner', 'GNNRunner'),
    'GNNPreProcessor': ('.preprocessor', 'GNNPreProcessor'),
    'GNNPostprocessor': ('.postprocessor', 'GNNPostprocessor'),
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

model_manifest(
    "GNN",
    config_adapter=GNNConfigAdapter,
    result_extractor=GNNResultExtractor,
)


if TYPE_CHECKING:
    from .postprocessor import GNNPostprocessor
    from .preprocessor import GNNPreProcessor
    from .runner import GNNRunner


__all__ = ['GNNRunner', 'GNNPreProcessor', 'GNNPostprocessor']
