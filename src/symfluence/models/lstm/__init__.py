# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""LSTM neural network model for streamflow prediction.

Uses recurrent neural networks to learn temporal patterns in forcing data
(precipitation, temperature) for hydrological prediction. Supports optional
attention mechanism and configurable architecture.
"""

from typing import TYPE_CHECKING

# Lazy import mapping — avoids importing PyTorch at module level
_LAZY_IMPORTS = {
    'LSTMRunner': ('.runner', 'LSTMRunner'),
    'LSTMPreProcessor': ('.preprocessor', 'LSTMPreProcessor'),
    'LSTMPostprocessor': ('.postprocessor', 'LSTMPostprocessor'),
    'LSTMModel': ('.model', 'LSTMModel'),
    'visualize_lstm': ('.visualizer', 'visualize_lstm'),
}

# Backward-compatibility aliases resolved lazily
_LAZY_ALIASES = {
    'FLASH': ('.runner', 'LSTMRunner'),
    'FlashRunner': ('.runner', 'LSTMRunner'),
    'FlashPreProcessor': ('.preprocessor', 'LSTMPreProcessor'),
    'FlashPostprocessor': ('.postprocessor', 'LSTMPostprocessor'),
}


def __getattr__(name: str):
    """Lazy import handler for LSTM module components."""
    target = _LAZY_IMPORTS.get(name) or _LAZY_ALIASES.get(name)
    if target:
        module_path, attr_name = target
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys()) + list(_LAZY_ALIASES.keys())


# Register all LSTM components via unified registry
from symfluence.core.registry import model_manifest

from .config import LSTMConfigAdapter
from .extractor import LSTMResultExtractor
from .plotter import LSTMPlotter

model_manifest(
    "LSTM",
    config_adapter=LSTMConfigAdapter,
    result_extractor=LSTMResultExtractor,
    plotter=LSTMPlotter,
)


if TYPE_CHECKING:
    from .model import LSTMModel
    from .postprocessor import LSTMPostprocessor
    from .preprocessor import LSTMPreProcessor
    from .runner import LSTMRunner
    from .visualizer import visualize_lstm


__all__ = [
    'LSTMRunner', 'LSTMPreProcessor', 'LSTMPostprocessor', 'LSTMModel',
    'visualize_lstm',
    'FLASH', 'FlashRunner', 'FlashPreProcessor', 'FlashPostprocessor',
]
