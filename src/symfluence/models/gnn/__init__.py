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
    return list(_LAZY_IMPORTS.keys()) + ['register_with_model_registry']


def register_with_model_registry():
    """Register GNN components with the ModelRegistry."""
    from symfluence.models.registry import ModelRegistry

    from .config import GNNConfigAdapter
    from .extractor import GNNResultExtractor
    from .postprocessor import GNNPostprocessor
    from .preprocessor import GNNPreProcessor
    from .runner import GNNRunner

    ModelRegistry.register_config_adapter('GNN')(GNNConfigAdapter)
    ModelRegistry.register_result_extractor('GNN')(GNNResultExtractor)
    ModelRegistry.register_preprocessor('GNN')(GNNPreProcessor)
    ModelRegistry.register_runner('GNN', method_name='run_gnn')(GNNRunner)
    ModelRegistry.register_postprocessor('GNN')(GNNPostprocessor)


# Eagerly register when module is imported
register_with_model_registry()


if TYPE_CHECKING:
    from .postprocessor import GNNPostprocessor
    from .preprocessor import GNNPreProcessor
    from .runner import GNNRunner


__all__ = ['GNNRunner', 'GNNPreProcessor', 'GNNPostprocessor', 'register_with_model_registry']
