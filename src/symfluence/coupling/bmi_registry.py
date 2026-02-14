"""Registry mapping SYMFLUENCE model names to dCoupler component adapters.

The BMIRegistry provides a single lookup table to instantiate the correct
dCoupler component adapter for any SYMFLUENCE model.
"""

from __future__ import annotations

import logging
from typing import Dict, Type

logger = logging.getLogger(__name__)


class BMIRegistry:
    """Maps SYMFLUENCE model identifiers to dCoupler component classes.

    Usage::

        registry = BMIRegistry()
        component_cls = registry.get("SUMMA")
        component = component_cls(name="summa", config=config_dict)
    """

    _PROCESS_MODELS = {
        "SUMMA": "symfluence.coupling.adapters.process_adapters.SUMMAProcessComponent",
        "MIZUROUTE": "symfluence.coupling.adapters.process_adapters.MizuRouteProcessComponent",
        "TROUTE": "symfluence.coupling.adapters.process_adapters.TRouteProcessComponent",
        "PARFLOW": "symfluence.coupling.adapters.process_adapters.ParFlowProcessComponent",
        "MODFLOW": "symfluence.coupling.adapters.process_adapters.MODFLOWProcessComponent",
        "MESH": "symfluence.coupling.adapters.process_adapters.MESHProcessComponent",
        "CLM": "symfluence.coupling.adapters.process_adapters.CLMProcessComponent",
    }

    _JAX_MODELS = {
        "SNOW17": "symfluence.coupling.adapters.jax_adapters.Snow17JAXComponent",
        "XAJ": "symfluence.coupling.adapters.jax_adapters.XAJJAXComponent",
        "XINANJIANG": "symfluence.coupling.adapters.jax_adapters.XAJJAXComponent",
        "SACSMA": "symfluence.coupling.adapters.jax_adapters.SacSmaJAXComponent",
        "SAC-SMA": "symfluence.coupling.adapters.jax_adapters.SacSmaJAXComponent",
    }

    def __init__(self):
        self._registry: Dict[str, str] = {}
        self._registry.update(self._PROCESS_MODELS)
        self._registry.update(self._JAX_MODELS)

    def get(self, model_name: str) -> Type:
        """Resolve a model name to its component class.

        Args:
            model_name: Model identifier (case-insensitive), e.g. "SUMMA", "XAJ"

        Returns:
            The component class (not an instance).

        Raises:
            KeyError: If the model is not registered.
            ImportError: If the component class cannot be imported.
        """
        key = model_name.upper().replace(" ", "").replace("-", "")
        # Try exact match first, then normalized
        class_path = self._registry.get(key) or self._registry.get(model_name.upper())
        if class_path is None:
            available = sorted(self._registry.keys())
            raise KeyError(
                f"Unknown model '{model_name}'. Available: {available}"
            )

        module_path, class_name = class_path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def register(self, model_name: str, class_path: str) -> None:
        """Register a custom model adapter.

        Args:
            model_name: Model identifier (will be uppercased)
            class_path: Fully qualified class path, e.g.
                "my_package.adapters.MyComponent"
        """
        self._registry[model_name.upper()] = class_path

    def is_jax_model(self, model_name: str) -> bool:
        """Check if a model uses JAX (differentiable) backend."""
        return model_name.upper() in self._JAX_MODELS

    def is_process_model(self, model_name: str) -> bool:
        """Check if a model is an external process."""
        return model_name.upper() in self._PROCESS_MODELS

    def available_models(self) -> list:
        """Return sorted list of all registered model names."""
        return sorted(self._registry.keys())
