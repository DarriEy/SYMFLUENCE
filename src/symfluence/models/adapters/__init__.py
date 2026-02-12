"""
Model Forcing Adapters for SYMFLUENCE.

This package provides adapters that convert CFIF (CF-Intermediate Format)
forcing data into model-specific formats.

Architecture:
    CFIF Data → ForcingAdapter → Model-Specific Format

Each model registers its own adapter in its package directory
(e.g., models/summa/forcing_adapter.py). The adapter registry
provides dynamic discovery of adapters without hardcoded model names.

Usage:
    >>> from symfluence.models.adapters import ForcingAdapterRegistry
    >>> adapter = ForcingAdapterRegistry.get_adapter('SUMMA', config)
    >>> model_forcing = adapter.transform(cfif_dataset)

See Also:
    - ForcingAdapter: Abstract base class for adapters
    - ForcingAdapterRegistry: Registry for adapter discovery
"""

from .base_adapter import ForcingAdapter
from .adapter_registry import ForcingAdapterRegistry

__all__ = [
    'ForcingAdapter',
    'ForcingAdapterRegistry',
]
