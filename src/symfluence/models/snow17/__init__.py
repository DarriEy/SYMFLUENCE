# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Snow-17 Temperature Index Snow Model for SYMFLUENCE.

A standalone JAX/NumPy dual-backend implementation of the Anderson (1973, 2006)
temperature-index model for snow accumulation and ablation.

Designed for coupling with any rainfall-runoff model (XAJ, GR4J, etc.) via:
- Functional API: ``snow17_step()``, ``snow17_simulate()`` for lax.scan coupling
- BMI-like class: ``Snow17BMI`` for imperative coupling

Usage:
    from symfluence.models.snow17 import Snow17BMI, snow17_simulate

    # BMI coupling
    snow = Snow17BMI(params={'SCF': 1.1}, latitude=51.17)
    snow.initialize()
    rain_plus_melt = snow.update(precip, temp, doy)

    # Functional coupling (JAX-compatible)
    rpm, state = snow17_simulate(precip, temp, doy, params, lat=51.17)

References:
    Anderson, E.A. (2006). Snow Accumulation and Ablation Model - SNOW-17.
    NWS River Forecast System User Manual.
"""

from typing import TYPE_CHECKING

_LAZY_IMPORTS = {
    # Configuration
    'Snow17Config': ('.config', 'Snow17Config'),
    'Snow17ConfigAdapter': ('.config', 'Snow17ConfigAdapter'),

    # BMI interface
    'Snow17BMI': ('.bmi', 'Snow17BMI'),

    # Parameters
    'Snow17State': ('.parameters', 'Snow17State'),
    'Snow17Params': ('.parameters', 'Snow17Params'),
    'SNOW17_PARAM_NAMES': ('.parameters', 'SNOW17_PARAM_NAMES'),
    'SNOW17_PARAM_BOUNDS': ('.parameters', 'SNOW17_PARAM_BOUNDS'),
    'SNOW17_DEFAULTS': ('.parameters', 'SNOW17_DEFAULTS'),
    'DEFAULT_ADC': ('.parameters', 'DEFAULT_ADC'),

    # Core model
    'snow17_step': ('.model', 'snow17_step'),
    'snow17_simulate': ('.model', 'snow17_simulate'),
    'snow17_simulate_jax': ('.model', 'snow17_simulate_jax'),
    'snow17_simulate_numpy': ('.model', 'snow17_simulate_numpy'),
    'seasonal_melt_factor': ('.model', 'seasonal_melt_factor'),
    'HAS_JAX': ('.model', 'HAS_JAX'),
    'create_initial_state': ('.model', 'create_initial_state'),
}


def __getattr__(name: str):
    """Lazy import handler for Snow-17 module components."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_IMPORTS.keys())


# Register all Snow-17 components via unified registry
from symfluence.core.registry import model_manifest

from .config import Snow17ConfigAdapter

model_manifest(
    "SNOW17",
    config_adapter=Snow17ConfigAdapter,
)


if TYPE_CHECKING:
    from .bmi import Snow17BMI
    from .config import Snow17Config, Snow17ConfigAdapter
    from .model import (
        HAS_JAX,
        create_initial_state,
        seasonal_melt_factor,
        snow17_simulate,
        snow17_simulate_jax,
        snow17_simulate_numpy,
        snow17_step,
    )
    from .parameters import (
        DEFAULT_ADC,
        SNOW17_DEFAULTS,
        SNOW17_PARAM_BOUNDS,
        SNOW17_PARAM_NAMES,
        Snow17Params,
        Snow17State,
    )


__all__ = [
    'Snow17Config', 'Snow17ConfigAdapter',
    'Snow17BMI',
    'Snow17State', 'Snow17Params',
    'SNOW17_PARAM_NAMES', 'SNOW17_PARAM_BOUNDS', 'SNOW17_DEFAULTS', 'DEFAULT_ADC',
    'snow17_step', 'snow17_simulate', 'snow17_simulate_jax', 'snow17_simulate_numpy',
    'seasonal_melt_factor', 'HAS_JAX', 'create_initial_state',
]
