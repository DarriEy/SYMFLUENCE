"""One-time bootstrap for static registrations.

Called once from ``symfluence/__init__.py`` to populate:

* Delineation strategy aliases
* BMI adapter lazy imports and aliases
* Metric registry entries with aliases

This module should be kept lightweight — no heavy dependencies.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_bootstrapped = False


def bootstrap() -> None:
    """Populate static registrations.  Safe to call multiple times."""
    global _bootstrapped  # noqa: PLW0603
    if _bootstrapped:
        return
    _bootstrapped = True

    from symfluence.core.registries import R

    _bootstrap_delineation_aliases(R)
    _bootstrap_bmi_adapters(R)
    _bootstrap_metrics(R)


def _bootstrap_delineation_aliases(R: type) -> None:  # noqa: N803
    """Register canonical delineation aliases."""
    aliases = {
        "delineate": "semidistributed",
        "distribute": "distributed",
        "subset": "semidistributed",
        "discretized": "semidistributed",
    }
    for alias, canonical in aliases.items():
        R.delineation_strategies.alias(alias, canonical)


def _bootstrap_bmi_adapters(R: type) -> None:  # noqa: N803
    """Register BMI/dCoupler adapters as lazy imports + aliases."""

    process_models = {
        "SUMMA": "symfluence.coupling.adapters.process_adapters.SUMMAProcessComponent",
        "MIZUROUTE": "symfluence.coupling.adapters.process_adapters.MizuRouteProcessComponent",
        "TROUTE": "symfluence.coupling.adapters.process_adapters.TRouteProcessComponent",
        "PARFLOW": "symfluence.coupling.adapters.process_adapters.ParFlowProcessComponent",
        "MODFLOW": "symfluence.coupling.adapters.process_adapters.MODFLOWProcessComponent",
        "MESH": "symfluence.coupling.adapters.process_adapters.MESHProcessComponent",
        "CLM": "symfluence.coupling.adapters.process_adapters.CLMProcessComponent",
    }
    jax_models = {
        "SNOW17": "symfluence.coupling.adapters.jax_adapters.Snow17JAXComponent",
        "XAJ": "symfluence.coupling.adapters.jax_adapters.XAJJAXComponent",
        "SACSMA": "symfluence.coupling.adapters.jax_adapters.SacSmaJAXComponent",
    }

    for name, path in process_models.items():
        R.bmi_adapters.add_lazy(name, path)
    for name, path in jax_models.items():
        R.bmi_adapters.add_lazy(name, path)

    # Aliases for common alternate names
    R.bmi_adapters.alias("XINANJIANG", "XAJ")
    R.bmi_adapters.alias("SAC-SMA", "SACSMA")


def _bootstrap_metrics(R: type) -> None:  # noqa: N803
    """Seed the unified metrics registry from the existing METRIC_REGISTRY.

    We import the existing metric registry dict and re-register each entry
    into ``R.metrics`` so that both old and new consumers see the same data.
    """
    try:
        from symfluence.evaluation.metrics_registry import METRIC_REGISTRY
    except ImportError:
        logger.debug("metrics_registry not available; skipping metric bootstrap")
        return

    # Primary entries (use exact casing from the dict keys)
    _primary_names = {
        "NSE", "logNSE", "KGE", "KGEp", "KGEnp", "VE",
        "RMSE", "NRMSE", "MAE", "MARE", "bias", "PBIAS",
        "correlation", "R2",
    }

    # Use identity normalization for metrics — preserve original casing
    # (metrics registry has mixed case keys like "logNSE", "KGEp", etc.)
    R.metrics._normalize = lambda s: s  # noqa: E731

    for name in _primary_names:
        if name in METRIC_REGISTRY:
            R.metrics.add(name, METRIC_REGISTRY[name])

    # Aliases (lowercase and alternative names)
    _aliases = {
        "kge": "KGE",
        "nse": "NSE",
        "kge_prime": "KGEp",
        "kge_np": "KGEnp",
        "r_squared": "R2",
        "log_nse": "logNSE",
    }
    for alias, canonical in _aliases.items():
        R.metrics.alias(alias, canonical)
