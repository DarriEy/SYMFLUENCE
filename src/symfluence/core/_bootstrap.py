# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""One-time bootstrap for static registrations.

Called once from ``symfluence/__init__.py`` to populate:

* Delineation strategy aliases
* BMI adapter lazy imports and aliases
* Metric registry entries with aliases
* External plugins discovered via ``importlib.metadata`` entry points

This module should be kept lightweight — no heavy dependencies.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_bootstrapped = False

#: Entry-point group that external packages use to register plugins.
PLUGIN_ENTRY_POINT_GROUP = "symfluence.plugins"


def bootstrap() -> None:
    """Populate static registrations.  Safe to call multiple times."""
    global _bootstrapped  # noqa: PLW0603
    if _bootstrapped:
        return
    _bootstrapped = True

    from symfluence.core.registries import R

    _bootstrap_delineation_aliases(R)
    _bootstrap_bmi_adapters(R)
    _bootstrap_model_aliases(R)
    _bootstrap_metrics(R)
    _discover_plugins()


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
        "HBV": "symfluence.coupling.adapters.jax_adapters.HBVJAXComponent",
        "HECHMS": "symfluence.coupling.adapters.jax_adapters.HecHmsJAXComponent",
        "TOPMODEL": "symfluence.coupling.adapters.jax_adapters.TopmodelJAXComponent",
    }

    for name, path in process_models.items():
        R.bmi_adapters.add_lazy(name, path)
    for name, path in jax_models.items():
        R.bmi_adapters.add_lazy(name, path)

    # Aliases for common alternate names
    R.bmi_adapters.alias("XINANJIANG", "XAJ")
    R.bmi_adapters.alias("SAC-SMA", "SACSMA")
    R.bmi_adapters.alias("HEC-HMS", "HECHMS")


def _bootstrap_model_aliases(R: type) -> None:  # noqa: N803
    """Alias hyphenated model names to their canonical registry keys.

    Some models register their components under a hyphen-free canonical name
    (e.g. ``jhechms`` registers ``HECHMS``, ``jsacsma`` registers ``SACSMA``,
    and the coupled land-surface/subsurface runner registers ``CLMPARFLOW``).
    A config using the conventional hyphenated spelling
    (``HYDROLOGICAL_MODEL: HEC-HMS`` or ``CLM-ParFlow``) would otherwise fail
    to resolve a runner.  Aliases are resolved lazily at lookup time, so they
    may be declared here before the plugin entry points register the canonical
    keys.

    Two kinds of alias live here. First, hyphenated spellings whose hyphen-free
    form is the *actual* canonical registration (``HEC-HMS`` -> ``HECHMS``).
    Note this differs from the BMI-adapter aliases above: e.g. the BMI adapter
    is registered as ``XAJ`` whereas the standalone runner is registered as
    ``XINANJIANG``, so no runner-level alias is added for it. Second, common
    alternate / short names a config (or a derived sensitivity-analysis label)
    may use for a model whose canonical key differs (``RHESS`` -> ``RHESSYS``;
    the SUMMA+MODFLOW coupling -> the ``COUPLED_GW`` calibration pipeline). The
    guard below additionally refuses to shadow a real registration with an alias.
    """
    # alias -> canonical, applied across every model-component registry
    model_aliases = {
        "HEC-HMS": "HECHMS",
        "SAC-SMA": "SACSMA",
        "CLM-PARFLOW": "CLMPARFLOW",
        # Alternate / short names whose canonical key differs from the spelling.
        "RHESS": "RHESSYS",
        "SUMMA-MODFLOW": "COUPLED_GW",
    }
    component_registries = (
        R.runners,
        R.preprocessors,
        R.postprocessors,
        R.optimizers,
        R.workers,
    )
    for alias_key, canonical in model_aliases.items():
        for registry in component_registries:
            # Never let an alias shadow a real registration of the same name.
            if alias_key.upper() in registry.keys():
                continue
            registry.alias(alias_key, canonical)


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


# ======================================================================
# External plugin discovery via entry points
# ======================================================================


def _discover_plugins() -> None:
    """Load external plugins registered under the ``symfluence.plugins`` group.

    Each entry point should reference a callable (typically a function)
    that performs its own registrations using ``R.*.add()``,
    ``model_manifest()``, or any other registry API.  The callable is
    invoked with no arguments.

    A failing plugin is logged and skipped — it never takes down the
    framework.

    **How to write a plugin** (in the external package's ``pyproject.toml``)::

        [project.entry-points."symfluence.plugins"]
        my_model = "my_package:register"

    Where ``my_package.register`` is a zero-arg function::

        # my_package/__init__.py
        def register():
            from symfluence.core.registries import R
            from .runner import MyRunner
            R.runners.add("MY_MODEL", MyRunner)
    """
    import sys

    if sys.version_info >= (3, 12):
        from importlib.metadata import entry_points
    else:
        # Python 3.9-3.11: entry_points() accepts the *group* keyword
        # starting from 3.9, but the return type changed in 3.12.
        from importlib.metadata import entry_points

    try:
        eps = entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
    except TypeError:
        # Very old importlib_metadata fallback (shouldn't happen on 3.11+)
        eps = entry_points().get(PLUGIN_ENTRY_POINT_GROUP, [])  # type: ignore[assignment]

    in_tree_loaded = 0
    for ep in eps:
        try:
            plugin_fn = ep.load()
            plugin_fn()
            logger.debug("Loaded plugin %r from %s", ep.name, ep.value)
            if ep.value.startswith("symfluence.models."):
                in_tree_loaded += 1
        except ImportError as exc:
            missing_module = getattr(exc, "name", None) or ""
            if (
                isinstance(exc, ModuleNotFoundError)
                and missing_module.partition(".")[0] not in ("", "symfluence")
            ):
                # A missing *third-party* module almost always means an optional
                # dependency isn't installed (e.g. an MPI/GPU model on a
                # laptop). Keep this quiet — the same models were debug-logged
                # by the old import loop — so it doesn't drown the logs on
                # every `import symfluence`.
                logger.debug(
                    "Plugin %r (%s) not loaded — optional dependency missing: %s",
                    ep.name,
                    ep.value,
                    exc,
                )
            else:
                # Any other ImportError (e.g. "cannot import name ... from
                # symfluence...") means the installed plugin was built against
                # a different SYMFLUENCE API. Burying this at DEBUG as an
                # "optional dependency" silently removes the plugin's models
                # from the registry, so calibration runs fail later with an
                # unhelpful "unknown model" error. Warn loudly instead.
                logger.warning(
                    "Plugin %r (%s) is incompatible with this SYMFLUENCE "
                    "version (%s). Its models will be unavailable — upgrade "
                    "the plugin package to a compatible release.",
                    ep.name,
                    ep.value,
                    exc,
                )
        except Exception:  # noqa: BLE001 — never let a broken plugin crash the framework
            logger.warning(
                "Failed to load symfluence plugin %r (%s); skipping.",
                ep.name,
                ep.value,
                exc_info=True,
            )

    # In-tree models register through these same entry points (declared in
    # SYMFLUENCE's own pyproject.toml). Discovering zero of them means the
    # installed dist metadata predates the entry-point declarations — almost
    # always a stale editable install. Without this signal the framework would
    # silently come up with no runnable models. We warn loudly rather than raise
    # so unusual-but-valid setups (e.g. partial vendoring) are not hard-blocked.
    if in_tree_loaded == 0:
        logger.error(
            "No in-tree SYMFLUENCE models were discovered via entry points. The "
            "installed package metadata is likely stale — reinstall with "
            "`pip install -e .` to regenerate it. Model runs will fail until then."
        )
