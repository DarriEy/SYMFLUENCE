# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Per-family contract versions for the extensibility surfaces of core.

Mirrors the acquisition-backend contract (``symfluence.data.backends.contract``,
the pattern proven by the CFS/CAS/CSFS/COS community services): each contract
family carries an independent semantic version, and an external package
declares the family version it targets so skew is detected cleanly at
registration time instead of failing mid-run.

Versioning semantics (identical to the acquisition contract): same major
required; while a family is pre-1.0, every MINOR bump is *additive-only* —
a package targeting an older-or-equal minor uses only surface the framework
still ships and is compatible, while forward skew (package newer than
framework) is declined. Bump a family's MINOR when its surface gains
additive capability; bump MAJOR at the first breaking change.

Families and their surfaces:

- ``models`` — the model-adapter contract: ``model_manifest()`` and the
  registry namespaces a model package populates (runners, preprocessors,
  workers, optimizers, parameter managers, config schemas, base settings,
  build instructions, calibration targets), plus forcing-artifact selection
  and model-output location under ``core.modeling``.
- ``calibration`` — the engine bases under ``core.calibration``:
  ``BaseModelOptimizer``, ``BaseWorker``/``InMemoryModelWorker``,
  ``BaseParameterManager``, the algorithm suite and its registration seam,
  the parameter-bounds seam (``register_model_bounds``), multi-gauge metrics,
  and parameter regionalization strategies/transfer functions.
- ``metrics`` — the ``core.metrics`` facade: the metric functions, the metric
  registry, ``MetricTransformer`` and ``StreamflowMetrics``.
- ``geospatial-utils`` — the geometry utilities model preprocessors build
  against (``core.geometry_utils``).

The acquisition-backend family keeps its own constant in
``symfluence.data.backends.contract`` (external packages already import it
from there); :func:`contract_version` surfaces it under ``acquisition`` for a
uniform read-only view.
"""
from __future__ import annotations

import sys
from typing import Callable, Dict, Mapping, Tuple, TypeVar

_PluginCallable = TypeVar("_PluginCallable", bound=Callable[[], object])
PLUGIN_CONTRACTS_ATTR = "__symfluence_contracts__"


class ContractCompatibilityError(RuntimeError):
    """A plugin targets a contract incompatible with this framework."""

#: Independent contract version per family. Bump deliberately; each entry's
#: history belongs in CHANGELOG.md under a "contracts" heading.
FAMILY_CONTRACTS: Dict[str, str] = {
    "models": "0.2.0",
    "calibration": "0.2.0",
    "metrics": "0.1.0",
    "geospatial-utils": "0.1.0",
}


def _parse_version(version: str) -> Tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"expected MAJOR.MINOR.PATCH, got {version!r}")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def contract_version(family: str) -> str:
    """The framework's current contract version for *family*.

    Raises:
        KeyError: If the family is unknown.
    """
    if family == "acquisition":
        from symfluence.data.backends.contract import PROTOCOL_VERSION

        return PROTOCOL_VERSION
    return FAMILY_CONTRACTS[family]


def is_compatible(family: str, target_version: str) -> bool:
    """True if a package targeting *target_version* of *family* may register.

    Same major required; pre-1.0, an older-or-equal minor is compatible and
    forward skew is declined (identical semantics to
    ``symfluence.data.backends.contract.is_compatible``).
    """
    try:
        target = _parse_version(target_version)
        current = _parse_version(contract_version(family))
    except (ValueError, KeyError):
        return False
    if target[0] != current[0]:
        return False
    if current[0] == 0 and target[1] > current[1]:
        return False
    return True


def assert_compatible(family: str, target_version: str) -> None:
    """Raise with a clear message when a package targets an incompatible version.

    Intended for a plugin's ``register()``: declining at registration time is
    what keeps uncapped pip requirements safe across independent releases.
    """
    if not is_compatible(family, target_version):
        raise ContractCompatibilityError(
            f"This package targets {family} contract {target_version}, but the "
            f"installed framework provides {contract_version(family)}. "
            f"Upgrade symfluence (forward skew) or the package (major skew)."
        )


def plugin_contracts(**targets: str) -> Callable[[_PluginCallable], _PluginCallable]:
    """Declare the contract-family versions targeted by a plugin entry point.

    Apply this decorator to the zero-argument callable published in the
    ``symfluence.plugins`` entry-point group. Discovery validates every declared
    family before invoking the callable, so an incompatible package cannot
    partially mutate registries.
    """
    unknown = set(targets) - (set(FAMILY_CONTRACTS) | {"acquisition"})
    if unknown:
        raise ValueError(f"unknown SYMFLUENCE contract families: {sorted(unknown)}")
    for family, version in targets.items():
        try:
            _parse_version(version)
        except ValueError as exc:
            raise ValueError(f"invalid {family} contract version {version!r}") from exc

    def decorate(plugin: _PluginCallable) -> _PluginCallable:
        setattr(plugin, PLUGIN_CONTRACTS_ATTR, dict(targets))
        return plugin

    return decorate


def declared_plugin_contracts(plugin: Callable[..., object]) -> Mapping[str, str]:
    """Return contract targets declared by a callable or its package.

    Callable declarations take precedence. For distributions exposing many
    entry points, the declaration may live once on an imported parent package
    (for example ``symfluence.models``); discovery checks already-loaded parent
    modules without importing additional plugin code.
    """
    targets = getattr(plugin, PLUGIN_CONTRACTS_ATTR, None)
    if targets is None:
        module_name = getattr(plugin, "__module__", "")
        parts = module_name.split(".") if isinstance(module_name, str) else []
        for size in range(len(parts), 0, -1):
            module = sys.modules.get(".".join(parts[:size]))
            if module is not None and hasattr(module, PLUGIN_CONTRACTS_ATTR):
                targets = getattr(module, PLUGIN_CONTRACTS_ATTR)
                break
    if targets is None:
        targets = {}
    if not isinstance(targets, Mapping):
        raise ContractCompatibilityError(
            f"plugin contract declaration must be a mapping, got {type(targets).__name__}"
        )
    return dict(targets)


def assert_plugin_compatible(plugin: Callable[..., object]) -> None:
    """Validate every contract target declared by a plugin callable."""
    for family, target_version in declared_plugin_contracts(plugin).items():
        if not isinstance(family, str) or not isinstance(target_version, str):
            raise ContractCompatibilityError(
                "plugin contract family and target version must both be strings"
            )
        assert_compatible(family, target_version)
