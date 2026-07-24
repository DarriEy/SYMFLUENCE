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
  build instructions, calibration targets).
- ``calibration`` — the engine bases under ``core.calibration``:
  ``BaseModelOptimizer``, ``BaseWorker``/``InMemoryModelWorker``,
  ``BaseParameterManager``, the algorithm suite and its registration seam,
  and the parameter-bounds seam (``register_model_bounds``).
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

from typing import Dict, Tuple

#: Independent contract version per family. Bump deliberately; each entry's
#: history belongs in CHANGELOG.md under a "contracts" heading.
FAMILY_CONTRACTS: Dict[str, str] = {
    "models": "0.1.0",
    "calibration": "0.1.0",
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
        raise RuntimeError(
            f"This package targets {family} contract {target_version}, but the "
            f"installed framework provides {contract_version(family)}. "
            f"Upgrade symfluence (forward skew) or the package (major skew)."
        )
