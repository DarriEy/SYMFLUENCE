# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Framework-side acquisition-backend selection (design §2).

Resolves which registered :class:`AcquisitionBackend` serves a dataset:

* Priority from ``DATA_ACCESS``: ``community`` → ``["community", "native"]``;
  anything else (``cloud``/``MAF``/unset) → ``["native"]``.
* Per-dataset flat-key override ``<DATASET>_BACKEND: native|community`` wins
  outright (a pinned backend that cannot serve raises instead of falling
  through).
* A backend may decline at capability time (does not claim the dataset, cannot
  serve the requested variables, or declares a coverage window excluding the
  request) → clean fallthrough to the next backend with an INFO log.
* Parity-gate policy: a non-native backend whose capability for the dataset is
  ungraded (``parity_grade is None``) is refused unless
  ``ALLOW_UNGATED_BACKENDS: true``. The native backend is the parity
  reference and is exempt.

No registry overwriting, no captured classes, no ordering dependence.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, List, Optional

from symfluence.core.registries import R
from symfluence.data.backends.contract import (
    AcquisitionBackend,
    DatasetCapability,
    is_compatible,
)
from symfluence.data.backends.errors import DatasetUnsupported

# Importing the native adapter registers it under R.acquisition_backends, so
# the fallback backend always exists by the time selection runs.
from symfluence.data.backends.native import NativeBackend  # noqa: F401

_logger = logging.getLogger(__name__)

_TRUTHY = {'1', 'true', 'yes', 'on'}


def _flat_get(config: Any, key: str, default: Any = None) -> Any:
    """Read a flat config key from a dict or a SymfluenceConfig-like object."""
    if config is None:
        return default
    if isinstance(config, Mapping):
        value = config.get(key, default)
        return default if value is None else value
    getter = getattr(config, 'get', None)
    if callable(getter):
        value = getter(key, default)
        return default if value is None else value
    return default


def _data_access(config: Any) -> str:
    """Resolve DATA_ACCESS (typed path first, flat key fallback; default 'cloud')."""
    try:
        value = config.domain.data_access
    except (AttributeError, KeyError, TypeError):
        value = None
    if not value:
        value = _flat_get(config, 'DATA_ACCESS', 'cloud')
    return str(value or 'cloud').lower()


def _allow_ungated(config: Any) -> bool:
    value = _flat_get(config, 'ALLOW_UNGATED_BACKENDS', False)
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY
    return bool(value)


def _backend_override(config: Any, dataset_id: str) -> Optional[str]:
    """Per-dataset flat-key override ``<DATASET>_BACKEND`` (same key as today)."""
    upper = dataset_id.upper()
    for key in (f"{upper}_BACKEND", f"{upper.replace('-', '_')}_BACKEND"):
        value = _flat_get(config, key)
        if value:
            return str(value).lower()
    return None


def _resolve_backend(name: str, config: Any, logger: logging.Logger) -> Optional[AcquisitionBackend]:
    """Materialize the registered backend (instantiating classes with (config, logger))."""
    entry = R.acquisition_backends.get(name)
    if entry is None:
        return None
    if isinstance(entry, type):
        return entry(config, logger)
    return entry


def _capability_for(backend: AcquisitionBackend, dataset_id: str) -> Optional[DatasetCapability]:
    wanted = dataset_id.lower()
    for cap in backend.capabilities():
        if cap.dataset_id.lower() == wanted:
            return cap
    return None


def _decline_reason(
    name: str,
    cap: Optional[DatasetCapability],
    *,
    variables: Optional[frozenset],
    window: Optional[tuple],
    allow_ungated: bool,
) -> Optional[str]:
    """Return why backend *name* cannot serve, or None if it can."""
    if cap is None:
        return "does not claim the dataset"
    if variables:
        missing = set(variables) - set(cap.variables)
        if missing:
            return f"cannot serve requested variables {sorted(missing)}"
    if window and cap.temporal:
        start, end = cap.temporal
        if window[0] < start or window[1] > end:
            return f"window {window} outside declared coverage [{start}, {end})"
    if name != 'native' and cap.parity_grade is None and not allow_ungated:
        return (
            "dataset is ungraded (parity_grade=None) on a non-native backend; "
            "set ALLOW_UNGATED_BACKENDS: true to permit"
        )
    return None


def select_backend(
    dataset_id: str,
    config: Any,
    *,
    variables: Optional[frozenset] = None,
    window: Optional[tuple] = None,
    logger: Optional[logging.Logger] = None,
) -> AcquisitionBackend:
    """Select the acquisition backend that will serve *dataset_id*.

    Args:
        dataset_id: Framework dataset name (case-insensitive).
        config: SymfluenceConfig or flat dict.
        variables: Required CFIF variable names (checked against capability).
        window: Requested ISO-UTC ``(start, end)`` window.
        logger: Optional logger (selection decisions are logged at INFO).

    Returns:
        The first backend in priority order that claims the dataset and
        survives the capability/parity checks.

    Raises:
        DatasetUnsupported: When no registered backend can serve the request.
    """
    log = logger or _logger
    allow_ungated = _allow_ungated(config)

    override = _backend_override(config, dataset_id)
    if override is not None:
        priority: List[str] = [override]
        pinned = True
    else:
        data_access = _data_access(config)
        priority = ['community', 'native'] if data_access == 'community' else ['native']
        pinned = False

    declined: List[str] = []
    for name in priority:
        backend = _resolve_backend(name, config, log)
        if backend is None:
            declined.append(f"{name}: not registered")
            if pinned:
                raise DatasetUnsupported(
                    f"Backend '{name}' pinned via {dataset_id.upper()}_BACKEND is not registered",
                    dataset_id=dataset_id,
                    backend=name,
                )
            continue
        if not is_compatible(getattr(backend, 'interface_version', '')):
            declined.append(f"{name}: incompatible interface_version "
                            f"{getattr(backend, 'interface_version', None)!r}")
            log.info(
                "Acquisition backend '%s' declined for %s: incompatible interface version",
                name, dataset_id,
            )
            continue

        reason = _decline_reason(
            name, _capability_for(backend, dataset_id),
            variables=variables, window=window, allow_ungated=allow_ungated,
        )
        if reason is not None:
            declined.append(f"{name}: {reason}")
            if pinned:
                raise DatasetUnsupported(
                    f"Backend '{name}' pinned via {dataset_id.upper()}_BACKEND "
                    f"cannot serve '{dataset_id}': {reason}",
                    dataset_id=dataset_id,
                    backend=name,
                )
            log.info(
                "Acquisition backend '%s' declined for %s (%s); falling through",
                name, dataset_id, reason,
            )
            continue

        log.info(
            "Acquisition backend '%s' serves dataset '%s' (priority=%s%s)",
            name, dataset_id, priority, ", pinned" if pinned else "",
        )
        return backend

    raise DatasetUnsupported(
        f"No registered acquisition backend can serve '{dataset_id}' "
        f"(tried: {'; '.join(declined) or 'none'})",
        dataset_id=dataset_id,
    )


__all__ = ["select_backend"]
