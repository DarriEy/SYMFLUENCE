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
* Admission policy for a non-native backend depends on the capability's
  ``source_kind``:
  - ``dataset_artifact`` → provenance gate (DOI + version + checksum + an
    open/attribution license).
  - ``provider_api`` → parity gate (a ``parity_grade`` validated against the
    native handler) OR, when there is no native handler to grade against, the
    posture-only gate (source license is open/attribution).
  Restricted redistribution is refused for every non-native backend. Anything
  still ungated needs ``ALLOW_UNGATED_BACKENDS: true``. The native backend is
  the parity reference and is exempt from all of the above.

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
    Redistribution,
    SourceKind,
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


def _resolve_backend(
    name: str, config: Any, logger: logging.Logger, registry: Any = None,
) -> Optional[AcquisitionBackend]:
    """Materialize the registered backend (instantiating classes with (config, logger))."""
    entry = (registry if registry is not None else R.acquisition_backends).get(name)
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
    # Source-data license gate (contract 0.4.0), checked first because it is a
    # legal constraint refused for ALL non-native backends and NOT waivable by
    # ALLOW_UNGATED_BACKENDS (that flag waives the parity grade, not the licence).
    # The native backend fetches from source on the user's behalf — never gated.
    if name != 'native' and getattr(cap, 'redistribution', None) == Redistribution.RESTRICTED:
        return (
            "source data is redistribution=restricted; a mirroring backend "
            "must not re-serve it (native backend fetches from source instead)"
        )
    # Admission for a non-native backend: EITHER a parity grade (validated
    # against the native pipeline) OR — when there is no native pipeline to be
    # parity-equivalent to — the posture-only gate: the SOURCE licence is
    # open/attribution, so a mirror may serve. An undeclared posture (UNKNOWN)
    # on an ungraded dataset still needs ALLOW_UNGATED_BACKENDS.
    if name != 'native' and cap.parity_grade is None and not allow_ungated:
        if getattr(cap, 'redistribution', None) not in (
            Redistribution.OPEN, Redistribution.ATTRIBUTION
        ):
            return (
                "dataset is ungraded (parity_grade=None) and its source licence "
                "posture is not open/attribution; declare the posture or set "
                "ALLOW_UNGATED_BACKENDS: true"
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


def _observation_decline_reason(
    name: str,
    cap: Optional[Any],
    *,
    kind: str,
    window: Optional[tuple],
    allow_ungated: bool,
) -> Optional[str]:
    """Return why an observation backend cannot serve, or None if it can."""
    if cap is None:
        return "does not claim the provider"
    if kind not in cap.kinds:
        return f"does not serve observation kind {kind!r} (kinds: {sorted(cap.kinds)})"
    if window and cap.temporal:
        start, end = cap.temporal
        if window[0] < start or window[1] > end:
            return f"window {window} outside declared coverage [{start}, {end})"
    # The native backend is the parity reference and fetches from source on the
    # user's behalf, so it is exempt from every gate below (mirrors the forcing
    # _decline_reason native exemption).
    if name == 'native':
        return None

    # Restricted redistribution is a legal constraint refused for ALL non-native
    # backends regardless of source kind, and NOT waivable by ALLOW_UNGATED.
    if getattr(cap, 'redistribution', None) == Redistribution.RESTRICTED:
        return (
            "source data is redistribution=restricted; a mirroring backend "
            "must not re-serve it (native backend fetches from source instead)"
        )

    source_kind = getattr(cap, 'source_kind', SourceKind.PROVIDER_API)
    if source_kind == SourceKind.DATASET_ARTIFACT:
        # Provenance gate (contract 0.5.0): a published artifact is admitted on
        # a verifiable DOI + version + content checksum and an open/attribution
        # license — NOT a parity grade (there is usually no native API to be
        # parity-equivalent to; the dataset is the source of truth).
        if not allow_ungated:
            missing = [
                f for f in ('dataset_doi', 'dataset_version', 'dataset_checksum')
                if not getattr(cap, f, '')
            ]
            if missing:
                return (
                    f"dataset artifact is missing provenance {missing}; declare "
                    "DOI/version/checksum or set ALLOW_UNGATED_BACKENDS: true"
                )
            if getattr(cap, 'redistribution', None) not in (
                Redistribution.OPEN, Redistribution.ATTRIBUTION
            ):
                return (
                    "dataset artifact license posture is not open/attribution; "
                    "declare it or set ALLOW_UNGATED_BACKENDS: true"
                )
        return None

    # PROVIDER_API: admitted by EITHER of two gates.
    #  * Parity gate: a parity_grade declares the backend was validated
    #    value/bit-identical against the native handler for this live API.
    #  * Posture-only gate: many live providers have NO native handler to be
    #    parity-equivalent to. Such a backend may still serve if the SOURCE
    #    data's license posture is open/attribution (a redistributable mirror) —
    #    the same posture bar the dataset-artifact tier clears. Restricted
    #    redistribution was already refused above; an unknown posture still needs
    #    ALLOW_UNGATED_BACKENDS.
    if cap.parity_grade is None and not allow_ungated:
        if getattr(cap, 'redistribution', None) not in (
            Redistribution.OPEN, Redistribution.ATTRIBUTION
        ):
            return (
                "provider is ungraded (parity_grade=None) and its source license "
                "posture is not open/attribution; declare the posture or set "
                "ALLOW_UNGATED_BACKENDS: true"
            )
    return None


def select_observation_backend(
    provider_id: str,
    config: Any,
    *,
    kind: str = 'streamflow',
    window: Optional[tuple] = None,
    logger: Optional[logging.Logger] = None,
):
    """Select the observation backend that will serve *provider_id* (contract 0.2.0).

    The observation-flavored sibling of :func:`select_backend`, consulted by
    the observed_processor's backend-first tier under ``DATA_ACCESS:
    community``. The ``native`` backend (``NativeObservationBackend``) is the
    parity reference and the last-priority fallthrough: when no community
    backend serves a provider that native exposes, selection resolves to
    native. When native does not yet expose the provider either,
    :class:`DatasetUnsupported` is raised and the caller falls through to the
    legacy registry/provider-branch tiers (the same inert-seam discipline as
    forcing — these tiers are retired in Phase 3 of the coverage plan).

    Per-provider pin ``<PROVIDER>_BACKEND: native`` resolves the native backend
    directly (or, if native does not expose the provider, raises so the caller
    falls through); pinning any other name requires that backend to serve, or
    it raises. The parity gate and the source-data redistribution gate apply to
    every NON-native backend; native is exempt from both (it is the reference
    and fetches from source rather than mirroring).

    Raises:
        DatasetUnsupported: when no registered observation backend can serve
            the request (the caller's signal to fall through to handlers).
    """
    log = logger or _logger
    allow_ungated = _allow_ungated(config)

    override = _backend_override(config, provider_id)
    if override is not None:
        # 'native' resolves the NativeObservationBackend like any other pin
        # (mirrors forcing select_backend); if native does not expose the
        # provider it raises below and the caller falls through to the legacy
        # tiers — same outcome the old 'native'-special-case produced.
        priority = [override]
        pinned = True
    else:
        names = R.observation_backends.keys()
        # Deterministic order with 'community' first (mirrors the forcing
        # priority list); other names follow alphabetically.
        priority = sorted(names, key=lambda n: (n != 'community', n))
        pinned = False

    declined: List[str] = []
    for name in priority:
        backend = _resolve_backend(name, config, log, registry=R.observation_backends)
        if backend is None:
            declined.append(f"{name}: not registered")
            if pinned:
                raise DatasetUnsupported(
                    f"Observation backend '{name}' pinned via "
                    f"{provider_id.upper()}_BACKEND is not registered",
                    dataset_id=provider_id,
                    backend=name,
                )
            continue
        if not is_compatible(getattr(backend, 'interface_version', '')):
            declined.append(f"{name}: incompatible interface_version "
                            f"{getattr(backend, 'interface_version', None)!r}")
            log.info(
                "Observation backend '%s' declined for %s: incompatible interface version",
                name, provider_id,
            )
            continue

        wanted = provider_id.lower()
        cap = next(
            (c for c in backend.capabilities() if c.provider_id.lower() == wanted),
            None,
        )
        reason = _observation_decline_reason(
            name, cap, kind=kind, window=window, allow_ungated=allow_ungated,
        )
        if reason is not None:
            declined.append(f"{name}: {reason}")
            if pinned:
                raise DatasetUnsupported(
                    f"Observation backend '{name}' pinned via "
                    f"{provider_id.upper()}_BACKEND cannot serve '{provider_id}': {reason}",
                    dataset_id=provider_id,
                    backend=name,
                )
            log.info(
                "Observation backend '%s' declined for %s (%s); falling through",
                name, provider_id, reason,
            )
            continue

        log.info(
            "Observation backend '%s' serves provider '%s' (kind=%s%s)",
            name, provider_id, kind, ", pinned" if pinned else "",
        )
        if getattr(cap, 'noncommercial', False):
            log.warning(
                "Provider '%s' carries a NonCommercial license clause (%s); the "
                "data may not be used for commercial purposes. Ensure your use "
                "complies.", provider_id, getattr(cap, 'data_license', '') or 'non-commercial',
            )
        return backend

    raise DatasetUnsupported(
        f"No registered observation backend can serve '{provider_id}' "
        f"(tried: {'; '.join(declined) or 'none'})",
        dataset_id=provider_id,
    )


def _attribute_decline_reason(
    cap: Optional[Any],
    *,
    attribute_ids: Optional[frozenset],
    allow_ungated: bool,
) -> Optional[str]:
    """Return why an attribute backend cannot serve, or None if it can.

    There is no native attribute backend (the in-tree geospatial/profile
    processors are the default path), so every backend reaching this gate is a
    mirroring/community backend and the licence + parity policy below applies to
    all of them — mirroring the non-native branch of
    :func:`_observation_decline_reason`.
    """
    if cap is None:
        return "does not claim the provider"
    if attribute_ids:
        missing = set(attribute_ids) - set(cap.attribute_ids)
        if missing:
            return f"cannot serve requested attribute ids {sorted(missing)}"
    # Restricted redistribution is a legal constraint refused for ALL community
    # backends and NOT waivable by ALLOW_UNGATED_BACKENDS (that flag waives the
    # parity grade, not the licence).
    if getattr(cap, 'redistribution', None) == Redistribution.RESTRICTED:
        return (
            "source data is redistribution=restricted; a mirroring backend "
            "must not re-serve it (use the in-tree processors instead)"
        )
    # Admission: EITHER a tolerance parity grade (validated against an in-tree
    # extraction) OR — since zonal stats often have no native reference to be
    # parity-equivalent to — the posture-only gate: the SOURCE licence is
    # open/attribution, so a mirror may serve. An undeclared posture (UNKNOWN)
    # on an ungraded provider still needs ALLOW_UNGATED_BACKENDS.
    if cap.parity_grade is None and not allow_ungated:
        if getattr(cap, 'redistribution', None) not in (
            Redistribution.OPEN, Redistribution.ATTRIBUTION
        ):
            return (
                "provider is ungraded (parity_grade=None) and its source licence "
                "posture is not open/attribution; declare the posture or set "
                "ALLOW_UNGATED_BACKENDS: true"
            )
    return None


def select_attribute_backend(
    provider_id: str,
    config: Any,
    *,
    attribute_ids: Optional[frozenset] = None,
    logger: Optional[logging.Logger] = None,
):
    """Select the attribute backend that will serve *provider_id* (contract 0.3.0).

    The attribute-flavored sibling of :func:`select_observation_backend`,
    consulted by the attribute pipeline under ``DATA_ACCESS: community``. As
    with observations there is **no native attribute backend**: the in-tree
    geospatial/profile processors are the default path, so callers treat
    :class:`DatasetUnsupported` as "no community attribute backend serves this"
    — the same inert-seam discipline (no backend registered → exactly today's
    behavior).

    Per-provider pin ``<PROVIDER>_BACKEND: native`` forces the default path
    (raises :class:`DatasetUnsupported`). Acceptance mirrors the forcing/
    observation gates: ``redistribution=restricted`` is refused first (a legal
    constraint, non-waivable), then a provider is admitted by EITHER a tolerance
    parity grade OR the posture-only gate (source licence open/attribution);
    an ungraded provider with an undeclared posture is refused unless
    ``ALLOW_UNGATED_BACKENDS: true``. A served provider that carries a
    NonCommercial clause is surfaced with a warning (not refused).

    Raises:
        DatasetUnsupported: when no registered attribute backend can serve the
            request (the caller's signal to use only the in-tree processors).
    """
    log = logger or _logger
    allow_ungated = _allow_ungated(config)

    override = _backend_override(config, provider_id)
    if override == 'native':
        raise DatasetUnsupported(
            f"Provider '{provider_id}' is pinned to the in-tree path via "
            f"{provider_id.upper()}_BACKEND: native",
            dataset_id=provider_id,
            backend='native',
        )
    if override is not None:
        priority = [override]
        pinned = True
    else:
        names = R.attribute_backends.keys()
        priority = sorted(names, key=lambda n: (n != 'community', n))
        pinned = False

    declined: List[str] = []
    for name in priority:
        backend = _resolve_backend(name, config, log, registry=R.attribute_backends)
        if backend is None:
            declined.append(f"{name}: not registered")
            if pinned:
                raise DatasetUnsupported(
                    f"Attribute backend '{name}' pinned via "
                    f"{provider_id.upper()}_BACKEND is not registered",
                    dataset_id=provider_id,
                    backend=name,
                )
            continue
        if not is_compatible(getattr(backend, 'interface_version', '')):
            declined.append(f"{name}: incompatible interface_version "
                            f"{getattr(backend, 'interface_version', None)!r}")
            log.info(
                "Attribute backend '%s' declined for %s: incompatible interface version",
                name, provider_id,
            )
            continue

        wanted = provider_id.lower()
        cap = next(
            (c for c in backend.capabilities() if c.provider_id.lower() == wanted),
            None,
        )
        reason = _attribute_decline_reason(
            cap, attribute_ids=attribute_ids, allow_ungated=allow_ungated,
        )
        if reason is not None:
            declined.append(f"{name}: {reason}")
            if pinned:
                raise DatasetUnsupported(
                    f"Attribute backend '{name}' pinned via "
                    f"{provider_id.upper()}_BACKEND cannot serve '{provider_id}': {reason}",
                    dataset_id=provider_id,
                    backend=name,
                )
            log.info(
                "Attribute backend '%s' declined for %s (%s); falling through",
                name, provider_id, reason,
            )
            continue

        log.info(
            "Attribute backend '%s' serves provider '%s'%s",
            name, provider_id, ", pinned" if pinned else "",
        )
        if getattr(cap, 'noncommercial', False):
            log.warning(
                "Provider '%s' carries a NonCommercial license clause (%s); the "
                "data may not be used for commercial purposes. Ensure your use "
                "complies.", provider_id, getattr(cap, 'data_license', '') or 'non-commercial',
            )
        return backend

    raise DatasetUnsupported(
        f"No registered attribute backend can serve '{provider_id}' "
        f"(tried: {'; '.join(declined) or 'none'})",
        dataset_id=provider_id,
    )


__all__ = ["select_backend", "select_observation_backend", "select_attribute_backend"]
