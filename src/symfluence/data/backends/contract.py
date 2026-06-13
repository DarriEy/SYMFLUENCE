# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""AcquisitionBackend protocol contract (Phase A).

The formal, versioned contract under which native and community data
acquisition are co-equal backends (design: ``acquisition_backend_protocol.md``,
ACCEPTED). Backends DECLARE what they can serve (:class:`DatasetCapability`)
and what they produced (:class:`AcquisitionResult`); nothing downstream ever
sniffs files again.

EXTRACTION CONSTRAINT (hard, guard-tested)
------------------------------------------
This module is destined for extraction into a tiny standalone
``symfluence-data-interface`` package at contract v1.0. It must therefore
import **only the Python standard library** (pydantic is permitted by the
design but not currently needed). No ``symfluence.*`` import may occur here,
directly or transitively — enforced by
``tests/conformance/test_extraction_readiness.py``, which executes this file
in an isolated subprocess with all ``symfluence`` imports blocked.

WINDOW SEMANTICS (normative, conformance item 4)
------------------------------------------------
``AcquisitionRequest.window`` is a **half-open UTC interval** ``[start, end)``:
every delivered timestep ``t`` must satisfy ``start <= t < end``, interpreted
in UTC. Timezone-naive timestamps on a delivered time axis ARE UTC by
definition of this contract; tz-aware timestamps must convert into the window.
Rationale: USGS NWIS treats its ``endDT`` request parameter as *inclusive* of
the end day, so native and community acquisitions of the same nominal window
disagreed on whether the final boundary bin belonged to the delivery — the
boundary-bin discrepancy surfaced by the USGS parity work. A single documented
inclusivity rule turns that class of silent off-by-one into a conformance
failure (``tests/conformance/test_window_bbox_semantics.py``).

BBOX SEMANTICS (normative, conformance item 4)
----------------------------------------------
``AcquisitionRequest.bbox`` is ``(lat_min, lon_min, lat_max, lon_max)``. The
delivered grid must cover the requested box up to a **one-native-pixel
tolerance per edge**: each edge of the delivered extent (outermost cell
centers ± half a native pixel) must lie within one native pixel of the
requested edge. Outward snapping to the native pixel grid — e.g. the DEM
tile-merge clip in ``data/acquisition/handlers/dem.py``, whose rasterio merge
aligns the requested bounds outward to the source pixel grid — and
cell-center subsetting (delivering only cells whose centers fall inside the
box) are both within tolerance; a larger shortfall or overshoot is a
conformance failure.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: Semver of the protocol contract defined in this module. Backends declare
#: the version they target via ``AcquisitionBackend.interface_version`` /
#: ``ObservationBackend.interface_version``.
#:
#: PRE-1.0 SEMANTICS: while the major version is 0, a MINOR bump is a
#: BREAKING change (enforced by :func:`is_compatible`). 0.2.0 added the
#: observation flavour (``ObservationCapability`` / ``ObservationRequest`` /
#: ``ObservationBackend``); 0.3.0 added the attribute flavour
#: (``AttributeCapability`` / ``AttributeRequest`` / ``AttributeBackend``,
#: schema :attr:`SchemaId.HRU_STATS_V1`). Backends hardcoding an older target
#: are *detected* as skew by the selection layer and declined — never silently
#: claimed compatible.
PROTOCOL_VERSION = "0.3.0"

#: Name of the sidecar manifest written next to acquired raw files. The
#: persisted, declared schema lets resumed runs dispatch preprocessing
#: without re-sniffing file contents.
MANIFEST_FILENAME = "acquisition_manifest.json"

#: Version of the sidecar-manifest layout itself.
MANIFEST_VERSION = 1


class GridClass(StrEnum):
    """Spatial structure class of a dataset, as served by a backend."""

    REGULAR_LATLON = "regular_latlon"
    PROJECTED = "projected"          # rotated-pole / LCC etc., 2-D coord rasters
    STATION = "station"              # point time series
    VECTOR = "vector"                # feature collections


class SchemaId(StrEnum):
    """Registered output schemas.

    Each id has a versioned spec + a CFIF mapping owned by the preprocessing
    layer. Backends DECLARE what they produced; downstream dispatches on the
    declared id, never on file probing.
    """

    NATIVE_RAW = "native-raw"        # per-dataset legacy layout: native-raw/<dataset>
    CANONICAL_V1 = "canonical-v1"    # CFS canonical (CF-aligned SI, fluxes)
    OBS_CSV_V1 = "obs-csv-v1"        # datetime,value,quality_flag (UTC, SI)
    HRU_STATS_V1 = "hru-stats-v1"    # per-HRU attribute table


@dataclass(frozen=True)
class CredentialContext:
    """Passive carrier of credentials already resolved by the framework.

    The framework owns credential *resolution* (netrc / environment /
    config — e.g. the lookup chains behind the native handlers'
    ``_get_earthdata_credentials`` / ``_get_earthdata_token``); a backend
    only *reads* the result from this carrier, keyed by auth-provider id
    (the same ids declared in :attr:`DatasetCapability.auth`, e.g.
    ``"earthdata"``, ``"cds"``). This type never scrapes the environment
    and never performs lookups itself.
    """

    #: provider id -> resolved secret material (e.g. ``{"username": ...,
    #: "password": ...}`` or ``{"token": ...}``). Empty = anonymous.
    providers: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def has(self, provider: str) -> bool:
        """Return True if resolved credentials exist for *provider*."""
        return bool(self.providers.get(provider))

    def get(self, provider: str) -> Mapping[str, str]:
        """Return the resolved secrets for *provider* (empty mapping if none)."""
        return self.providers.get(provider, {})


@dataclass(frozen=True)
class DatasetCapability:
    """One dataset a backend claims to be able to serve."""

    dataset_id: str                  # framework name, e.g. "ERA5" (case-insensitive)
    grid_class: GridClass
    schema: SchemaId                 # what acquire() will produce for this dataset
    variables: frozenset[str]        # CFIF names servable (forcing) / obs kinds / products
    temporal: tuple[str, str] | None  # coverage [start, end); None = static or undeclared
    auth: frozenset[str]             # e.g. {"earthdata"}, {"cds"}; empty = anonymous
    parity_grade: str | None         # "bit-identical" | "value-identical:<tol>" | None = ungated
    notes: str = ""


@dataclass(frozen=True)
class AcquisitionRequest:
    """A single acquisition request, fully framework-resolved."""

    dataset_id: str
    bbox: tuple[float, float, float, float] | None   # (lat_min, lon_min, lat_max, lon_max)
    window: tuple[str, str] | None                   # ISO UTC [start, end)
    variables: frozenset[str] | None                 # REQUIRED CFIF names; None = dataset default
    target_dir: Path
    credentials: CredentialContext = field(default_factory=CredentialContext)
    options: Mapping[str, Any] = field(default_factory=dict)  # dataset-specific, declared in capability notes


@dataclass(frozen=True)
class AcquisitionResult:
    """What a backend actually produced for a request."""

    paths: tuple[Path, ...]
    schema: SchemaId                  # DECLARED, drives preprocessing dispatch
    dataset_id: str
    backend: str
    provenance: Mapping[str, str]     # upstream URL/version/access time/checksums
    variables_delivered: frozenset[str]
    warnings: tuple[str, ...] = ()


@runtime_checkable
class AcquisitionBackend(Protocol):
    """The provider protocol every acquisition backend implements."""

    name: str                         # "native" | "community" | future others
    interface_version: str            # semver of THIS protocol the backend targets

    def capabilities(self) -> Sequence[DatasetCapability]:
        """Datasets servable: id, grid class, variables, schema, auth, coverage."""
        ...

    def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        """Acquire *request* and return paths + declared schema + provenance."""
        ...


# ======================================================================
# Observation flavour (contract 0.2.0)
# ======================================================================

@dataclass(frozen=True)
class ObservationCapability:
    """One observation provider a backend claims to be able to serve.

    The observation-flavored sibling of :class:`DatasetCapability` (design
    §3: "same protocol shape, obs-flavored request"). ``provider_id`` is the
    framework provider name (the ``STREAMFLOW_DATA_PROVIDER`` value, e.g.
    ``"USGS"``), matched case-insensitively.
    """

    provider_id: str                  # framework provider name, e.g. "USGS"
    kinds: frozenset[str]             # observation kinds servable, e.g. {"streamflow"}
    station_id_scheme: str            # human-readable id scheme, e.g. "8-digit USGS site number"
    temporal: tuple[str, str] | None  # coverage [start, end); None = undeclared
    auth: frozenset[str]              # auth-provider ids; empty = anonymous
    parity_grade: str | None          # "bit-identical" | "value-identical:<tol>" | None = ungated
    notes: str = ""


@dataclass(frozen=True)
class ObservationRequest:
    """A single observation-acquisition request, fully framework-resolved.

    ``window`` follows the same normative rule as forcing requests: half-open
    UTC ``[start, end)`` (see WINDOW SEMANTICS above — the rule exists
    because of the USGS NWIS inclusive-``endDT`` boundary-bin discrepancy).
    """

    provider_id: str
    station_ids: tuple[str, ...]      # provider-scheme station ids; () = backend resolves from config
    kind: str                         # e.g. "streamflow"
    window: tuple[str, str] | None    # ISO UTC [start, end)
    target_dir: Path
    credentials: CredentialContext = field(default_factory=CredentialContext)
    options: Mapping[str, Any] = field(default_factory=dict)  # provider-specific, declared in capability notes


@runtime_checkable
class ObservationBackend(Protocol):
    """The provider protocol every observation backend implements.

    Mirrors :class:`AcquisitionBackend` (same method names, observation
    types); the two roles are distinguished by the registry a backend is
    registered under (``R.observation_backends``), never structurally.
    ``acquire()`` returns a reused :class:`AcquisitionResult` whose
    ``dataset_id`` carries the provider id and whose ``schema`` is typically
    :attr:`SchemaId.OBS_CSV_V1`; the sidecar manifest helpers below are
    shared between both flavours.
    """

    name: str                         # "community" | future others
    interface_version: str            # semver of THIS protocol the backend targets

    def capabilities(self) -> Sequence[ObservationCapability]:
        """Providers servable: id, kinds, station-id scheme, auth, coverage."""
        ...

    def acquire(self, request: ObservationRequest) -> AcquisitionResult:
        """Acquire *request* and return paths + declared schema + provenance."""
        ...


# ======================================================================
# Attribute flavour (contract 0.3.0)
# ======================================================================

@dataclass(frozen=True)
class AttributeCapability:
    """One attribute provider a backend claims to be able to serve.

    The attribute-flavored sibling of :class:`DatasetCapability` /
    :class:`ObservationCapability`. ``provider_id`` is the framework provider
    name (e.g. ``"CAS"``), matched case-insensitively. ``attribute_ids`` are
    the backend's own product ids (e.g. ``"isric_soilgrids:clay_0-5cm"``); the
    framework does not interpret them, it only asks the backend to deliver them
    per-HRU.

    ``output_kind`` is fixed to ``"per_hru_stats"``: the backend delivers one
    zonal statistic per HRU geometry (the :attr:`SchemaId.HRU_STATS_V1` table).

    PARITY SEMANTICS (normative): unlike forcing/observations, attribute zonal
    statistics are NOT bitwise-reproducible against a native reference — they
    depend on resampling, masking, and source-grid alignment. ``parity_grade``
    therefore carries a TOLERANCE semantics (e.g. ``"value-within:1%"``) rather
    than ``"bit-identical"``; ``None`` is ungated and refused by the selection
    parity gate unless ``ALLOW_UNGATED_BACKENDS: true``.
    """

    provider_id: str                  # framework provider name, e.g. "CAS"
    attribute_ids: frozenset[str]     # backend product ids servable per-HRU
    output_kind: str                  # fixed "per_hru_stats"
    schema: SchemaId                  # what acquire() produces (HRU_STATS_V1)
    auth: frozenset[str]              # auth-provider ids; empty = anonymous
    parity_grade: str | None          # tolerance grade "value-within:<tol>" | None = ungated
    notes: str = ""


@dataclass(frozen=True)
class AttributeRequest:
    """A single attribute-acquisition request, fully framework-resolved.

    The backend reduces every HRU geometry to one per-HRU statistic per
    requested attribute id. Geometries are supplied either inline (``geometries``
    — GeoJSON-like mappings in EPSG:4326, aligned 1:1 with ``hru_ids``) or via
    the catchment shapefile path (``catchment_path``) the backend reads itself;
    a backend should prefer ``geometries`` when present.
    """

    provider_id: str
    attribute_ids: tuple[str, ...]    # backend product ids to extract
    hru_ids: tuple[Any, ...]          # HRU ids aligned 1:1 with geometries
    geometries: tuple[Mapping[str, Any], ...]  # GeoJSON-like, EPSG:4326; () = use catchment_path
    catchment_path: Path | None       # HRU/catchment shapefile (fallback geometry source)
    target_dir: Path
    lumped: bool = False              # DOMAIN_DEFINITION_METHOD == 'lumped'
    credentials: CredentialContext = field(default_factory=CredentialContext)
    options: Mapping[str, Any] = field(default_factory=dict)  # provider-specific, declared in capability notes


@runtime_checkable
class AttributeBackend(Protocol):
    """The provider protocol every attribute backend implements.

    Mirrors :class:`AcquisitionBackend` / :class:`ObservationBackend` (same
    method names); distinguished by the registry it is registered under
    (``R.attribute_backends``). ``acquire()`` returns a reused
    :class:`AcquisitionResult` whose ``dataset_id`` carries the provider id and
    whose ``schema`` is :attr:`SchemaId.HRU_STATS_V1` (a per-HRU attribute
    table written as one or more CSV files plus the shared sidecar manifest).
    """

    name: str                         # "community" | future others
    interface_version: str            # semver of THIS protocol the backend targets

    def capabilities(self) -> Sequence[AttributeCapability]:
        """Providers servable: id, attribute ids, output kind, auth, coverage."""
        ...

    def acquire(self, request: AttributeRequest) -> AcquisitionResult:
        """Acquire *request* and return paths + declared schema + provenance."""
        ...


# ======================================================================
# Protocol version compatibility
# ======================================================================

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a strict ``MAJOR.MINOR.PATCH`` semver string.

    Raises:
        ValueError: if *version* is not strict semver.
    """
    match = _SEMVER_RE.match(version or "")
    if match is None:
        raise ValueError(f"not a semver string: {version!r}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def is_compatible(backend_version: str, protocol_version: str = PROTOCOL_VERSION) -> bool:
    """Return True if a backend targeting *backend_version* may register.

    Same major version always required. While the contract is pre-1.0
    (major == 0), each MINOR bump is *additive-only* and never removes or
    changes an existing flavour's types — 0.2.0 added observations on top of
    0.1.0's forcing surface, 0.3.0 added attributes on top of 0.2.0. So a
    backend targeting an OLDER-or-equal minor than the running protocol is
    compatible (it uses only types the protocol still ships); FORWARD skew —
    a backend targeting a NEWER minor than the framework's protocol — is the
    breaking case and is declined (the framework lacks the newer flavour's
    types). This is what lets a contract-0.2.0 forcing/observation backend
    keep working unchanged when the in-tree protocol advances to 0.3.0, while
    a 0.3.0 backend is still detected as skew on a 0.2.0 framework.
    """
    try:
        backend = parse_version(backend_version)
        protocol = parse_version(protocol_version)
    except ValueError:
        return False
    if backend[0] != protocol[0]:
        return False
    if protocol[0] == 0 and backend[1] > protocol[1]:
        return False
    return True


# ======================================================================
# Sidecar manifest (minimal NATIVE_RAW spec, Phase A)
# ======================================================================

#: required manifest field -> required Python type
_MANIFEST_FIELDS: dict[str, type | tuple[type, ...]] = {
    "manifest_version": int,
    "interface_version": str,
    "dataset_id": str,
    "schema": str,
    "backend": str,
    "paths": list,
    "variables_delivered": list,
    "provenance": dict,
}


def build_manifest(result: AcquisitionResult) -> dict[str, Any]:
    """Build the JSON-serializable sidecar manifest payload for *result*."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "interface_version": PROTOCOL_VERSION,
        "dataset_id": result.dataset_id,
        "schema": str(result.schema),
        "backend": result.backend,
        "paths": [str(p) for p in result.paths],
        "variables_delivered": sorted(result.variables_delivered),
        "provenance": dict(result.provenance),
        "warnings": list(result.warnings),
    }


def validate_manifest(payload: Any) -> None:
    """Validate a sidecar-manifest payload against the minimal spec.

    Raises:
        ValueError: listing every problem found, if the payload is invalid.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"manifest must be a mapping, got {type(payload).__name__}")

    problems: list[str] = []
    for name, expected in _MANIFEST_FIELDS.items():
        if name not in payload:
            problems.append(f"missing field {name!r}")
            continue
        value = payload[name]
        if not isinstance(value, expected):
            problems.append(f"field {name!r} must be {expected}, got {type(value).__name__}")

    schema = payload.get("schema")
    if isinstance(schema, str):
        try:
            SchemaId(schema)
        except ValueError:
            problems.append(f"unknown schema id {schema!r} (known: {[s.value for s in SchemaId]})")

    paths = payload.get("paths")
    if isinstance(paths, list) and not all(isinstance(p, str) and p for p in paths):
        problems.append("every entry in 'paths' must be a non-empty string")

    variables = payload.get("variables_delivered")
    if isinstance(variables, list) and not all(isinstance(v, str) and v for v in variables):
        problems.append("every entry in 'variables_delivered' must be a non-empty string")

    warnings_field = payload.get("warnings", [])
    if not isinstance(warnings_field, list):
        problems.append("'warnings' must be a list when present")

    if problems:
        raise ValueError("invalid acquisition manifest: " + "; ".join(problems))


def write_manifest(result: AcquisitionResult, target_dir: Path) -> Path:
    """Write the sidecar manifest for *result* into *target_dir* and return its path."""
    payload = build_manifest(result)
    validate_manifest(payload)
    manifest_path = Path(target_dir) / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def read_manifest(path: Path) -> dict[str, Any]:
    """Read and validate a sidecar manifest from *path* (file or directory)."""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload


__all__ = [
    "PROTOCOL_VERSION",
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "GridClass",
    "SchemaId",
    "CredentialContext",
    "DatasetCapability",
    "AcquisitionRequest",
    "AcquisitionResult",
    "AcquisitionBackend",
    "ObservationCapability",
    "ObservationRequest",
    "ObservationBackend",
    "AttributeCapability",
    "AttributeRequest",
    "AttributeBackend",
    "parse_version",
    "is_compatible",
    "build_manifest",
    "validate_manifest",
    "write_manifest",
    "read_manifest",
]
