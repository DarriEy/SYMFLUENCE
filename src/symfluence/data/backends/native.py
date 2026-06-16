# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""NativeBackend — the existing in-tree acquisition handlers exposed through the protocol.

A thin adapter (Phase A, design §3): handlers stay UNTOUCHED; this class
enumerates the registered forcing datasets as :class:`DatasetCapability`
entries (``schema=NATIVE_RAW``) and translates an :class:`AcquisitionRequest`
into exactly the handler instantiation the ``AcquisitionService`` performs
today, wrapping the output into an :class:`AcquisitionResult` with a sidecar
manifest. Internal failures are mapped onto the protocol error taxonomy with
the original exception preserved as ``__cause__``.

Capability facts (grid class, auth, CFIF variables) come from a static table
encoding the verified inventory in ``community_services_full_coverage_plan.md``
§2; unknowns are marked conservatively in the entry notes. ``parity_grade`` is
``None`` for every native entry: the native backend IS the parity reference,
and the selection policy exempts it from the parity gate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from symfluence.core.registries import R
from symfluence.data.backends.contract import (
    PROTOCOL_VERSION,
    AcquisitionRequest,
    AcquisitionResult,
    DatasetCapability,
    GridClass,
    ObservationCapability,
    ObservationRequest,
    Redistribution,
    SchemaId,
    write_manifest,
)
from symfluence.data.backends.errors import (
    AcquisitionError,
    AuthRequired,
    DatasetUnsupported,
    IntegrityError,
    UpstreamOutage,
)

_logger = logging.getLogger(__name__)

# CFIF variable names (vocabulary: symfluence.data.preprocessing.cfif).
_GRID_STANDARD_8 = frozenset({
    'air_temperature',
    'precipitation_flux',
    'specific_humidity',
    'surface_air_pressure',
    'surface_downwelling_shortwave_flux',
    'surface_downwelling_longwave_flux',
    'eastward_wind',
    'northward_wind',
})


@dataclass(frozen=True)
class _DatasetFacts:
    """Static capability facts for one natively-handled forcing dataset."""

    grid_class: GridClass
    variables: frozenset[str]
    auth: frozenset[str] = frozenset()
    aliases: Tuple[str, ...] = ()
    notes: str = ""
    temporal: Optional[Tuple[str, str]] = None  # Phase A: coverage metadata deferred (None = undeclared)


# Inventory per community_services_full_coverage_plan.md §2 (forcing — cloud
# path + supplementary). MAF-only datasets (CASR, CONUS_I/II, ESPO-G6-R2, …)
# have no registry handler and are deliberately absent; 'local' is not an
# acquisition. Unknown/unverified facts are flagged in notes rather than
# guessed.
_FORCING_FACTS: dict[str, _DatasetFacts] = {
    'ERA5': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=_GRID_STANDARD_8,
        notes="ARCO pathway (anonymous GCS); the CDS pathway is the separate ERA5_CDS dataset.",
    ),
    'ERA5_CDS': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=_GRID_STANDARD_8,
        auth=frozenset({'cds'}),
    ),
    'ERA5_LAND': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=_GRID_STANDARD_8,
        auth=frozenset({'cds'}),
        aliases=('ERA5-LAND',),
        notes="Supplementary dataset.",
    ),
    'NLDAS': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=_GRID_STANDARD_8,
        auth=frozenset({'earthdata'}),
        aliases=('NLDAS2', 'NLDAS-2'),
    ),
    'AORC': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=_GRID_STANDARD_8,
        notes="Primary lat/lon pathway; pre-2002 windows fall back to the NWM-projected source natively.",
    ),
    'NEX-GDDP-CMIP6': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=_GRID_STANDARD_8 | frozenset({
            'air_temperature_min', 'air_temperature_max', 'wind_speed',
        }),
        aliases=('NEX-GDDP',),
        notes="Daily CMIP6 downscaling; surface pressure synthesized by the native handler.",
    ),
    'RDRS': _DatasetFacts(
        grid_class=GridClass.PROJECTED,
        variables=_GRID_STANDARD_8 | frozenset({'wind_speed'}),
        aliases=('RDRS_v3.1',),
        notes="Rotated-pole (CaSR v3.2 family).",
    ),
    'CARRA': _DatasetFacts(
        grid_class=GridClass.PROJECTED,
        variables=_GRID_STANDARD_8,
        auth=frozenset({'cds'}),
        notes="Rotated-pole Arctic reanalysis.",
    ),
    'CERRA': _DatasetFacts(
        grid_class=GridClass.PROJECTED,
        variables=_GRID_STANDARD_8,
        auth=frozenset({'cds'}),
        notes="Rotated-pole European reanalysis.",
    ),
    'CONUS404': _DatasetFacts(
        grid_class=GridClass.PROJECTED,
        variables=_GRID_STANDARD_8,
        notes="LCC 4 km; auth requirements unverified (assumed anonymous).",
    ),
    'HRRR': _DatasetFacts(
        grid_class=GridClass.PROJECTED,
        variables=_GRID_STANDARD_8,
        notes="LCC 3 km, anonymous AWS.",
    ),
    'DAYMET': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=frozenset({
            'precipitation_flux', 'air_temperature_min', 'air_temperature_max',
            'surface_downwelling_shortwave_flux',
        }),
        auth=frozenset({'earthdata'}),
        notes="LCC-native upstream served as lat/lon by the native handler; grid classification to be confirmed.",
    ),
    'NWM3_RETROSPECTIVE': _DatasetFacts(
        grid_class=GridClass.PROJECTED,
        variables=_GRID_STANDARD_8,
        notes="LCC 1 km.",
    ),
    'MSWEP': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=frozenset({'precipitation_flux'}),
        auth=frozenset({'rclone-gdrive'}),
        notes="Requires a configured rclone Google Drive remote; live access currently blocked upstream.",
    ),
    'EM-EARTH': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=frozenset({'precipitation_flux', 'air_temperature'}),
        auth=frozenset({'em-earth'}),
        aliases=('EM_EARTH',),
        notes="S3/FRDR access currently blocked upstream; auth requirements unverified.",
    ),
    'CHIRPS': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=frozenset({'precipitation_flux'}),
        notes="Supplementary precipitation dataset.",
    ),
    'GPM_IMERG': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=frozenset({'precipitation_flux'}),
        auth=frozenset({'earthdata'}),
        notes="Supplementary precipitation dataset.",
    ),
    'MERRA2': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=_GRID_STANDARD_8,
        auth=frozenset({'earthdata'}),
        notes="Supplementary dataset.",
    ),
    'WORLDCLIM': _DatasetFacts(
        grid_class=GridClass.REGULAR_LATLON,
        variables=frozenset({
            'precipitation_flux', 'air_temperature',
            'air_temperature_min', 'air_temperature_max',
        }),
        aliases=('WORLDCLIM_V21',),
        notes="Static monthly climatology (supplementary).",
    ),
}


def _ensure_handlers_registered() -> None:
    """Trigger in-tree handler registration (idempotent, heavy deps stay lazy)."""
    import symfluence.data.acquisition  # noqa: F401 — import populates R.acquisition_handlers


def _facts_for(dataset_id: str) -> Optional[_DatasetFacts]:
    """Look up static facts for *dataset_id* (case-insensitive, alias-aware)."""
    wanted = dataset_id.lower()
    for primary, facts in _FORCING_FACTS.items():
        if wanted == primary.lower() or any(wanted == alias.lower() for alias in facts.aliases):
            return facts
    return None


@dataclass
class NativeBackend:
    """Adapter exposing the existing acquisition-handler registry via the protocol.

    ``config``/``logger`` follow the handler constructor signature used by
    ``AcquisitionService`` today; ``capabilities()`` needs neither, ``acquire()``
    needs both (the native handlers read everything from config).
    """

    config: Any = None
    logger: Any = field(default=None, repr=False)

    name: str = field(default="native", init=False)
    interface_version: str = field(default=PROTOCOL_VERSION, init=False)

    def __post_init__(self) -> None:
        if self.logger is None:
            self.logger = _logger

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    def capabilities(self) -> Sequence[DatasetCapability]:
        """Enumerate the natively-registered forcing datasets.

        One capability per registered registry key (aliases included, so a
        request for ``NLDAS2`` matches directly). Under the protocol no
        plugin overwrites these registry slots (registry-overwrite shadow
        wrappers were deleted with Phase A step 5), so every registered key
        is this backend's.
        """
        _ensure_handlers_registered()
        registry = R.acquisition_handlers
        caps: list[DatasetCapability] = []
        for primary, facts in _FORCING_FACTS.items():
            for key in (primary, *facts.aliases):
                handler_cls = registry.get(key)
                if handler_cls is None:
                    continue
                caps.append(DatasetCapability(
                    dataset_id=key,
                    grid_class=facts.grid_class,
                    schema=SchemaId.NATIVE_RAW,
                    variables=facts.variables,
                    temporal=facts.temporal,
                    auth=facts.auth,
                    parity_grade=None,  # native is the parity reference; exempt from the gate
                    notes=facts.notes,
                ))
        return tuple(caps)

    def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        """Acquire via the registered native handler, exactly as the service does today.

        Instantiates ``AcquisitionRegistry.get_handler(dataset, config, logger)``
        and calls ``handler.download(target_dir)`` — the handler itself stays
        untouched and continues to read bbox/window/variables from config.
        The wrapping layer adds the result envelope and the sidecar manifest.
        """
        if self.config is None:
            raise AcquisitionError(
                "NativeBackend.acquire() requires a framework config "
                "(native handlers are configuration-driven)"
            )
        _ensure_handlers_registered()

        from symfluence.core.exceptions import DataAcquisitionError
        from symfluence.data.acquisition.registry import AcquisitionRegistry

        dataset = request.dataset_id
        try:
            handler = AcquisitionRegistry.get_handler(dataset, self.config, self.logger)
        except DataAcquisitionError as exc:
            raise DatasetUnsupported(
                f"No native acquisition handler registered for dataset '{dataset}'",
                dataset_id=dataset,
                backend=self.name,
            ) from exc

        target_dir = Path(request.target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        facts = _facts_for(dataset)
        try:
            output = handler.download(target_dir)
        except AcquisitionError:
            raise  # already protocol taxonomy — pass through untouched
        except PermissionError as exc:
            provider = next(iter(facts.auth), "") if facts else ""
            raise AuthRequired(
                f"Native handler for '{dataset}' was denied access: {exc}",
                provider=provider,
            ) from exc
        except (ConnectionError, TimeoutError) as exc:
            raise UpstreamOutage(
                f"Upstream failure while acquiring '{dataset}': {exc}",
                upstream=dataset,
            ) from exc
        except DataAcquisitionError as exc:
            raise AcquisitionError(f"Native handler for '{dataset}' failed: {exc}") from exc
        except (OSError, ValueError, KeyError, TypeError, RuntimeError,
                ImportError, AttributeError, IndexError) as exc:
            raise AcquisitionError(f"Native handler for '{dataset}' failed: {exc}") from exc

        # Declared (not file-sniffed) variable set: the request's explicit
        # asks win; otherwise the static capability declaration. NATIVE_RAW
        # files keep native upstream variable names (the rename map lives in
        # preprocessing), so the CFIF names declared here cannot be checked
        # against file contents — what CAN be cheaply verified is done in
        # _verify_delivery() and recorded in provenance.
        variables = request.variables if request.variables else (
            facts.variables if facts else frozenset()
        )

        paths = self._collect_paths(output)
        verified, netcdf_variables = self._verify_delivery(paths, dataset)

        provenance = {
            'handler': f"{type(handler).__module__}.{type(handler).__qualname__}",
            'acquired_at': datetime.now(timezone.utc).isoformat(),
            'dataset_id': dataset,
            'honesty_checks': verified,
        }
        if netcdf_variables:
            # Native (upstream) names actually present in NetCDF outputs —
            # the auditable ground truth behind the declared CFIF set above.
            provenance['netcdf_variables'] = ','.join(sorted(netcdf_variables))

        result = AcquisitionResult(
            paths=paths,
            schema=SchemaId.NATIVE_RAW,
            dataset_id=dataset,
            backend=self.name,
            provenance=provenance,
            variables_delivered=frozenset(variables),
        )
        write_manifest(result, target_dir)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _verify_delivery(paths: Tuple[Path, ...], dataset: str) -> Tuple[str, frozenset]:
        """Cheap honesty checks on what the handler actually delivered (item 3).

        CHECKED (failures raise :class:`IntegrityError`, the WSC-pagination
        class — silent partial delivery must never look like success):

        * every reported path exists and is non-empty;
        * every NetCDF output (``.nc``/``.nc4``) opens and contains at least
          one data variable (native upstream names are collected so the
          manifest provenance records the auditable file contents).

        NOT checked (documented limits, deliberately cheap):

        * presence of the declared CFIF names — NATIVE_RAW files keep native
          upstream variable names and the rename map is owned by the
          preprocessing layer;
        * timestep completeness against the request window (native handlers
          read their window from config; window conformance is item 4);
        * contents of non-NetCDF outputs (CSV/GRIB/zarr) beyond existence.

        Returns:
            ``(honesty_checks, netcdf_variables)`` — a short provenance tag of
            what was verified, and the union of native variable names found in
            NetCDF outputs (empty when no NetCDF was delivered).
        """
        problems: list[str] = []
        netcdf_names: set[str] = set()
        checked_netcdf = False
        for path in paths:
            p = Path(path)
            if not p.exists():
                problems.append(f"reported path does not exist: {p}")
                continue
            if p.is_file() and p.stat().st_size == 0:
                problems.append(f"reported output is empty: {p}")
                continue
            if p.suffix.lower() not in ('.nc', '.nc4'):
                continue
            checked_netcdf = True
            try:
                import xarray as xr  # lazy: heavy dependency
                with xr.open_dataset(p, decode_times=False) as ds:
                    names = {str(name) for name in ds.data_vars}
            except (OSError, ValueError, KeyError, ImportError) as exc:
                problems.append(f"unreadable NetCDF output {p.name}: {exc}")
                continue
            if not names:
                problems.append(f"NetCDF output {p.name} contains no data variables")
            netcdf_names |= names

        if problems:
            raise IntegrityError(
                f"Native handler for '{dataset}' reported outputs that do not "
                f"match what it delivered: " + "; ".join(problems)
            )

        verified = 'paths-exist,files-non-empty' + (
            ',netcdf-readable,netcdf-has-variables' if checked_netcdf else ''
        )
        return verified, frozenset(netcdf_names)

    @staticmethod
    def _collect_paths(output: Path) -> Tuple[Path, ...]:
        """Normalize a handler's return (file or directory) into result paths."""
        output = Path(output)
        if output.is_dir():
            from symfluence.data.backends.contract import MANIFEST_FILENAME
            files = tuple(sorted(
                p for p in output.rglob('*') if p.is_file() and p.name != MANIFEST_FILENAME
            ))
            return files or (output,)
        return (output,)


# ======================================================================
# Observation flavour — NativeObservationBackend (contract 0.4.0)
# ======================================================================

@dataclass(frozen=True)
class _ProviderFacts:
    """Static observation-capability facts for one natively-served provider.

    ``data_license`` / ``redistribution`` document the SOURCE posture; native
    is never gated on redistribution (it fetches from source on the user's
    behalf, not a mirror), but declaring it lets the result/manifest carry the
    obligation and lets the parity work compare native vs community on equal
    footing.
    """

    station_id_scheme: str
    data_license: str = ""
    attribution: str = ""
    redistribution: Redistribution = Redistribution.UNKNOWN
    notes: str = ""


# Legacy streamflow providers exposed through the protocol, added incrementally
# as each is driven to a parity gate (community_services_full_coverage_plan.md).
# USGS first (Phase 1c); WSC/SMHI/LAMAH_ICE/VI/CARAVANS follow as they are
# verified. A provider appears here only when native genuinely serves it via a
# registered observation handler.
_OBSERVATION_FACTS: dict[str, _ProviderFacts] = {
    'USGS': _ProviderFacts(
        station_id_scheme="USGS NWIS site number (e.g. '06191500')",
        data_license="public-domain",
        attribution="U.S. Geological Survey, National Water Information System (NWIS)",
        redistribution=Redistribution.OPEN,
        notes="NWIS discharge; the native handler reads the station id and window from config.",
    ),
}

#: Registry keys to probe for a provider's native handler, in order. The
#: formalized observation handlers register under both the bare provider name
#: and a ``<provider>_streamflow`` key; either satisfies the capability claim.
def _native_handler_key(provider: str) -> Optional[str]:
    registry = R.observation_handlers
    for key in (provider.lower(), f"{provider.lower()}_streamflow"):
        if registry.get(key) is not None:
            return key
    return None


def _ensure_observation_handlers_registered() -> None:
    """Trigger in-tree observation-handler registration (idempotent)."""
    import symfluence.data.observation.handlers  # noqa: F401 — populates R.observation_handlers


@dataclass
class NativeObservationBackend:
    """Adapter exposing the in-tree observation handlers via the protocol.

    The observation-flavored sibling of :class:`NativeBackend`: handlers stay
    UNTOUCHED; ``acquire()`` performs exactly the registry dispatch the
    ``ObservedDataProcessor`` registry tier performs today
    (``ObservationRegistry.get_handler(provider).acquire()`` then
    ``.process()``), wrapping the processed CSV into an
    :class:`AcquisitionResult` (``schema=OBS_CSV_V1``) with a sidecar manifest.

    This is what makes native streamflow a co-equal protocol backend: under
    ``DATA_ACCESS: community`` the selection layer can resolve to ``native``
    (parity reference) when no community backend serves a provider, instead of
    silently falling through to the legacy tiers. ``parity_grade`` is ``None``
    for every native entry — native IS the reference and the selection policy
    exempts it from the parity gate.
    """

    config: Any = None
    logger: Any = field(default=None, repr=False)

    name: str = field(default="native", init=False)
    interface_version: str = field(default=PROTOCOL_VERSION, init=False)

    def __post_init__(self) -> None:
        if self.logger is None:
            self.logger = _logger

    def capabilities(self) -> Sequence[ObservationCapability]:
        """Enumerate the natively-served streamflow providers exposed so far.

        One capability per entry in :data:`_OBSERVATION_FACTS` that has a
        registered native handler (so the claim never outruns what native can
        actually serve).
        """
        _ensure_observation_handlers_registered()
        caps: list[ObservationCapability] = []
        for provider, facts in _OBSERVATION_FACTS.items():
            if _native_handler_key(provider) is None:
                continue
            caps.append(ObservationCapability(
                provider_id=provider,
                kinds=frozenset({'streamflow'}),
                station_id_scheme=facts.station_id_scheme,
                temporal=None,
                auth=frozenset(),
                parity_grade=None,  # native is the parity reference; exempt from the gate
                notes=facts.notes,
                data_license=facts.data_license,
                attribution=facts.attribution,
                redistribution=facts.redistribution,
            ))
        return tuple(caps)

    def acquire(self, request: ObservationRequest) -> AcquisitionResult:
        """Acquire via the registered native observation handler.

        Mirrors :meth:`NativeBackend.acquire`: the handler reads station id /
        window / bbox from config (the request fields are framework-resolved
        copies of the same values), so this adapter only orchestrates the
        existing ``acquire()`` → ``process()`` pair and wraps the output.
        """
        if self.config is None:
            raise AcquisitionError(
                "NativeObservationBackend.acquire() requires a framework config "
                "(native observation handlers are configuration-driven)"
            )
        _ensure_observation_handlers_registered()

        from symfluence.core.exceptions import DataAcquisitionError
        from symfluence.data.observation.registry import ObservationRegistry

        provider = request.provider_id
        key = _native_handler_key(provider)
        if key is None:
            raise DatasetUnsupported(
                f"No native observation handler registered for provider '{provider}'",
                dataset_id=provider,
                backend=self.name,
            )

        try:
            handler = ObservationRegistry.get_handler(key, self.config, self.logger)
        except DataAcquisitionError as exc:
            raise DatasetUnsupported(
                f"No native observation handler registered for provider '{provider}'",
                dataset_id=provider,
                backend=self.name,
            ) from exc

        target_dir = Path(request.target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            raw_path = handler.acquire()
            processed_path = handler.process(raw_path)
        except AcquisitionError:
            raise  # already protocol taxonomy
        except PermissionError as exc:
            raise AuthRequired(
                f"Native observation handler for '{provider}' was denied access: {exc}",
            ) from exc
        except (ConnectionError, TimeoutError) as exc:
            raise UpstreamOutage(
                f"Upstream failure while acquiring '{provider}' observations: {exc}",
                upstream=provider,
            ) from exc
        except (DataAcquisitionError, OSError, ValueError, KeyError, TypeError,
                RuntimeError, ImportError, AttributeError, IndexError) as exc:
            raise AcquisitionError(
                f"Native observation handler for '{provider}' failed: {exc}"
            ) from exc

        paths = self._verify_observation_delivery(processed_path, provider)
        facts = _OBSERVATION_FACTS.get(provider.upper())

        provenance = {
            'handler': f"{type(handler).__module__}.{type(handler).__qualname__}",
            'acquired_at': datetime.now(timezone.utc).isoformat(),
            'provider_id': provider,
            'kind': request.kind,
            'honesty_checks': 'processed-path-exists,non-empty',
        }
        result = AcquisitionResult(
            paths=paths,
            schema=SchemaId.OBS_CSV_V1,
            dataset_id=provider,
            backend=self.name,
            provenance=provenance,
            variables_delivered=frozenset({request.kind}),
            data_license=facts.data_license if facts else "",
            attribution=facts.attribution if facts else "",
            redistribution=facts.redistribution if facts else Redistribution.UNKNOWN,
        )
        write_manifest(result, target_dir)
        return result

    @staticmethod
    def _verify_observation_delivery(processed_path: Any, provider: str) -> Tuple[Path, ...]:
        """Honesty check: the handler's reported processed CSV exists and is non-empty."""
        if processed_path is None:
            raise IntegrityError(
                f"Native observation handler for '{provider}' returned no processed path"
            )
        p = Path(processed_path)
        if not p.exists():
            raise IntegrityError(
                f"Native observation handler for '{provider}' reported a path that "
                f"does not exist: {p}"
            )
        if p.is_file() and p.stat().st_size == 0:
            raise IntegrityError(
                f"Native observation handler for '{provider}' produced an empty output: {p}"
            )
        return (p,)


# Register under the protocol-backend registries (importing this module is enough).
R.acquisition_backends.add('native', NativeBackend)
R.observation_backends.add('native', NativeObservationBackend)

__all__ = ["NativeBackend", "NativeObservationBackend"]
