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
    SchemaId,
    write_manifest,
)
from symfluence.data.backends.errors import (
    AcquisitionError,
    AuthRequired,
    DatasetUnsupported,
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


def _is_native_entry(handler_cls: Any) -> bool:
    """True when the registered handler is in-tree (not a plugin shadow wrapper)."""
    module = getattr(handler_cls, '__module__', '') or ''
    return module.startswith('symfluence.')


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
        request for ``NLDAS2`` matches directly). Entries whose registry slot
        is currently occupied by a plugin shadow wrapper (module path not
        under ``symfluence.``) are excluded — they are not *this* backend's.
        """
        _ensure_handlers_registered()
        registry = R.acquisition_handlers
        caps: list[DatasetCapability] = []
        for primary, facts in _FORCING_FACTS.items():
            for key in (primary, *facts.aliases):
                handler_cls = registry.get(key)
                if handler_cls is None or not _is_native_entry(handler_cls):
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
        # asks win; otherwise the static capability declaration. Honesty
        # checking (manifest vs files) is conformance item 3, later phase.
        variables = request.variables if request.variables else (
            facts.variables if facts else frozenset()
        )

        result = AcquisitionResult(
            paths=self._collect_paths(output),
            schema=SchemaId.NATIVE_RAW,
            dataset_id=dataset,
            backend=self.name,
            provenance={
                'handler': f"{type(handler).__module__}.{type(handler).__qualname__}",
                'acquired_at': datetime.now(timezone.utc).isoformat(),
                'dataset_id': dataset,
            },
            variables_delivered=frozenset(variables),
        )
        write_manifest(result, target_dir)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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


# Register under the protocol-backend registry (importing this module is enough).
R.acquisition_backends.add('native', NativeBackend)

__all__ = ["NativeBackend"]
