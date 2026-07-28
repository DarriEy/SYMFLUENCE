# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Detect a mizuRoute topology file that no longer matches its source geofabric.

``topology.nc`` is written from the river-network and river-basins shapefiles by
:meth:`MizuRouteTopologyGenerator.create_network_topology_file`, which always
regenerates. Staleness therefore arises when the domain is re-delineated and
mizuRoute preprocessing is *not* re-run before the next routed simulation: the
run silently routes over a superseded network.

This is not hypothetical. On the Iceland national domain a topology written in
February survived a re-delineation later that month and was still being used by a
calibration two months on — routing over 366 reaches and 127 HRUs that no longer
existed in the domain, and omitting 6 basins that did. Nothing detected it,
because nothing looked.

Two independent signals are checked:

* **Counts** — ``segId``/``hruId`` sizes against the shapefile feature counts.
  This is definitive: a mismatch means the topology cannot describe the current
  geofabric, whatever the timestamps say.
* **Timestamps** — topology older than a source shapefile. Advisory only, since
  copying a settings directory (as parallel calibration workers do) rewrites
  mtimes without changing content, and some filesystems preserve them on copy.

Counts are the load-bearing check; timestamps catch the case where the shapefile
was rewritten with the same feature count.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

#: Recognised values for ``MIZUROUTE_TOPOLOGY_STALENESS``.
STALENESS_ACTIONS = ('warn', 'error', 'regenerate', 'ignore')


@dataclass
class FreshnessReport:
    """Outcome of a topology freshness check."""

    topology: Path
    problems: List[str] = field(default_factory=list)
    #: True when a count mismatch was found, i.e. the topology provably does not
    #: describe the current geofabric (as opposed to merely being older).
    definitive: bool = False

    @property
    def is_stale(self) -> bool:
        return bool(self.problems)

    def message(self) -> str:
        """Operator-facing description, including how to fix it."""
        lead = (
            'mizuRoute topology does not match the current geofabric'
            if self.definitive
            else 'mizuRoute topology may be out of date'
        )
        lines = [f'{lead}: {self.topology}']
        lines.extend(f'  - {p}' for p in self.problems)
        lines.append(
            '  Re-run mizuRoute preprocessing to rebuild the topology from the '
            'current shapefiles, or set MIZUROUTE_TOPOLOGY_STALENESS: regenerate '
            'to rebuild it automatically.'
        )
        return '\n'.join(lines)


def _feature_count(path: Path) -> Optional[int]:
    """Number of features in a vector file, without loading geometry when possible."""
    try:
        from pyogrio import read_info

        return int(read_info(str(path))['features'])
    except Exception:  # noqa: BLE001 - fall back to the slower reader
        try:
            import geopandas as gpd

            return len(gpd.read_file(path))
        except Exception:  # noqa: BLE001 - unreadable source is not our error to raise
            return None


def _topology_counts(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """(n_segments, n_hrus) declared by a mizuRoute topology file."""
    try:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            n_seg = int(ds['segId'].size) if 'segId' in ds.variables else None
            n_hru = int(ds['hruId'].size) if 'hruId' in ds.variables else None
            return n_seg, n_hru
    except Exception:  # noqa: BLE001 - an unreadable topology fails later, and louder
        return None, None


def resolve_source_shapefiles(component: Any) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve the (river-network, river-basins) shapefiles a topology is built from.

    Mirrors the resolution in
    :meth:`MizuRouteTopologyGenerator.create_network_topology_file` so the check
    inspects the same files the generator would read.

    Args:
        component: Any object exposing ``_get_config_value``, ``_get_method_suffix``,
            ``project_dir`` and ``domain_name`` (model runners and preprocessors do).

    Returns:
        Tuple of paths; either element is None when it cannot be resolved.
    """
    get = component._get_config_value
    domain_name = getattr(component, 'domain_name', None) or get(
        lambda: None, default=None, dict_key='DOMAIN_NAME')
    if not domain_name:
        return None, None

    project_dir = getattr(component, 'project_dir', None)
    if project_dir is None:
        # Calibration executors and workers carry config but no project_dir.
        data_dir = get(lambda: None, default=None, dict_key='SYMFLUENCE_DATA_DIR')
        if not data_dir:
            return None, None
        project_dir = Path(data_dir) / f'domain_{domain_name}'
    project_dir = Path(project_dir)

    try:
        suffix = component._get_method_suffix()
    except Exception:  # noqa: BLE001 - fall back to the commonest layout
        suffix = 'semidistributed'

    # A lumped domain with river-network routing is built from the delineated
    # network rather than the lumped one.
    method = get(lambda: None, default='semidistributed', dict_key='DOMAIN_DEFINITION_METHOD')
    routing = get(lambda: None, default='lumped', dict_key='ROUTING_DELINEATION')
    if method == 'lumped' and routing == 'river_network':
        suffix = 'delineate'

    net_dir = get(lambda: None, default='default', dict_key='RIVER_NETWORK_SHP_PATH')
    net_name = get(lambda: None, default='default', dict_key='RIVER_NETWORK_SHP_NAME')
    basin_dir = get(lambda: None, default='default', dict_key='RIVER_BASINS_PATH')
    basin_name = get(lambda: None, default='default', dict_key='RIVER_BASINS_NAME')

    if not net_name or net_name == 'default':
        net_name = f'{domain_name}_riverNetwork_{suffix}.shp'
    if not basin_name or basin_name == 'default':
        basin_name = f'{domain_name}_riverBasins_{suffix}.shp'

    net_base = (project_dir / 'shapefiles/river_network'
                if not net_dir or net_dir == 'default' else Path(net_dir))
    basin_base = (project_dir / 'shapefiles/river_basins'
                  if not basin_dir or basin_dir == 'default' else Path(basin_dir))

    network = net_base / net_name
    basins = basin_base / basin_name
    return (network if network.exists() else None,
            basins if basins.exists() else None)


def check_topology_freshness(
    topology_path: Path,
    river_network_path: Optional[Path],
    river_basins_path: Optional[Path],
    logger: Optional[logging.Logger] = None,
) -> FreshnessReport:
    """Compare a topology file against the shapefiles it should have been built from.

    A missing topology is *not* stale — it has simply not been generated yet, and
    the generator handles that. Sources that cannot be resolved or read are
    skipped rather than guessed at, so this never invents a failure.

    Args:
        topology_path: Path to ``topology.nc``.
        river_network_path: Source river-network shapefile, or None if unresolved.
        river_basins_path: Source river-basins shapefile, or None if unresolved.
        logger: Optional logger for debug detail.

    Returns:
        A :class:`FreshnessReport`; ``is_stale`` is False when nothing was found.
    """
    report = FreshnessReport(topology=Path(topology_path))
    topology_path = Path(topology_path)
    if not topology_path.exists():
        return report

    n_seg, n_hru = _topology_counts(topology_path)
    topo_mtime = topology_path.stat().st_mtime

    for label, source, declared, unit in (
        ('river network', river_network_path, n_seg, 'segments'),
        ('river basins', river_basins_path, n_hru, 'HRUs'),
    ):
        if source is None:
            if logger is not None:
                logger.debug(f'Topology freshness: {label} shapefile not resolved, skipping')
            continue

        actual = _feature_count(Path(source))
        if actual is not None and declared is not None and actual != declared:
            report.definitive = True
            report.problems.append(
                f'topology declares {declared:,} {unit} but {label} '
                f'{Path(source).name} has {actual:,} features '
                f'(difference {declared - actual:+,})'
            )
        elif Path(source).stat().st_mtime > topo_mtime:
            report.problems.append(
                f'{label} {Path(source).name} is newer than the topology '
                f'(shapefile modified after topology was written)'
            )

    return report


class _DictComponent:
    """Adapter exposing the component interface over a plain config dict.

    Calibration workers execute mizuRoute directly from a config dict with no
    runner object to hand; this lets them reuse the same resolution and check.
    """

    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger

    def _get_config_value(self, _typed, default=None, dict_key=None):
        if dict_key is None:
            return default
        try:
            value = self.config.get(dict_key, default)
        except AttributeError:
            value = getattr(self.config, dict_key, default)
        return default if value is None else value

    def _get_method_suffix(self) -> str:
        return self._get_config_value(
            None, default='semidistributed', dict_key='DOMAIN_DEFINITION_METHOD')


def component_from_config(config: Any, logger: Optional[logging.Logger] = None) -> Any:
    """Wrap a config mapping so it can be passed to the freshness helpers."""
    return _DictComponent(config, logger)


#: Topologies already reported this process, keyed by (path, mtime). Calibration
#: runs mizuRoute once per trial; without this the same warning would be emitted
#: thousands of times and each emission would re-read the source shapefiles.
_REPORTED: dict = {}


def reset_freshness_cache() -> None:
    """Clear the once-per-topology report cache (used by tests)."""
    _REPORTED.clear()


def enforce_topology_freshness(
    component: Any,
    topology_path: Path,
    action: str = 'warn',
    logger: Optional[logging.Logger] = None,
    once: bool = True,
) -> FreshnessReport:
    """Check topology freshness and apply the configured action.

    Args:
        component: Runner/preprocessor used to resolve sources and, for
            ``regenerate``, to rebuild the topology.
        topology_path: Path to the topology file to check.
        action: One of :data:`STALENESS_ACTIONS`. Unknown values fall back to
            ``warn`` so a typo cannot silently disable the guard.
        logger: Logger for the warning/error message.

    Returns:
        The :class:`FreshnessReport`, so callers can inspect what was found.

    Raises:
        ModelExecutionError: When ``action == 'error'`` and the topology is stale.
    """
    log = logger if logger is not None else getattr(component, 'logger', None)
    action = (action or 'warn').lower()
    if action not in STALENESS_ACTIONS:
        if log is not None:
            log.warning(
                f"Unknown MIZUROUTE_TOPOLOGY_STALENESS '{action}'; "
                f"expected one of {', '.join(STALENESS_ACTIONS)}. Falling back to 'warn'."
            )
        action = 'warn'

    if action == 'ignore':
        return FreshnessReport(topology=Path(topology_path))

    topology_path = Path(topology_path)
    cache_key = None
    if once and topology_path.exists():
        cache_key = (str(topology_path), topology_path.stat().st_mtime, action)
        if cache_key in _REPORTED:
            cached = _REPORTED[cache_key]
            # 'error' must keep raising: a cached verdict cannot let a later
            # trial slip through after the first one was rejected.
            if action == 'error' and cached.is_stale:
                from symfluence.core.exceptions import ModelExecutionError

                raise ModelExecutionError(cached.message())
            return cached

    network, basins = resolve_source_shapefiles(component)
    report = check_topology_freshness(topology_path, network, basins, log)
    if cache_key is not None:
        _REPORTED[cache_key] = report

    if not report.is_stale:
        if log is not None:
            log.debug(f'mizuRoute topology is consistent with the geofabric: {topology_path}')
        return report

    if action == 'error':
        from symfluence.core.exceptions import ModelExecutionError

        raise ModelExecutionError(report.message())

    if action == 'regenerate':
        if log is not None:
            log.warning(report.message())
            log.info('Regenerating mizuRoute topology from the current geofabric')
        _regenerate_topology(component, log)
        return report

    if log is not None:
        log.warning(report.message())
    return report


def _regenerate_topology(component: Any, logger: Optional[logging.Logger]) -> None:
    """Rebuild the topology file from the current shapefiles."""
    from symfluence.models.mizuroute.preprocessor import MizuRoutePreProcessor

    preprocessor = MizuRoutePreProcessor(component.config, logger or logging.getLogger(__name__))
    preprocessor.topology_generator.create_network_topology_file()
