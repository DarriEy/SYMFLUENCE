# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for the mizuRoute topology staleness guard.

``topology.nc`` is only rewritten by mizuRoute preprocessing, so re-delineating a
domain without re-running that step leaves a topology describing the previous
geofabric. A routed run then silently uses the superseded network. These tests
pin the two detection signals — definitive count mismatch and advisory timestamp
skew — and the configured actions.
"""
from __future__ import annotations

import os

import pytest

gpd = pytest.importorskip("geopandas")
xr = pytest.importorskip("xarray")

import numpy as np  # noqa: E402
from shapely.geometry import LineString, Polygon  # noqa: E402

from symfluence.models.mizuroute.topology_freshness import (  # noqa: E402
    STALENESS_ACTIONS,
    check_topology_freshness,
    component_from_config,
    enforce_topology_freshness,
    reset_freshness_cache,
    resolve_source_shapefiles,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _isolate_report_cache():
    """The once-per-topology cache is process-global; keep tests independent."""
    reset_freshness_cache()
    yield
    reset_freshness_cache()


def _write_topology(path, n_seg, n_hru):
    ds = xr.Dataset(
        {
            "segId": ("seg", np.arange(1, n_seg + 1)),
            "downSegId": ("seg", np.full(n_seg, -1)),
            "hruId": ("hru", np.arange(1, n_hru + 1)),
            "hruToSegId": ("hru", np.arange(1, n_hru + 1)),
        }
    )
    ds.to_netcdf(path)
    ds.close()
    return path


def _write_network(path, n):
    gpd.GeoDataFrame(
        {"LINKNO": np.arange(1, n + 1)},
        geometry=[LineString([(i, 0), (i + 1, 1)]) for i in range(n)],
        crs="EPSG:4326",
    ).to_file(path)
    return path


def _write_basins(path, n):
    gpd.GeoDataFrame(
        {"GRU_ID": np.arange(1, n + 1)},
        geometry=[Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(n)],
        crs="EPSG:4326",
    ).to_file(path)
    return path


@pytest.fixture
def domain(tmp_path):
    """A consistent topology + geofabric: 5 segments, 5 basins."""
    net = _write_network(tmp_path / "net.shp", 5)
    bas = _write_basins(tmp_path / "bas.shp", 5)
    topo = _write_topology(tmp_path / "topology.nc", 5, 5)
    # Topology written last, as a fresh preprocessing run would leave it.
    now = os.stat(topo).st_mtime
    os.utime(net, (now - 100, now - 100))
    os.utime(bas, (now - 100, now - 100))
    return topo, net, bas


def test_consistent_topology_is_not_stale(domain):
    topo, net, bas = domain

    report = check_topology_freshness(topo, net, bas)

    assert not report.is_stale
    assert not report.definitive


def test_segment_count_mismatch_is_definitive(domain):
    """The Iceland failure: geofabric re-delineated, topology left behind."""
    topo, net, bas = domain
    _write_network(net, 3)  # re-delineated to fewer reaches

    report = check_topology_freshness(topo, net, bas)

    assert report.is_stale
    assert report.definitive
    assert any("5" in p and "3" in p and "segments" in p for p in report.problems)


def test_hru_count_mismatch_is_definitive(domain):
    topo, net, bas = domain
    _write_basins(bas, 9)

    report = check_topology_freshness(topo, net, bas)

    assert report.definitive
    assert any("HRUs" in p for p in report.problems)


def test_newer_shapefile_with_matching_counts_is_advisory(domain):
    """Same feature count but rewritten later — flagged, but not definitive."""
    topo, net, bas = domain
    future = os.stat(topo).st_mtime + 500
    os.utime(net, (future, future))

    report = check_topology_freshness(topo, net, bas)

    assert report.is_stale
    assert not report.definitive
    assert any("newer than the topology" in p for p in report.problems)


def test_missing_topology_is_not_stale(tmp_path, domain):
    """Not yet generated is not the same as out of date."""
    _, net, bas = domain

    report = check_topology_freshness(tmp_path / "absent.nc", net, bas)

    assert not report.is_stale


def test_unresolved_sources_are_skipped_not_guessed(domain):
    topo, _, _ = domain

    report = check_topology_freshness(topo, None, None)

    assert not report.is_stale


def test_message_names_the_file_and_the_remedy(domain):
    topo, net, bas = domain
    _write_network(net, 2)

    msg = check_topology_freshness(topo, net, bas).message()

    assert "topology.nc" in msg
    assert "Re-run mizuRoute preprocessing" in msg
    assert "MIZUROUTE_TOPOLOGY_STALENESS" in msg


# --------------------------------------------------------------------------
# enforce_topology_freshness / actions
# --------------------------------------------------------------------------


class _Component:
    """Minimal stand-in for a runner: config lookup + path/domain attributes."""

    def __init__(self, project_dir, domain_name="testdom"):
        self.project_dir = project_dir
        self.domain_name = domain_name
        self.config = {}
        self.logger = _Log()

    def _get_config_value(self, _lambda, default=None, dict_key=None):
        return self.config.get(dict_key, default)

    def _get_method_suffix(self):
        return "semidistributed"


class _Log:
    def __init__(self):
        self.warnings, self.infos, self.debugs = [], [], []

    def warning(self, m):
        self.warnings.append(str(m))

    def info(self, m):
        self.infos.append(str(m))

    def debug(self, m):
        self.debugs.append(str(m))


def _stale_component(tmp_path):
    (tmp_path / "shapefiles/river_network").mkdir(parents=True)
    (tmp_path / "shapefiles/river_basins").mkdir(parents=True)
    _write_network(tmp_path / "shapefiles/river_network/testdom_riverNetwork_semidistributed.shp", 3)
    _write_basins(tmp_path / "shapefiles/river_basins/testdom_riverBasins_semidistributed.shp", 3)
    topo = _write_topology(tmp_path / "topology.nc", 5, 5)
    return _Component(tmp_path), topo


def test_resolve_uses_domain_and_method_suffix_defaults(tmp_path):
    component, _ = _stale_component(tmp_path)

    net, bas = resolve_source_shapefiles(component)

    assert net is not None and net.name == "testdom_riverNetwork_semidistributed.shp"
    assert bas is not None and bas.name == "testdom_riverBasins_semidistributed.shp"


def test_warn_action_logs_and_returns(tmp_path):
    component, topo = _stale_component(tmp_path)

    report = enforce_topology_freshness(component, topo, action="warn")

    assert report.definitive
    assert len(component.logger.warnings) == 1
    assert "does not match" in component.logger.warnings[0]


def test_error_action_raises(tmp_path):
    from symfluence.core.exceptions import ModelExecutionError

    component, topo = _stale_component(tmp_path)

    with pytest.raises(ModelExecutionError, match="does not match the current geofabric"):
        enforce_topology_freshness(component, topo, action="error")


def test_ignore_action_skips_entirely(tmp_path):
    component, topo = _stale_component(tmp_path)

    report = enforce_topology_freshness(component, topo, action="ignore")

    assert not report.is_stale
    assert component.logger.warnings == []


def test_unknown_action_falls_back_to_warn(tmp_path):
    """A typo must not silently disable the guard."""
    component, topo = _stale_component(tmp_path)

    report = enforce_topology_freshness(component, topo, action="regenrate")

    assert report.is_stale
    assert any("Unknown MIZUROUTE_TOPOLOGY_STALENESS" in w for w in component.logger.warnings)
    assert any("does not match" in w for w in component.logger.warnings)


def test_fresh_topology_produces_no_warning(tmp_path):
    (tmp_path / "shapefiles/river_network").mkdir(parents=True)
    (tmp_path / "shapefiles/river_basins").mkdir(parents=True)
    net = _write_network(
        tmp_path / "shapefiles/river_network/testdom_riverNetwork_semidistributed.shp", 4)
    bas = _write_basins(
        tmp_path / "shapefiles/river_basins/testdom_riverBasins_semidistributed.shp", 4)
    topo = _write_topology(tmp_path / "topology.nc", 4, 4)
    now = os.stat(topo).st_mtime
    os.utime(net, (now - 50, now - 50))
    os.utime(bas, (now - 50, now - 50))

    report = enforce_topology_freshness(_Component(tmp_path), topo, action="warn")

    assert not report.is_stale


# --------------------------------------------------------------------------
# Dict-config path (calibration workers) and once-per-topology reporting
# --------------------------------------------------------------------------


def test_dict_config_resolves_via_data_dir_and_domain_name(tmp_path):
    """Calibration workers hold a config dict and no project_dir."""
    data_dir = tmp_path / "data"
    project = data_dir / "domain_testdom"
    (project / "shapefiles/river_network").mkdir(parents=True)
    (project / "shapefiles/river_basins").mkdir(parents=True)
    _write_network(project / "shapefiles/river_network/testdom_riverNetwork_semidistributed.shp", 3)
    _write_basins(project / "shapefiles/river_basins/testdom_riverBasins_semidistributed.shp", 3)

    component = component_from_config({
        "SYMFLUENCE_DATA_DIR": str(data_dir),
        "DOMAIN_NAME": "testdom",
        "DOMAIN_DEFINITION_METHOD": "semidistributed",
    })
    net, bas = resolve_source_shapefiles(component)

    assert net is not None and net.name == "testdom_riverNetwork_semidistributed.shp"
    assert bas is not None


def test_dict_config_without_data_dir_resolves_nothing(tmp_path):
    """Missing context must yield no finding rather than a guessed path."""
    component = component_from_config({"DOMAIN_NAME": "testdom"})

    assert resolve_source_shapefiles(component) == (None, None)


def test_stale_topology_is_reported_once_across_trials(tmp_path):
    """Calibration runs mizuRoute per trial; the warning must not repeat."""
    component, topo = _stale_component(tmp_path)

    for _ in range(6):
        enforce_topology_freshness(component, topo, action="warn")

    assert len(component.logger.warnings) == 1


def test_error_action_keeps_raising_on_repeat(tmp_path):
    """A cached verdict must not let a later trial slip past an 'error' policy."""
    from symfluence.core.exceptions import ModelExecutionError

    component, topo = _stale_component(tmp_path)

    for _ in range(3):
        with pytest.raises(ModelExecutionError):
            enforce_topology_freshness(component, topo, action="error")


def test_rebuilt_topology_is_rechecked(tmp_path):
    """The cache keys on mtime, so a regenerated topology gets a fresh verdict."""
    component, topo = _stale_component(tmp_path)
    enforce_topology_freshness(component, topo, action="warn")
    assert len(component.logger.warnings) == 1

    # Topology rebuilt to match the geofabric (3 segments / 3 basins).
    _write_topology(topo, 3, 3)
    now = os.stat(topo).st_mtime + 10
    os.utime(topo, (now, now))

    report = enforce_topology_freshness(component, topo, action="warn")

    assert not report.is_stale
    assert len(component.logger.warnings) == 1  # no new warning


def test_once_false_bypasses_the_cache(tmp_path):
    component, topo = _stale_component(tmp_path)

    enforce_topology_freshness(component, topo, action="warn", once=False)
    enforce_topology_freshness(component, topo, action="warn", once=False)

    assert len(component.logger.warnings) == 2


def test_all_documented_actions_are_accepted(tmp_path):
    """Every value in STALENESS_ACTIONS must be handled, not fall through to warn."""
    assert set(STALENESS_ACTIONS) == {"warn", "error", "regenerate", "ignore"}
    for action in ("warn", "ignore"):
        component, topo = _stale_component(tmp_path / action)
        enforce_topology_freshness(component, topo, action=action)
        assert not any("Unknown" in w for w in component.logger.warnings)
