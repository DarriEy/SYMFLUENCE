# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""MERIT-Basins handler tests.

Two regressions are pinned here (2026-06 parity campaign):

1. The Pfafstetter Level-1 region table must follow the OFFICIAL
   MERIT-Basins ReadMe mapping (1 Africa, 2 Europe, 3-5 Asia(+),
   6 South America, 7 North America, 8 Arctic, 9 Greenland). The handler
   previously shipped 1=Amazon ... 9=Australia, contradicting the shipped
   data (the pfaf_9 extent is Greenland, verified empirically).

2. The legacy bulk-download host (hydrology.princeton.edu) is DNS-dead.
   Missing region files must raise ONE actionable error listing the
   current distribution channels (Google Drive folder, Globus,
   reachhydro.org) and the local staging directory — no retry loop, no
   network attempt.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from symfluence.data.acquisition.handlers.merit_basins import (
    _PFAFSTETTER_L1_BBOXES,
    MERITBasinsAcquirer,
)

OFFICIAL_TABLE = {
    1: (-36.0, -20.0, 38.0, 56.0),    # Africa
    2: (12.0, -25.0, 72.0, 70.0),     # Europe
    3: (40.0, 55.0, 84.0, 180.0),     # North Asia / Siberia
    4: (0.0, 55.0, 56.0, 150.0),      # South Asia
    5: (-56.0, 90.0, 21.0, 180.0),    # Oceania + South Asian islands
    6: (-56.0, -93.0, 15.0, -32.0),   # South America
    7: (5.0, -170.0, 62.0, -52.0),    # North America
    8: (50.0, -180.0, 84.0, -50.0),   # Arctic (N. American Arctic)
    9: (58.0, -75.0, 84.0, -10.0),    # Greenland
}


def _make_acquirer(tmp_path, pour_point='39.75/-105.2', bbox=None):
    bbox = bbox or {
        'lat_min': 39.5, 'lat_max': 40.0,
        'lon_min': -105.5, 'lon_max': -105.0,
    }
    config = {
        'DOMAIN_NAME': 'test_domain',
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'POUR_POINT_COORDS': pour_point,
        'BOUNDING_BOX_COORDS': (
            f"{bbox['lat_max']}/{bbox['lon_min']}/"
            f"{bbox['lat_min']}/{bbox['lon_max']}"
        ),
        'EXPERIMENT_TIME_START': '2020-01-01',
        'EXPERIMENT_TIME_END': '2020-12-31',
    }
    acq = MERITBasinsAcquirer(config, MagicMock())
    acq.data_dir = tmp_path
    acq.bbox = dict(bbox)
    return acq


def _stage(geofabric_dir, code, legacy=False):
    geofabric_dir.mkdir(parents=True, exist_ok=True)
    if legacy:
        names = [f"merit_cat_pfaf_{code}.shp", f"merit_riv_pfaf_{code}.shp"]
    else:
        names = [
            f"pfaf_{code}_MERIT_Hydro_v07_Basins_v01.shp",
            f"pfaf_{code}_MERIT_Hydro_v07_Basins_v01_rivernet.shp",
        ]
    for n in names:
        (geofabric_dir / n).touch()


class TestOfficialRegionTable:
    """The region table is pinned to the official ReadMe mapping."""

    def test_table_matches_official_mapping(self):
        assert _PFAFSTETTER_L1_BBOXES == OFFICIAL_TABLE

    def test_greenland_is_region_9(self, tmp_path):
        """pfaf_9 is Greenland (verified empirically), NOT Australia."""
        acq = _make_acquirer(
            tmp_path, pour_point='70.0/-45.0',
            bbox={'lat_min': 69.5, 'lat_max': 70.5,
                  'lon_min': -46.0, 'lon_max': -44.0},
        )
        assert acq._find_pfafstetter_regions(70.0, -45.0) == [9]

    def test_amazon_is_region_6(self, tmp_path):
        """South America is region 6, NOT region 1."""
        acq = _make_acquirer(
            tmp_path, pour_point='-3.0/-60.0',
            bbox={'lat_min': -3.5, 'lat_max': -2.5,
                  'lon_min': -60.5, 'lon_max': -59.5},
        )
        assert acq._find_pfafstetter_regions(-3.0, -60.0) == [6]

    def test_africa_is_region_1(self, tmp_path):
        acq = _make_acquirer(
            tmp_path, pour_point='0.0/20.0',
            bbox={'lat_min': -0.5, 'lat_max': 0.5,
                  'lon_min': 19.5, 'lon_max': 20.5},
        )
        assert acq._find_pfafstetter_regions(0.0, 20.0) == [1]

    def test_europe_is_region_2(self, tmp_path):
        acq = _make_acquirer(
            tmp_path, pour_point='47.0/8.0',
            bbox={'lat_min': 46.5, 'lat_max': 47.5,
                  'lon_min': 7.5, 'lon_max': 8.5},
        )
        assert acq._find_pfafstetter_regions(47.0, 8.0) == [2]

    def test_north_america_is_region_7(self, tmp_path):
        acq = _make_acquirer(tmp_path)  # Colorado Front Range default
        assert acq._find_pfafstetter_regions(39.75, -105.2) == [7]

    def test_australia_is_region_5(self, tmp_path):
        """Australia belongs to region 5 (Oceania), NOT region 9."""
        acq = _make_acquirer(
            tmp_path, pour_point='-25.0/135.0',
            bbox={'lat_min': -25.5, 'lat_max': -24.5,
                  'lon_min': 134.5, 'lon_max': 135.5},
        )
        assert acq._find_pfafstetter_regions(-25.0, 135.0) == [5]

    def test_siberia_is_region_3(self, tmp_path):
        acq = _make_acquirer(
            tmp_path, pour_point='65.0/100.0',
            bbox={'lat_min': 64.5, 'lat_max': 65.5,
                  'lon_min': 99.5, 'lon_max': 100.5},
        )
        assert acq._find_pfafstetter_regions(65.0, 100.0) == [3]

    def test_bbox_spanning_regions_adds_codes(self, tmp_path):
        """A domain bbox overlapping a second region pulls it in."""
        acq = _make_acquirer(
            tmp_path, pour_point='60.0/-150.0',
            bbox={'lat_min': 55.0, 'lat_max': 65.0,
                  'lon_min': -155.0, 'lon_max': -145.0},
        )
        codes = acq._find_pfafstetter_regions(60.0, -150.0)
        assert 7 in codes and 8 in codes


class TestDeadHostReplacement:
    """Missing files raise one actionable error — no dead-host download."""

    def test_missing_files_raise_actionable_error(self, tmp_path):
        acq = _make_acquirer(tmp_path)
        with pytest.raises(RuntimeError) as excinfo:
            acq.download(tmp_path)
        msg = str(excinfo.value)
        assert "drive.google.com/drive/folders/1uCQFmdxFbjwoT9OYJxw-pXaP8q_GYH1a" in msg
        assert "Globus" in msg
        assert "reachhydro.org" in msg
        assert "hydrology.princeton.edu" in msg  # explains WHY it cannot download
        assert "pfaf_7_MERIT_Hydro_v07_Basins_v01.zip" in msg
        # The staging directory the handler reads from is spelled out
        assert "geofabric" in msg and "merit_basins" in msg

    def test_no_network_attempt_or_retry_loop(self, tmp_path, monkeypatch):
        """The failure path must not touch the network or sleep-retry."""
        import socket
        import time as _time

        def _no_net(*a, **k):
            raise AssertionError("network access attempted")

        def _no_sleep(*a, **k):
            raise AssertionError("retry sleep attempted")

        monkeypatch.setattr(socket.socket, "connect", _no_net)
        monkeypatch.setattr(_time, "sleep", _no_sleep)

        acq = _make_acquirer(tmp_path)
        with pytest.raises(RuntimeError, match="not staged locally"):
            acq.download(tmp_path)

    def test_staged_upstream_names_resolve(self, tmp_path):
        acq = _make_acquirer(tmp_path)
        geofabric_dir = acq._attribute_dir("geofabric") / "merit_basins"
        _stage(geofabric_dir, 7)
        out = acq.download(tmp_path)
        assert out == geofabric_dir

    def test_staged_legacy_names_resolve(self, tmp_path):
        acq = _make_acquirer(tmp_path)
        geofabric_dir = acq._attribute_dir("geofabric") / "merit_basins"
        _stage(geofabric_dir, 7, legacy=True)
        out = acq.download(tmp_path)
        assert out == geofabric_dir

    def test_partial_staging_still_raises(self, tmp_path):
        """Catchments staged but rivers missing -> still actionable error."""
        acq = _make_acquirer(tmp_path)
        geofabric_dir = acq._attribute_dir("geofabric") / "merit_basins"
        geofabric_dir.mkdir(parents=True, exist_ok=True)
        (geofabric_dir / "pfaf_7_MERIT_Hydro_v07_Basins_v01.shp").touch()
        with pytest.raises(RuntimeError, match="not staged locally"):
            acq.download(tmp_path)

    def test_no_dead_host_in_module(self):
        """The dead host must not be used as a download URL anywhere."""
        import inspect

        import symfluence.data.acquisition.handlers.merit_basins as mod
        src = inspect.getsource(mod)
        # It may be MENTIONED in error text/docs, but never as a URL template
        assert "http://hydrology.princeton.edu" not in src
