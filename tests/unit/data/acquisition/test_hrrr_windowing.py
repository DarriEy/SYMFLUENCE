# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""HRRR spatial windowing tests.

The hrrrzarr variable groups carry NO latitude/longitude coordinates — only
1-D ``projection_y/x_coordinate`` arrays on the fixed HRRR Lambert Conformal
Conic grid. The old handler masked on a latitude coordinate that never
exists there, so the bbox windowing silently no-opped and every acquisition
downloaded the full CONUS grid (~1.3 GB/day; parity campaign exp14,
2026-06).

These tests run the handler against a synthetic zarr store shaped like
hrrrzarr (no latitude coordinate, LCC projection coords, chunked data) and
assert that:

a) the index window is computed from a pyproj LCC transform of the bbox and
   applied BEFORE any data chunk is fetched (a store spy counts which chunk
   keys are actually read);
b) when no spatial window can be determined, the handler RAISES instead of
   silently fetching the full domain.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from pyproj import Transformer

from symfluence.data.acquisition.handlers.hrrr import HRRRAcquirer

# Colorado Front Range box (the exp14 case)
BBOX = {
    'lat_min': 39.5,
    'lat_max': 40.0,
    'lon_min': -105.5,
    'lon_max': -105.0,
}

GRID_N = 40          # 40x40 synthetic grid
GRID_SPACING = 3000  # meters, like HRRR
CHUNK = 10           # 10x10 chunks -> 16 chunks total


# The synthetic store is read through whichever zarr major the environment
# resolves (zarr is an explicit runtime dependency). The spy records every
# key actually fetched so the tests can assert the bbox window is applied
# before any data chunk is downloaded. zarr v3 stores expose an async ``get``;
# v2 stores are mappings keyed via ``__getitem__`` — subclass whichever matches.
if int(zarr.__version__.split(".", 1)[0]) >= 3:
    class SpyStore(zarr.storage.LocalStore):
        """Local zarr v3 store that records every key fetched via ``get``."""

        def __init__(self, root, read_log):
            super().__init__(root, read_only=True)
            # object.__setattr__: zarr Store dataclasses can freeze attributes
            object.__setattr__(self, 'read_log', read_log)

        async def get(self, key, prototype, byte_range=None):
            self.read_log.append(key)
            return await super().get(key, prototype, byte_range)
else:
    class SpyStore(zarr.storage.DirectoryStore):
        """Local zarr v2 store that records every key fetched via ``__getitem__``."""

        def __init__(self, root, read_log):
            super().__init__(str(root))
            self.read_log = read_log

        def __getitem__(self, key):
            self.read_log.append(key)
            return super().__getitem__(key)


def _data_chunk_keys(read_log, var: str):
    """Keys of *data* chunks read for ``var`` (metadata excluded)."""
    return {
        k for k in read_log
        if k.startswith(f"{var}/")
        and not k.endswith("zarr.json")          # zarr v3 metadata
        and not k.split("/")[-1].startswith(".")  # zarr v2 metadata
    }


def _make_store(tmp_path, with_proj_coords=True):
    """Write a synthetic store mimicking the real hrrrzarr layout.

    The archive splits each variable into a parent group carrying the 1-D
    projection coordinate arrays and a scalar time (path
    ``.../{level}/{var}``) and a subgroup carrying ONLY the chunked data
    array (path ``.../{level}/{var}/{level}``). Crucially there is no
    latitude/longitude anywhere.
    """
    tr = Transformer.from_crs(
        "EPSG:4326", HRRRAcquirer.HRRR_LCC_PROJ, always_xy=True
    )
    cx, cy = tr.transform(
        (BBOX['lon_min'] + BBOX['lon_max']) / 2,
        (BBOX['lat_min'] + BBOX['lat_max']) / 2,
    )
    half = GRID_N // 2
    proj_x = cx + (np.arange(GRID_N) - half) * float(GRID_SPACING)
    proj_y = cy + (np.arange(GRID_N) - half) * float(GRID_SPACING)

    parent = tmp_path / "synthetic_parent.zarr"
    sub = tmp_path / "synthetic_sub.zarr"

    coords = {'time': pd.Timestamp('2023-06-01 00:00')}
    if with_proj_coords:
        coords['projection_x_coordinate'] = proj_x
        coords['projection_y_coordinate'] = proj_y
    xr.Dataset(coords=coords).to_zarr(parent, consolidated=False, mode='w')

    data = np.arange(GRID_N * GRID_N, dtype=np.float32).reshape(GRID_N, GRID_N)
    ds_data = xr.Dataset(
        {'TMP': (('projection_y_coordinate', 'projection_x_coordinate'), data)},
    )
    ds_data.to_zarr(
        sub,
        encoding={'TMP': {'chunks': (CHUNK, CHUNK)}},
        consolidated=False,
        mode='w',
    )
    return parent, sub


def _make_acquirer(bbox=BBOX):
    config = {
        'DOMAIN_NAME': 'test_frontrange',
        'BOUNDING_BOX_COORDS': (
            f"{bbox['lat_max']}/{bbox['lon_min']}/"
            f"{bbox['lat_min']}/{bbox['lon_max']}"
        ),
        'EXPERIMENT_TIME_START': '2023-06-01 00:00',
        'EXPERIMENT_TIME_END': '2023-06-01 00:00',
        'HRRR_VARS': ['TMP'],
    }
    acq = HRRRAcquirer(config, MagicMock())
    acq.bbox = dict(bbox)
    acq.start_date = pd.Timestamp('2023-06-01 00:00')
    acq.end_date = pd.Timestamp('2023-06-01 00:00')
    return acq


def _patch_s3(parent_path, sub_path, read_log):
    """Route the handler's S3 access to the local spy-wrapped stores.

    The handler opens ``.../{level}/{var}/{level}`` (data subgroup) and
    ``.../{level}/{var}`` (coords parent group); map them onto the two
    local synthetic stores.
    """
    def fake_s3map(path, s3=None, **kwargs):
        root = sub_path if path.endswith('/TMP/2m_above_ground') else parent_path
        return SpyStore(root, read_log)

    return [
        patch(
            'symfluence.data.acquisition.handlers.hrrr.s3fs.S3FileSystem',
            return_value=MagicMock(),
        ),
        patch(
            'symfluence.data.acquisition.handlers.hrrr.s3fs.S3Map',
            side_effect=fake_s3map,
        ),
    ]


class TestLccWindowPreFetch:
    """(a) The bbox window must be applied before any chunk download."""

    def test_only_window_chunks_are_read(self, tmp_path):
        parent, sub = _make_store(tmp_path)
        read_log: list = []
        acq = _make_acquirer()

        patches = _patch_s3(parent, sub, read_log)
        with patches[0], patches[1]:
            out = acq.download(tmp_path / "out")

        total_chunks = (GRID_N // CHUNK) ** 2
        read = _data_chunk_keys(read_log, 'TMP')
        assert len(read) > 0, "no data chunks read at all"
        assert len(read) < total_chunks, (
            f"window not applied pre-fetch: read {len(read)}/{total_chunks} "
            f"chunks ({sorted(read)})"
        )
        # 0.5 deg x 0.5 deg around the grid center plus a 2-cell buffer spans
        # well under half the 40x40 synthetic grid in each direction.
        assert len(read) <= 9

        with xr.open_dataset(out) as ds:
            assert ds.sizes['projection_y_coordinate'] < GRID_N
            assert ds.sizes['projection_x_coordinate'] < GRID_N
            # The windowed grid must still cover the bbox (buffered)
            assert ds.sizes['projection_y_coordinate'] >= 2
            assert ds.sizes['projection_x_coordinate'] >= 2
            assert 'latitude' in ds.coords and 'longitude' in ds.coords
            lat = ds['latitude'].values
            lon = ds['longitude'].values
            assert lat.min() < BBOX['lat_min'] and lat.max() > BBOX['lat_max']
            assert lon.min() < BBOX['lon_min'] and lon.max() > BBOX['lon_max']

    def test_window_matches_direct_lcc_computation(self, tmp_path):
        """The applied window equals the pure pyproj LCC index window."""
        parent, _sub = _make_store(tmp_path)
        with xr.open_zarr(parent, consolidated=False) as ds:
            proj_y = ds['projection_y_coordinate'].values
            proj_x = ds['projection_x_coordinate'].values

        acq = _make_acquirer()
        y_slice, x_slice = acq._lcc_index_window(proj_y, proj_x, BBOX)

        # Window is interior to the synthetic grid and non-degenerate
        assert 0 < y_slice.start < y_slice.stop < GRID_N
        assert 0 < x_slice.start < x_slice.stop < GRID_N

        # All bbox corners fall inside the windowed LCC extent
        tr = Transformer.from_crs(
            "EPSG:4326", HRRRAcquirer.HRRR_LCC_PROJ, always_xy=True
        )
        for lon, lat in [
            (BBOX['lon_min'], BBOX['lat_min']),
            (BBOX['lon_max'], BBOX['lat_max']),
        ]:
            x, y = tr.transform(lon, lat)
            assert proj_x[x_slice][0] <= x <= proj_x[x_slice][-1]
            assert proj_y[y_slice][0] <= y <= proj_y[y_slice][-1]

    def test_tiny_bbox_falls_back_to_nearest_cell(self, tmp_path):
        parent, _sub = _make_store(tmp_path)
        with xr.open_zarr(parent, consolidated=False) as ds:
            proj_y = ds['projection_y_coordinate'].values
            proj_x = ds['projection_x_coordinate'].values

        center_lat = (BBOX['lat_min'] + BBOX['lat_max']) / 2
        center_lon = (BBOX['lon_min'] + BBOX['lon_max']) / 2
        tiny = {
            'lat_min': center_lat, 'lat_max': center_lat + 1e-4,
            'lon_min': center_lon, 'lon_max': center_lon + 1e-4,
        }
        acq = _make_acquirer(tiny)
        y_slice, x_slice = acq._lcc_index_window(proj_y, proj_x, tiny)
        # Buffered window still resolves; degenerate case yields >= 1 cell
        assert y_slice.stop - y_slice.start >= 1
        assert x_slice.stop - x_slice.start >= 1

    def test_bbox_off_grid_raises(self, tmp_path):
        parent, _sub = _make_store(tmp_path)
        with xr.open_zarr(parent, consolidated=False) as ds:
            proj_y = ds['projection_y_coordinate'].values
            proj_x = ds['projection_x_coordinate'].values

        europe = {'lat_min': 46.0, 'lat_max': 47.0, 'lon_min': 8.0, 'lon_max': 9.0}
        acq = _make_acquirer(europe)
        with pytest.raises(ValueError, match="does not intersect"):
            acq._lcc_index_window(proj_y, proj_x, europe)


class TestNoWindowRaises:
    """(b) Undeterminable window must raise — never full-domain download."""

    def test_store_without_coords_raises(self, tmp_path):
        parent, sub = _make_store(tmp_path, with_proj_coords=False)
        read_log: list = []
        acq = _make_acquirer()

        patches = _patch_s3(parent, sub, read_log)
        with patches[0], patches[1]:
            with pytest.raises(ValueError, match="Refusing to download the full"):
                acq.download(tmp_path / "out")

        # And crucially: no data chunks were fetched before the raise
        assert _data_chunk_keys(read_log, 'TMP') == set()

    def test_determine_xy_window_raises_without_any_coords(self):
        acq = _make_acquirer()
        ds = xr.Dataset(
            {'TMP': (('projection_y_coordinate', 'projection_x_coordinate'),
                     np.zeros((4, 4), dtype=np.float32))}
        )
        with pytest.raises(ValueError, match="Refusing to download the full"):
            acq._determine_xy_window(ds, BBOX)

    def test_unknown_dims_raise(self):
        acq = _make_acquirer()
        ds = xr.Dataset({'TMP': (('a', 'b'), np.zeros((4, 4), dtype=np.float32))})
        with pytest.raises(ValueError, match="spatial dimensions"):
            acq._determine_xy_window(ds, BBOX)


class TestLatLonMaskFallback:
    """Renamed grids with 2-D lat/lon still window via the geographic mask."""

    def test_mask_route(self):
        acq = _make_acquirer()
        lat = np.linspace(39.0, 40.5, 30)
        lon = np.linspace(-106.0, -104.5, 30)
        lon2d, lat2d = np.meshgrid(lon, lat)
        ds = xr.Dataset(
            {'TMP': (('y', 'x'), np.zeros((30, 30), dtype=np.float32))},
            coords={'latitude': (('y', 'x'), lat2d),
                    'longitude': (('y', 'x'), lon2d)},
        )
        y_slice, x_slice = acq._determine_xy_window(ds, BBOX)
        sub = ds.isel(y=y_slice, x=x_slice)
        assert 0 < sub.sizes['y'] < 30
        assert 0 < sub.sizes['x'] < 30
