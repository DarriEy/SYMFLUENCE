# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression test for grid-snapped DEM merge bounds.

Copernicus DEM tiles carry a half-pixel-offset grid: pixel *centers* sit
on degree lines, so pixel *edges* sit at half-pixel offsets from round
coordinates. ``_merge_tiles`` used to pass the raw request bbox straight
to ``rasterio.merge(bounds=...)``; rasterio anchors the output grid at
the requested west/north bound, and when that grid is offset from the
source pixel grid, rounding can leave the easternmost column (or
southernmost row) of the mosaic uncovered by any source. With
``nodata=None`` (Copernicus tiles) the uncovered strip is silently
zero-filled — elevation 0 m, invisible to most downstream checks.

Pin the fix: ``_snap_bounds_to_grid`` expands the request outward to the
first source's pixel edges (at most one native pixel per edge) before
merging, so every output pixel is covered by a source.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.transform import from_origin

pytestmark = [pytest.mark.unit, pytest.mark.quick]

RES = 0.01
# Half-pixel-offset grid a la Copernicus: pixel centers on degree lines,
# pixel edges at -22.005, -21.995, ... / 64.005, 63.995, ...
X_OFF = -22.0 - RES / 2
Y_OFF = 64.0 + RES / 2
# Request bbox in round-ish numbers, as users supply them. Against the
# half-pixel-offset grid every edge lands exactly mid-pixel — the tie
# case where rasterio.merge's window rounding leaves an output row/column
# uncovered (reproduced with rasterio 1.5: the southernmost row comes
# back fully zero-filled).
REQUEST = (-21.8, 62.7, -21.3, 63.4)


def _write_tile(path, x_origin, y_origin, size=100, fill=1.0, nodata=None):
    """Write a synthetic 1° x 1° GeoTIFF on a half-pixel-offset grid."""
    transform = from_origin(x_origin, y_origin, RES, RES)
    data = np.full((size, size), fill, dtype="float32")
    profile = {
        "driver": "GTiff", "height": size, "width": size,
        "count": 1, "dtype": "float32", "crs": "EPSG:4326",
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data, 1)


def _make_offset_tiles(tmp_path):
    """Two north/south-adjacent tiles on the half-pixel-offset grid.

    Top tile (fill 1) covers lat 63.005..64.005, bottom tile (fill 2)
    covers 62.005..63.005; both span lon -22.005..-21.005. Like real
    Copernicus tiles, neither declares a nodata value.
    """
    top = tmp_path / "tile_top.tif"
    bottom = tmp_path / "tile_bottom.tif"
    _write_tile(top, X_OFF, Y_OFF, fill=1.0)
    _write_tile(bottom, X_OFF, Y_OFF - 1.0, fill=2.0)
    return [top, bottom]


def _make_probe():
    """Minimal object exposing _merge_tiles; the rest of the acquirer
    is irrelevant here."""
    from symfluence.data.acquisition.handlers.dem import _TileDownloadMixin

    class _Probe(_TileDownloadMixin):
        def __init__(self):
            self.logger = MagicMock()

    return _Probe()


def test_raw_bounds_merge_exhibits_zero_filled_edge(tmp_path):
    """Fixture sanity check: document the pre-fix symptom.

    Passing the raw (mid-pixel) bbox straight to rasterio.merge leaves
    an output column/row uncovered and zero-filled. If this ever stops
    reproducing (rasterio fixed upstream), skip rather than fail — the
    snap then becomes belt-and-braces.
    """
    srcs = [rasterio.open(p) for p in _make_offset_tiles(tmp_path)]
    try:
        mosaic, _ = rio_merge(srcs, bounds=REQUEST)
    finally:
        for s in srcs:
            s.close()
    band = mosaic[0]
    col_dead = ~(band != 0).any(axis=0)
    row_dead = ~(band != 0).any(axis=1)
    if not (col_dead.any() or row_dead.any()):
        pytest.skip("rasterio.merge no longer zero-fills off-grid edge pixels")
    # The symptom is the easternmost column and/or southernmost row.
    assert col_dead[-1] or row_dead[-1]


def test_merge_tiles_snaps_bounds_full_coverage(tmp_path):
    """_merge_tiles with a mid-pixel bbox must produce a fully covered
    mosaic: no zero-filled column/row, grid-aligned transform, and
    bounds within one native pixel of the request."""
    tiles = _make_offset_tiles(tmp_path)
    out = tmp_path / "merged.tif"
    _make_probe()._merge_tiles(tiles, out, bounds=REQUEST)

    with rasterio.open(out) as ds:
        band = ds.read(1)
        t = ds.transform
        b = ds.bounds

    # 1. Full coverage: tiles are filled with 1.0 / 2.0, merge fill is 0.
    #    Every pixel must come from a source — no zero anywhere, and in
    #    particular no dead edge column/row.
    assert (band != 0).all(), (
        f"{np.count_nonzero(band == 0)} zero-filled pixels — "
        "merge bounds not aligned with the source pixel grid"
    )

    # 2. Output grid is aligned with the source pixel grid.
    col_steps = (t.c - X_OFF) / RES
    row_steps = (Y_OFF - t.f) / RES
    assert abs(col_steps - round(col_steps)) < 1e-6, f"x origin off-grid by {col_steps}"
    assert abs(row_steps - round(row_steps)) < 1e-6, f"y origin off-grid by {row_steps}"

    # 3. Bounds enclose the request, expanded outward by < 1 native pixel.
    west, south, east, north = REQUEST
    assert west - RES < b.left <= west
    assert east <= b.right < east + RES
    assert south - RES < b.bottom <= south
    assert north <= b.top < north + RES


def test_merge_tiles_snap_preserves_nodata_handling(tmp_path):
    """With explicit nodata the snap must not disturb nodata plumbing:
    output carries the source nodata and is still fully covered."""
    top = tmp_path / "tile_top.tif"
    bottom = tmp_path / "tile_bottom.tif"
    _write_tile(top, X_OFF, Y_OFF, fill=1.0, nodata=-9999.0)
    _write_tile(bottom, X_OFF, Y_OFF - 1.0, fill=2.0, nodata=-9999.0)
    out = tmp_path / "merged.tif"
    _make_probe()._merge_tiles([top, bottom], out, bounds=REQUEST)

    with rasterio.open(out) as ds:
        assert ds.nodata == -9999.0
        band = ds.read(1)
    assert ((band == 1.0) | (band == 2.0)).all()


def test_snap_bounds_outward_and_bounded_by_one_pixel():
    """The helper expands every edge outward by less than one pixel and
    lands exactly on grid lines."""
    from symfluence.data.acquisition.handlers.dem import _snap_bounds_to_grid

    transform = from_origin(X_OFF, Y_OFF, RES, RES)
    west, south, east, north = _snap_bounds_to_grid(REQUEST, transform)

    rw, rs, re, rn = REQUEST
    assert rw - RES < west <= rw
    assert rs - RES < south <= rs
    assert re <= east < re + RES
    assert rn <= north < rn + RES
    for val, ref in ((west, X_OFF), (east, X_OFF)):
        steps = (val - ref) / RES
        assert abs(steps - round(steps)) < 1e-6
    for val in (south, north):
        steps = (Y_OFF - val) / RES
        assert abs(steps - round(steps)) < 1e-6


def test_snap_bounds_is_identity_on_aligned_bounds():
    """Bounds already on the pixel grid must pass through unchanged
    (no creeping expansion on already-aligned requests)."""
    from symfluence.data.acquisition.handlers.dem import _snap_bounds_to_grid

    transform = from_origin(X_OFF, Y_OFF, RES, RES)
    aligned = (X_OFF + 20 * RES, Y_OFF - 130 * RES, X_OFF + 71 * RES, Y_OFF - 60 * RES)
    snapped = _snap_bounds_to_grid(aligned, transform)
    assert snapped == pytest.approx(aligned, abs=RES * 1e-9)
