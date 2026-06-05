# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for GeometryProcessor.remove_spikes / despike_geodataframe."""
from __future__ import annotations

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("scipy")

from shapely.geometry import LineString, Polygon  # noqa: E402

from symfluence.geospatial.geofabric.processors.geometry_processor import GeometryProcessor  # noqa: E402

pytestmark = [pytest.mark.unit]


def _body_with_tentacle():
    """A compact 2x2 km body plus a thin ~100 m-wide tentacle running diagonally
    ~25 km to a far corner (projected, metres) — fill ratio < 0.05, like the real
    watershed artifacts."""
    body = Polygon([(0, 0), (2000, 0), (2000, 2000), (0, 2000)])
    tentacle = LineString([(1000, 1000), (20000, 20000)]).buffer(50)
    return body.union(tentacle)


def test_remove_spikes_strips_tentacle():
    spiky = _body_with_tentacle()
    bb = spiky.bounds
    fill_before = spiky.area / ((bb[2] - bb[0]) * (bb[3] - bb[1]))
    assert fill_before < 0.05  # genuinely spiky
    cleaned = GeometryProcessor.remove_spikes(spiky, resolution=50.0, max_iterations=4)
    cbb = cleaned.bounds
    fill_after = cleaned.area / ((cbb[2] - cbb[0]) * (cbb[3] - cbb[1]))
    assert fill_after > fill_before  # despiked: more compact
    assert (cbb[2] - cbb[0]) < 6000  # the ~25 km tentacle is gone
    assert cleaned.area >= 0.6 * (2000 * 2000)  # body preserved


def test_remove_spikes_leaves_compact_polygon_untouched():
    compact = Polygon([(0, 0), (3000, 0), (3000, 3000), (0, 3000)])
    out = GeometryProcessor.remove_spikes(compact, resolution=50.0)
    assert out is compact  # short-circuits, returns the same object


def test_remove_spikes_preserves_a_genuinely_thin_basin():
    # A long thin (but real, area-filled) basin must NOT be erased by the guard.
    thin = Polygon([(0, 0), (12000, 0), (12000, 400), (0, 400)])
    out = GeometryProcessor.remove_spikes(thin, resolution=50.0, keep_area_fraction=0.6)
    assert out.area >= 0.6 * thin.area  # not destroyed


def test_despike_geodataframe_roundtrips_crs():
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(geometry=[_body_with_tentacle()], crs="EPSG:3057")
    out = GeometryProcessor.despike_geodataframe(gdf, resolution=50.0, max_iterations=4)
    assert out.crs == gdf.crs
    assert len(out) == 1
