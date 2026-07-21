# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""A failed watershed delineation must not silently widen the grid domain.

Grid clipping trims the bounding-box mesh down to the watershed. When the
delineation failed, the clip used to log a warning and return the FULL bbox
grid — a different, much larger domain — and the workflow reported success.
Observed in the paper reproduction: a broken TauDEM install produced 6900
cells instead of 2332, and the published cell count silently tripled.
"""
from __future__ import annotations

import logging

import pytest

from symfluence.core.exceptions import DiscretizationError
from symfluence.geospatial.geofabric.delineators.grid_delineator import GridDelineator


class _Harness(GridDelineator):
    """Bypass __init__; exercise only the clip-failure branch."""

    def __init__(self, basin_path=None):
        self.logger = logging.getLogger("test_grid_clip")
        self.config = {}
        self._basin_path = basin_path


@pytest.fixture
def grid_gdf():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    return gpd.GeoDataFrame(
        {"GRU_ID": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )


def test_failed_delineation_raises_instead_of_returning_full_grid(monkeypatch, grid_gdf):
    """The regression: a failed delineation must not yield the unclipped grid."""
    class _FailingLumped:
        def __init__(self, *a, **k):
            pass

        def delineate_lumped_watershed(self):
            return None, None  # delineation failed

    monkeypatch.setattr(
        "symfluence.geospatial.geofabric.delineators.lumped_delineator."
        "LumpedWatershedDelineator",
        _FailingLumped,
    )
    h = _Harness()
    with pytest.raises(DiscretizationError, match="could not be delineated"):
        h._clip_grid_to_watershed(grid_gdf)


def test_error_names_the_opt_out(monkeypatch, grid_gdf):
    """The message must tell the user how to intentionally model the bbox."""
    class _FailingLumped:
        def __init__(self, *a, **k):
            pass

        def delineate_lumped_watershed(self):
            return None, None

    monkeypatch.setattr(
        "symfluence.geospatial.geofabric.delineators.lumped_delineator."
        "LumpedWatershedDelineator",
        _FailingLumped,
    )
    with pytest.raises(DiscretizationError) as exc:
        _Harness()._clip_grid_to_watershed(grid_gdf)
    assert "CLIP_GRID_TO_WATERSHED" in str(exc.value)
