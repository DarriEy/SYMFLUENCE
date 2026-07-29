"""An area column in the wrong units must not beat the geometry.

``GRU_area`` is only trustworthy if it was written in m². The point
delineator stored ``polygon.area`` straight off an EPSG:4326 geometry, so
``paradise_snotel_wa`` carried 0.0004 square degrees where every consumer
expects m². Divided by 1e6 that reads as 4e-10 km² for a 3.39 km² domain,
and discharge unit conversion silently fell back to the 1 km² default —
making every streamflow metric on that domain meaningless.

Geometry carries a CRS and is therefore self-describing, so it arbitrates
when the two disagree by more than an order of magnitude.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics

DOMAIN = "UnitsDomain"
# 40 km x 40 km in UTM = 1600 km2.
SQUARE = Polygon([(0, 0), (40000, 0), (40000, 40000), (0, 40000)])
TRUE_KM2 = 1600.0


def _basin(tmp_path, area_value, crs="EPSG:32611", geom=SQUARE):
    path = tmp_path / "shapefiles" / "river_basins" / f"{DOMAIN}_riverBasins_point.shp"
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame({"GRU_area": [area_value]}, geometry=[geom], crs=crs)
    gdf.to_file(path)
    return tmp_path


def _area(project_dir):
    return StreamflowMetrics().get_catchment_area(
        {"DOMAIN_NAME": DOMAIN, "EXPERIMENT_ID": "run_1"}, project_dir, DOMAIN
    )


def test_square_degree_column_is_rejected_in_favour_of_geometry(tmp_path):
    """The paradise_snotel_wa failure: a deg2 column against an m2 assumption."""
    # 1600 km2 expressed (wrongly) as square degrees is a tiny number.
    project_dir = _basin(tmp_path, 0.0004)
    assert _area(project_dir) == pytest.approx(TRUE_KM2, rel=1e-6)


def test_a_correct_square_metre_column_is_used(tmp_path):
    """The normal case must be untouched — the column still wins."""
    project_dir = _basin(tmp_path, TRUE_KM2 * 1e6)
    assert _area(project_dir) == pytest.approx(TRUE_KM2, rel=1e-6)


def test_small_disagreements_keep_the_column(tmp_path):
    """HRU columns legitimately differ a little from the dissolved outline."""
    project_dir = _basin(tmp_path, TRUE_KM2 * 1e6 * 1.02)
    assert _area(project_dir) == pytest.approx(TRUE_KM2 * 1.02, rel=1e-6)


def test_absurdly_large_column_is_also_rejected(tmp_path):
    """The guard is symmetric — a column 1000x too large is equally wrong."""
    project_dir = _basin(tmp_path, TRUE_KM2 * 1e6 * 1000)
    assert _area(project_dir) == pytest.approx(TRUE_KM2, rel=1e-6)
