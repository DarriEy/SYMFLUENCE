"""Catchment area must come from one source across calibration and evaluation.

``StreamflowMetrics`` resolved an experiment-scoped catchment/HRU shapefile
while ``StreamflowEvaluator`` read the delineated river basin. Both convert
mm/day to m3/s, so any disagreement scales simulated discharge and shifts
KGE's bias and variability terms.

On the P3 Bow-at-Banff domain the two differed by 1.837% (2248.0606 vs
2207.5038 km2), which surfaced as FUSE reporting a calibration KGE of
0.909638 for a simulation the final evaluation scored at 0.904377 — the
same bytes, scored twice, on two different areas.
"""
from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics

DOMAIN = "TestDomain"
# A square in a projected CRS: 40 km x 40 km = 1600 km2.
BASIN_AREA_M2 = 1_600_000_000.0
# What a stale HRU shapefile might claim instead.
STALE_AREA_M2 = 1_800_000_000.0


def _write(path, area_m2, column):
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {column: [area_m2]},
        geometry=[Polygon([(0, 0), (40000, 0), (40000, 40000), (0, 40000)])],
        crs="EPSG:32611",
    )
    gdf.to_file(path)


@pytest.fixture
def project_dir(tmp_path):
    """A domain carrying both a river basin and a divergent HRU shapefile."""
    _write(
        tmp_path / "shapefiles" / "river_basins" / f"{DOMAIN}_riverBasins_lumped.shp",
        BASIN_AREA_M2,
        "GRU_area",
    )
    _write(
        tmp_path / "shapefiles" / "catchment" / "lumped" / "run_1" / f"{DOMAIN}_HRUs_GRUs.shp",
        STALE_AREA_M2,
        "GRU_area",
    )
    return tmp_path


def test_river_basin_wins_over_a_divergent_hru_shapefile(project_dir):
    """The delineation is authoritative; a stale HRU file must not win."""
    area = StreamflowMetrics().get_catchment_area(
        {"DOMAIN_NAME": DOMAIN, "EXPERIMENT_ID": "run_1"}, project_dir, DOMAIN
    )
    assert area == pytest.approx(BASIN_AREA_M2 / 1e6, rel=1e-6)
    assert area != pytest.approx(STALE_AREA_M2 / 1e6, rel=1e-6)


def test_falls_back_to_the_catchment_shapefile_without_a_river_basin(project_dir):
    """Domains delineated without a river-basin layer still resolve."""
    for f in (project_dir / "shapefiles" / "river_basins").iterdir():
        f.unlink()
    area = StreamflowMetrics().get_catchment_area(
        {"DOMAIN_NAME": DOMAIN, "EXPERIMENT_ID": "run_1"}, project_dir, DOMAIN
    )
    assert area == pytest.approx(STALE_AREA_M2 / 1e6, rel=1e-6)


def test_fixed_catchment_area_still_overrides_everything(project_dir):
    """An explicit override must keep winning — it is the documented escape hatch."""
    area = StreamflowMetrics().get_catchment_area(
        {"DOMAIN_NAME": DOMAIN, "EXPERIMENT_ID": "run_1", "FIXED_CATCHMENT_AREA": 5_000_000_000.0},
        project_dir,
        DOMAIN,
    )
    assert area == pytest.approx(5000.0, rel=1e-9)


def test_geometry_used_when_the_basin_carries_no_area_column(tmp_path):
    """A river basin without GRU_area falls back to its projected geometry."""
    _write(
        tmp_path / "shapefiles" / "river_basins" / f"{DOMAIN}_riverBasins_lumped.shp",
        BASIN_AREA_M2,
        "unrelated_col",
    )
    area = StreamflowMetrics().get_catchment_area(
        {"DOMAIN_NAME": DOMAIN, "EXPERIMENT_ID": "run_1"}, tmp_path, DOMAIN
    )
    assert area == pytest.approx(1600.0, rel=1e-6)
