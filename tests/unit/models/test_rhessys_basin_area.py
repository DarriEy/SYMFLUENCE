"""RHESSys must convert units with the same basin area it calibrates against.

``RHESSysWorker`` scores calibration trials using the shared catchment-area
lookup (the delineated river basin), while this target converted mm/day to
m³/s using the area baked into the RHESSys *worldfile*. The worldfile is a
generated input: it captures whatever area preprocessing saw and is not
rewritten when the domain is re-discretised, so the two drift apart.

On Bow_at_Banff_lumped_era5 the worldfile still carried 2248.0606 km²
against the delineation's 2207.5038 km² — 1.837% — which made the final
evaluation score a run at 0.835738 that the calibration objective had
scored 0.851883. Same simulation, two different basins.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from symfluence.models.rhessys.calibration.targets import RHESSysStreamflowTarget

DOMAIN = "RhDomain"
BASIN_M2 = 1_600_000_000.0        # 1600 km2, the delineated truth
WORLDFILE_M2 = 1_800_000_000.0    # 1800 km2, a stale worldfile


@pytest.fixture
def domain(tmp_path):
    """A domain with a delineated basin and a divergent worldfile."""
    basin = tmp_path / f"domain_{DOMAIN}" / "shapefiles" / "river_basins"
    basin.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"GRU_area": [BASIN_M2]},
        geometry=[Polygon([(0, 0), (40000, 0), (40000, 40000), (0, 40000)])],
        crs="EPSG:32611",
    ).to_file(basin / f"{DOMAIN}_riverBasins_lumped.shp")

    wf = tmp_path / f"domain_{DOMAIN}" / "settings" / "RHESSys" / "worldfiles"
    wf.mkdir(parents=True)
    (wf / f"{DOMAIN}.world").write_text(f"         {WORLDFILE_M2:.8f}    area\n")

    sim = tmp_path / f"domain_{DOMAIN}" / "optimization" / "RHESSys" / "dds_run_1"
    sim.mkdir(parents=True)
    return tmp_path, sim / "rhessys_basin.daily"


def _target(root):
    cfg = {"DOMAIN_NAME": DOMAIN, "SYMFLUENCE_DATA_DIR": str(root), "EXPERIMENT_ID": "run_1"}
    return RHESSysStreamflowTarget(cfg, root / f"domain_{DOMAIN}", logging.getLogger("test"))


def test_prefers_the_delineated_basin_over_the_worldfile(domain):
    """The stale worldfile must not win — this is the reported failure."""
    root, sim_file = domain
    assert _target(root)._get_basin_area(sim_file) == pytest.approx(BASIN_M2, rel=1e-6)


def test_falls_back_to_the_worldfile_without_a_delineated_basin(domain):
    """Domains with no river-basin layer must still convert units."""
    root, sim_file = domain
    for f in (root / f"domain_{DOMAIN}" / "shapefiles" / "river_basins").iterdir():
        f.unlink()
    area = _target(root)._get_basin_area(sim_file)
    assert area == pytest.approx(WORLDFILE_M2, rel=1e-6)


def test_dict_config_does_not_raise(domain):
    """A dict config must resolve, not AttributeError.

    The previous implementation read ``self.config.domain.name`` directly.
    Against a dict config that raised, the caller swallowed it and returned
    streamflow *unconverted* — still in mm/day — with only a warning.
    """
    root, sim_file = domain
    assert _target(root)._get_basin_area(sim_file) is not None
