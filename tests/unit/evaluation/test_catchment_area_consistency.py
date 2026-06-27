# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Catchment-area resolution contract for the worker metrics path.

Trial metrics (StreamflowMetrics) and the final evaluation
(StreamflowEvaluator) resolve catchment area independently, and a
disagreement makes the calibration score and the final-evaluation score
incomparable. The contract: FIXED_CATCHMENT_AREA always wins, and the
last-resort default (1 km²) is identical in both paths.
"""
from __future__ import annotations

import logging

import pytest

from symfluence.evaluation.utilities.streamflow_metrics import StreamflowMetrics


@pytest.fixture
def metrics():
    return StreamflowMetrics()


class TestFixedAreaOverride:
    def test_fixed_catchment_area_wins(self, metrics, tmp_path):
        """FIXED_CATCHMENT_AREA (m²) must drive trial metrics too, not just
        the final-evaluation chain."""
        config = {'FIXED_CATCHMENT_AREA': 2.207e9}  # 2207 km² in m²
        area = metrics.get_catchment_area(config, tmp_path, 'nodomain')
        assert area == pytest.approx(2207.0)

    def test_fixed_area_skips_shapefile_lookup(self, metrics, tmp_path):
        """No shapefile exists under tmp_path; the override must not care."""
        config = {'FIXED_CATCHMENT_AREA': 5.0e8}
        area = metrics.get_catchment_area(
            config, tmp_path, 'nodomain', source='shapefile'
        )
        assert area == pytest.approx(500.0)


class TestDiscretisedCatchmentArea:
    """An HRU-discretised catchment (e.g. elevation bands) repeats the per-GRU
    basin area on every HRU row. Summing GRU_area across HRU rows multiplies the
    basin area by the HRU count, which silently inflates discharge unit
    conversions. The area must equal the true basin area, not N x it."""

    def _write_catchment(self, path, n_hrus, basin_area_m2):
        import geopandas as gpd
        from shapely.geometry import box
        rows = []
        for i in range(n_hrus):
            rows.append({
                'GRU_ID': 1,                       # single GRU, many HRUs
                'HRU_ID': i + 1,
                'GRU_area': basin_area_m2,          # basin area, duplicated per HRU
                'HRU_area': basin_area_m2 / n_hrus,  # sums to the basin area
                'geometry': box(i, 0, i + 1, 1),
            })
        gpd.GeoDataFrame(rows, crs='EPSG:4326').to_file(path)

    def test_hru_discretised_area_not_multiplied(self, metrics, tmp_path):
        basin_area_m2 = 2.207e9  # 2207 km²
        cat_dir = tmp_path / 'shapefiles' / 'catchment' / 'lumped' / 'run_1'
        cat_dir.mkdir(parents=True)
        shp = cat_dir / 'dom_HRUs_elevation.shp'
        self._write_catchment(shp, n_hrus=12, basin_area_m2=basin_area_m2)

        config = {
            'DOMAIN_DEFINITION_METHOD': 'lumped', 'EXPERIMENT_ID': 'run_1',
            'CATCHMENT_PATH': str(cat_dir), 'CATCHMENT_SHP_NAME': shp.name,
        }
        area = metrics.get_catchment_area(config, tmp_path, 'dom', source='shapefile')
        # Must be ~2207 km², NOT 12 x 2207.
        assert area == pytest.approx(2207.0, rel=1e-3)


class TestDefaultConsistency:
    def test_last_resort_default_matches_evaluator(self, metrics, tmp_path, caplog):
        """With no area source at all, the fallback is 1 km² — the same
        last resort the StreamflowEvaluator chain uses — and it warns."""
        with caplog.at_level(logging.WARNING):
            area = metrics.get_catchment_area({}, tmp_path, 'nodomain')
        assert area == 1.0
        assert any('FIXED_CATCHMENT_AREA' in r.message for r in caplog.records)
