"""Unit Tests for HYPE parameter regionalization adapter."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.optimization]


@pytest.fixture
def geoclass_path(tmp_path):
    """Create a minimal GeoClass.txt with 3 LU types and 2 soil types."""
    content = "! SLC\tLandUse\tSoilType\tArea\n"
    content += "1\t1\t1\t0.3\n"
    content += "2\t5\t1\t0.2\n"
    content += "3\t10\t2\t0.25\n"
    content += "4\t1\t2\t0.25\n"
    path = tmp_path / "GeoClass.txt"
    path.write_text(content)
    return path


@pytest.fixture
def hype_bounds():
    return {
        'ttmp': (-3.0, 3.0),
        'cmlt': (0.5, 15.0),
        'cevp': (0.01, 1.0),
        'srrcs': (0.001, 0.5),
        'wcwp': (0.01, 0.5),
        'wcfc': (0.1, 0.6),
        'wcep': (0.1, 0.7),
        'rrcs1': (0.01, 1.0),
        'rrcs2': (0.001, 0.5),
        'lp': (0.1, 1.0),
        'epotdist': (0.1, 10.0),
        'rcgrw': (0.001, 0.5),
        'rivvel': (0.1, 20.0),
        'damp': (0.01, 1.0),
    }


@pytest.fixture
def test_logger():
    logger = logging.getLogger('test_hype_regionalization')
    logger.setLevel(logging.DEBUG)
    return logger


class TestLoadLuAttributes:
    def test_loads_correct_count(self, geoclass_path, test_logger):
        from symfluence.models.hype.calibration.hype_regionalization import (
            load_lu_attributes,
        )
        df, lu_ids = load_lu_attributes(geoclass_path, test_logger)
        assert len(lu_ids) == 3
        assert set(lu_ids) == {1, 5, 10}
        assert len(df) == 3
        assert 'lai' in df.columns
        assert 'vegetation_height' in df.columns


class TestLoadSoilAttributes:
    def test_loads_correct_count(self, geoclass_path, test_logger):
        from symfluence.models.hype.calibration.hype_regionalization import (
            load_soil_attributes,
        )
        df, soil_ids = load_soil_attributes(geoclass_path, logger=test_logger)
        assert len(soil_ids) == 2
        assert len(df) == 2
        assert 'clay_fraction' in df.columns

    def test_uses_csv_when_provided(self, geoclass_path, tmp_path, test_logger):
        from symfluence.models.hype.calibration.hype_regionalization import (
            load_soil_attributes,
        )
        csv_path = tmp_path / "soil_attrs.csv"
        pd.DataFrame({
            'soil_id': [1, 2],
            'clay_fraction': [0.35, 0.15],
            'sand_fraction': [0.25, 0.55],
        }).to_csv(csv_path, index=False)
        df, soil_ids = load_soil_attributes(geoclass_path, csv_path, test_logger)
        assert df['clay_fraction'].iloc[0] == pytest.approx(0.35)


class TestCreateHypeRegionalization:
    def test_lumped_mode(self, geoclass_path, hype_bounds, test_logger):
        from symfluence.models.hype.calibration.hype_regionalization import (
            create_hype_regionalization,
        )
        reg = create_hype_regionalization(
            method='lumped', param_bounds=hype_bounds,
            geoclass_path=geoclass_path, logger=test_logger,
        )
        cal_params = reg.get_calibration_parameters()
        assert len(cal_params) == len(hype_bounds)

    def test_transfer_function_mode(self, geoclass_path, hype_bounds, test_logger):
        from symfluence.models.hype.calibration.hype_regionalization import (
            create_hype_regionalization,
        )
        reg = create_hype_regionalization(
            method='transfer_function', param_bounds=hype_bounds,
            geoclass_path=geoclass_path, logger=test_logger,
        )
        cal_params = reg.get_calibration_parameters()
        assert 'ttmp_a' in cal_params
        assert 'ttmp_b' in cal_params
        assert 'wcwp_a' in cal_params
        assert 'lp' in cal_params
        assert len(cal_params) > len(hype_bounds)

    def test_expand_to_par_values_lumped(self, geoclass_path, hype_bounds, test_logger):
        from symfluence.models.hype.calibration.hype_regionalization import (
            create_hype_regionalization,
        )
        reg = create_hype_regionalization(
            method='lumped', param_bounds=hype_bounds,
            geoclass_path=geoclass_path, logger=test_logger,
        )
        params = {k: (v[0] + v[1]) / 2 for k, v in hype_bounds.items()}
        result = reg.expand_to_par_values(params)
        # LU params produce lists (one per LU class position)
        assert 'ttmp' in result
        assert isinstance(result['ttmp'], list)
        assert len(result['ttmp']) == 10  # max LU ID
        # Global params produce scalars
        assert 'lp' in result
        assert isinstance(result['lp'], float)
        # Soil params produce lists
        assert 'wcwp' in result
        assert isinstance(result['wcwp'], list)

    def test_expand_to_par_values_tf(self, geoclass_path, hype_bounds, test_logger):
        from symfluence.models.hype.calibration.hype_regionalization import (
            create_hype_regionalization,
        )
        reg = create_hype_regionalization(
            method='transfer_function', param_bounds=hype_bounds,
            geoclass_path=geoclass_path, logger=test_logger,
        )
        cal_params = reg.get_calibration_parameters()
        coeffs = {k: (v[0] + v[1]) / 2 for k, v in cal_params.items()}
        result = reg.expand_to_par_values(coeffs)
        assert 'ttmp' in result
        assert isinstance(result['ttmp'], list)
        assert len(result['ttmp']) == 10  # max LU ID

    def test_unknown_method_raises(self, geoclass_path, hype_bounds, test_logger):
        from symfluence.models.hype.calibration.hype_regionalization import (
            create_hype_regionalization,
        )
        with pytest.raises(ValueError, match="Unsupported"):
            create_hype_regionalization(
                method='distributed', param_bounds=hype_bounds,
                geoclass_path=geoclass_path, logger=test_logger,
            )
