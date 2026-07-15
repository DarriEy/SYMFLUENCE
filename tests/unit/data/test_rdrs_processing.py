from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.constants import PhysicalConstants
from symfluence.data.preprocessing.dataset_handlers.rdrs_utils import RDRSHandler


@pytest.fixture
def rdrs_handler():
    config = {'DOMAIN_NAME': 'test_domain'}
    import logging
    logger = logging.getLogger('test')
    project_dir = Path('/tmp/test_project')
    return RDRSHandler(config, logger, project_dir)

def test_rdrs_v21_processing(rdrs_handler):
    # Mock RDRS v2.1 dataset
    times = pd.date_range('2015-01-01', periods=2, freq='h')
    ds = xr.Dataset(
        data_vars={
            'RDRS_v2.1_P_TT_1.5m': (['time'], [10.0, 15.0]), # Celsius
            'RDRS_v2.1_P_P0_SFC': (['time'], [1013.0, 1012.0]), # mb
            'RDRS_v2.1_A_PR0_SFC': (['time'], [1.0, 2.0]), # mm/hr
        },
        coords={'time': times}
    )

    processed = rdrs_handler.process_dataset(ds)

    assert processed.air_temperature.values[0] == 10.0 + PhysicalConstants.KELVIN_OFFSET
    assert processed.surface_air_pressure.values[0] == 1013.0 * 100
    assert processed.precipitation_flux.values[0] == 1.0 / 3600.0

def test_rdrs_v31_processing(rdrs_handler):
    # Mock RDRS v3.1 dataset (short names, already in standard units)
    times = pd.date_range('2015-01-01', periods=2, freq='h')
    ds = xr.Dataset(
        data_vars={
            'TT': (['time'], [283.15, 288.15]), # Kelvin
            'P0': (['time'], [101325.0, 101200.0]), # Pa
            'PR0': (['time'], [0.0001, 0.0002]), # mm/s
        },
        coords={'time': times}
    )

    processed = rdrs_handler.process_dataset(ds)

    assert processed.air_temperature.values[0] == 283.15
    assert processed.surface_air_pressure.values[0] == 101325.0
    assert processed.precipitation_flux.values[0] == 0.0001


def _casr_v32_raw_dataset() -> xr.Dataset:
    """A CaSR v3.2 dataset with the raw ECCC-archive names and native units."""
    times = pd.date_range('2015-01-01', periods=2, freq='h')
    return xr.Dataset(
        data_vars={
            'CaSR_v3.2_P_TT_1.5m': (['time'], [10.0, 15.0], {'units': 'deg_C'}),
            'CaSR_v3.2_P_P0_SFC': (['time'], [850.0, 851.0], {'units': 'mb'}),
            'CaSR_v3.2_P_HU_1.5m': (['time'], [0.004, 0.005], {'units': 'kg kg**-1'}),
            'CaSR_v3.2_P_UVC_10m': (['time'], [4.0, 5.0], {'units': 'kts'}),
            'CaSR_v3.2_A_PR0_SFC': (['time'], [0.001, 0.002], {'units': 'm'}),
            'CaSR_v3.2_P_FB_SFC': (['time'], [100.0, 200.0], {'units': 'W m**-2'}),
            'CaSR_v3.2_P_FI_SFC': (['time'], [250.0, 260.0], {'units': 'W m**-2'}),
        },
        coords={'time': times},
    )


def test_casr_v32_raw_processing(rdrs_handler):
    # Raw CaSR v3.2 archive names must standardise to CF names with the
    # correct unit conversions (the tiled-acquisition fallback path).
    processed = rdrs_handler.process_dataset(_casr_v32_raw_dataset())

    expected_cf = {
        'air_temperature', 'surface_air_pressure', 'specific_humidity',
        'wind_speed', 'precipitation_flux',
        'surface_downwelling_shortwave_flux', 'surface_downwelling_longwave_flux',
    }
    assert expected_cf.issubset(set(processed.data_vars))
    # No raw CaSR names survive.
    assert not any(str(v).startswith('CaSR_v3.2_') for v in processed.data_vars)

    # deg_C -> K
    assert processed.air_temperature.values[0] == pytest.approx(10.0 + PhysicalConstants.KELVIN_OFFSET)
    # mb -> Pa
    assert processed.surface_air_pressure.values[0] == pytest.approx(850.0 * 100)
    # metres/hour of accumulated precip -> kg m-2 s-1 (x1000 / 3600)
    assert processed.precipitation_flux.values[0] == pytest.approx(0.001 * 1000.0 / 3600.0)
    # knots -> m/s
    assert processed.wind_speed.values[0] == pytest.approx(4.0 * 0.514444)
    # humidity already kg/kg — unchanged
    assert processed.specific_humidity.values[0] == pytest.approx(0.004)


def test_point_scale_extractor_standardizes_raw_casr(rdrs_handler):
    # Regression: a raw CaSR-named file reaching the point-scale extractor
    # (e.g. the consolidated acquisition artifact globbed from merged_path)
    # must be CF-standardised before extraction rather than passed through
    # with raw names (which broke model-specific preprocessing downstream).
    import logging

    from symfluence.data.preprocessing.resampling.point_scale_extractor import (
        PointScaleForcingExtractor,
    )

    extractor = PointScaleForcingExtractor(
        config={'DOMAIN_NAME': 'test_domain', 'FORCING_DATASET': 'RDRS'},
        project_dir=Path('/tmp/test_project'),
        dataset_handler=rdrs_handler,
        logger=logging.getLogger('test'),
    )

    standardized = extractor._standardize_if_raw(
        _casr_v32_raw_dataset(), Path('domain_x_RDRS_2015_2020.nc')
    )
    assert 'air_temperature' in standardized.data_vars
    assert 'precipitation_flux' in standardized.data_vars
    assert not any(str(v).startswith('CaSR_v3.2_') for v in standardized.data_vars)

    # Idempotent: an already-CF file is returned untouched (no double conversion).
    already_cf = standardized
    again = extractor._standardize_if_raw(already_cf, Path('RDRS_monthly_201501.nc'))
    assert again.air_temperature.values[0] == already_cf.air_temperature.values[0]
