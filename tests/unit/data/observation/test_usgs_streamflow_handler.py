# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for the USGS streamflow handler.

Covers the two bugs found by the 2026-06 native-vs-CSFS parity experiment:

1. NWIS RDB timestamps are gauge-local clock time with a per-row ``tz_cd``
   code; the handler previously ignored the code, leaving processed
   timestamps in local time (with DST discontinuities) while forcing is UTC.
2. ``_download_data`` previously hardcoded ``endDT=datetime.now()``,
   downloading years past the experiment window.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from symfluence.data.observation.handlers.usgs import USGSStreamflowHandler

pytestmark = [pytest.mark.unit, pytest.mark.data]


@pytest.fixture
def mock_config(tmp_path):
    return {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'EXPERIMENT_TIME_START': '2023-06-01 00:00',
        'EXPERIMENT_TIME_END': '2023-06-15 00:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_TIME_STEP_SIZE': 3600,
        'BOUNDING_BOX_COORDS': '46/-111/44/-110',
        'STATION_ID': '06191500',
    }


@pytest.fixture
def handler(mock_config):
    return USGSStreamflowHandler(mock_config, logging.getLogger('test_usgs'))


def _rdb(rows: str) -> str:
    """Build a minimal NWIS RDB document around tab-separated data rows."""
    header = "agency_cd\tsite_no\tdatetime\ttz_cd\t147720_00060\t147720_00060_cd"
    fmt = "5s\t15s\t20d\t6s\t14n\t10s"
    return "\n".join([
        "# Data provided for site 06191500",
        "#  TS_ID    Parameter  Description",
        header,
        fmt,
        rows,
    ]) + "\n"


class TestTimezoneHandling:
    def test_rdb_local_time_converted_to_utc(self, handler, tmp_path):
        """MDT rows shift +6 h, MST rows +7 h — DST handled per row."""
        raw = tmp_path / "usgs_06191500_raw.rdb"
        raw.write_text(_rdb(
            "USGS\t06191500\t2023-06-01 12:00\tMDT\t3530\tA\n"
            "USGS\t06191500\t2023-06-01 13:00\tMDT\t3540\tA\n"
            "USGS\t06191500\t2023-01-15 12:00\tMST\t900\tA"
        ))

        out = handler.process(raw)
        df = pd.read_csv(out, index_col='datetime', parse_dates=True)

        # 12:00 MDT == 18:00 UTC; 12:00 MST == 19:00 UTC. The observed values
        # must land in the UTC-shifted hourly bins (resampling pads the index,
        # so assert on values, not index membership).
        factor = 0.028316846592
        assert df.loc[pd.Timestamp('2023-06-01 18:00'), 'discharge_cms'] == pytest.approx(3530 * factor)
        assert df.loc[pd.Timestamp('2023-06-01 19:00'), 'discharge_cms'] == pytest.approx(3540 * factor)
        assert df.loc[pd.Timestamp('2023-01-15 19:00'), 'discharge_cms'] == pytest.approx(900 * factor)

    def test_unknown_tz_code_warns_and_passes_through(self, mock_config, tmp_path, caplog):
        logger = logging.getLogger('test_usgs_unknown_tz')
        h = USGSStreamflowHandler(mock_config, logger)
        raw = tmp_path / "usgs_06191500_raw.rdb"
        raw.write_text(_rdb("USGS\t06191500\t2023-06-01 12:00\tXYZ\t3530\tA"))

        with caplog.at_level(logging.WARNING, logger='test_usgs_unknown_tz'):
            out = h.process(raw)

        assert any('XYZ' in r.message for r in caplog.records)
        df = pd.read_csv(out, index_col='datetime', parse_dates=True)
        # Treated as UTC: timestamp unchanged
        assert pd.Timestamp('2023-06-01 12:00') in df.index

    def test_csv_without_tz_column_unchanged(self, handler, tmp_path):
        """Manual CSV exports without tz info keep their timestamps as-is."""
        raw = tmp_path / "usgs_06191500_raw.csv"
        pd.DataFrame({
            'datetime': ['2023-06-01 12:00', '2023-06-01 13:00'],
            'discharge': [100.0, 110.0],
        }).to_csv(raw, index=False)

        out = handler.process(raw)
        df = pd.read_csv(out, index_col='datetime', parse_dates=True)
        assert pd.Timestamp('2023-06-01 12:00') in df.index

    def test_exact_unit_conversion(self, handler, tmp_path):
        raw = tmp_path / "usgs_06191500_raw.rdb"
        raw.write_text(_rdb("USGS\t06191500\t2023-06-01 12:00\tMDT\t1000\tA"))

        out = handler.process(raw)
        df = pd.read_csv(out, index_col='datetime', parse_dates=True)
        assert df['discharge_cms'].iloc[0] == pytest.approx(28.316846592, rel=1e-12)


class TestDownloadWindow:
    def test_end_dt_uses_experiment_end_not_now(self, handler, tmp_path):
        captured = {}

        def fake_get(url, timeout):
            captured['url'] = url
            resp = MagicMock()
            resp.text = _rdb("USGS\t06191500\t2023-06-01 12:00\tMDT\t3530\tA")
            resp.raise_for_status = MagicMock()
            return resp

        with patch('symfluence.data.observation.handlers.usgs.requests.get', side_effect=fake_get):
            handler._download_data('06191500', tmp_path / 'out.rdb')

        assert 'startDT=2023-06-01' in captured['url']
        assert 'endDT=2023-06-15' in captured['url']
