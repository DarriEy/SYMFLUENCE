# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for the WSC streamflow handler GeoMet acquisition path.

Pins the fix for the silent-data-corruption bug found by the 2026-06
parity-validation sweep: ``_download_from_geomet`` paged the GeoMet OGC API
with ``offset`` but no ``sortby``, and the backend ordering is unstable —
observed live as 9,243 duplicated rows / 9,243 silently missing rows on a
42k-record station. The fix adds ``sortby=DATE``, a server-side ``datetime``
window filter, and a hard integrity guard (numberMatched + duplicate DATEs).
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.data.observation.handlers.wsc import WSCStreamflowHandler

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
        'BOUNDING_BOX_COORDS': '52/-116/50/-114',
        'STATION_ID': '05BB001',
    }


@pytest.fixture
def handler(mock_config):
    return WSCStreamflowHandler(mock_config, logging.getLogger('test_wsc'))


def _geomet_response(dates, number_matched=None):
    """Build a mock GeoMet items response carrying the given DATE values."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        'features': [
            {'properties': {'DATE': d, 'DISCHARGE': 1.0, 'STATION_NUMBER': '05BB001'}}
            for d in dates
        ],
        'numberMatched': number_matched if number_matched is not None else len(dates),
        'numberReturned': len(dates),
    }
    return resp


class TestGeoMetRequestParams:
    def test_sortby_and_datetime_window_sent(self, handler, tmp_path):
        """Every page request must carry sortby=DATE and the experiment window."""
        dates = ['2023-06-01', '2023-06-02', '2023-06-03']
        with patch('symfluence.data.observation.handlers.wsc.requests.get') as mock_get:
            mock_get.return_value = _geomet_response(dates)
            handler._download_from_geomet('05BB001', tmp_path / 'out.csv')

        assert mock_get.call_count == 1
        params = mock_get.call_args.kwargs.get('params') or mock_get.call_args.args[1]
        assert params['sortby'] == 'DATE'
        # Window is padded by one day at the start (boundary/timezone safety)
        assert params['datetime'] == '2023-05-31T00:00:00Z/2023-06-15T00:00:00Z'
        assert params['STATION_NUMBER'] == '05BB001'

    def test_no_datetime_filter_when_dates_unconfigured(self, handler, tmp_path):
        """Full-record behaviour is preserved when no window is configured."""
        handler.start_date = None
        handler.end_date = None
        with patch('symfluence.data.observation.handlers.wsc.requests.get') as mock_get:
            mock_get.return_value = _geomet_response(['2023-06-01'])
            handler._download_from_geomet('05BB001', tmp_path / 'out.csv')

        params = mock_get.call_args.kwargs.get('params') or mock_get.call_args.args[1]
        assert 'datetime' not in params
        assert params['sortby'] == 'DATE'


class TestIntegrityGuard:
    def test_number_matched_mismatch_raises(self, handler, tmp_path):
        """Row count != numberMatched must hard-fail, not warn-and-continue."""
        dates = ['2023-06-01', '2023-06-02']
        with patch('symfluence.data.observation.handlers.wsc.requests.get') as mock_get:
            mock_get.return_value = _geomet_response(dates, number_matched=5)
            with pytest.raises(DataAcquisitionError, match='numberMatched'):
                handler._download_from_geomet('05BB001', tmp_path / 'out.csv')

    def test_duplicate_dates_raise(self, handler, tmp_path):
        """Duplicate DATE values indicate unstable paging and must hard-fail."""
        dates = ['2023-06-01', '2023-06-02', '2023-06-02']
        with patch('symfluence.data.observation.handlers.wsc.requests.get') as mock_get:
            mock_get.return_value = _geomet_response(dates)
            with pytest.raises(DataAcquisitionError, match='duplicate DATE'):
                handler._download_from_geomet('05BB001', tmp_path / 'out.csv')

    def test_clean_response_writes_csv(self, handler, tmp_path):
        dates = ['2023-06-01', '2023-06-02', '2023-06-03']
        out = tmp_path / 'out.csv'
        with patch('symfluence.data.observation.handlers.wsc.requests.get') as mock_get:
            mock_get.return_value = _geomet_response(dates)
            result = handler._download_from_geomet('05BB001', out)

        assert result == out
        assert out.exists()
        import pandas as pd
        df = pd.read_csv(out)
        assert len(df) == 3
        assert df['DATE'].is_unique
