# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for the SMHI streamflow handler idempotency guard.

acquire() is invoked twice per process_observed_data() run; without the
skip-if-raw-exists guard the handler re-downloaded the full ~73 MB
corrected-archive on every call.
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from symfluence.data.observation.handlers.smhi import SMHIStreamflowHandler

pytestmark = [pytest.mark.unit, pytest.mark.data]


def _make_config(tmp_path, **extra):
    cfg = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'EXPERIMENT_TIME_START': '2015-01-01 00:00',
        'EXPERIMENT_TIME_END': '2015-12-31 00:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_TIME_STEP_SIZE': 3600,
        'BOUNDING_BOX_COORDS': '60/14/59/15',
        'STATION_ID': '2357',
        'DATA_ACCESS': 'cloud',
    }
    cfg.update(extra)
    return cfg


def _raw_file(handler):
    return handler.project_observations_dir / 'streamflow' / 'raw_data' / 'smhi_2357_raw.csv'


def test_existing_raw_file_skips_download(tmp_path):
    handler = SMHIStreamflowHandler(_make_config(tmp_path), logging.getLogger('test_smhi'))
    raw = _raw_file(handler)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text('date,discharge_m3s,quality_code\n2015-01-01,1.0,G\n')

    with patch.object(handler, '_download_from_smhi') as mock_dl:
        result = handler.acquire()

    mock_dl.assert_not_called()
    assert result == raw


def test_missing_raw_file_downloads(tmp_path):
    handler = SMHIStreamflowHandler(_make_config(tmp_path), logging.getLogger('test_smhi'))
    raw = _raw_file(handler)

    with patch.object(handler, '_download_from_smhi', return_value=raw) as mock_dl:
        result = handler.acquire()

    mock_dl.assert_called_once()
    assert result == raw


def test_force_download_overrides_guard(tmp_path):
    handler = SMHIStreamflowHandler(
        _make_config(tmp_path, FORCE_DOWNLOAD=True), logging.getLogger('test_smhi'))
    raw = _raw_file(handler)
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text('date,discharge_m3s,quality_code\n2015-01-01,1.0,G\n')

    with patch.object(handler, '_download_from_smhi', return_value=raw) as mock_dl:
        handler.acquire()

    mock_dl.assert_called_once()
