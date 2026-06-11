# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for the EM-Earth S3 acquisition handler.

Pins the four fixes from the 2026-06 parity-validation sweep:

1. PermissionError (an OSError subclass) was swallowed by the generic
   ``except (OSError, ...)``, surfacing 48 silent 403s as a misleading
   "No EM-Earth data downloaded".
2. ``anon=True`` was hardcoded — now configurable via EM_EARTH_S3_ANON.
3. The handler read EM_EARTH_REGION_FOLDER and could insert a region path
   component; the live bucket has NO region subfolders, so the component
   is dropped entirely.
4. All 12 months of every year were probed regardless of the requested
   window, and post-2019 windows silently produced nothing (the record
   ends 2019-12).
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.exceptions import DataAcquisitionError
from symfluence.data.acquisition.handlers.em_earth import EMEarthAcquirer

pytestmark = [pytest.mark.unit, pytest.mark.data]


def _make_config(tmp_path, start='2018-06-05 00:00', end='2018-06-20 00:00'):
    return {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'EXPERIMENT_TIME_START': start,
        'EXPERIMENT_TIME_END': end,
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'FORCING_DATASET': 'EM-EARTH',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_TIME_STEP_SIZE': 86400,
        'BOUNDING_BOX_COORDS': '52/-116/50/-114',
    }


def _make_handler(tmp_path, **kwargs):
    cfg = _make_config(tmp_path, **{k: v for k, v in kwargs.items() if k in ('start', 'end')})
    cfg.update({k: v for k, v in kwargs.items() if k not in ('start', 'end')})
    return EMEarthAcquirer(cfg, logging.getLogger('test_em_earth'))


class TestPermissionError:
    def test_permission_error_raises_actionable_message(self, tmp_path):
        """A 403 from the bucket must surface as DataAcquisitionError, not be
        swallowed into 'No EM-Earth data downloaded'."""
        handler = _make_handler(tmp_path)
        mock_fs = MagicMock()
        mock_fs.exists.side_effect = PermissionError('Access Denied')
        with patch(
            'symfluence.data.acquisition.handlers.em_earth.s3fs.S3FileSystem',
            return_value=mock_fs,
        ):
            with pytest.raises(DataAcquisitionError, match='frdr-dfdr.ca'):
                handler.download(tmp_path / 'out')


class TestAnonConfigurable:
    def test_anon_defaults_to_true(self, tmp_path):
        handler = _make_handler(tmp_path)
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        with patch(
            'symfluence.data.acquisition.handlers.em_earth.s3fs.S3FileSystem',
            return_value=mock_fs,
        ) as mock_s3:
            with pytest.raises(DataAcquisitionError):
                handler.download(tmp_path / 'out')
        mock_s3.assert_called_once_with(anon=True)

    def test_anon_false_uses_credential_chain(self, tmp_path):
        handler = _make_handler(tmp_path, EM_EARTH_S3_ANON=False)
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        with patch(
            'symfluence.data.acquisition.handlers.em_earth.s3fs.S3FileSystem',
            return_value=mock_fs,
        ) as mock_s3:
            with pytest.raises(DataAcquisitionError):
                handler.download(tmp_path / 'out')
        mock_s3.assert_called_once_with(anon=False)


class TestKeyConstruction:
    def test_keys_have_no_region_component(self, tmp_path):
        """Bucket keys are emearth/nc/<product>/<var>/<fname> directly —
        the live bucket has no region subfolders (verified via live LIST)."""
        handler = _make_handler(tmp_path)
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        with patch(
            'symfluence.data.acquisition.handlers.em_earth.s3fs.S3FileSystem',
            return_value=mock_fs,
        ):
            with pytest.raises(DataAcquisitionError):
                handler.download(tmp_path / 'out')

        probed = [call.args[0] for call in mock_fs.exists.call_args_list]
        assert (
            'emearth/nc/deterministic_raw_daily/prcp/'
            'EM_Earth_deterministic_daily_prcp_201806.nc'
        ) in probed
        for key in probed:
            parts = key.split('/')
            # emearth / nc / deterministic_raw_daily / <var> / <fname>
            assert len(parts) == 5, f"unexpected key shape (region folder?): {key}"


class TestWindowHandling:
    def test_only_requested_months_probed(self, tmp_path):
        """A 16-day window in one month must probe exactly one key per variable."""
        handler = _make_handler(tmp_path)
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        with patch(
            'symfluence.data.acquisition.handlers.em_earth.s3fs.S3FileSystem',
            return_value=mock_fs,
        ):
            with pytest.raises(DataAcquisitionError):
                handler.download(tmp_path / 'out')

        probed = [call.args[0] for call in mock_fs.exists.call_args_list]
        assert len(probed) == 4  # prcp, tmean, trange, tdew x 1 month
        assert all(key.endswith('_201806.nc') for key in probed)

    def test_window_spanning_months_probes_each_month(self, tmp_path):
        handler = _make_handler(
            tmp_path, start='2018-05-20 00:00', end='2018-07-05 00:00')
        mock_fs = MagicMock()
        mock_fs.exists.return_value = False
        with patch(
            'symfluence.data.acquisition.handlers.em_earth.s3fs.S3FileSystem',
            return_value=mock_fs,
        ):
            with pytest.raises(DataAcquisitionError):
                handler.download(tmp_path / 'out')
        probed = [call.args[0] for call in mock_fs.exists.call_args_list]
        assert len(probed) == 12  # 4 vars x 3 months (May, Jun, Jul)

    def test_window_after_record_end_fails_loudly(self, tmp_path):
        """The record ends 2019-12: a fully out-of-record window must raise a
        clear error, not silently produce nothing."""
        handler = _make_handler(
            tmp_path, start='2021-01-01 00:00', end='2021-12-31 00:00')
        mock_fs = MagicMock()
        with patch(
            'symfluence.data.acquisition.handlers.em_earth.s3fs.S3FileSystem',
            return_value=mock_fs,
        ):
            with pytest.raises(DataAcquisitionError, match='entirely outside'):
                handler.download(tmp_path / 'out')
        mock_fs.exists.assert_not_called()

    def test_window_before_record_start_fails_loudly(self, tmp_path):
        handler = _make_handler(
            tmp_path, start='1900-01-01 00:00', end='1910-12-31 00:00')
        mock_fs = MagicMock()
        with patch(
            'symfluence.data.acquisition.handlers.em_earth.s3fs.S3FileSystem',
            return_value=mock_fs,
        ):
            with pytest.raises(DataAcquisitionError, match='entirely outside'):
                handler.download(tmp_path / 'out')
        mock_fs.exists.assert_not_called()
