# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Unit tests for MSWEP remote path construction.

Pins the fix for the path bug found by the 2026-06 parity-validation sweep:
the handler inserted a spurious ``{year}/`` subfolder and mangled the
3-hourly filename. Per the official GloH2O V3.16 docs the layout is
``{VERSION}/{Past|Past_nogauge|NRT}/{Hourly|3hourly|Daily|Monthly}/`` with
filenames ``YYYYDOY.HH.nc`` (3-hourly) / ``YYYYDOY.nc`` (daily) and NO
per-year subfolders; worked example ``MSWEP_V315/Past/Hourly/2020116.18.nc``.
"""
from __future__ import annotations

import logging

import pytest

from symfluence.data.acquisition.handlers.mswep import MSWEPAcquirer

pytestmark = [pytest.mark.unit, pytest.mark.data]


def _make_handler(tmp_path, start, end, **extra):
    cfg = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'SYMFLUENCE_CODE_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'EXPERIMENT_TIME_START': start,
        'EXPERIMENT_TIME_END': end,
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'FORCING_DATASET': 'MSWEP',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_TIME_STEP_SIZE': 10800,
        'BOUNDING_BOX_COORDS': '52/-116/50/-114',
    }
    cfg.update(extra)
    return MSWEPAcquirer(cfg, logging.getLogger('test_mswep'))


class TestFileListConstruction:
    def test_3hourly_matches_documented_worked_example(self, tmp_path):
        """2020-04-25 is DOY 116; hour 18 must yield Past/3hourly/2020116.18.nc
        (the docs' worked example uses the Hourly folder but the same
        YYYYDOY.HH.nc filename convention)."""
        handler = _make_handler(tmp_path, '2020-04-25 00:00', '2020-04-25 23:00')
        files = handler._generate_file_list('3hourly', 'Past')

        paths = [f['relative_path'] for f in files]
        assert 'Past/3hourly/2020116.18.nc' in paths
        assert len(paths) == 8  # one day, 3-hourly
        for p in paths:
            assert '/2020/' not in p, f"spurious per-year subfolder in {p}"

    def test_daily_no_year_subfolder(self, tmp_path):
        handler = _make_handler(tmp_path, '2020-04-25 00:00', '2020-04-26 23:00')
        files = handler._generate_file_list('daily', 'Past')

        paths = [f['relative_path'] for f in files]
        assert paths == ['Past/Daily/2020116.nc', 'Past/Daily/2020117.nc']

    def test_monthly_layout(self, tmp_path):
        handler = _make_handler(tmp_path, '2020-04-01 00:00', '2020-05-15 00:00')
        files = handler._generate_file_list('monthly', 'Past')

        paths = [f['relative_path'] for f in files]
        assert paths == ['Past/Monthly/202004.nc', 'Past/Monthly/202005.nc']


class TestVersionFolder:
    def test_default_version_is_v316(self, tmp_path):
        handler = _make_handler(tmp_path, '2020-04-25 00:00', '2020-04-25 23:00')
        assert handler._get_gdrive_folder() == 'MSWEP_V316'

    def test_version_config_overridable(self, tmp_path):
        handler = _make_handler(
            tmp_path, '2020-04-25 00:00', '2020-04-25 23:00', MSWEP_VERSION='V315')
        assert handler._get_gdrive_folder() == 'MSWEP_V315'

    def test_unknown_version_falls_back_to_prefix(self, tmp_path):
        handler = _make_handler(
            tmp_path, '2020-04-25 00:00', '2020-04-25 23:00', MSWEP_VERSION='V999')
        assert handler._get_gdrive_folder() == 'MSWEP_V999'
