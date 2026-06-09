# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""DataManager triggers the LaMAH-Ice download at observation acquisition.

Earlier wiring put the auto-download inside the FUSE multi-gauge
calibration worker. That meant any config that hit the FUSE-internal
SCE path (FUSE_RUN_INTERNAL_CALIBRATION=True) — which 09_large_domain
uses — never triggered the download. Move the trigger to
``DataManager.process_observed_data`` so the dataset lands at
observation acquisition time regardless of which optimiser the user
picks.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _make_dm(obs_dir_value):
    """Build a DataManager bypassing __init__ with just the helper's deps."""
    from types import SimpleNamespace

    from symfluence.data.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    # Plain namespaces so the typed-config getter raises AttributeError
    # cleanly and the dict_key fallback runs.
    dm.config = SimpleNamespace()
    dm.logger = logging.getLogger("test_multi_gauge_dl")

    def _get(getter, default=None, dict_key=None):
        try:
            v = getter()
            if v is not None:
                return v
        except (AttributeError, TypeError):
            pass
        if dict_key == 'MULTI_GAUGE_OBS_DIR':
            return obs_dir_value
        return default

    dm._get_config_value = _get
    return dm


def test_no_multi_gauge_obs_dir_is_a_noop(tmp_path):
    dm = _make_dm(None)
    with patch(
        'symfluence.data.observation.handlers.lamah_ice.ensure_lamah_ice_streamflow',
    ) as m:
        dm._ensure_multi_gauge_dataset_present()
    m.assert_not_called()


def test_present_obs_dir_is_a_noop(tmp_path):
    """If the dir already exists, don't redownload."""
    obs_dir = tmp_path / "lamah_ice" / "D_gauges" / "2_timeseries" / "daily"
    obs_dir.mkdir(parents=True)
    dm = _make_dm(str(obs_dir))
    with patch(
        'symfluence.data.observation.handlers.lamah_ice.ensure_lamah_ice_streamflow',
    ) as m:
        dm._ensure_multi_gauge_dataset_present()
    m.assert_not_called()


def test_non_lamah_obs_dir_is_a_noop(tmp_path):
    """An obs_dir without 'D_gauges' in the path is some other
    multi-gauge source (CAMELS, etc.) — we're not the right helper
    for that."""
    obs_dir = tmp_path / "camels" / "streamflow"
    dm = _make_dm(str(obs_dir))
    with patch(
        'symfluence.data.observation.handlers.lamah_ice.ensure_lamah_ice_streamflow',
    ) as m:
        dm._ensure_multi_gauge_dataset_present()
    m.assert_not_called()


def test_missing_lamah_obs_dir_triggers_download(tmp_path):
    """The 09_large_domain case: obs_dir points at a missing
    LaMAH-Ice subtree — auto-download fires at obs-acquisition time
    so the dataset is on disk before any optimiser starts."""
    obs_dir = tmp_path / "lamah_ice" / "D_gauges" / "2_timeseries" / "daily"
    dm = _make_dm(str(obs_dir))
    with patch(
        'symfluence.data.observation.handlers.lamah_ice.ensure_lamah_ice_streamflow',
    ) as m:
        dm._ensure_multi_gauge_dataset_present()
    m.assert_called_once()
    # Helper must be called with the dataset root (lamah_ice/), not
    # the inner 2_timeseries/daily/ subdir.
    args, _ = m.call_args
    called_with = args[0]
    assert called_with == tmp_path / "lamah_ice", \
        f"helper should be called with dataset root, got {called_with}"


def test_download_failure_does_not_break_obs_step(tmp_path, caplog):
    """If the auto-download itself fails, we log a warning and let
    the existing observation acquisition surface the missing-data
    error — don't add a new failure mode."""
    obs_dir = tmp_path / "lamah_ice" / "D_gauges" / "2_timeseries" / "daily"
    dm = _make_dm(str(obs_dir))
    with patch(
        'symfluence.data.observation.handlers.lamah_ice.ensure_lamah_ice_streamflow',
        side_effect=RuntimeError("zenodo down"),
    ):
        caplog.set_level(logging.WARNING)
        dm._ensure_multi_gauge_dataset_present()  # must not raise
    msg = "\n".join(r.getMessage() for r in caplog.records)
    assert "auto-download skipped" in msg
    assert "zenodo down" in msg
