# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression tests for the final-evaluation held-out window recompute.

These guard the bug where FUSE/HEC-HMS calibration runs wrote a blank or
missing ``<run>_<algo>_final_evaluation.json`` because the generic calibration
target did not yield a usable held-out ``Eval_*`` slice. The fix adds
``BaseModelOptimizer._recompute_final_eval_windows`` which reads the produced
``final_evaluation/`` output and slices it independently to CALIBRATION_PERIOD
and EVALUATION_PERIOD, so the evaluation score is a genuine out-of-sample slice
(distinct from calibration), never blank and never a copy.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.calibration.optimizers.base_model_optimizer import (
    BaseModelOptimizer,
)
from symfluence.evaluation.evaluators.streamflow import StreamflowEvaluator

CALIB_PERIOD = "2010-01-01, 2013-12-31"
EVAL_PERIOD = "2014-01-01, 2015-12-31"


class _Recomputer:
    """Minimal stand-in exercising only the recompute helpers under test.

    Binds the real ``BaseModelOptimizer`` methods so the test drives the exact
    production code path without constructing a full optimizer (which needs
    param managers, parallel dirs, an installed model, etc.).
    """

    _recompute_final_eval_windows = BaseModelOptimizer._recompute_final_eval_windows
    _final_eval_daily_series = staticmethod(
        BaseModelOptimizer._final_eval_daily_series
    )

    def __init__(self, target, config):
        self.calibration_target = target
        self.logger = logging.getLogger("test-final-eval")
        self._config = config

    @property
    def config(self):
        return self._config

    def _get_config_value(self, accessor, default=None, dict_key=None):
        if dict_key and isinstance(self._config, dict) and dict_key in self._config:
            return self._config[dict_key]
        return default


def _write_hechms_style_output(final_dir, obs_series, eval_scale):
    """Write a HEC-HMS/generic style ``*_output.nc`` (streamflow in m3/s).

    Simulated flow equals the observed flow over the calibration window (so
    KGE there is ~1) but is scaled during the evaluation window, so a genuine
    held-out slice must produce a *different* KGE.
    """
    sim = obs_series.copy()
    eval_mask = sim.index >= pd.Timestamp(EVAL_PERIOD.split(",")[0].strip())
    sim.loc[eval_mask] = sim.loc[eval_mask] * eval_scale

    ds = xr.Dataset(
        {"streamflow": (["time"], sim.values.astype(np.float32))},
        coords={"time": sim.index},
    )
    ds["streamflow"].attrs["units"] = "m3/s"
    final_dir.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(final_dir / "testdom_hechms_output.nc")


@pytest.fixture
def obs_series():
    idx = pd.date_range("2010-01-01", "2015-12-31", freq="D")
    # Positive, non-constant seasonal signal.
    base = 20.0 + 15.0 * np.sin(np.arange(len(idx)) * 2 * np.pi / 365.25)
    return pd.Series(base, index=idx, name="discharge_cms")


@pytest.fixture
def target(tmp_path, obs_series):
    """Real StreamflowEvaluator backed by a plain-dict config + on-disk obs."""
    obs_csv = tmp_path / "obs.csv"
    obs_series.to_frame("discharge_cms").to_csv(obs_csv, index_label="datetime")

    config = {
        "CALIBRATION_PERIOD": CALIB_PERIOD,
        "EVALUATION_PERIOD": EVAL_PERIOD,
        "CALIBRATION_TIMESTEP": "daily",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "ROUTING_DELINEATION": "lumped",
        "OBSERVATIONS_PATH": str(obs_csv),
        "HYDROLOGICAL_MODEL": "HECHMS",
        "REQUIRE_EXPLICIT_CATCHMENT_AREA": False,
    }
    evaluator = StreamflowEvaluator(config, tmp_path, logging.getLogger("test-eval"))
    evaluator.domain_name = "testdom"
    return evaluator


def test_recompute_yields_distinct_calib_and_eval(tmp_path, target, obs_series):
    """The recompute must populate BOTH windows with a genuine held-out slice."""
    final_dir = tmp_path / "final_evaluation"
    # Build the simulated output from the clean fixture series (a real model's
    # NetCDF carries a proper time coordinate) so sim matches obs in the
    # calibration window.
    _write_hechms_style_output(final_dir, obs_series, eval_scale=2.0)

    recomputer = _Recomputer(target, target.config)
    result = recomputer._recompute_final_eval_windows(final_dir)

    assert result is not None, "recompute returned nothing"
    assert "Calib" in result and "Eval" in result
    assert "KGE" in result["Calib"] and "KGE" in result["Eval"]

    calib_kge = result["Calib"]["KGE"]
    eval_kge = result["Eval"]["KGE"]
    # Calibration window: sim == obs, so near-perfect.
    assert calib_kge == pytest.approx(1.0, abs=1e-6)
    # Evaluation window is a genuine held-out slice (sim scaled x2), so it must
    # differ from the calibration score — the core bug was Eval echoing Calib.
    assert eval_kge < 0.5
    assert abs(eval_kge - calib_kge) > 0.1


def test_recompute_returns_none_without_output(tmp_path, target):
    """No produced output -> None (never a penalty/echo, never a raise)."""
    empty_dir = tmp_path / "final_evaluation_empty"
    empty_dir.mkdir()
    recomputer = _Recomputer(target, target.config)
    assert recomputer._recompute_final_eval_windows(empty_dir) is None


def test_daily_series_normalises_subdaily_and_timezone():
    """Sub-daily, tz-aware input collapses to a tz-naive daily-mean series."""
    idx = pd.date_range("2010-01-01", periods=48, freq="h", tz="UTC")
    s = pd.Series(np.arange(48, dtype=float), index=idx)
    out = BaseModelOptimizer._final_eval_daily_series(s)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is None
    assert len(out) == 2  # two calendar days
