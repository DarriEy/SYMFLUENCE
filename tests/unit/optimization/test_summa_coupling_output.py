# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for SUMMA groundwater-coupling output-control guard.

When SUMMA is coupled to an external groundwater model (GROUNDWATER_MODEL, e.g.
MODFLOW), the coupler reads a recharge flux (scalarSoilDrainage by default) from
SUMMA's output. A calibration-target rewrite can leave the per-run outputControl
with only the streamflow variable, so _ensure_coupling_output_vars re-adds the
coupling variables right before each SUMMA run. See
optimization/workers/summa/model_execution.py.
"""
from __future__ import annotations

import logging

from symfluence.models.summa.calibration.worker_impl.model_execution import (
    _ensure_coupling_output_vars,
)

LOG = logging.getLogger("test")


def _write(p, text):
    p.write_text(text, encoding="utf-8")
    return p


def test_adds_recharge_var_when_coupled(tmp_path):
    oc = _write(tmp_path / "outputControl.txt", "averageRoutedRunoff | 1\n")
    _ensure_coupling_output_vars(oc, {"GROUNDWATER_MODEL": "MODFLOW"}, LOG)
    content = oc.read_text()
    assert "scalarSoilDrainage | 1" in content
    assert "scalarSurfaceRunoff | 1" in content
    assert "averageRoutedRunoff | 1" in content  # preserved


def test_respects_custom_recharge_variable(tmp_path):
    oc = _write(tmp_path / "outputControl.txt", "averageRoutedRunoff | 1\n")
    _ensure_coupling_output_vars(
        oc, {"GROUNDWATER_MODEL": "MODFLOW",
             "MODFLOW_RECHARGE_VARIABLE": "scalarAquiferRecharge"}, LOG)
    assert "scalarAquiferRecharge | 1" in oc.read_text()


def test_noop_without_groundwater_model(tmp_path):
    oc = _write(tmp_path / "outputControl.txt", "averageRoutedRunoff | 1\n")
    _ensure_coupling_output_vars(oc, {"GROUNDWATER_MODEL": "none"}, LOG)
    assert "scalarSoilDrainage" not in oc.read_text()
    _ensure_coupling_output_vars(oc, {}, LOG)            # key absent
    assert "scalarSoilDrainage" not in oc.read_text()


def test_idempotent_when_already_present(tmp_path):
    oc = _write(
        tmp_path / "outputControl.txt",
        "averageRoutedRunoff | 1\nscalarSoilDrainage | 1\nscalarSurfaceRunoff | 1\n",
    )
    before = oc.read_text()
    _ensure_coupling_output_vars(oc, {"GROUNDWATER_MODEL": "MODFLOW"}, LOG)
    assert oc.read_text() == before  # no duplicate lines


def test_handles_missing_trailing_newline(tmp_path):
    oc = _write(tmp_path / "outputControl.txt", "averageRoutedRunoff | 1")  # no \n
    _ensure_coupling_output_vars(oc, {"GROUNDWATER_MODEL": "MODFLOW"}, LOG)
    lines = [ln for ln in oc.read_text().splitlines() if ln.strip()]
    assert "averageRoutedRunoff | 1" in lines
    assert "scalarSoilDrainage | 1" in lines
