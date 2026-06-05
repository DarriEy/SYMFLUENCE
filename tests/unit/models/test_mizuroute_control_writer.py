# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tests for the mizuRoute ControlFileWriter runoff-variable auto-detection.

The writer should self-heal a stale/incorrect SETTINGS_MIZU_ROUTING_VAR by
inspecting the actual model output NetCDF (the SUMMA-side analogue of the FUSE
runner fix, PR #69), while still respecting a correct config and falling back
gracefully when the output file does not exist yet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from symfluence.models.mizuroute.control_writer import ControlFileWriter

EXPERIMENT_ID = 'full_spatial_001'
DOMAIN_NAME = 'test_domain'


def _write_summa_output(project_dir: Path, var_name: str) -> Path:
    """Write a minimal SUMMA timestep file containing ``var_name``."""
    out_dir = project_dir / f"simulations/{EXPERIMENT_ID}" / 'SUMMA'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_ID}_timestep.nc"
    ds = xr.Dataset(
        {var_name: xr.DataArray(np.random.rand(3, 2), dims=('time', 'gru'))},
        coords={'time': np.arange(3), 'gru': np.arange(2)},
    )
    ds.to_netcdf(out_path)
    return out_path


def _make_config(overrides: dict) -> dict:
    """Build a minimal but valid SYMFLUENCE config dict with overrides."""
    base = dict(
        DOMAIN_DEFINITION_METHOD='lumped',
        DOMAIN_NAME=DOMAIN_NAME,
        EXPERIMENT_ID=EXPERIMENT_ID,
        EXPERIMENT_TIME_START='2010-01-01 00:00',
        EXPERIMENT_TIME_END='2010-01-31 23:00',
        FORCING_DATASET='ERA5',
        HYDROLOGICAL_MODEL='SUMMA',
        SUB_GRID_DISCRETIZATION='grus',
        SYMFLUENCE_CODE_DIR='/tmp/code',
        SYMFLUENCE_DATA_DIR='/tmp/data',
    )
    base.update(overrides)
    return base


def _make_writer(tmp_path: Path, overrides: dict) -> ControlFileWriter:
    setup_dir = tmp_path / 'settings' / 'mizuRoute'
    setup_dir.mkdir(parents=True, exist_ok=True)
    project_dir = tmp_path
    return ControlFileWriter(
        config=_make_config(overrides),
        setup_dir=setup_dir,
        project_dir=project_dir,
        experiment_id=EXPERIMENT_ID,
        domain_name=DOMAIN_NAME,
    )


def _read_field(control_path: Path, tag: str) -> str:
    """Return the value following a ``<tag>`` line in the control file."""
    for line in control_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(tag):
            return stripped[len(tag):].split('!')[0].strip()
    raise AssertionError(f"{tag} not found in control file")


def test_overrides_stale_routing_var_from_output(tmp_path):
    """A FUSE-style 'q_routed' in a SUMMA config is corrected to the real var."""
    _write_summa_output(tmp_path, 'averageRoutedRunoff')
    config = {
        'SETTINGS_MIZU_ROUTING_VAR': 'q_routed',
        'SETTINGS_MIZU_ROUTING_UNITS': 'mm/d',
    }
    writer = _make_writer(tmp_path, config)

    control_path = writer.write_control_file(model_type='summa')

    assert _read_field(control_path, '<vname_qsim>') == 'averageRoutedRunoff'
    # Units were untrustworthy alongside the wrong var → reset to SUMMA default.
    assert _read_field(control_path, '<units_qsim>') == 'm/s'


def test_respects_correct_config_var(tmp_path):
    """When the configured variable is present, it is kept (no override)."""
    _write_summa_output(tmp_path, 'averageRoutedRunoff')
    config = {
        'SETTINGS_MIZU_ROUTING_VAR': 'averageRoutedRunoff',
        'SETTINGS_MIZU_ROUTING_UNITS': 'm/s',
    }
    writer = _make_writer(tmp_path, config)

    control_path = writer.write_control_file(model_type='summa')

    assert _read_field(control_path, '<vname_qsim>') == 'averageRoutedRunoff'
    assert _read_field(control_path, '<units_qsim>') == 'm/s'


def test_falls_back_to_config_when_output_missing(tmp_path):
    """With no model output yet (preprocessing time), keep the configured value."""
    # Deliberately do not write any SUMMA output file.
    config = {
        'SETTINGS_MIZU_ROUTING_VAR': 'q_routed',
        'SETTINGS_MIZU_ROUTING_UNITS': 'mm/d',
    }
    writer = _make_writer(tmp_path, config)

    control_path = writer.write_control_file(model_type='summa')

    # No file to detect from → configured value is preserved unchanged.
    assert _read_field(control_path, '<vname_qsim>') == 'q_routed'
    assert _read_field(control_path, '<units_qsim>') == 'mm/d'


def test_detects_alternate_summa_runoff_var(tmp_path):
    """A different valid SUMMA runoff var is detected when config var is absent."""
    _write_summa_output(tmp_path, 'scalarTotalRunoff')
    config = {'SETTINGS_MIZU_ROUTING_VAR': 'q_routed'}
    writer = _make_writer(tmp_path, config)

    control_path = writer.write_control_file(model_type='summa')

    assert _read_field(control_path, '<vname_qsim>') == 'scalarTotalRunoff'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
