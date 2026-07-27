# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The mizuRoute control writer serves runoff metadata from the schema.

``control_writer`` used to carry its own ``ModelRunoffConfig`` class and a
four-entry ``MODEL_CONFIGS`` dict — a third copy of the same per-model table,
already served by ``runoff_loader.get_model_config`` from each model's
registered ``ModelConfigSchema``. The copy held identical values for
SUMMA/FUSE/GR/NGEN and simply omitted HYPE, so a HYPE source model was rejected
here while routing resolved it everywhere else.

These tests pin the collapse: one source of truth, and a model becomes writable
by registering a schema rather than by editing this module.

CHANGED: the disagreement about HYPE is now settled the other way. HYPE is not
a routable source — its runoff declaration named a NetCDF the adapter never
writes, and its ``cout`` is already routed discharge at subbasin outlets, so a
converter would double-route. The declaration is gone, so HYPE is rejected here
*and* everywhere else, which is the agreement that matters.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from symfluence.core.exceptions import ConfigValidationError
from symfluence.core.modeling.utilities import runoff_loader
from symfluence.models.mizuroute import control_writer

pytestmark = [pytest.mark.unit]

EXPERIMENT_ID = 'schema_source_001'
DOMAIN_NAME = 'test_domain'


def _make_writer(tmp_path: Path) -> control_writer.ControlFileWriter:
    setup_dir = tmp_path / 'settings' / 'mizuRoute'
    setup_dir.mkdir(parents=True, exist_ok=True)
    config = dict(
        DOMAIN_DEFINITION_METHOD='lumped',
        DOMAIN_NAME=DOMAIN_NAME,
        EXPERIMENT_ID=EXPERIMENT_ID,
        EXPERIMENT_TIME_START='2010-01-01 00:00',
        EXPERIMENT_TIME_END='2010-01-31 23:00',
        FORCING_DATASET='ERA5',
        HYDROLOGICAL_MODEL='HYPE',
        SYMFLUENCE_CODE_DIR='/tmp/code',
        SYMFLUENCE_DATA_DIR='/tmp/data',
    )
    return control_writer.ControlFileWriter(
        config=config,
        setup_dir=setup_dir,
        project_dir=tmp_path,
        experiment_id=EXPERIMENT_ID,
        domain_name=DOMAIN_NAME,
    )


def _read_field(control_path: Path, tag: str) -> str:
    for line in control_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(tag):
            return stripped[len(tag):].split('!')[0].strip()
    raise AssertionError(f"{tag} not found in control file")


def test_module_no_longer_owns_a_runoff_table():
    """The names stay importable, but they are the runoff_loader's objects."""
    assert control_writer.MODEL_CONFIGS is runoff_loader.MODEL_CONFIGS
    assert control_writer.ModelRunoffConfig is runoff_loader.ModelRunoffConfig


@pytest.mark.parametrize("model_type", ["summa", "fuse", "gr", "ngen"])
def test_every_registered_runoff_model_is_writable(tmp_path, model_type):
    """Every model with a runoff declaration, and only those."""
    writer = _make_writer(tmp_path)

    control_path = writer.write_control_file(
        model_type=model_type,
        control_file_name=f'mizuroute_{model_type}.control',
    )

    expected = runoff_loader.get_model_config(model_type)
    assert _read_field(control_path, '<vname_qsim>') == expected.default_var
    assert _read_field(control_path, '<units_qsim>') == expected.default_units
    assert _read_field(control_path, '<dt_qsim>') == expected.default_dt
    assert _read_field(control_path, '<dname_hruid>') == expected.hru_dim
    assert _read_field(control_path, '<vname_hruid>') == expected.hru_var
    # FUSE truncates/hashes FMODEL_ID to 6 chars; every other model uses the
    # experiment id verbatim.
    file_id = EXPERIMENT_ID
    if model_type == 'fuse':
        import hashlib
        file_id = hashlib.md5(
            EXPERIMENT_ID.encode(), usedforsecurity=False
        ).hexdigest()[:6]
    assert _read_field(control_path, '<fname_qsim>') == (
        expected.output_file_pattern.format(
            experiment_id=file_id, domain_name=DOMAIN_NAME
        )
    )


def test_unregistered_source_model_is_still_a_config_error(tmp_path):
    """The unknown-model failure mode is preserved, not widened."""
    writer = _make_writer(tmp_path)

    with pytest.raises(ConfigValidationError, match="not_a_model"):
        writer.write_control_file(model_type='not_a_model')


def test_hype_is_rejected_as_a_source(tmp_path):
    """A model with no runoff declaration cannot have a control file written.

    The error lists the models that *can* be routed, so the message names the
    alternatives rather than leaving 'why not HYPE?' to the reader.
    """
    writer = _make_writer(tmp_path)

    with pytest.raises(ConfigValidationError) as excinfo:
        writer.write_control_file(model_type='hype')

    message = str(excinfo.value)
    assert 'hype' in message
    assert 'summa' in message and 'fuse' in message
