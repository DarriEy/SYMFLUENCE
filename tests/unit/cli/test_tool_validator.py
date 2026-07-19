"""Unit tests for ToolValidator's required-tool handling."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def optional_tool_defs():
    """One optional tool, shaped like the optional entries in the real config.

    `crhm`, `gsflow`, `mhm`, `prms`, `rhessys` and `swat` are all marked
    optional but are part of the Paper 3 set, so this is the combination that
    matters.
    """
    return {
        'crhm': {
            'name': 'CRHM',
            'description': 'Cold Regions Hydrological Model',
            'install_dir': 'crhm',
            'default_path_suffix': 'installs/crhm/bin',
            'default_exe': 'crhm',
            'optional': True,
            'verify_install': {
                'check_type': 'exists_any',
                'file_paths': ['crhm'],
            },
        },
    }


def _validator(tool_defs, data_dir):
    """Build a ToolValidator whose data dir points at *data_dir*."""
    from symfluence.cli.services.tool_validator import ToolValidator

    validator = ToolValidator(external_tools=tool_defs)
    validator._load_config = MagicMock(return_value={})
    validator._get_data_dir = MagicMock(return_value=data_dir)
    return validator


def test_missing_optional_tool_is_skipped_by_default(optional_tool_defs, tmp_path):
    """Without a required set, a missing optional tool is skipped and passes.

    This is the pre-existing behavior and stays intact: a user who never asked
    for CRHM should not have validation fail because it is absent.
    """
    validator = _validator(optional_tool_defs, tmp_path)

    result = validator.validate()

    assert result is True


def test_missing_optional_tool_fails_when_required(optional_tool_defs, tmp_path):
    """Declaring an optional tool required makes its absence a failure.

    Regression test: `binary validate` used to skip optional tools that were
    not installed, so a --paper-repro bundle missing 6 of its 13 binaries
    validated clean and the bootstrap reported success over a broken install.
    """
    validator = _validator(optional_tool_defs, tmp_path)

    result = validator.validate(required_tools=['crhm'])

    assert result is not True, "a missing required tool must not validate clean"
    assert 'crhm' not in result['skipped_tools']
    assert 'crhm' in result['missing_tools']


def test_present_required_tool_still_validates(optional_tool_defs, tmp_path):
    """A required tool that is actually installed validates normally."""
    bindir = tmp_path / 'installs' / 'crhm' / 'bin'
    bindir.mkdir(parents=True)
    (bindir / 'crhm').write_text('#!/bin/sh\n')

    validator = _validator(optional_tool_defs, tmp_path)

    result = validator.validate(required_tools=['crhm'])

    assert result is True


def test_required_set_scopes_the_verdict(tmp_path):
    """Missing tools outside the required set are reported but do not fail.

    --paper-repro builds 13 of the 26 tools by design, so the ones it
    deliberately skips must not make a correct bundle look broken.
    """
    defs = {
        'summa': {
            'name': 'SUMMA',
            'description': '',
            'install_dir': 'summa',
            'default_path_suffix': 'installs/summa/bin',
            'default_exe': 'summa.exe',
            'verify_install': {'check_type': 'exists_any', 'file_paths': ['summa.exe']},
        },
        'ngen': {  # not required, not installed
            'name': 'NGEN',
            'description': '',
            'install_dir': 'ngen',
            'default_path_suffix': 'installs/ngen/cmake_build',
            'default_exe': 'ngen',
            'verify_install': {'check_type': 'exists_any', 'file_paths': ['ngen']},
        },
    }
    bindir = tmp_path / 'installs' / 'summa' / 'bin'
    bindir.mkdir(parents=True)
    (bindir / 'summa.exe').write_text('#!/bin/sh\n')

    validator = _validator(defs, tmp_path)

    # ngen is missing, but only summa was required.
    assert validator.validate(required_tools=['summa']) is True

    # Without a required set, the same missing ngen is a failure.
    result = validator.validate()
    assert result is not True
    assert 'ngen' in result['missing_tools']


def test_unknown_required_tool_is_reported(optional_tool_defs, tmp_path):
    """A required name that is not a known tool fails instead of passing silently."""
    validator = _validator(optional_tool_defs, tmp_path)

    result = validator.validate(required_tools=['not_a_real_tool'])

    assert result is not True
    assert 'not_a_real_tool' in result['missing_tools']
