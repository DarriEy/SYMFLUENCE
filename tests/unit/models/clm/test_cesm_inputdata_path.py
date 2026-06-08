# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for CLM CESM-inputdata path resolution.

The CESM inputdata root must default to a writable, SYMFLUENCE-managed location
(under the data dir, alongside model installs) rather than the home directory,
which is often not writable from HPC compute nodes — while still honouring an
explicit CLM_CESM_INPUTDATA_PATH override.
"""

from __future__ import annotations

from pathlib import Path

from symfluence.models.clm.nuopc_generator import (
    DEFAULT_CESM_INPUTDATA,
    _resolve_cesm_inputdata,
)


class _StubPreprocessor:
    """Minimal stand-in exposing the two hooks _resolve_cesm_inputdata uses."""

    def __init__(self, configured="default", code_dir=None, project_dir=None):
        self._configured = configured
        self._code_dir = code_dir
        self.project_dir = project_dir

    def _get_config_value(self, _lambda, default=None, dict_key=None):
        return {
            "CLM_CESM_INPUTDATA_PATH": self._configured,
            "SYMFLUENCE_CODE_DIR": self._code_dir,
        }.get(dict_key, default)


def test_explicit_override_is_used(tmp_path):
    override = tmp_path / "cesm-inputdata"
    pp = _StubPreprocessor(configured=str(override))
    assert _resolve_cesm_inputdata(pp) == override


def test_default_resolves_next_to_installs_via_code_dir(tmp_path):
    code_dir = tmp_path / "SYMFLUENCE"
    pp = _StubPreprocessor(configured="default", code_dir=str(code_dir))
    assert _resolve_cesm_inputdata(pp) == (
        tmp_path / "SYMFLUENCE_data" / "installs" / "cesm-inputdata"
    )


def test_default_resolves_under_data_dir_via_project_dir(tmp_path):
    pp = _StubPreprocessor(
        configured="default",
        project_dir=tmp_path / "SYMFLUENCE_data" / "domain_X",
    )
    # Mirrors CLMPreProcessor._get_install_path(): project_dir.parents[1]/installs
    assert _resolve_cesm_inputdata(pp) == tmp_path / "installs" / "cesm-inputdata"


def test_default_falls_back_to_home_when_no_project():
    pp = _StubPreprocessor(configured="default")
    assert _resolve_cesm_inputdata(pp) == DEFAULT_CESM_INPUTDATA
