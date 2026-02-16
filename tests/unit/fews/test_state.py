"""Tests for FEWS state file exchange."""

import pytest

from symfluence.fews.exceptions import StateExchangeError
from symfluence.fews.state import export_states, import_states


class TestImportStates:
    def test_import_copies_files(self, sample_state_files, tmp_path):
        model_dir = tmp_path / "model_states"
        copied = import_states(sample_state_files, model_dir)
        assert len(copied) == 2
        assert (model_dir / "state_snow.nc").exists()
        assert (model_dir / "state_soil.nc").exists()

    def test_import_none_dir(self, tmp_path):
        copied = import_states(None, tmp_path / "model_states")
        assert copied == []

    def test_import_missing_dir(self, tmp_path):
        copied = import_states(tmp_path / "nonexistent", tmp_path / "model_states")
        assert copied == []

    def test_import_creates_model_dir(self, sample_state_files, tmp_path):
        model_dir = tmp_path / "deep" / "nested" / "states"
        copied = import_states(sample_state_files, model_dir)
        assert model_dir.is_dir()
        assert len(copied) == 2


class TestExportStates:
    def test_export_copies_files(self, sample_state_files, tmp_path):
        fews_out = tmp_path / "fews_states_out"
        copied = export_states(sample_state_files, fews_out)
        assert len(copied) == 2
        assert (fews_out / "state_snow.nc").exists()

    def test_export_none_dir(self, sample_state_files):
        copied = export_states(sample_state_files, None)
        assert copied == []

    def test_export_missing_model_dir(self, tmp_path):
        copied = export_states(tmp_path / "nonexistent", tmp_path / "fews_out")
        assert copied == []

    def test_export_creates_fews_dir(self, sample_state_files, tmp_path):
        fews_out = tmp_path / "new_dir" / "states"
        copied = export_states(sample_state_files, fews_out)
        assert fews_out.is_dir()
        assert len(copied) == 2
