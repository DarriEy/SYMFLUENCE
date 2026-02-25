"""
Unit tests for stage_marker module.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from symfluence.core.stage_marker import (
    STAGE_CONFIG_SECTIONS,
    StageMarker,
    clear_markers,
    compute_config_hash,
    is_stage_current,
    read_marker,
    write_marker,
)

# ---------------------------------------------------------------------------
# Lightweight config stub
# ---------------------------------------------------------------------------


class _Section:
    """Minimal stand-in for a Pydantic config section."""

    def __init__(self, data: dict):
        self._data = data

    def model_dump(self, *, by_alias: bool = False) -> dict:
        return dict(self._data)


class _FakeConfig:
    """Fake SymfluenceConfig with a few sections for hashing tests."""

    def __init__(self, **sections):
        for name, data in sections.items():
            setattr(self, name, _Section(data))


# ---------------------------------------------------------------------------
# compute_config_hash
# ---------------------------------------------------------------------------


class TestComputeConfigHash:
    def test_deterministic(self):
        cfg = _FakeConfig(domain={"name": "bow"}, data={"source": "era5"})
        h1 = compute_config_hash(cfg, ["domain", "data"])
        h2 = compute_config_hash(cfg, ["domain", "data"])
        assert h1 == h2

    def test_changes_when_config_changes(self):
        cfg_a = _FakeConfig(domain={"name": "bow"})
        cfg_b = _FakeConfig(domain={"name": "columbia"})
        assert compute_config_hash(cfg_a, ["domain"]) != compute_config_hash(
            cfg_b, ["domain"]
        )

    def test_section_order_irrelevant(self):
        cfg = _FakeConfig(domain={"name": "bow"}, data={"source": "era5"})
        h1 = compute_config_hash(cfg, ["domain", "data"])
        h2 = compute_config_hash(cfg, ["data", "domain"])
        assert h1 == h2

    def test_missing_section_ignored(self):
        cfg = _FakeConfig(domain={"name": "bow"})
        h1 = compute_config_hash(cfg, ["domain"])
        h2 = compute_config_hash(cfg, ["domain", "nonexistent"])
        assert h1 == h2

    def test_empty_sections(self):
        cfg = _FakeConfig()
        h = compute_config_hash(cfg, [])
        assert isinstance(h, str) and len(h) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# write_marker / read_marker
# ---------------------------------------------------------------------------


class TestMarkerIO:
    def test_write_read_roundtrip(self, tmp_path):
        write_marker(tmp_path, "setup_project", "abc123", git_commit="deadbeef")
        marker = read_marker(tmp_path, "setup_project")

        assert marker is not None
        assert marker.stage == "setup_project"
        assert marker.config_hash == "abc123"
        assert marker.git_commit == "deadbeef"
        assert marker.completed_utc  # non-empty

    def test_missing_marker_returns_none(self, tmp_path):
        assert read_marker(tmp_path, "no_such_stage") is None

    def test_corrupt_json_returns_none(self, tmp_path):
        marker_dir = tmp_path / ".symfluence" / "stage_markers"
        marker_dir.mkdir(parents=True)
        (marker_dir / "broken.json").write_text("NOT JSON", encoding="utf-8")
        assert read_marker(tmp_path, "broken") is None

    def test_missing_key_returns_none(self, tmp_path):
        marker_dir = tmp_path / ".symfluence" / "stage_markers"
        marker_dir.mkdir(parents=True)
        (marker_dir / "partial.json").write_text(
            json.dumps({"stage": "partial"}), encoding="utf-8"
        )
        assert read_marker(tmp_path, "partial") is None

    def test_write_without_git_commit(self, tmp_path):
        with patch(
            "symfluence.core.stage_marker._current_git_commit", return_value=None
        ):
            write_marker(tmp_path, "run_models", "hash456")
        marker = read_marker(tmp_path, "run_models")
        assert marker is not None
        assert marker.git_commit is None


# ---------------------------------------------------------------------------
# is_stage_current
# ---------------------------------------------------------------------------


class TestIsStageCurrent:
    def test_true_when_hash_matches(self, tmp_path):
        write_marker(tmp_path, "define_domain", "myhash")
        assert is_stage_current(tmp_path, "define_domain", "myhash") is True

    def test_false_when_hash_differs(self, tmp_path):
        write_marker(tmp_path, "define_domain", "oldhash")
        assert is_stage_current(tmp_path, "define_domain", "newhash") is False

    def test_false_when_no_marker(self, tmp_path):
        assert is_stage_current(tmp_path, "define_domain", "any") is False


# ---------------------------------------------------------------------------
# clear_markers
# ---------------------------------------------------------------------------


class TestClearMarkers:
    def test_clear_all(self, tmp_path):
        write_marker(tmp_path, "a", "h1")
        write_marker(tmp_path, "b", "h2")
        clear_markers(tmp_path)
        assert read_marker(tmp_path, "a") is None
        assert read_marker(tmp_path, "b") is None

    def test_clear_selective(self, tmp_path):
        write_marker(tmp_path, "a", "h1")
        write_marker(tmp_path, "b", "h2")
        clear_markers(tmp_path, stage_names=["a"])
        assert read_marker(tmp_path, "a") is None
        assert read_marker(tmp_path, "b") is not None

    def test_clear_nonexistent_is_noop(self, tmp_path):
        # Should not raise
        clear_markers(tmp_path)
        clear_markers(tmp_path, stage_names=["ghost"])


# ---------------------------------------------------------------------------
# Coverage of orchestrator step names
# ---------------------------------------------------------------------------


EXPECTED_STAGES = [
    "setup_project",
    "create_pour_point",
    "acquire_attributes",
    "define_domain",
    "discretize_domain",
    "process_observed_data",
    "acquire_forcings",
    "run_model_agnostic_preprocessing",
    "build_model_ready_store",
    "preprocess_models",
    "run_models",
    "postprocess_results",
    "calibrate_model",
    "run_benchmarking",
    "run_decision_analysis",
    "run_sensitivity_analysis",
]


class TestStageConfigSectionsCoverage:
    @pytest.mark.parametrize("stage", EXPECTED_STAGES)
    def test_stage_has_sections(self, stage):
        assert stage in STAGE_CONFIG_SECTIONS
        assert len(STAGE_CONFIG_SECTIONS[stage]) > 0

    def test_no_system_or_paths_sections(self):
        for sections in STAGE_CONFIG_SECTIONS.values():
            assert "system" not in sections
            assert "paths" not in sections
