"""Tests for the conservative workspace-cleanup utility."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[3] / "scripts" / "clean_workspace.py"
SPEC = importlib.util.spec_from_file_location("clean_workspace", SCRIPT)
assert SPEC and SPEC.loader
clean_workspace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clean_workspace)


def test_collect_only_known_generated_artifacts(tmp_path):
    generated_log = tmp_path / "--versionhyss_probe.log"
    generated_log.write_text("probe")
    ds_store = tmp_path / "nested" / ".DS_Store"
    ds_store.parent.mkdir()
    ds_store.write_text("metadata")
    cache = tmp_path / "src" / "__pycache__"
    cache.mkdir(parents=True)
    environment_cache = tmp_path / ".venv" / "lib" / "__pycache__"
    environment_cache.mkdir(parents=True)
    preserved = tmp_path / "results" / "valuable.log"
    preserved.parent.mkdir()
    preserved.write_text("keep")

    found = set(clean_workspace.collect(tmp_path))

    assert {generated_log, ds_store, cache} <= found
    assert preserved not in found
    assert environment_cache not in found


def test_collect_does_not_include_workspace_root(tmp_path):
    assert tmp_path not in clean_workspace.collect(tmp_path)
