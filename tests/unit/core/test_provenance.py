"""Tests for symfluence.core.provenance."""

import json
import platform
import re
import time
from pathlib import Path

import pytest

from symfluence.core.provenance import (
    RunProvenance,
    _dependency_versions,
    _git_info,
    _platform_info,
    capture_provenance,
    finalize,
    make_run_id,
    record_step,
)

# ---------------------------------------------------------------------------
# _git_info
# ---------------------------------------------------------------------------

def test_git_info_returns_dict():
    """_git_info should return a dict with the expected keys."""
    info = _git_info()
    assert set(info.keys()) == {"commit", "commit_short", "branch", "dirty"}
    # If run inside a git repo the short hash is 8 chars
    if info["commit"] is not None:
        assert len(info["commit_short"]) == 8


def test_git_info_graceful_outside_repo(tmp_path):
    """_git_info should return Nones when pointed at a non-repo dir."""
    info = _git_info(tmp_path)
    assert info["commit"] is None
    assert info["commit_short"] is None
    assert info["branch"] is None
    assert info["dirty"] is None


# ---------------------------------------------------------------------------
# _dependency_versions / _platform_info
# ---------------------------------------------------------------------------

def test_dependency_versions():
    """numpy should be present; a made-up package should not."""
    deps = _dependency_versions()
    assert "numpy" in deps
    assert "nonexistent_pkg_xyz" not in deps


def test_platform_info():
    """Keys exist and python version matches the running interpreter."""
    info = _platform_info()
    assert set(info.keys()) == {"os", "os_version", "arch", "hostname", "python"}
    assert info["python"] == platform.python_version()


# ---------------------------------------------------------------------------
# make_run_id
# ---------------------------------------------------------------------------

def test_make_run_id_format():
    """Run ID matches expected pattern: <exp>_<YYYYMMDD_HHMMSS>_<hash>."""
    rid = make_run_id("bow_banff", git_short="abcd1234")
    assert re.match(r"bow_banff_\d{8}_\d{6}_abcd1234$", rid)


def test_make_run_id_no_git():
    """Without git info the run ID ends with '_nogit'."""
    rid = make_run_id("test_exp")
    assert rid.endswith("_nogit")


# ---------------------------------------------------------------------------
# capture_provenance
# ---------------------------------------------------------------------------

def test_capture_provenance():
    """capture_provenance returns a populated RunProvenance."""
    prov = capture_provenance("exp1", "domain1", config_path="/tmp/cfg.yaml")
    assert prov.experiment_id == "exp1"
    assert prov.domain_name == "domain1"
    assert prov.config_path == "/tmp/cfg.yaml"
    assert prov.symfluence_version  # non-empty
    assert prov.start_utc  # non-empty
    assert prov.steps == []
    assert prov.status == "running"


# ---------------------------------------------------------------------------
# record_step
# ---------------------------------------------------------------------------

def test_record_step():
    """record_step appends entries and captures errors."""
    prov = capture_provenance("e", "d")
    record_step(prov, "step_a", 1.23)
    record_step(prov, "step_b", 0.0, status="failed", error="boom")

    assert len(prov.steps) == 2
    assert prov.steps[0]["name"] == "step_a"
    assert prov.steps[0]["status"] == "completed"
    assert prov.steps[1]["error"] == "boom"

    # No-op on None
    record_step(None, "x", 0.0)


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------

def test_finalize():
    """finalize stamps end_utc, elapsed, and status."""
    prov = capture_provenance("e", "d")
    time.sleep(0.01)  # ensure measurable elapsed
    finalize(prov, "completed")

    assert prov.end_utc is not None
    assert prov.elapsed_seconds >= 0
    assert prov.status == "completed"

    # None provenance is a no-op
    assert finalize(None, "completed") is None


# ---------------------------------------------------------------------------
# write / to_dict
# ---------------------------------------------------------------------------

def test_write_manifest(tmp_path):
    """write() produces a valid JSON file that round-trips."""
    prov = capture_provenance("e", "d")
    record_step(prov, "a", 0.5)
    finalize(prov, "completed")

    path = prov.write(tmp_path)
    assert path.exists()
    assert path.name == "run_manifest.json"

    data = json.loads(path.read_text())
    assert data["experiment_id"] == "e"
    assert len(data["steps"]) == 1


def test_to_dict_serializable():
    """to_dict() output must be JSON-serializable."""
    prov = capture_provenance("e", "d")
    record_step(prov, "s", 0.1)
    finalize(prov, "completed", errors=["oops"])

    # This will raise if any value isn't serializable
    text = json.dumps(prov.to_dict())
    assert "oops" in text
