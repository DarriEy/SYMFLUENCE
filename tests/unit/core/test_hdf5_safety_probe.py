"""Tests for compatibility-aware bundled HDF5 detection."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import symfluence.core.hdf5_safety as safety


def _fake_libraries(tmp_path: Path) -> tuple[Path, Path]:
    h5py_lib = tmp_path / "h5py.libs" / "libhdf5-h5py.so"
    nc4_lib = tmp_path / "netcdf4.libs" / "libhdf5-netcdf.so"
    h5py_lib.parent.mkdir()
    nc4_lib.parent.mkdir()
    h5py_lib.write_bytes(b"h5py")
    nc4_lib.write_bytes(b"netcdf4")
    return h5py_lib, nc4_lib


def test_compatibility_probe_is_isolated_and_cached(tmp_path, monkeypatch):
    h5py_lib, nc4_lib = _fake_libraries(tmp_path)
    cache = tmp_path / "cache" / "compatibility.json"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(safety, "_compatibility_cache_path", lambda: cache)
    monkeypatch.setattr(safety.subprocess, "run", fake_run)

    assert safety._probe_bundled_hdf5_compatibility(h5py_lib, nc4_lib)
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[-2:] == ["-m", "symfluence.core.hdf5_probe"]
    assert kwargs["env"]["SYMFLUENCE_HDF5_PROBE"] == "1"
    assert kwargs["timeout"] == 30

    monkeypatch.setattr(
        safety.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache missed")),
    )
    assert safety._probe_bundled_hdf5_compatibility(h5py_lib, nc4_lib)
    assert json.loads(cache.read_text(encoding="utf-8"))["compatible"] is True


def test_compatibility_probe_fails_closed_on_timeout(tmp_path, monkeypatch):
    h5py_lib, nc4_lib = _fake_libraries(tmp_path)
    monkeypatch.setattr(
        safety,
        "_compatibility_cache_path",
        lambda: tmp_path / "cache" / "compatibility.json",
    )
    monkeypatch.setattr(
        safety.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("probe", 30)
        ),
    )

    assert not safety._probe_bundled_hdf5_compatibility(h5py_lib, nc4_lib)


def test_different_bundled_paths_are_accepted_when_probe_passes(
    tmp_path, monkeypatch
):
    h5py_lib, nc4_lib = _fake_libraries(tmp_path)
    monkeypatch.setattr(
        safety,
        "_find_bundled_libhdf5",
        lambda package: h5py_lib if package == "h5py" else nc4_lib,
    )
    monkeypatch.setattr(
        safety, "_probe_bundled_hdf5_compatibility", lambda *args: True
    )
    safety.hdf5_library_conflict = False

    safety._check_hdf5_library_conflict()

    assert safety.hdf5_library_conflict is False


def test_different_bundled_paths_are_rejected_when_probe_fails(
    tmp_path, monkeypatch
):
    h5py_lib, nc4_lib = _fake_libraries(tmp_path)
    monkeypatch.setattr(
        safety,
        "_find_bundled_libhdf5",
        lambda package: h5py_lib if package == "h5py" else nc4_lib,
    )
    monkeypatch.setattr(
        safety, "_probe_bundled_hdf5_compatibility", lambda *args: False
    )
    monkeypatch.setattr(safety, "_conflict_already_reported", lambda *args: True)
    safety.hdf5_library_conflict = False

    safety._check_hdf5_library_conflict()

    assert safety.hdf5_library_conflict is True
