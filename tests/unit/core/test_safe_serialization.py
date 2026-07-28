# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

from __future__ import annotations

import os

import numpy as np
import pytest

from symfluence.core.safe_serialization import (
    SafeSerializationError,
    dump_json_atomic,
    load_json,
)


def test_round_trip_numpy_values(tmp_path):
    path = tmp_path / "cache.json"
    dump_json_atomic({"scalar": np.float64(1.5), "array": np.array([1, 2])}, path)
    assert load_json(path) == {"scalar": 1.5, "array": [1, 2]}


def test_rejects_executable_objects(tmp_path):
    with pytest.raises(SafeSerializationError, match="Unsupported"):
        dump_json_atomic({"unsafe": object()}, tmp_path / "cache.json")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_owner_only_permissions(tmp_path):
    path = tmp_path / "cache.json"
    dump_json_atomic({"ok": True}, path)
    assert path.stat().st_mode & 0o777 == 0o600


def test_rejects_symlinked_output_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(SafeSerializationError, match="symlink"):
        dump_json_atomic({"unsafe": True}, link / "cache.json")
    assert not (real / "cache.json").exists()
