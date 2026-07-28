"""End-to-end MPI validation for non-executable file hand-offs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.optimization]


def test_real_mpi_json_handoff(tmp_path, monkeypatch):
    pytest.importorskip("mpi4py")
    if not (shutil.which("mpiexec") or shutil.which("mpirun")):
        pytest.skip("MPI launcher unavailable")

    tests_dir = Path(__file__).resolve().parents[2]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(tests_dir)
    if existing_pythonpath:
        pythonpath = f"{pythonpath}{os.pathsep}{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)

    from mpi_json_worker import double_task

    from symfluence.optimization.mixins.parallel.execution_strategies.mpi import (
        MPIExecutionStrategy,
    )

    strategy = MPIExecutionStrategy(tmp_path, num_processes=2)
    results = strategy.execute(
        [
            {"individual_id": 1, "value": 2.5, "params": {"x": 1}},
            {"individual_id": 2, "value": 4.0, "params": {"x": 2}},
        ],
        double_task,
        max_workers=2,
    )

    assert sorted(result["score"] for result in results) == [5.0, 8.0]
    assert not list((tmp_path / "temp_mpi").glob("*.pkl"))
