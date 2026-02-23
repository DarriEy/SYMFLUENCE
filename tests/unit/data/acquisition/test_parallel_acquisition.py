"""Tests for concurrent data acquisition in AcquisitionService.

Tests _run_parallel_tasks() helper and the MAX_ACQUISITION_WORKERS config field.
"""

import logging
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from symfluence.core.config.models.data import DataConfig

# ---------------------------------------------------------------------------
# DataConfig field validation
# ---------------------------------------------------------------------------

class TestMaxAcquisitionWorkersConfig:
    """Validate MAX_ACQUISITION_WORKERS Pydantic field on DataConfig."""

    def test_default_value(self):
        cfg = DataConfig()
        assert cfg.max_acquisition_workers == 3

    def test_from_alias(self):
        cfg = DataConfig.model_validate({"MAX_ACQUISITION_WORKERS": 5})
        assert cfg.max_acquisition_workers == 5

    def test_minimum_is_one(self):
        with pytest.raises(Exception):  # ValidationError
            DataConfig.model_validate({"MAX_ACQUISITION_WORKERS": 0})

    def test_maximum_is_eight(self):
        with pytest.raises(Exception):  # ValidationError
            DataConfig.model_validate({"MAX_ACQUISITION_WORKERS": 9})

    def test_boundary_values(self):
        cfg1 = DataConfig.model_validate({"MAX_ACQUISITION_WORKERS": 1})
        assert cfg1.max_acquisition_workers == 1
        cfg8 = DataConfig.model_validate({"MAX_ACQUISITION_WORKERS": 8})
        assert cfg8.max_acquisition_workers == 8


# ---------------------------------------------------------------------------
# _run_parallel_tasks() helper
# ---------------------------------------------------------------------------

def _make_service(max_workers=3):
    """Create a minimal AcquisitionService with mocked config for testing."""
    from symfluence.data.acquisition.acquisition_service import AcquisitionService

    mock_config = MagicMock()
    mock_config.data.max_acquisition_workers = max_workers
    mock_config.system.data_dir = "/tmp/test"
    mock_config.domain.name = "test"

    logger = logging.getLogger("test_parallel")
    logger.setLevel(logging.DEBUG)

    with patch.object(AcquisitionService, "__init__", lambda self, *a, **kw: None):
        svc = AcquisitionService.__new__(AcquisitionService)

    svc._config = mock_config
    svc.config = mock_config
    svc.logger = logger
    return svc


class TestRunParallelTasks:
    """Unit tests for AcquisitionService._run_parallel_tasks()."""

    def test_all_succeed(self):
        svc = _make_service(max_workers=3)
        tasks = [
            ("a", lambda: "result_a"),
            ("b", lambda: "result_b"),
            ("c", lambda: "result_c"),
        ]
        results = svc._run_parallel_tasks(tasks, desc="test")
        assert results == {"a": "result_a", "b": "result_b", "c": "result_c"}

    def test_error_collected_not_raised(self):
        """Errors are collected in results dict, not raised."""
        svc = _make_service(max_workers=2)

        def _fail():
            raise ValueError("boom")

        tasks = [
            ("ok", lambda: 42),
            ("bad", _fail),
        ]
        results = svc._run_parallel_tasks(tasks, desc="test")
        assert results["ok"] == 42
        assert isinstance(results["bad"], ValueError)

    def test_serial_fallback_with_one_worker(self):
        """Workers=1 uses serial path (no ThreadPoolExecutor)."""
        svc = _make_service(max_workers=1)
        call_order = []

        def _task_a():
            call_order.append("a")
            return "a"

        def _task_b():
            call_order.append("b")
            return "b"

        tasks = [("a", _task_a), ("b", _task_b)]
        results = svc._run_parallel_tasks(tasks, desc="serial")

        # Serial path preserves submission order
        assert call_order == ["a", "b"]
        assert results == {"a": "a", "b": "b"}

    @patch("symfluence.data.acquisition.acquisition_service.sys")
    def test_macos_forces_serial(self, mock_sys):
        """On macOS (darwin), workers should be forced to 1."""
        mock_sys.platform = "darwin"
        svc = _make_service(max_workers=4)
        call_order = []

        def _task_a():
            call_order.append("a")
            return "a"

        def _task_b():
            call_order.append("b")
            return "b"

        tasks = [("a", _task_a), ("b", _task_b)]
        results = svc._run_parallel_tasks(tasks, desc="macos")
        # Should be serial on macOS
        assert call_order == ["a", "b"]
        assert results == {"a": "a", "b": "b"}

    def test_empty_task_list(self):
        svc = _make_service(max_workers=3)
        results = svc._run_parallel_tasks([], desc="empty")
        assert results == {}

    def test_single_task(self):
        svc = _make_service(max_workers=3)
        tasks = [("only", lambda: "done")]
        results = svc._run_parallel_tasks(tasks, desc="single")
        assert results == {"only": "done"}

    def test_mixed_success_and_failure(self):
        """Multiple tasks with mixed outcomes."""
        svc = _make_service(max_workers=3)

        def _fail_runtime():
            raise RuntimeError("network error")

        def _fail_key():
            raise KeyError("missing")

        tasks = [
            ("ok1", lambda: "yes"),
            ("fail1", _fail_runtime),
            ("ok2", lambda: 123),
            ("fail2", _fail_key),
        ]
        results = svc._run_parallel_tasks(tasks, desc="mixed")
        assert results["ok1"] == "yes"
        assert results["ok2"] == 123
        assert isinstance(results["fail1"], RuntimeError)
        assert isinstance(results["fail2"], KeyError)

    def test_none_result_preserved(self):
        """A task returning None should have None in results, not be treated as failure."""
        svc = _make_service(max_workers=2)
        tasks = [("returns_none", lambda: None)]
        results = svc._run_parallel_tasks(tasks, desc="none")
        assert "returns_none" in results
        assert results["returns_none"] is None

    def test_workers_capped_at_task_count(self):
        """max_workers should be capped at the number of tasks."""
        svc = _make_service(max_workers=8)
        tasks = [("a", lambda: 1), ("b", lambda: 2)]
        # Should not error even though max_workers > len(tasks)
        results = svc._run_parallel_tasks(tasks, desc="capped")
        assert len(results) == 2
