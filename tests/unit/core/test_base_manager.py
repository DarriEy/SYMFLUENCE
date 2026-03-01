"""Tests for symfluence.core.base_manager module."""

import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from symfluence.core.config.models import SymfluenceConfig


def _make_config(tmp_path):
    """Create a minimal SymfluenceConfig for testing."""
    config_dict = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path / "data"),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "EXPERIMENT_ID": "bm_test",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-12-31 23:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "HYDROLOGICAL_MODEL": "SUMMA",
        "FORCING_DATASET": "ERA5",
        "FORCING_TIME_STEP_SIZE": 3600,
    }
    return SymfluenceConfig(**config_dict)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def bm_config(temp_dir):
    return _make_config(temp_dir)


@pytest.fixture
def mock_logger():
    logger = Mock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


# We need a concrete subclass to test the abstract BaseManager
def _make_manager(config, logger, reporting_manager=None):
    """Create a concrete BaseManager subclass instance."""
    from symfluence.core.base_manager import BaseManager

    class ConcreteManager(BaseManager):
        """Minimal concrete implementation for testing."""
        pass

    return ConcreteManager(config, logger, reporting_manager=reporting_manager)


class TestBaseManagerInit:
    """Tests for BaseManager initialization."""

    def test_init_with_config(self, bm_config, mock_logger):
        mgr = _make_manager(bm_config, mock_logger)
        assert mgr is not None
        assert mgr.logger is mock_logger

    def test_init_stores_reporting_manager(self, bm_config, mock_logger):
        rm = MagicMock()
        mgr = _make_manager(bm_config, mock_logger, reporting_manager=rm)
        assert mgr.reporting_manager is rm

    def test_init_calls_initialize_services(self, bm_config, mock_logger):
        from symfluence.core.base_manager import BaseManager

        class TrackingManager(BaseManager):
            initialized = False
            def _initialize_services(self):
                TrackingManager.initialized = True

        mgr = TrackingManager(bm_config, mock_logger)
        assert TrackingManager.initialized


class TestExecuteWorkflow:
    """Tests for _execute_workflow."""

    def test_executes_handler_for_each_item(self, bm_config, mock_logger):
        mgr = _make_manager(bm_config, mock_logger)
        handler = Mock(side_effect=lambda x: x.upper())
        results = mgr._execute_workflow(["a", "b", "c"], handler, "test_op")
        assert results == ["A", "B", "C"]
        assert handler.call_count == 3

    def test_filters_none_results(self, bm_config, mock_logger):
        mgr = _make_manager(bm_config, mock_logger)
        handler = Mock(side_effect=lambda x: x if x != "skip" else None)
        results = mgr._execute_workflow(["a", "skip", "b"], handler, "test_op")
        assert results == ["a", "b"]

    def test_empty_items_returns_empty(self, bm_config, mock_logger):
        mgr = _make_manager(bm_config, mock_logger)
        results = mgr._execute_workflow([], Mock(), "empty_op")
        assert results == []


class TestSafeVisualize:
    """Tests for _safe_visualize."""

    def test_returns_none_without_reporting_manager(self, bm_config, mock_logger):
        mgr = _make_manager(bm_config, mock_logger)
        result = mgr._safe_visualize(lambda: "viz")
        assert result is None

    def test_calls_viz_func_with_reporting_manager(self, bm_config, mock_logger):
        rm = MagicMock()
        mgr = _make_manager(bm_config, mock_logger, reporting_manager=rm)
        viz_func = Mock(return_value="chart")
        result = mgr._safe_visualize(viz_func, "arg1", key="val")
        viz_func.assert_called_once_with("arg1", key="val")
        assert result == "chart"

    def test_catches_viz_exceptions(self, bm_config, mock_logger):
        rm = MagicMock()
        mgr = _make_manager(bm_config, mock_logger, reporting_manager=rm)
        viz_func = Mock(side_effect=RuntimeError("render failed"))
        result = mgr._safe_visualize(viz_func)
        assert result is None
        mock_logger.warning.assert_called()


class TestGetService:
    """Tests for _get_service."""

    def test_creates_service_instance(self, bm_config, mock_logger):
        mgr = _make_manager(bm_config, mock_logger)

        class FakeService:
            def __init__(self, a, b):
                self.a = a
                self.b = b

        svc = mgr._get_service(FakeService, 1, b=2)
        assert isinstance(svc, FakeService)
        assert svc.a == 1
        assert svc.b == 2


class TestValidateReadinessAndGetStatus:
    """Tests for validate_readiness and get_status."""

    def test_validate_readiness_returns_empty_dict(self, bm_config, mock_logger):
        mgr = _make_manager(bm_config, mock_logger)
        assert mgr.validate_readiness() == {}

    def test_get_status_returns_basic_info(self, bm_config, mock_logger):
        mgr = _make_manager(bm_config, mock_logger)
        status = mgr.get_status()
        assert "manager" in status
        assert status["manager"] == "ConcreteManager"
        assert "project_dir" in status
        assert "experiment_id" in status
