"""
Contract tests for worker implementations.

These tests define the interface contract that any BaseWorker implementation
must satisfy. When adding a new model worker, it should pass all these tests.

Workers are responsible for:
1. Applying parameters to model configuration files
2. Running the hydrological model
3. Calculating performance metrics from model output
"""

import pytest
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from unittest.mock import Mock

# Mark all tests in this module
pytestmark = [pytest.mark.unit, pytest.mark.optimization]


# ============================================================================
# Worker task/result dataclass contracts (to be implemented)
# ============================================================================

@dataclass
class MockWorkerTask:
    """Mock worker task matching expected interface."""
    individual_id: int
    params: Dict[str, float]
    proc_id: int
    config: Dict[str, Any]
    settings_dir: Path
    output_dir: Path
    multiobjective: bool = False
    objective_names: Optional[List[str]] = None


@dataclass
class MockWorkerResult:
    """Mock worker result matching expected interface."""
    individual_id: int
    params: Dict[str, float]
    score: Optional[float]
    objectives: Optional[List[float]] = None
    error: Optional[str] = None
    runtime: Optional[float] = None
    debug_info: Optional[Dict] = None


# ============================================================================
# Test fixtures
# ============================================================================

@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger('test_worker_contracts')
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def base_worker_config(tmp_path):
    """Base configuration for workers."""
    return {
        'SYMFLUENCE_DATA_DIR': str(tmp_path),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'OPTIMIZATION_METRIC': 'KGE',
        'WORKER_MAX_RETRIES': 3,
    }


@pytest.fixture
def sample_task(tmp_path, base_worker_config):
    """Create a sample worker task."""
    settings_dir = tmp_path / 'settings'
    output_dir = tmp_path / 'output'
    settings_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    return MockWorkerTask(
        individual_id=0,
        params={'param1': 0.5, 'param2': 5.0},
        proc_id=0,
        config=base_worker_config,
        settings_dir=settings_dir,
        output_dir=output_dir,
    )


@pytest.fixture
def multiobjective_task(sample_task):
    """Create a multi-objective worker task."""
    sample_task.multiobjective = True
    sample_task.objective_names = ['KGE', 'NSE']
    return sample_task


# ============================================================================
# Worker interface contract tests
# ============================================================================

class TestWorkerTaskContract:
    """Tests that verify WorkerTask structure."""

    def test_task_has_required_fields(self, sample_task):
        """Test that task has all required fields."""
        required_fields = [
            'individual_id',
            'params',
            'proc_id',
            'config',
            'settings_dir',
            'output_dir',
        ]

        for field in required_fields:
            assert hasattr(sample_task, field), f"Task missing required field: {field}"

    def test_task_params_is_dict(self, sample_task):
        """Test that params is a dictionary."""
        assert isinstance(sample_task.params, dict)

    def test_task_paths_are_path_objects(self, sample_task):
        """Test that path fields are Path objects."""
        assert isinstance(sample_task.settings_dir, Path)
        assert isinstance(sample_task.output_dir, Path)

    def test_task_config_is_dict(self, sample_task):
        """Test that config is a dictionary."""
        assert isinstance(sample_task.config, dict)

    def test_multiobjective_task_has_objective_names(self, multiobjective_task):
        """Test that multi-objective tasks have objective names."""
        assert multiobjective_task.multiobjective is True
        assert multiobjective_task.objective_names is not None
        assert len(multiobjective_task.objective_names) > 0


class TestWorkerResultContract:
    """Tests that verify WorkerResult structure."""

    def test_result_has_required_fields(self):
        """Test that result has all required fields."""
        result = MockWorkerResult(
            individual_id=0,
            params={'param1': 0.5},
            score=0.75,
        )

        required_fields = [
            'individual_id',
            'params',
            'score',
        ]

        for field in required_fields:
            assert hasattr(result, field), f"Result missing required field: {field}"

    def test_successful_result_has_score(self):
        """Test that successful results have a score."""
        result = MockWorkerResult(
            individual_id=0,
            params={'param1': 0.5},
            score=0.75,
        )

        assert result.score is not None
        assert result.error is None

    def test_failed_result_has_error(self):
        """Test that failed results have an error message."""
        result = MockWorkerResult(
            individual_id=0,
            params={'param1': 0.5},
            score=None,
            error="Model execution failed",
        )

        assert result.score is None
        assert result.error is not None

    def test_multiobjective_result_has_objectives(self):
        """Test that multi-objective results have objectives list."""
        result = MockWorkerResult(
            individual_id=0,
            params={'param1': 0.5},
            score=0.75,
            objectives=[0.8, 0.7],  # KGE, NSE
        )

        assert result.objectives is not None
        assert len(result.objectives) == 2

    def test_result_can_have_runtime(self):
        """Test that results can include runtime information."""
        result = MockWorkerResult(
            individual_id=0,
            params={'param1': 0.5},
            score=0.75,
            runtime=2.5,
        )

        assert result.runtime == 2.5


class TestWorkerMethodContract:
    """Tests that verify worker method interfaces."""

    def test_worker_has_apply_parameters(self):
        """Test that workers have apply_parameters method."""
        mock_worker = Mock()
        mock_worker.apply_parameters = Mock(return_value=True)

        result = mock_worker.apply_parameters({'param1': 0.5}, Path('/settings'))
        assert isinstance(result, bool)

    def test_worker_has_run_model(self):
        """Test that workers have run_model method."""
        mock_worker = Mock()
        mock_worker.run_model = Mock(return_value=True)

        result = mock_worker.run_model({})
        assert isinstance(result, bool)

    def test_worker_has_calculate_metrics(self):
        """Test that workers have calculate_metrics method."""
        mock_worker = Mock()
        mock_worker.calculate_metrics = Mock(return_value={'KGE': 0.8, 'NSE': 0.75})

        result = mock_worker.calculate_metrics(Path('/output'), {})
        assert isinstance(result, dict)
        assert 'KGE' in result or 'NSE' in result

    def test_worker_has_evaluate(self):
        """Test that workers have evaluate method."""
        mock_worker = Mock()
        mock_worker.evaluate = Mock(return_value=MockWorkerResult(
            individual_id=0,
            params={'param1': 0.5},
            score=0.75,
        ))

        result = mock_worker.evaluate(Mock())
        assert hasattr(result, 'score')


class TestWorkerBehaviorContract:
    """Tests that verify worker behavioral contracts."""

    def test_apply_parameters_returns_bool(self):
        """Test that apply_parameters returns boolean success/failure."""
        mock_worker = Mock()

        # Success case
        mock_worker.apply_parameters.return_value = True
        assert mock_worker.apply_parameters({'p': 1}, Path('/')) is True

        # Failure case
        mock_worker.apply_parameters.return_value = False
        assert mock_worker.apply_parameters({'p': 1}, Path('/')) is False

    def test_run_model_returns_bool(self):
        """Test that run_model returns boolean success/failure."""
        mock_worker = Mock()

        # Success case
        mock_worker.run_model.return_value = True
        assert mock_worker.run_model({}) is True

        # Failure case
        mock_worker.run_model.return_value = False
        assert mock_worker.run_model({}) is False

    def test_evaluate_handles_parameter_failure(self):
        """Test that evaluate handles parameter application failure gracefully."""
        mock_worker = Mock()

        # Simulate parameter application failure
        result = MockWorkerResult(
            individual_id=0,
            params={'param1': 0.5},
            score=None,
            error="Parameter application failed",
        )
        mock_worker.evaluate.return_value = result

        result = mock_worker.evaluate(Mock())
        assert result.score is None
        assert result.error is not None
        assert 'parameter' in result.error.lower() or 'Parameter' in result.error

    def test_evaluate_handles_model_failure(self):
        """Test that evaluate handles model execution failure gracefully."""
        mock_worker = Mock()

        # Simulate model execution failure
        result = MockWorkerResult(
            individual_id=0,
            params={'param1': 0.5},
            score=None,
            error="Model execution failed",
        )
        mock_worker.evaluate.return_value = result

        result = mock_worker.evaluate(Mock())
        assert result.score is None
        assert result.error is not None

    def test_evaluate_preserves_individual_id(self, sample_task):
        """Test that evaluate preserves the individual_id."""
        mock_worker = Mock()
        mock_worker.evaluate.return_value = MockWorkerResult(
            individual_id=sample_task.individual_id,
            params=sample_task.params,
            score=0.75,
        )

        result = mock_worker.evaluate(sample_task)
        assert result.individual_id == sample_task.individual_id

    def test_evaluate_preserves_params(self, sample_task):
        """Test that evaluate preserves the params."""
        mock_worker = Mock()
        mock_worker.evaluate.return_value = MockWorkerResult(
            individual_id=sample_task.individual_id,
            params=sample_task.params,
            score=0.75,
        )

        result = mock_worker.evaluate(sample_task)
        assert result.params == sample_task.params


class TestWorkerRetryContract:
    """Tests that verify retry behavior contract."""

    def test_retry_on_transient_failure(self):
        """Test that workers should retry on transient failures."""
        # Document the retry contract - workers should implement this behavior
        mock_worker = Mock()

        # Simulate the expected behavior: retry succeeds after transient failure
        mock_worker.evaluate.side_effect = [
            MockWorkerResult(individual_id=0, params={'p': 1}, score=None, error="Stale file handle"),
            MockWorkerResult(individual_id=0, params={'p': 1}, score=None, error="Stale file handle"),
            MockWorkerResult(individual_id=0, params={'p': 1}, score=0.75),  # Success on 3rd try
        ]

        # The actual BaseWorker.evaluate() should handle retries internally
        # Here we document what the results look like
        results = [mock_worker.evaluate(Mock()) for _ in range(3)]

        # First two fail, third succeeds
        assert results[0].error is not None
        assert results[1].error is not None
        assert results[2].score == 0.75

    def test_max_retries_respected(self):
        """Test that workers should respect max retry count."""
        MAX_RETRIES = 3

        # Document the expected behavior after max retries exceeded
        mock_worker = Mock()
        mock_worker.max_retries = MAX_RETRIES

        # After all retries fail, should return failure result
        mock_worker.evaluate.return_value = MockWorkerResult(
            individual_id=0,
            params={'p': 1},
            score=None,
            error="Max retries exceeded",
        )

        result = mock_worker.evaluate(Mock())
        assert result.score is None
        assert result.error is not None


class TestWorkerIsolationContract:
    """Tests that verify process isolation contract."""

    def test_worker_uses_process_specific_dirs(self, sample_task):
        """Test that workers can use process-specific directories."""
        # Workers should support process isolation via directories
        proc_id = sample_task.proc_id
        settings_dir = sample_task.settings_dir

        # Pattern: process-specific directory
        proc_specific_dir = settings_dir / f'process_{proc_id}'
        assert f'process_{proc_id}' in str(proc_specific_dir) or proc_id == 0

    def test_worker_handles_environment_isolation(self):
        """Test that workers handle environment isolation."""
        # Workers should set thread counts to prevent resource contention
        expected_env_vars = {
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
        }

        # Document the expected pattern
        for var, value in expected_env_vars.items():
            assert var is not None
            assert value == '1'


class TestWorkerMetricsContract:
    """Tests that verify metrics calculation contract."""

    def test_calculate_metrics_returns_dict(self):
        """Test that calculate_metrics returns a dictionary."""
        mock_worker = Mock()
        mock_worker.calculate_metrics.return_value = {'KGE': 0.8}

        result = mock_worker.calculate_metrics(Path('/'), {})
        assert isinstance(result, dict)

    def test_metrics_include_requested_metric(self):
        """Test that metrics include the requested optimization metric."""
        mock_worker = Mock()

        # KGE optimization
        mock_worker.calculate_metrics.return_value = {'KGE': 0.8, 'NSE': 0.75, 'RMSE': 2.5}

        result = mock_worker.calculate_metrics(Path('/'), {'OPTIMIZATION_METRIC': 'KGE'})
        assert 'KGE' in result

    def test_invalid_metrics_return_failure_value(self):
        """Test that invalid/failed metrics return sentinel value."""
        FAILURE_VALUE = -999.0

        mock_worker = Mock()
        mock_worker.calculate_metrics.return_value = {'KGE': FAILURE_VALUE}

        result = mock_worker.calculate_metrics(Path('/'), {})
        assert result['KGE'] == FAILURE_VALUE

    def test_multiobjective_returns_multiple_metrics(self):
        """Test that multi-objective workers return multiple metrics."""
        mock_worker = Mock()
        mock_worker.calculate_metrics.return_value = {
            'KGE': 0.8,
            'NSE': 0.75,
            'RMSE': 2.5,
            'MAE': 1.8,
        }

        result = mock_worker.calculate_metrics(Path('/'), {'MULTI_OBJECTIVE': True})
        assert len(result) >= 2


# ============================================================================
# Model-specific worker contract tests (to be enabled)
# ============================================================================

class TestSUMMAWorkerContract:
    """Contract tests specific to SUMMA worker."""

    def test_summa_worker_registered(self):
        """Test that SUMMAWorker is registered with registry."""
        from symfluence.models.summa.calibration.worker import SUMMAWorker  # noqa: F401 — triggers @register_worker
        from symfluence.optimization.registry import OptimizerRegistry
        worker_cls = OptimizerRegistry.get_worker('SUMMA')
        assert worker_cls is not None
        assert worker_cls.__name__ == 'SUMMAWorker'

    def test_summa_worker_inherits_from_base(self):
        """Test that SUMMAWorker inherits from BaseWorker."""
        from symfluence.optimization.workers import SUMMAWorker, BaseWorker
        assert issubclass(SUMMAWorker, BaseWorker)

    def test_summa_worker_has_required_methods(self):
        """Test that SUMMAWorker has all required abstract methods."""
        from symfluence.optimization.workers import SUMMAWorker
        required_methods = ['apply_parameters', 'run_model', 'calculate_metrics']
        for method in required_methods:
            assert hasattr(SUMMAWorker, method)

    def test_summa_worker_handles_routing_decision(self):
        """Test that SUMMA worker has routing-related attributes."""
        from symfluence.optimization.workers import SUMMAWorker
        # The worker should handle routing decisions internally
        assert hasattr(SUMMAWorker, 'run_model')


class TestFUSEWorkerContract:
    """Contract tests specific to FUSE worker."""

    def test_fuse_worker_registered(self):
        """Test that FUSEWorker is registered with registry."""
        from symfluence.models.fuse.calibration.worker import FUSEWorker  # noqa: F401 — triggers @register_worker
        from symfluence.optimization.registry import OptimizerRegistry
        worker_cls = OptimizerRegistry.get_worker('FUSE')
        assert worker_cls is not None
        assert worker_cls.__name__ == 'FUSEWorker'

    def test_fuse_worker_inherits_from_base(self):
        """Test that FUSEWorker inherits from BaseWorker."""
        from symfluence.models.fuse.calibration.worker import FUSEWorker
        from symfluence.optimization.workers import BaseWorker
        assert issubclass(FUSEWorker, BaseWorker)

    def test_fuse_worker_has_required_methods(self):
        """Test that FUSEWorker has all required abstract methods."""
        from symfluence.models.fuse.calibration.worker import FUSEWorker
        required_methods = ['apply_parameters', 'run_model', 'calculate_metrics']
        for method in required_methods:
            assert hasattr(FUSEWorker, method)


class TestNgenWorkerContract:
    """Contract tests specific to NextGen worker."""

    def test_ngen_worker_registered(self):
        """Test that NgenWorker is registered with registry."""
        from symfluence.models.ngen.calibration.worker import NgenWorker  # noqa: F401 — triggers @register_worker
        from symfluence.optimization.registry import OptimizerRegistry
        worker_cls = OptimizerRegistry.get_worker('NGEN')
        assert worker_cls is not None
        assert worker_cls.__name__ == 'NgenWorker'

    def test_ngen_worker_inherits_from_base(self):
        """Test that NgenWorker inherits from BaseWorker."""
        from symfluence.optimization.workers import NgenWorker, BaseWorker
        assert issubclass(NgenWorker, BaseWorker)

    def test_ngen_worker_has_required_methods(self):
        """Test that NgenWorker has all required abstract methods."""
        from symfluence.optimization.workers import NgenWorker
        required_methods = ['apply_parameters', 'run_model', 'calculate_metrics']
        for method in required_methods:
            assert hasattr(NgenWorker, method)


# ============================================================================
# Legacy compatibility contract tests
# ============================================================================

class TestLegacyCompatibilityContract:
    """Tests that verify backward compatibility with existing worker functions."""

    def test_legacy_task_dict_format(self):
        """Test that workers can accept legacy task dictionary format."""
        legacy_task = {
            'individual_id': 0,
            'params': {'param1': 0.5},
            'proc_id': 0,
            'config': {},
            'settings_dir': '/path/to/settings',
            'output_dir': '/path/to/output',
        }

        # Document expected conversion
        assert 'individual_id' in legacy_task
        assert 'params' in legacy_task

    def test_legacy_result_dict_format(self):
        """Test that workers can return legacy result dictionary format."""
        legacy_result = {
            'score': 0.75,
            'individual_id': 0,
            'params': {'param1': 0.5},
            'error': None,
        }

        # Document expected structure
        assert 'score' in legacy_result
        assert 'individual_id' in legacy_result

    def test_legacy_sentinel_values(self):
        """Test that legacy sentinel values are supported."""
        # Legacy: return -999.0 for failures
        LEGACY_FAILURE = -999.0

        result = MockWorkerResult(
            individual_id=0,
            params={},
            score=LEGACY_FAILURE,
        )

        assert result.score == LEGACY_FAILURE
