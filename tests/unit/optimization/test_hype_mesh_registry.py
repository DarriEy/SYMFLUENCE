"""
Test HYPE and MESH optimizer/worker registration

Verifies that the new components are properly registered and can be instantiated.
"""

import logging
import tempfile
from pathlib import Path

import pytest
from symfluence.optimization.registry import OptimizerRegistry


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger('test_registry')


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def hype_config(temp_dir):
    """Create a minimal HYPE config."""
    return {
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'SYMFLUENCE_DATA_DIR': str(temp_dir),
        'SYMFLUENCE_CODE_DIR': str(temp_dir),
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'EXPERIMENT_TIME_START': '2010-01-01 00:00',
        'EXPERIMENT_TIME_END': '2010-12-31 23:00',
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'HYPE',
        'HYPE_PARAMS_TO_CALIBRATE': 'ttmp,cmlt,cevp',
        'OPTIMIZATION_TARGET': 'streamflow',
        'OPTIMIZATION_METRIC': 'KGE',
    }


@pytest.fixture
def mesh_config(temp_dir):
    """Create a minimal MESH config."""
    return {
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_exp',
        'SYMFLUENCE_DATA_DIR': str(temp_dir),
        'SYMFLUENCE_CODE_DIR': str(temp_dir),
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'SUB_GRID_DISCRETIZATION': 'lumped',
        'EXPERIMENT_TIME_START': '2010-01-01 00:00',
        'EXPERIMENT_TIME_END': '2010-12-31 23:00',
        'FORCING_DATASET': 'ERA5',
        'HYDROLOGICAL_MODEL': 'MESH',
        'MESH_PARAMS_TO_CALIBRATE': 'ZSNL,MANN',
        'OPTIMIZATION_TARGET': 'streamflow',
        'OPTIMIZATION_METRIC': 'KGE',
    }


class TestHYPERegistration:
    """Test HYPE optimizer and worker registration."""

    def test_hype_optimizer_registered(self):
        """Test that HYPE optimizer is registered."""
        optimizer_cls = OptimizerRegistry.get_optimizer('HYPE')
        assert optimizer_cls is not None
        assert optimizer_cls.__name__ == 'HYPEModelOptimizer'

    def test_hype_worker_registered(self):
        """Test that HYPE worker is registered."""
        worker_cls = OptimizerRegistry.get_worker('HYPE')
        assert worker_cls is not None
        assert worker_cls.__name__ == 'HYPEWorker'

    def test_hype_optimizer_can_instantiate(self, hype_config, logger, temp_dir):
        """Test that HYPE optimizer can be instantiated."""
        from symfluence.models.hype.calibration.optimizer import HYPEModelOptimizer

        optimizer = HYPEModelOptimizer(hype_config, logger, temp_dir)
        assert optimizer is not None
        assert optimizer._get_model_name() == 'HYPE'

    def test_hype_worker_can_instantiate(self, hype_config, logger):
        """Test that HYPE worker can be instantiated."""
        from symfluence.models.hype.calibration.worker import HYPEWorker

        worker = HYPEWorker(hype_config, logger)
        assert worker is not None

    def test_hype_optimizer_creates_hype_parameter_manager(self, hype_config, logger, temp_dir):
        """Test that HYPE optimizer creates HYPEParameterManager."""
        from symfluence.models.hype.calibration.optimizer import HYPEModelOptimizer
        from symfluence.optimization.parameter_managers import HYPEParameterManager

        optimizer = HYPEModelOptimizer(hype_config, logger, temp_dir)
        param_manager = optimizer._create_parameter_manager()

        assert isinstance(param_manager, HYPEParameterManager)


class TestMESHRegistration:
    """Test MESH optimizer and worker registration."""

    def test_mesh_optimizer_registered(self):
        """Test that MESH optimizer is registered."""
        # Import to trigger registration

        optimizer_cls = OptimizerRegistry.get_optimizer('MESH')
        assert optimizer_cls is not None
        assert optimizer_cls.__name__ == 'MESHModelOptimizer'

    def test_mesh_worker_registered(self):
        """Test that MESH worker is registered."""
        # Import to trigger registration

        worker_cls = OptimizerRegistry.get_worker('MESH')
        assert worker_cls is not None
        assert worker_cls.__name__ == 'MESHWorker'

    def test_mesh_optimizer_can_instantiate(self, mesh_config, logger, temp_dir):
        """Test that MESH optimizer can be instantiated."""
        from symfluence.models.mesh.calibration.optimizer import MESHModelOptimizer

        optimizer = MESHModelOptimizer(mesh_config, logger, temp_dir)
        assert optimizer is not None
        assert optimizer._get_model_name() == 'MESH'

    def test_mesh_worker_can_instantiate(self, mesh_config, logger):
        """Test that MESH worker can be instantiated."""
        from symfluence.models.mesh.calibration.worker import MESHWorker

        worker = MESHWorker(mesh_config, logger)
        assert worker is not None

    def test_mesh_optimizer_creates_mesh_parameter_manager(self, mesh_config, logger, temp_dir):
        """Test that MESH optimizer creates MESHParameterManager."""
        from symfluence.models.mesh.calibration.optimizer import MESHModelOptimizer
        from symfluence.optimization.parameter_managers import MESHParameterManager

        optimizer = MESHModelOptimizer(mesh_config, logger, temp_dir)
        param_manager = optimizer._create_parameter_manager()

        assert isinstance(param_manager, MESHParameterManager)


class TestRegistryIntegrity:
    """Test that all expected optimizers are registered."""

    def test_all_optimizers_registered(self):
        """Test that all expected model optimizers are registered."""
        expected_optimizers = ['SUMMA', 'FUSE', 'NGEN', 'HYPE', 'MESH']

        for model in expected_optimizers:
            optimizer_cls = OptimizerRegistry.get_optimizer(model)
            assert optimizer_cls is not None, f"{model} optimizer not registered"

    def test_all_workers_registered(self):
        """Test that all expected workers are registered."""
        expected_workers = ['SUMMA', 'FUSE', 'NGEN', 'HYPE', 'MESH']

        for model in expected_workers:
            worker_cls = OptimizerRegistry.get_worker(model)
            assert worker_cls is not None, f"{model} worker not registered"
