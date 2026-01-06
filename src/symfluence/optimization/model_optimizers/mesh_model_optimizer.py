"""
MESH Model Optimizer

MESH-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with MESH.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..optimizers.base_model_optimizer import BaseModelOptimizer
from ..workers.mesh_worker import MESHWorker
from ..registry import OptimizerRegistry


@OptimizerRegistry.register_optimizer('MESH')
class MESHModelOptimizer(BaseModelOptimizer):
    """
    MESH-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution

    Example:
        optimizer = MESHModelOptimizer(config, logger)
        results_path = optimizer.run_dds()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize MESH optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.info(f"MESHModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'MESH'

    def _create_parameter_manager(self):
        """Create MESH parameter manager."""
        from ..parameter_managers import MESHParameterManager
        return MESHParameterManager(
            self.config,
            self.logger,
            self.optimization_settings_dir
        )

    def _create_calibration_target(self):
        """Create MESH calibration target based on configuration."""
        from ..calibration_targets import (
            StreamflowTarget, MultivariateTarget
        )

        target_type = self.config.get('OPTIMIZATION_TARGET', 'streamflow').lower()

        if target_type == 'multivariate':
            return MultivariateTarget(self.config, self.project_dir, self.logger)
        else:
            return StreamflowTarget(self.config, self.project_dir, self.logger)

    def _create_worker(self) -> MESHWorker:
        """Create MESH worker."""
        return MESHWorker(self.config, self.logger)

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run MESH for final evaluation."""
        return self.worker.run_model(
            self.config,
            self.project_dir / 'settings' / 'MESH',
            output_dir,
            mode='run_def'
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to MESH input file (similar to file manager)."""
        mesh_input = self.config.get('SETTINGS_MESH_INPUT', 'MESH_input_run_options.ini')
        if mesh_input == 'default':
            mesh_input = 'MESH_input_run_options.ini'
        return self.project_dir / 'settings' / 'MESH' / mesh_input

    def _setup_parallel_dirs(self) -> None:
        """Setup MESH-specific parallel directories."""
        base_dir = self.project_dir / 'simulations' / f'run_{self.experiment_id}'
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'MESH',
            self.experiment_id
        )

        # Copy MESH forcing directory to each parallel directory
        source_forcing = self.project_dir / 'forcing' / 'MESH_input'
        if source_forcing.exists():
            import shutil
            for parallel_dir in self.parallel_dirs:
                dest_forcing = parallel_dir / 'forcing' / 'MESH_input'
                dest_forcing.parent.mkdir(parents=True, exist_ok=True)
                if dest_forcing.exists():
                    shutil.rmtree(dest_forcing)
                shutil.copytree(source_forcing, dest_forcing)
                self.logger.debug(f"Copied MESH forcing to {dest_forcing}")
