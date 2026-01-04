"""
HYPE Model Optimizer

HYPE-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with HYPE.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..optimizers.base_model_optimizer import BaseModelOptimizer
from ..workers.hype_worker import HYPEWorker
from ..registry import OptimizerRegistry


@OptimizerRegistry.register_optimizer('HYPE')
class HYPEModelOptimizer(BaseModelOptimizer):
    """
    HYPE-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution

    Example:
        optimizer = HYPEModelOptimizer(config, logger)
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
        Initialize HYPE optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.info(f"HYPEModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'HYPE'

    def _create_parameter_manager(self):
        """Create HYPE parameter manager."""
        from ..parameter_managers import HYPEParameterManager
        return HYPEParameterManager(
            self.config,
            self.logger,
            self.optimization_settings_dir
        )

    def _create_calibration_target(self):
        """Create HYPE calibration target based on configuration."""
        from ..calibration_targets import (
            StreamflowTarget, MultivariateTarget
        )

        target_type = self.config.get('OPTIMIZATION_TARGET', 'streamflow').lower()

        if target_type == 'multivariate':
            return MultivariateTarget(self.config, self.project_dir, self.logger)
        else:
            return StreamflowTarget(self.config, self.project_dir, self.logger)

    def _create_worker(self) -> HYPEWorker:
        """Create HYPE worker."""
        return HYPEWorker(self.config, self.logger)

    def _setup_parallel_dirs(self) -> None:
        """Setup HYPE-specific parallel directories."""
        base_dir = self.project_dir / 'simulations' / f'run_{self.experiment_id}'
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'HYPE',
            self.experiment_id
        )

        # Copy HYPE settings to each parallel directory
        source_settings = self.project_dir / 'settings' / 'HYPE'
        if source_settings.exists():
            self.copy_base_settings(source_settings, self.parallel_dirs, 'HYPE')
