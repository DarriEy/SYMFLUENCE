"""
NextGen Model Optimizer

NGEN-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with NextGen.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..optimizers.base_model_optimizer import BaseModelOptimizer
from ..workers.ngen_worker import NgenWorker
from ..registry import OptimizerRegistry


@OptimizerRegistry.register_optimizer('NGEN')
class NgenModelOptimizer(BaseModelOptimizer):
    """
    NextGen-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution
    - run_adam(): Adam gradient-based optimization
    - run_lbfgs(): L-BFGS gradient-based optimization

    Example:
        optimizer = NgenModelOptimizer(config, logger)
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
        Initialize NextGen optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        # NGEN-specific paths
        self.ngen_sim_dir = self.project_dir / 'simulations' / self.experiment_id / 'NGEN'
        self.ngen_setup_dir = self.project_dir / 'settings' / 'NGEN'

        self.logger.info(f"NgenModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'NGEN'

    def _create_parameter_manager(self):
        """Create NGEN parameter manager."""
        from ..parameter_managers import NgenParameterManager
        return NgenParameterManager(
            self.config,
            self.logger,
            self.optimization_settings_dir
        )

    def _create_calibration_target(self):
        """Create NGEN calibration target based on configuration."""
        from ..calibration_targets import NgenStreamflowTarget

        target_type = self.config.get('OPTIMIZATION_TARGET', 'streamflow').lower()

        # Currently only streamflow is supported for NGEN
        if target_type not in ['streamflow', 'discharge', 'flow']:
            self.logger.warning(
                f"Unknown target {target_type} for NGEN, defaulting to streamflow"
            )

        return NgenStreamflowTarget(self.config, self.project_dir, self.logger)

    def _create_worker(self) -> NgenWorker:
        """Create NGEN worker."""
        return NgenWorker(self.config, self.logger)

    def _setup_parallel_dirs(self) -> None:
        """Setup NGEN-specific parallel directories."""
        base_dir = self.project_dir / 'simulations' / f'run_{self.experiment_id}'
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'NGEN',
            self.experiment_id
        )

        # Copy NGEN settings to each parallel directory
        if self.ngen_setup_dir.exists():
            self.copy_base_settings(self.ngen_setup_dir, self.parallel_dirs, 'NGEN')


# Backward compatibility alias
NgenOptimizer = NgenModelOptimizer
