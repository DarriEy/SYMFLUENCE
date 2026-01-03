"""
FUSE Model Optimizer

FUSE-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with FUSE.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..optimizers.base_model_optimizer import BaseModelOptimizer
from ..workers.fuse_worker import FUSEWorker
from ..registry import OptimizerRegistry


@OptimizerRegistry.register_optimizer('FUSE')
class FUSEModelOptimizer(BaseModelOptimizer):
    """
    FUSE-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution
    - run_adam(): Adam gradient-based optimization
    - run_lbfgs(): L-BFGS gradient-based optimization

    Example:
        optimizer = FUSEModelOptimizer(config, logger)
        results_path = optimizer.run_pso()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize FUSE optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        # FUSE-specific paths
        self.fuse_exe_path = self._get_fuse_executable_path()
        self.fuse_sim_dir = self.project_dir / 'simulations' / self.experiment_id / 'FUSE'
        self.fuse_setup_dir = self.project_dir / 'settings' / 'FUSE'

        self.logger.info(f"FUSEModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'FUSE'

    def _create_parameter_manager(self):
        """Create FUSE parameter manager."""
        from ..fuse_parameter_manager import FUSEParameterManager
        return FUSEParameterManager(
            self.config,
            self.logger,
            self.optimization_settings_dir
        )

    def _create_calibration_target(self):
        """Create FUSE calibration target based on configuration."""
        from ..fuse_calibration_targets import (
            FUSEStreamflowTarget, FUSESnowTarget
        )

        target_type = self.config.get('OPTIMIZATION_TARGET', 'streamflow').lower()

        if target_type in ['snow', 'swe', 'sca', 'snow_depth']:
            return FUSESnowTarget(self.config, self.project_dir, self.logger)
        else:
            return FUSEStreamflowTarget(self.config, self.project_dir, self.logger)

    def _create_worker(self) -> FUSEWorker:
        """Create FUSE worker."""
        return FUSEWorker(self.config, self.logger)

    def _get_fuse_executable_path(self) -> Path:
        """Get path to FUSE executable."""
        fuse_install = self.config.get('FUSE_INSTALL_PATH', 'default')
        if fuse_install == 'default':
            return self.data_dir / 'installs' / 'fuse' / 'bin' / 'fuse.exe'
        return Path(fuse_install) / 'fuse.exe'

    def _setup_parallel_dirs(self) -> None:
        """Setup FUSE-specific parallel directories."""
        base_dir = self.project_dir / 'simulations' / f'run_{self.experiment_id}'
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'FUSE',
            self.experiment_id
        )

        # Copy FUSE settings to each parallel directory
        if self.fuse_setup_dir.exists():
            self.copy_base_settings(self.fuse_setup_dir, self.parallel_dirs, 'FUSE')


# Backward compatibility alias
FUSEOptimizer = FUSEModelOptimizer
