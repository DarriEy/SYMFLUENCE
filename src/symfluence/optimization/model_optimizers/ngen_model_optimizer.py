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
        # Initialize NGEN-specific paths BEFORE super().__init__() so they're available in _setup_parallel_dirs
        # Compute paths using config directly (same logic as BaseModelOptimizer)
        data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
        domain_name = config.get('DOMAIN_NAME', 'default')
        experiment_id = config.get('EXPERIMENT_ID', 'optimization')

        project_dir = data_dir / f"domain_{domain_name}"
        self.ngen_sim_dir = project_dir / 'simulations' / experiment_id / 'NGEN'
        self.ngen_setup_dir = project_dir / 'settings' / 'NGEN'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug(f"NgenModelOptimizer initialized")

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
        """Create NGEN calibration target using registry-based factory.

        Uses the centralized create_calibration_target factory which:
        1. Checks OptimizerRegistry for registered targets
        2. Falls back to model-specific target mappings
        3. Returns appropriate default targets if not found
        """
        from ..calibration_targets import create_calibration_target

        target_type = self.config.get('OPTIMIZATION_TARGET', 'streamflow').lower()

        # Currently only streamflow is supported for NGEN
        if target_type not in ['streamflow', 'discharge', 'flow']:
            self.logger.warning(
                f"Unknown target {target_type} for NGEN, defaulting to streamflow"
            )
            target_type = 'streamflow'

        return create_calibration_target(
            model_name='NGEN',
            target_type=target_type,
            config=self.config,
            project_dir=self.project_dir,
            logger=self.logger
        )

    def _create_worker(self) -> NgenWorker:
        """Create NGEN worker."""
        return NgenWorker(self.config, self.logger)

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run NGEN for final evaluation."""
        return self.worker.run_model(
            self.config,
            self.ngen_setup_dir,
            output_dir,
            mode='run_def'
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to NGEN realization file (similar to file manager)."""
        ngen_realization = self.config.get('SETTINGS_NGEN_REALIZATION', 'realization.json')
        if ngen_realization == 'default':
            ngen_realization = 'realization.json'
        return self.ngen_setup_dir / ngen_realization

    def _setup_parallel_dirs(self) -> None:
        """Setup NGEN-specific parallel directories."""
        # Use algorithm-specific directory (matching base_model_optimizer pattern)
        algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'optimization').lower()
        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'
        
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'NGEN',
            self.experiment_id
        )

        # Copy NGEN settings to each parallel directory
        # ngen_setup_dir is set to a placeholder initially, so only proceed if it's a valid Path
        if self.ngen_setup_dir is not None and self.ngen_setup_dir.exists():
            self.copy_base_settings(self.ngen_setup_dir, self.parallel_dirs, 'NGEN')


# Backward compatibility alias
NgenOptimizer = NgenModelOptimizer
