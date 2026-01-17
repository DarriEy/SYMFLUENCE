"""
HYPE Model Optimizer

HYPE-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with HYPE.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.core.file_utils import copy_file
from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import HYPEWorker  # noqa: F401 - Import to trigger worker registration


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

        self.logger.debug("HYPEModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'HYPE'

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run HYPE for final evaluation."""
        return self.worker.run_model(
            self.config,
            self.project_dir / 'settings' / 'HYPE',
            output_dir,
            mode='run_def'
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to HYPE info file (similar to file manager)."""
        hype_info = self._get_config_value(lambda: self.config.model.hype.info_file, default='info.txt', dict_key='SETTINGS_HYPE_INFO')
        if hype_info == 'default':
            hype_info = 'info.txt'
        return self.project_dir / 'settings' / 'HYPE' / hype_info

    def _setup_parallel_dirs(self) -> None:
        """Setup HYPE-specific parallel directories."""
        algorithm = self._get_config_value(lambda: self.config.optimization.algorithm, default='optimization', dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM').lower()
        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir.absolute(),
            'HYPE',
            self.experiment_id
        )

        # Copy HYPE settings to each parallel directory
        source_settings = self.project_dir / 'settings' / 'HYPE'
        if source_settings.exists():
            self.copy_base_settings(source_settings.absolute(), self.parallel_dirs, 'HYPE')

        # Update HYPE info files with process-specific paths
        self.update_file_managers(
            self.parallel_dirs,
            'HYPE',
            self.experiment_id,
            self._get_config_value(lambda: self.config.model.hype.info_file, default='info.txt', dict_key='SETTINGS_HYPE_INFO')
        )

        # If routing needed, also copy and configure mizuRoute settings
        routing_model = self._get_config_value(lambda: self.config.model.routing_model, default='none', dict_key='ROUTING_MODEL')
        if routing_model == 'mizuRoute':
            mizu_settings = self.project_dir / 'settings' / 'mizuRoute'
            if mizu_settings.exists():
                for proc_id, dirs in self.parallel_dirs.items():
                    mizu_dest = dirs['root'] / 'settings' / 'mizuRoute'
                    mizu_dest.mkdir(parents=True, exist_ok=True)
                    for item in mizu_settings.iterdir():
                        if item.is_file():
                            copy_file(item, mizu_dest / item.name)

                # Update mizuRoute control files with process-specific paths
                self.update_mizuroute_controls(
                    self.parallel_dirs,
                    'HYPE',
                    self.experiment_id
                )
