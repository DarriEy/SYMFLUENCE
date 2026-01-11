"""
RHESSys Model Optimizer

RHESSys-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with RHESSys.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from ..optimizers.base_model_optimizer import BaseModelOptimizer
from ..workers.rhessys_worker import RHESSysWorker
from ..registry import OptimizerRegistry


@OptimizerRegistry.register_optimizer('RHESSys')
class RHESSysModelOptimizer(BaseModelOptimizer):
    """
    RHESSys-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution

    Example:
        optimizer = RHESSysModelOptimizer(config, logger)
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
        Initialize RHESSys optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)
        self.logger.debug("RHESSysModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'RHESSys'

    def _create_parameter_manager(self):
        """Create RHESSys parameter manager."""
        from ..parameter_managers.rhessys_parameter_manager import RHESSysParameterManager
        return RHESSysParameterManager(
            self.config,
            self.logger,
            self.optimization_settings_dir
        )

    def _create_calibration_target(self):
        """Create RHESSys calibration target using registry-based factory.

        Uses the centralized create_calibration_target factory which:
        1. Checks OptimizerRegistry for registered targets
        2. Falls back to model-specific target mappings
        3. Returns appropriate default targets if not found
        """
        from ..calibration_targets import create_calibration_target

        target_type = self.config.get('OPTIMIZATION_TARGET', 'streamflow').lower()

        return create_calibration_target(
            model_name='RHESSYS',
            target_type=target_type,
            config=self.config,
            project_dir=self.project_dir,
            logger=self.logger
        )

    def _create_worker(self) -> RHESSysWorker:
        """Create RHESSys worker."""
        return RHESSysWorker(self.config, self.logger)

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run RHESSys for final evaluation."""
        rhessys_input_dir = self.project_dir / 'RHESSys_input'
        return self.worker.run_model(
            self.config,
            rhessys_input_dir,
            output_dir
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to RHESSys world header file."""
        domain_name = self.config.get('DOMAIN_NAME')
        return self.project_dir / 'RHESSys_input' / 'worldfiles' / f'{domain_name}.world.hdr'

    def _setup_parallel_dirs(self) -> None:
        """Setup RHESSys-specific parallel directories."""
        algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'optimization').lower()
        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir.absolute(),
            'RHESSys',
            self.experiment_id
        )

        # Copy RHESSys definition files to each parallel directory
        source_defs = self.project_dir / 'RHESSys_input' / 'defs'
        if source_defs.exists():
            self._copy_defs_to_parallel_dirs(source_defs)

        # Set default_sim_dir to first proc's sim directory
        if self.parallel_dirs:
            first_proc = min(self.parallel_dirs.keys())
            self.default_sim_dir = self.parallel_dirs[first_proc]['sim_dir']

    def _copy_defs_to_parallel_dirs(self, source_defs: Path) -> None:
        """
        Copy RHESSys definition files to parallel directories.

        Each parallel worker needs its own copy of .def files for parameter modification.

        Args:
            source_defs: Path to source definition files
        """
        for proc_id, dirs in self.parallel_dirs.items():
            # Create defs directory in settings_dir
            proc_defs = dirs['settings_dir'] / 'defs'
            proc_defs.mkdir(parents=True, exist_ok=True)

            # Copy all .def files
            for def_file in source_defs.glob('*.def'):
                shutil.copy2(def_file, proc_defs / def_file.name)

            self.logger.debug(f"Copied RHESSys defs to proc_{proc_id}: {proc_defs}")

    def _apply_best_parameters_for_final(self, best_params: Dict[str, float]) -> bool:
        """
        Apply best parameters for final evaluation.

        RHESSys-specific override: uses RHESSys_input directory for defs.
        """
        try:
            # RHESSys defs are in RHESSys_input/defs, not settings/RHESSys
            rhessys_input_dir = self.project_dir / 'RHESSys_input'
            return self.worker.apply_parameters(
                best_params,
                rhessys_input_dir,  # This directory should have defs/ subdirectory
                config=self.config
            )
        except Exception as e:
            self.logger.error(f"Error applying parameters for final evaluation: {e}")
            return False
