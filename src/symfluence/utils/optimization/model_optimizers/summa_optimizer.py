"""
SUMMA Model Optimizer

SUMMA-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with SUMMA.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..optimizers.base_model_optimizer import BaseModelOptimizer
from ..workers.summa_worker import SUMMAWorker
from ..registry import OptimizerRegistry


@OptimizerRegistry.register_optimizer('SUMMA')
class SUMMAModelOptimizer(BaseModelOptimizer):
    """
    SUMMA-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution
    - run_adam(): Adam gradient-based optimization
    - run_lbfgs(): L-BFGS gradient-based optimization

    Example:
        optimizer = SUMMAModelOptimizer(config, logger)
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
        Initialize SUMMA optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        # SUMMA-specific paths
        self.summa_exe_path = self._get_summa_executable_path()
        self.mizuroute_exe_path = self._get_mizuroute_executable_path()

        # Determine if routing is needed
        self._routing_needed = self._check_routing_needed()

        self.logger.info(f"SUMMAModelOptimizer initialized")
        self.logger.info(f"Routing needed: {self._routing_needed}")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'SUMMA'

    def _create_parameter_manager(self):
        """Create SUMMA parameter manager."""
        from ..core.parameter_manager import ParameterManager
        return ParameterManager(
            self.config,
            self.logger,
            self.optimization_settings_dir
        )

    def _create_calibration_target(self):
        """Create SUMMA calibration target based on configuration."""
        from ..calibration_targets import (
            StreamflowTarget, ETTarget, SnowTarget,
            GroundwaterTarget, SoilMoistureTarget, MultivariateTarget
        )

        target_type = self.config.get('OPTIMIZATION_TARGET', 'streamflow').lower()

        if target_type == 'multivariate':
            return MultivariateTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['et', 'evapotranspiration']:
            return ETTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['snow', 'swe', 'sca']:
            return SnowTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['groundwater', 'gw']:
            return GroundwaterTarget(self.config, self.project_dir, self.logger)
        elif target_type in ['soil_moisture', 'sm']:
            return SoilMoistureTarget(self.config, self.project_dir, self.logger)
        else:
            return StreamflowTarget(self.config, self.project_dir, self.logger)

    def _create_worker(self) -> SUMMAWorker:
        """Create SUMMA worker."""
        return SUMMAWorker(self.config, self.logger)

    def _get_summa_executable_path(self) -> Path:
        """Get path to SUMMA executable."""
        summa_install = self.config.get('SUMMA_INSTALL_PATH', 'default')
        if summa_install == 'default':
            return self.data_dir / 'installs' / 'summa' / 'bin' / 'summa.exe'
        return Path(summa_install)

    def _get_mizuroute_executable_path(self) -> Path:
        """Get path to mizuRoute executable."""
        mizu_install = self.config.get('MIZUROUTE_INSTALL_PATH', 'default')
        if mizu_install == 'default':
            return self.data_dir / 'installs' / 'mizuroute' / 'bin' / 'mizuroute.exe'
        return Path(mizu_install)

    def _check_routing_needed(self) -> bool:
        """Determine if routing is needed based on configuration."""
        calibration_var = self.config.get('CALIBRATION_VARIABLE', 'streamflow')

        if calibration_var != 'streamflow':
            return False

        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
        routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')

        if domain_method not in ['point', 'lumped']:
            return True
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True

        return False

    @property
    def needs_routing(self) -> bool:
        """Check if routing is needed for this optimization."""
        return self._routing_needed

    def _setup_parallel_dirs(self) -> None:
        """Setup SUMMA-specific parallel directories."""
        base_dir = self.project_dir / 'simulations' / f'run_{self.experiment_id}'
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'SUMMA',
            self.experiment_id
        )

        # Copy SUMMA settings to each parallel directory
        source_settings = self.project_dir / 'settings' / 'SUMMA'
        if source_settings.exists():
            self.copy_base_settings(source_settings, self.parallel_dirs, 'SUMMA')

        # If routing needed, also copy mizuRoute settings
        if self._routing_needed:
            mizu_settings = self.project_dir / 'settings' / 'mizuRoute'
            if mizu_settings.exists():
                for proc_id, dirs in self.parallel_dirs.items():
                    mizu_dest = dirs['root'] / 'settings' / 'mizuRoute'
                    mizu_dest.mkdir(parents=True, exist_ok=True)
                    import shutil
                    for item in mizu_settings.iterdir():
                        if item.is_file():
                            shutil.copy2(item, mizu_dest / item.name)
