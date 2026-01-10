"""
FUSE Model Optimizer

FUSE-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with FUSE.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.core.file_utils import copy_file
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
        # Initialize FUSE-specific paths before super().__init__ 
        # because parent calls _setup_parallel_dirs()
        self.experiment_id = config.get('EXPERIMENT_ID')
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        self.fuse_sim_dir = self.project_dir / 'simulations' / self.experiment_id / 'FUSE'
        self.fuse_setup_dir = self.project_dir / 'settings' / 'FUSE'
        self.fuse_exe_path = self._get_fuse_executable_path_pre_init(config)
        self.fuse_id = config.get('FUSE_FILE_ID', self.experiment_id)

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug(f"FUSEModelOptimizer initialized")

    def _get_fuse_executable_path_pre_init(self, config: Dict[str, Any]) -> Path:
        """Helper to get FUSE executable path before full initialization."""
        fuse_install = config.get('FUSE_INSTALL_PATH', 'default')
        if fuse_install == 'default':
            return self.data_dir / 'installs' / 'fuse' / 'bin' / 'fuse.exe'
        return Path(fuse_install) / 'fuse.exe'

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'FUSE'

    def _create_parameter_manager(self):
        """Create FUSE parameter manager."""
        from ..parameter_managers import FUSEParameterManager
        return FUSEParameterManager(
            self.config,
            self.logger,
            self.fuse_setup_dir
        )

    def _create_calibration_target(self):
        """Create FUSE calibration target based on configuration."""
        from ..calibration_targets import (
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

    def _check_routing_needed(self) -> bool:
        """
        Determine if routing is needed for FUSE calibration.

        Returns:
            True if mizuRoute routing should be used
        """
        # Check FUSE routing integration setting
        routing_integration = self.config.get('FUSE_ROUTING_INTEGRATION', 'none')

        # If 'default', inherit from ROUTING_MODEL
        if routing_integration == 'default':
            routing_model = self.config.get('ROUTING_MODEL', 'none')
            routing_integration = 'mizuRoute' if routing_model == 'mizuRoute' else routing_integration

        if routing_integration != 'mizuRoute':
            return False

        # Check calibration variable (only streamflow calibration uses routing)
        calibration_var = self.config.get('CALIBRATION_VARIABLE', 'streamflow')
        if calibration_var != 'streamflow':
            return False

        # Check spatial mode and routing delineation
        spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')
        routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')

        # Distributed or semi-distributed modes need routing
        if spatial_mode in ['semi_distributed', 'distributed']:
            return True

        # Lumped with river network routing needs routing
        if spatial_mode == 'lumped' and routing_delineation == 'river_network':
            return True

        return False

    def _copy_default_initial_params_to_sce(self):
        """Helper to ensure para_sce.nc exists by copying para_def.nc."""
        if self.fuse_sim_dir.exists():
            default_params = self.fuse_sim_dir / f"{self.domain_name}_{self.fuse_id}_para_def.nc"
            sce_params = self.fuse_sim_dir / f"{self.domain_name}_{self.fuse_id}_para_sce.nc"
            if default_params.exists() and not sce_params.exists():
                copy_file(default_params, sce_params)
                self.logger.info("Initialized para_sce.nc from default parameters")

    def _apply_best_parameters_for_final(self, best_params: Dict[str, float]) -> bool:
        """
        Apply best parameters for final evaluation.

        Overrides base class to use param_manager which knows the correct
        FUSE parameter file path (in simulations dir, not settings dir).
        """
        try:
            # Use param_manager.update_model_files() which uses the correct path
            # (self.fuse_sim_dir / domain_name_fuse_id_para_def.nc)
            return self.param_manager.update_model_files(best_params)
        except Exception as e:
            self.logger.error(f"Error applying FUSE parameters for final evaluation: {e}")
            return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run FUSE for final evaluation."""
        self._copy_default_initial_params_to_sce()
        return self.worker.run_model(
            self.config,
            self.fuse_setup_dir,
            output_dir,
            mode='run_def'
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to FUSE file manager."""
        fuse_fm = self.config.get('SETTINGS_FUSE_FILEMANAGER', 'fm_catch.txt')
        if fuse_fm == 'default':
            fuse_fm = 'fm_catch.txt'
        return self.fuse_setup_dir / fuse_fm

    def _setup_parallel_dirs(self) -> None:
        """Setup FUSE-specific parallel directories."""
        # Use algorithm-specific directory (consistent with SUMMA)
        algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'optimization').lower()
        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'FUSE',
            self.experiment_id
        )

        # Copy FUSE settings to each parallel directory
        if self.fuse_setup_dir.exists():
            self.copy_base_settings(self.fuse_setup_dir, self.parallel_dirs, 'FUSE')

        # Copy parameter file to each parallel directory
        # This is critical for parallel workers to modify parameters in isolation
        fuse_id = self.config.get('FUSE_FILE_ID', self.experiment_id)
        param_file = self.fuse_sim_dir / f"{self.domain_name}_{fuse_id}_para_def.nc"
        
        if param_file.exists():
            for proc_id, dirs in self.parallel_dirs.items():
                dest_file = dirs['settings_dir'] / param_file.name
                try:
                    copy_file(param_file, dest_file)
                    self.logger.debug(f"Copied parameter file to {dest_file}")
                except Exception as e:
                    self.logger.error(f"Failed to copy parameter file to {dest_file}: {e}")
        else:
            self.logger.warning(f"Parameter file not found: {param_file} - Parallel workers will likely fail apply_parameters")

        # If routing needed, also copy and configure mizuRoute settings
        if self._check_routing_needed():
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
                    'FUSE',
                    self.experiment_id
                )
                self.logger.info("Copied and configured mizuRoute settings for parallel processes")


# Backward compatibility alias
FUSEOptimizer = FUSEModelOptimizer
