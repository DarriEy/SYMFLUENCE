"""
cFUSE Model Optimizer

cFUSE-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with cFUSE.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import CFUSEWorker  # noqa: F401 - Import to trigger worker registration
from .targets import CFUSEStreamflowTarget  # noqa: F401 - Import to trigger target registration


@OptimizerRegistry.register_optimizer('CFUSE')
class CFUSEModelOptimizer(BaseModelOptimizer):
    """
    cFUSE-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution
    - run_adam(): Adam gradient-based optimization (native Enzyme AD gradients)
    - run_lbfgs(): L-BFGS gradient-based optimization (native Enzyme AD gradients)

    cFUSE is a PyTorch/Enzyme AD implementation of FUSE that supports automatic
    differentiation for gradient-based calibration methods.

    Example:
        optimizer = CFUSEModelOptimizer(config, logger)
        results_path = optimizer.run_adam()  # Use native gradients
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize cFUSE optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        # Initialize cFUSE-specific paths before super().__init__
        _exp_id = config.get('EXPERIMENT_ID')  # noqa: F841
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        self.cfuse_setup_dir = self.project_dir / 'settings' / 'CFUSE'
        self.cfuse_forcing_dir = self.project_dir / 'forcing' / 'CFUSE_input'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("CFUSEModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'CFUSE'

    def _create_parameter_manager(self):
        """Create cFUSE parameter manager."""
        from .parameter_manager import CFUSEParameterManager
        return CFUSEParameterManager(
            self.config,
            self.logger,
            self.cfuse_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """
        Determine if routing is needed for cFUSE calibration.

        cFUSE can use internal routing or external mizuRoute.

        Returns:
            True if external mizuRoute routing should be used
        """
        # Check routing configuration
        routing_model = self._get_config_value(
            lambda: self.config.model.routing_model,
            default='none',
            dict_key='ROUTING_MODEL'
        )

        if routing_model != 'mizuRoute':
            return False

        # Check spatial mode
        spatial_mode = self._get_config_value(
            lambda: self.config.model.cfuse.spatial_mode,
            default='lumped',
            dict_key='CFUSE_SPATIAL_MODE'
        )

        # Distributed mode may need external routing
        if spatial_mode == 'distributed':
            # Check if internal routing is enabled
            enable_routing = self._get_config_value(
                lambda: self.config.model.cfuse.enable_routing,
                default=False,
                dict_key='CFUSE_ENABLE_ROUTING'
            )
            # If internal routing disabled, use external
            return not enable_routing

        return False

    def _apply_best_parameters_for_final(self, best_params: Dict[str, float]) -> bool:
        """
        Apply best parameters for final evaluation.

        For cFUSE, parameters are passed directly to the model during simulation.
        We must call worker.apply_parameters() to set the worker's _current_params,
        which will be used by run_model() in the final evaluation.
        Additionally, we update parameter files for record-keeping.
        """
        try:
            # Apply parameters to the worker (sets _current_params for run_model)
            if not self.worker.apply_parameters(
                best_params,
                self.cfuse_setup_dir,
                config=self.config
            ):
                self.logger.error("Failed to apply best parameters to cFUSE worker")
                return False

            # Also update parameter files for record-keeping
            self.param_manager.update_model_files(best_params)
            return True
        except Exception as e:
            self.logger.error(f"Error applying cFUSE parameters for final evaluation: {e}")
            return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run cFUSE for final evaluation."""
        return self.worker.run_model(
            self.config,
            self.cfuse_setup_dir,
            output_dir
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to cFUSE configuration file (placeholder for cFUSE).

        cFUSE doesn't use a file manager in the same way as SUMMA/FUSE.
        It runs with in-memory parameters, so we return a placeholder
        file path that won't interfere with the base optimizer's
        file update operations (which check if the file exists).
        """
        return self.cfuse_setup_dir / 'cfuse_config.txt'

    def _setup_parallel_dirs(self) -> None:
        """Setup cFUSE-specific parallel directories."""
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            default='optimization',
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM'
        ).lower()
        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'CFUSE',
            self.experiment_id
        )

        # Copy cFUSE settings to each parallel directory
        if self.cfuse_setup_dir.exists():
            self.copy_base_settings(self.cfuse_setup_dir, self.parallel_dirs, 'CFUSE')

        # If external routing needed, also copy mizuRoute settings
        if self._check_routing_needed():
            mizu_settings = self.project_dir / 'settings' / 'mizuRoute'
            if mizu_settings.exists():
                from symfluence.core.file_utils import copy_file
                for proc_id, dirs in self.parallel_dirs.items():
                    mizu_dest = dirs['root'] / 'settings' / 'mizuRoute'
                    mizu_dest.mkdir(parents=True, exist_ok=True)
                    for item in mizu_settings.iterdir():
                        if item.is_file():
                            copy_file(item, mizu_dest / item.name)

                self.update_mizuroute_controls(
                    self.parallel_dirs,
                    'CFUSE',
                    self.experiment_id
                )
                self.logger.debug("Copied and configured mizuRoute settings for parallel processes")


# Backward compatibility alias
CFUSEOptimizer = CFUSEModelOptimizer
