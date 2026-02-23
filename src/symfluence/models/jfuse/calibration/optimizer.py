"""
jFUSE Model Optimizer

jFUSE-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with jFUSE.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry

# Note: JFUSEWorker registration is triggered by calibration/__init__.py import,
# not here. Importing it here at module level causes a circular import when
# _register_optimizers() runs from model_optimizers/__init__.py.


@OptimizerRegistry.register_optimizer('JFUSE')
class JFUSEModelOptimizer(BaseModelOptimizer):
    """
    jFUSE-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution
    - run_adam(): Adam gradient-based optimization (native JAX gradients)
    - run_lbfgs(): L-BFGS gradient-based optimization (native JAX gradients)

    jFUSE is a JAX-based implementation of FUSE that supports automatic
    differentiation for gradient-based calibration methods.

    Example:
        optimizer = JFUSEModelOptimizer(config, logger)
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
        Initialize jFUSE optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        # Initialize jFUSE-specific paths before super().__init__
        # Store the raw config dict for passing to parameter manager
        self._raw_config = config if isinstance(config, dict) else {}
        _exp_id = config.get('EXPERIMENT_ID')  # noqa: F841
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        self.jfuse_setup_dir = self.project_dir / 'settings' / 'JFUSE'
        self.jfuse_forcing_dir = self.project_forcing_dir / 'JFUSE_input'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("JFUSEModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'JFUSE'

    def _create_parameter_manager(self):
        """Create jFUSE parameter manager."""
        from .parameter_manager import JFUSEParameterManager
        # Pass raw dict config so parameter manager can read flat YAML keys
        # (SymfluenceConfig doesn't preserve JFUSE_PARAMS_TO_CALIBRATE as a flat key)
        config_for_pm = self._raw_config if self._raw_config else self.config
        return JFUSEParameterManager(
            config_for_pm,
            self.logger,
            self.jfuse_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """
        Determine if routing is needed for jFUSE calibration.

        jFUSE can use internal routing or external mizuRoute.

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
            lambda: self.config.model.jfuse.spatial_mode,
            default='lumped',
            dict_key='JFUSE_SPATIAL_MODE'
        )

        # Distributed mode may need external routing
        if spatial_mode == 'distributed':
            # Check if internal routing is enabled
            enable_routing = self._get_config_value(
                lambda: self.config.model.jfuse.enable_routing,
                default=False,
                dict_key='JFUSE_ENABLE_ROUTING'
            )
            # If internal routing disabled, use external
            return not enable_routing

        return False

    def _apply_best_parameters_for_final(self, best_params: Dict[str, float]) -> bool:
        """
        Apply best parameters for final evaluation.

        For jFUSE, parameters are passed directly to the model during simulation.
        We must call worker.apply_parameters() to set the worker's _current_params,
        which will be used by run_model() in the final evaluation.
        Additionally, we update parameter files for record-keeping.
        """
        try:
            # Apply parameters to the worker (sets _current_params for run_model)
            if not self.worker.apply_parameters(
                best_params,
                self.jfuse_setup_dir,
                config=self.config
            ):
                self.logger.error("Failed to apply best parameters to jFUSE worker")
                return False

            # Also update parameter files for record-keeping
            self.param_manager.update_model_files(best_params)
            return True
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying jFUSE parameters for final evaluation: {e}")
            return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run jFUSE for final evaluation."""
        return self.worker.run_model(
            self.config,
            self.jfuse_setup_dir,
            output_dir
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to jFUSE configuration file (placeholder for jFUSE).

        jFUSE doesn't use a file manager in the same way as SUMMA/FUSE.
        It runs with in-memory parameters, so we return a placeholder
        file path that won't interfere with the base optimizer's
        file update operations (which check if the file exists).
        """
        return self.jfuse_setup_dir / 'jfuse_config.txt'

    def _setup_parallel_dirs(self) -> None:
        """Setup jFUSE-specific parallel directories."""
        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            default='optimization',
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM'
        ).lower()
        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'JFUSE',
            self.experiment_id
        )

        # Copy jFUSE settings to each parallel directory
        if self.jfuse_setup_dir.exists():
            self.copy_base_settings(self.jfuse_setup_dir, self.parallel_dirs, 'JFUSE')

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
                    'JFUSE',
                    self.experiment_id
                )
                self.logger.debug("Copied and configured mizuRoute settings for parallel processes")


# Backward compatibility alias
JFUSEOptimizer = JFUSEModelOptimizer
