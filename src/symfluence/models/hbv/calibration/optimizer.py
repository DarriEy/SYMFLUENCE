"""
HBV Model Optimizer

HBV-specific optimizer inheriting from BaseModelOptimizer.
Supports gradient-based optimization when JAX is available.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import HBVWorker  # noqa: F401 - Import to trigger worker registration


@OptimizerRegistry.register_optimizer('HBV')
class HBVModelOptimizer(BaseModelOptimizer):
    """
    HBV-specific optimizer using the unified BaseModelOptimizer framework.

    Supports:
    - Standard iterative optimization (DDS, PSO, SCE-UA, DE)
    - Gradient-based optimization with JAX autodiff
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        self.experiment_id = config.get('EXPERIMENT_ID')
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        self.hbv_setup_dir = self.project_dir / 'settings' / 'HBV'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("HBVModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'HBV'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to HBV configuration (dummy for HBV)."""
        # HBV doesn't use a file manager - it runs in-memory.
        # Return a placeholder path in the setup directory.
        return self.hbv_setup_dir / 'hbv_config.txt'

    def _create_parameter_manager(self):
        """Create HBV parameter manager."""
        from symfluence.models.hbv.calibration.parameter_manager import HBVParameterManager
        return HBVParameterManager(
            self.config,
            self.logger,
            self.hbv_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """Determine if routing is needed based on configuration."""
        routing_integration = self._get_config_value(
            lambda: self.config.model.hbv.routing_integration,
            default='none',
            dict_key='HBV_ROUTING_INTEGRATION'
        )
        global_routing = self._get_config_value(
            lambda: self.config.model.routing_model,
            default='none',
            dict_key='ROUTING_MODEL'
        )
        spatial_mode = self._get_config_value(
            lambda: self.config.model.hbv.spatial_mode,
            default='auto',
            dict_key='HBV_SPATIAL_MODE'
        )

        # Handle 'auto' mode - resolve from DOMAIN_DEFINITION_METHOD
        if spatial_mode in (None, 'auto', 'default'):
            domain_method = self._get_config_value(
                lambda: self.config.domain.definition_method,
                default='lumped',
                dict_key='DOMAIN_DEFINITION_METHOD'
            )
            if domain_method == 'delineate':
                spatial_mode = 'distributed'
            else:
                spatial_mode = 'lumped'

        if spatial_mode != 'distributed':
            return False

        return (routing_integration.lower() == 'mizuroute' or
                global_routing.lower() == 'mizuroute')

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run HBV for final evaluation using best parameters."""
        # Get best parameters from results if available
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        # Apply parameters first (required for worker.run_model)
        self.worker.apply_parameters(best_params, self.hbv_setup_dir)

        # For HBV, use the worker's run_model method with save_output=True
        return self.worker.run_model(
            self.config,
            self.hbv_setup_dir,
            output_dir,
            save_output=True  # Save output for calibration_target to read
        )
