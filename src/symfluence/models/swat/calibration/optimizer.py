"""
SWAT Model Optimizer

SWAT-specific optimizer inheriting from BaseModelOptimizer.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import SWATWorker  # noqa: F401 - Import to trigger worker registration


@OptimizerRegistry.register_optimizer('SWAT')
class SWATModelOptimizer(BaseModelOptimizer):
    """
    SWAT-specific optimizer using the unified BaseModelOptimizer framework.

    Supports all standard optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution
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

        txtinout_name = config.get('SWAT_TXTINOUT_DIR', 'TxtInOut')
        self.swat_txtinout_dir = self.project_dir / 'SWAT_input' / txtinout_name

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("SWATModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'SWAT'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to SWAT TxtInOut directory (used as settings path)."""
        return self.swat_txtinout_dir

    def _create_parameter_manager(self):
        """Create SWAT parameter manager."""
        from .parameter_manager import SWATParameterManager
        return SWATParameterManager(
            self.config,
            self.logger,
            self.swat_txtinout_dir
        )

    def _check_routing_needed(self) -> bool:
        """Determine if routing is needed for SWAT.

        SWAT includes built-in channel routing, so external routing
        is typically not needed.
        """
        routing_integration = self._get_config_value(
            lambda: self.config.model.swat.routing_integration,
            default='none',
            dict_key='SWAT_ROUTING_INTEGRATION'
        )
        return routing_integration.lower() != 'none'

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run SWAT for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.swat_txtinout_dir)

        return self.worker.run_model(
            self.config,
            self.swat_txtinout_dir,
            output_dir
        )
