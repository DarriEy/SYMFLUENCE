"""
VIC Model Optimizer

VIC-specific optimizer inheriting from BaseModelOptimizer.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry

from .worker import VICWorker  # noqa: F401 - Import to trigger worker registration


@OptimizerRegistry.register_optimizer('VIC')
class VICModelOptimizer(BaseModelOptimizer):
    """
    VIC-specific optimizer using the unified BaseModelOptimizer framework.

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
        # Extract config values before super().__init__ (which may reference them).
        # Supports both typed SymfluenceConfig and plain dict.
        if isinstance(config, dict):
            self.experiment_id = config.get('EXPERIMENT_ID')
            self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            self.domain_name = config.get('DOMAIN_NAME')
        else:
            self.experiment_id = config.domain.experiment_id
            self.data_dir = Path(config.system.data_dir)
            self.domain_name = config.domain.name
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        self.vic_setup_dir = self.project_dir / 'settings' / 'VIC'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("VICModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'VIC'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to VIC global parameter file."""
        global_file = self._get_config_value(
            lambda: self.config.model.vic.global_param_file,
            default='vic_global.txt',
            dict_key='VIC_GLOBAL_PARAM_FILE'
        )
        return self.vic_setup_dir / global_file

    def _create_parameter_manager(self):
        """Create VIC parameter manager."""
        from .parameter_manager import VICParameterManager
        return VICParameterManager(
            self.config,
            self.logger,
            self.vic_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """Determine if routing is needed for VIC."""
        routing_integration = self._get_config_value(
            lambda: self.config.model.vic.routing_integration,
            default='none',
            dict_key='VIC_ROUTING_INTEGRATION'
        )
        global_routing = self._get_config_value(
            lambda: self.config.model.routing_model,
            default='none',
            dict_key='ROUTING_MODEL'
        )
        return (routing_integration.lower() == 'mizuroute' or
                global_routing.lower() == 'mizuroute')

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run VIC for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.vic_setup_dir)

        return self.worker.run_model(
            self.config,
            self.vic_setup_dir,
            output_dir
        )
