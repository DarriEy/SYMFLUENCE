"""
mHM Model Optimizer

mHM-specific optimizer inheriting from BaseModelOptimizer.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import MHMWorker  # noqa: F401 - Import to trigger worker registration


@OptimizerRegistry.register_optimizer('MHM')
class MHMModelOptimizer(BaseModelOptimizer):
    """
    mHM-specific optimizer using the unified BaseModelOptimizer framework.

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

        self.mhm_setup_dir = self.project_dir / 'MHM_input' / 'settings'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("MHMModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'MHM'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to mHM namelist file."""
        namelist_file = self._get_config_value(
            lambda: self.config.model.mhm.namelist_file,
            default='mhm.nml',
            dict_key='MHM_NAMELIST_FILE'
        )
        return self.mhm_setup_dir / namelist_file

    def _create_parameter_manager(self):
        """Create mHM parameter manager."""
        from .parameter_manager import MHMParameterManager
        return MHMParameterManager(
            self.config,
            self.logger,
            self.mhm_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """Determine if external routing is needed for mHM.

        mHM has built-in mRM routing, so external routing is typically
        not needed unless explicitly configured.
        """
        routing_integration = self._get_config_value(
            lambda: self.config.model.mhm.routing_integration,
            default='none',
            dict_key='MHM_ROUTING_INTEGRATION'
        )
        global_routing = self._get_config_value(
            lambda: self.config.model.routing_model,
            default='none',
            dict_key='ROUTING_MODEL'
        )
        return (routing_integration.lower() == 'mizuroute' or
                global_routing.lower() == 'mizuroute')

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run mHM for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.mhm_setup_dir)

        return self.worker.run_model(
            self.config,
            self.mhm_setup_dir,
            output_dir
        )
