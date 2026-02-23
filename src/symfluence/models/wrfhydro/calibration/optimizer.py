"""
WRF-Hydro Model Optimizer

WRF-Hydro-specific optimizer inheriting from BaseModelOptimizer.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry

from .worker import WRFHydroWorker  # noqa: F401 - Import to trigger worker registration


@OptimizerRegistry.register_optimizer('WRFHYDRO')
class WRFHydroModelOptimizer(BaseModelOptimizer):
    """
    WRF-Hydro-specific optimizer using the unified BaseModelOptimizer framework.

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

        self.wrfhydro_setup_dir = self.project_dir / 'settings' / 'WRFHYDRO'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("WRFHydroModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'WRFHYDRO'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to WRF-Hydro hydro namelist file."""
        hydro_namelist = self._get_config_value(
            lambda: self.config.model.wrfhydro.hydro_namelist,
            default='hydro.namelist',
            dict_key='WRFHYDRO_HYDRO_NAMELIST'
        )
        return self.wrfhydro_setup_dir / hydro_namelist

    def _create_parameter_manager(self):
        """Create WRF-Hydro parameter manager."""
        from .parameter_manager import WRFHydroParameterManager
        return WRFHydroParameterManager(
            self.config,
            self.logger,
            self.wrfhydro_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """Determine if external routing is needed for WRF-Hydro.

        WRF-Hydro has built-in channel routing (Muskingum-Cunge or diffusive wave),
        so external routing is never needed.
        """
        return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run WRF-Hydro for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.wrfhydro_setup_dir)

        success = self.worker.run_model(
            self.config,
            self.wrfhydro_setup_dir,
            output_dir
        )

        if success:
            self.logger.info(f"WRF-Hydro final evaluation output in {output_dir}")

        return success

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Run final evaluation using WRF-Hydro worker metrics."""
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)

        try:
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            if not self._run_model_for_final_evaluation(final_output_dir):
                self.logger.error("WRF-Hydro run failed during final evaluation")
                return None

            metrics = self.worker.calculate_metrics(
                final_output_dir, self.config,
                sim_dir=final_output_dir
            )

            if not metrics or metrics.get('kge', -999) <= -999:
                self.logger.error("Failed to calculate final evaluation metrics")
                return None

            calib_metrics = {"KGE_Calib": metrics.get('kge', -999)}
            eval_metrics = {"KGE_Eval": metrics.get('kge', -999)}

            final_result = {
                'final_metrics': metrics,
                'calibration_metrics': calib_metrics,
                'evaluation_metrics': eval_metrics,
                'success': True,
                'best_params': best_params
            }

            self.logger.info(f"Final evaluation KGE: {metrics.get('kge', 'N/A')}")
            return final_result

        except Exception as e:
            self.logger.error(f"Error in final evaluation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
