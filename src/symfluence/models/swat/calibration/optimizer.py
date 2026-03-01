# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SWAT Model Optimizer

SWAT-specific optimizer inheriting from BaseModelOptimizer.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

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

        # Standard layout: settings in settings/SWAT/, forcing in data/forcing/SWAT_input/
        self.swat_settings_dir = self.project_dir / 'settings' / 'SWAT'
        self.swat_forcing_dir = self.project_dir / 'data' / 'forcing' / 'SWAT_input'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("SWATModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'SWAT'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to SWAT settings directory."""
        return self.swat_settings_dir

    def _create_parameter_manager(self):
        """Create SWAT parameter manager."""
        from .parameter_manager import SWATParameterManager
        return SWATParameterManager(
            self.config,
            self.logger,
            self.swat_settings_dir
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
        """Run SWAT for final evaluation using best parameters.

        Assembles a complete TxtInOut (settings + forcing) in output_dir,
        applies the best parameters there, and runs SWAT.  This mirrors
        what calibration workers do and avoids running in the template
        settings directory which only contains empty forcing stubs.

        The SWAT arm64 binary is non-deterministic due to uninitialised
        Fortran variables, so we allow extra retries for the final
        evaluation (controlled by SWAT_FINAL_EVAL_RETRIES, default 10).
        """
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        max_retries = int(self.config.get('SWAT_FINAL_EVAL_RETRIES', 10))

        # Build a plain-dict config copy with the higher retry budget.
        # SymfluenceConfig is a frozen Pydantic model that does not support
        # item assignment, so we convert to a mutable dict first.
        if hasattr(self.config, 'to_dict'):
            eval_config = self.config.to_dict()
        elif isinstance(self.config, dict):
            eval_config = self.config.copy()
        else:
            eval_config = dict(self.config)
        eval_config['SWAT_MAX_RETRIES'] = max_retries

        # Apply parameters to output_dir — this copies fresh settings +
        # forcing files into output_dir then modifies parameters in-place,
        # keeping the template settings/SWAT/ directory untouched.
        self.worker.apply_parameters(
            best_params, output_dir, config=eval_config
        )

        success = self.worker.run_model(
            eval_config,
            output_dir,
            output_dir
        )

        return success

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Run final evaluation using SWAT worker metrics instead of base evaluator."""
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)

        try:
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            # Run model with best params and copy outputs
            if not self._run_model_for_final_evaluation(final_output_dir):
                self.logger.error("SWAT run failed during final evaluation")
                return None

            # Use worker's calculate_metrics (knows how to parse output.rch)
            metrics = self.worker.calculate_metrics(
                final_output_dir, self.config,
                sim_dir=final_output_dir
            )

            if not metrics or metrics.get('kge', -999) <= -999:
                self.logger.error("Failed to calculate final evaluation metrics")
                return None

            # Build result structure compatible with results_saver
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

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in final evaluation: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
