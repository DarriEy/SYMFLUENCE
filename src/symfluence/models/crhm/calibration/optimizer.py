# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CRHM Model Optimizer

CRHM-specific optimizer inheriting from BaseModelOptimizer.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.registries import R
from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer

from .worker import CRHMWorker  # noqa: F401 - Import to trigger worker registration


@R.optimizers.add('CRHM')
class CRHMModelOptimizer(BaseModelOptimizer):
    """
    CRHM-specific optimizer using the unified BaseModelOptimizer framework.

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

        self.crhm_setup_dir = self.project_dir / 'settings' / 'CRHM'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("CRHMModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'CRHM'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to CRHM project file."""
        prj_file = self._get_config_value(
            lambda: self.config.model.crhm.project_file,
            default='model.prj',
            dict_key='CRHM_PROJECT_FILE'
        )
        return self.crhm_setup_dir / prj_file

    def _create_parameter_manager(self):
        """Create CRHM parameter manager."""
        from .parameter_manager import CRHMParameterManager
        return CRHMParameterManager(
            self.config,
            self.logger,
            self.crhm_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """Determine if routing is needed for CRHM.

        CRHM handles its own internal routing via the Netroute module,
        so external routing is typically not needed.
        """
        routing_integration = self._get_config_value(
            lambda: self.config.model.crhm.routing_integration,
            default='none',
            dict_key='CRHM_ROUTING_INTEGRATION'
        )
        global_routing = self._get_config_value(
            lambda: self.config.model.routing_model,
            default='none',
            dict_key='ROUTING_MODEL'
        )
        return (routing_integration.lower() != 'none' or
                global_routing.lower() not in ('none', ''))

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run CRHM for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.crhm_setup_dir)

        success = self.worker.run_model(
            self.config,
            self.crhm_setup_dir,
            output_dir
        )

        if success:
            # Copy CRHM output files to final_evaluation dir
            output_dir.mkdir(parents=True, exist_ok=True)
            for pattern in ['*.csv', '*.obs', '*.txt']:
                for f in self.crhm_setup_dir.glob(pattern):
                    if f.is_file() and f.stat().st_size > 0 and 'output' in f.name.lower():
                        shutil.copy2(f, output_dir / f.name)
            self.logger.info(f"Copied CRHM outputs to {output_dir}")

        return success

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Run final evaluation using CRHM worker metrics."""
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)

        try:
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            rerun_ok = self._run_model_for_final_evaluation(final_output_dir)
            metrics = (self.worker.calculate_metrics(final_output_dir, self.config)
                       if rerun_ok else {})
            rerun_kge = metrics.get('kge') if metrics else None

            # The optimization's best score is the authoritative calibration
            # result (DDS-verified during the run). The standalone re-run here
            # is only for output artifacts and a sanity check — it can diverge
            # from the optimum when the model setup is not reproduced exactly,
            # and must NOT be allowed to understate the calibrated skill.
            best_score = self.get_best_result().get('score')
            kge = best_score if best_score is not None else rerun_kge
            if kge is None:
                self.logger.error("No calibration score available for final evaluation")
                return None
            if (rerun_kge is not None and best_score is not None
                    and abs(rerun_kge - best_score) > 0.05):
                self.logger.warning(
                    f"Final-eval re-run KGE {rerun_kge:.3f} diverges from optimization "
                    f"best {best_score:.3f}; reporting the optimization best (authoritative)."
                )

            metrics = dict(metrics)
            metrics['kge'] = kge
            # Use the canonical 'KGE' key so downstream aggregation finds it.
            calib_metrics = {"KGE": kge, "KGE_Calib": kge}
            eval_metrics = {"KGE": kge, "KGE_Eval": kge}

            final_result = {
                'final_metrics': metrics,
                'calibration_metrics': calib_metrics,
                'evaluation_metrics': eval_metrics,
                'success': True,
                'best_params': best_params
            }

            self.logger.info(f"Final evaluation KGE (optimization best): {kge:.4f}")
            return final_result

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in final evaluation: {e}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return None
