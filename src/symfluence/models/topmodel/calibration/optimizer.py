# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
TOPMODEL Optimizer.

TOPMODEL-specific optimizer inheriting from BaseModelOptimizer.
Supports DDS and other iterative optimization algorithms.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from symfluence.evaluation.metrics import calculate_all_metrics
from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry

from .worker import TopmodelWorker  # noqa: F401 - Import to trigger worker registration


@OptimizerRegistry.register_optimizer('TOPMODEL')
class TopmodelModelOptimizer(BaseModelOptimizer):
    """
    TOPMODEL-specific optimizer using the unified BaseModelOptimizer framework.

    Supports:
    - Standard iterative optimization (DDS, PSO, SCE-UA, DE)
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

        self.topmodel_setup_dir = self.project_dir / 'settings' / 'TOPMODEL'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("TopmodelModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'TOPMODEL'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to TOPMODEL configuration (placeholder for in-memory model)."""
        return self.topmodel_setup_dir / 'topmodel_config.txt'

    def _create_parameter_manager(self):
        """Create TOPMODEL parameter manager."""
        from symfluence.models.topmodel.calibration.parameter_manager import TopmodelParameterManager
        return TopmodelParameterManager(
            self.config,
            self.logger,
            self.topmodel_setup_dir
        )

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run TOPMODEL for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.topmodel_setup_dir)

        return self.worker.run_model(
            self.config,
            self.topmodel_setup_dir,
            output_dir,
            save_output=True
        )

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Run final evaluation with consistent warmup handling for TOPMODEL.

        Calculates separate metrics for calibration and evaluation periods,
        matching the in-memory warmup handling used during optimization.
        """
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)

        try:
            if not self.worker._initialized:
                if not self.worker.initialize():
                    self.logger.error("Failed to initialize TOPMODEL worker for final evaluation")
                    return None

            if not self.worker.apply_parameters(best_params, self.topmodel_setup_dir):
                self.logger.error("Failed to apply best parameters for final evaluation")
                return None

            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            runoff = self.worker._run_simulation(
                self.worker._forcing,
                best_params
            )

            self.worker.save_output_files(
                runoff[self.worker.warmup_days:],
                final_output_dir,
                self.worker._time_index[self.worker.warmup_days:] if self.worker._time_index is not None else None
            )

            # Get time index and observations
            time_index = self.worker._time_index
            observations = self.worker._observations

            if time_index is None or observations is None:
                self.logger.error("Missing time index or observations for metric calculation")
                return None

            # Parse calibration and evaluation periods
            calib_period = self._parse_period_config('calibration_period', 'CALIBRATION_PERIOD')
            eval_period = self._parse_period_config('evaluation_period', 'EVALUATION_PERIOD')

            # Calculate metrics for calibration period (with warmup skip)
            calib_metrics = self._calculate_period_metrics_inmemory(
                runoff, observations, time_index,
                calib_period, 'Calib',
                skip_warmup=True
            )

            # Calculate metrics for evaluation period (no warmup skip needed)
            eval_metrics = {}
            if eval_period[0] and eval_period[1]:
                eval_metrics = self._calculate_period_metrics_inmemory(
                    runoff, observations, time_index,
                    eval_period, 'Eval',
                    skip_warmup=False
                )

            # Combine all metrics
            all_metrics = {**calib_metrics, **eval_metrics}

            # Add unprefixed versions for backward compatibility
            for k, v in calib_metrics.items():
                unprefixed = k.replace('Calib_', '')
                if unprefixed not in all_metrics:
                    all_metrics[unprefixed] = v

            # Log results
            self._log_final_evaluation_results(
                {k: v for k, v in calib_metrics.items()},
                {k: v for k, v in eval_metrics.items()}
            )

            return {
                'final_metrics': all_metrics,
                'calibration_metrics': calib_metrics,
                'evaluation_metrics': eval_metrics,
                'success': True,
                'best_params': best_params,
                'output_dir': str(final_output_dir),
            }

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Final evaluation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _parse_period_config(self, attr_name: str, dict_key: str):
        """Parse a period configuration string into start/end timestamps."""
        period_str = self._get_config_value(
            lambda: getattr(self.config.domain, attr_name, ''),
            default='',
            dict_key=dict_key
        )
        if not period_str:
            return (None, None)

        try:
            dates = [d.strip() for d in period_str.split(',')]
            if len(dates) >= 2:
                return (pd.Timestamp(dates[0]), pd.Timestamp(dates[1]))
        except (ValueError, AttributeError) as e:
            self.logger.debug(f"Could not parse period string '{period_str}': {e}")
        return (None, None)

    def _calculate_period_metrics_inmemory(
        self,
        runoff: np.ndarray,
        observations: np.ndarray,
        time_index: pd.DatetimeIndex,
        period: tuple,
        prefix: str,
        skip_warmup: bool = True
    ) -> Dict[str, float]:
        """Calculate metrics for a specific period using in-memory data.

        Warmup is skipped from the full simulation start before filtering
        to the target period, matching the calibration behavior.
        """
        try:
            if skip_warmup and len(runoff) > self.worker.warmup_days:
                runoff = runoff[self.worker.warmup_days:]
                observations = observations[self.worker.warmup_days:]
                time_index = time_index[self.worker.warmup_days:]

            sim_series = pd.Series(runoff, index=time_index)
            obs_series = pd.Series(observations, index=time_index)

            if period[0] and period[1]:
                period_mask = (time_index >= period[0]) & (time_index <= period[1])
                sim_period = sim_series[period_mask]
                obs_period = obs_series[period_mask]
                self.logger.info(
                    f"{prefix} period: {period[0].date()} to {period[1].date()}, "
                    f"{len(sim_period)} points"
                )
            else:
                sim_period = sim_series
                obs_period = obs_series

            common_idx = sim_period.index.intersection(obs_period.index)
            if len(common_idx) == 0:
                self.logger.warning(f"No common indices for {prefix} period")
                return {}

            sim_aligned = sim_period.loc[common_idx].values
            obs_aligned = obs_period.loc[common_idx].values

            valid_mask = ~(np.isnan(sim_aligned) | np.isnan(obs_aligned))
            sim_valid = sim_aligned[valid_mask]
            obs_valid = obs_aligned[valid_mask]

            if len(sim_valid) < 10:
                self.logger.warning(f"Insufficient valid points for {prefix} metrics: {len(sim_valid)}")
                return {}

            metrics_result = calculate_all_metrics(
                pd.Series(obs_valid),
                pd.Series(sim_valid)
            )

            prefixed_metrics = {}
            for key, value in metrics_result.items():
                prefixed_metrics[f"{prefix}_{key}"] = value

            return prefixed_metrics

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating {prefix} metrics: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {}
