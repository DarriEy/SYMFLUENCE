"""
HBV Model Optimizer

HBV-specific optimizer inheriting from BaseModelOptimizer.
Supports gradient-based optimization when JAX is available.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from symfluence.evaluation.metrics import calculate_all_metrics
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

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Run final evaluation with consistent warmup handling for HBV.

        This override ensures that the final evaluation metrics are calculated
        using the same warmup handling as during calibration. The issue with the
        base implementation is that:

        1. During calibration: metrics are calculated in-memory, skipping the first
           `warmup_days` via array slicing
        2. During final eval (base): metrics are calculated from saved files with
           date-based filtering, but no warmup skipping from the calibration period

        This method uses the worker's in-memory data with consistent warmup handling
        to ensure calibration metrics match between optimization and final evaluation.

        Args:
            best_params: Best parameters from optimization

        Returns:
            Final evaluation results dict, or None if failed
        """
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)
        self.logger.info("Running model with best parameters over full simulation period...")

        try:
            # Ensure worker is initialized
            if not self.worker._initialized:
                if not self.worker.initialize():
                    self.logger.error("Failed to initialize HBV worker for final evaluation")
                    return None

            # Apply best parameters
            if not self.worker.apply_parameters(best_params, self.hbv_setup_dir):
                self.logger.error("Failed to apply best parameters for final evaluation")
                return None

            # Setup output directory
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            # Run model with best parameters
            runoff = self.worker._run_simulation(
                self.worker._forcing,
                best_params
            )

            # Save output for reference
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

            # Calculate metrics for calibration period with warmup handling
            calib_metrics = self._calculate_period_metrics_inmemory(
                runoff, observations, time_index,
                calib_period, 'Calib',
                skip_warmup=True  # Match calibration behavior
            )

            # Calculate metrics for evaluation period (no warmup skip - different period)
            eval_metrics = {}
            if eval_period[0] and eval_period[1]:
                eval_metrics = self._calculate_period_metrics_inmemory(
                    runoff, observations, time_index,
                    eval_period, 'Eval',
                    skip_warmup=False  # Evaluation period doesn't need warmup skip
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

            final_result = {
                'final_metrics': all_metrics,
                'calibration_metrics': calib_metrics,
                'evaluation_metrics': eval_metrics,
                'success': True,
                'best_params': best_params
            }

            return final_result

        except Exception as e:
            self.logger.error(f"Error in HBV final evaluation: {e}")
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

        This method ensures consistent warmup handling with the calibration process.

        IMPORTANT: Warmup is skipped from the FULL simulation period BEFORE
        filtering to the evaluation period. This matches how calibration works:
        1. Run full simulation (e.g., 2002-2009)
        2. Skip warmup from start (e.g., skip 365 days from 2002 → start at 2003)
        3. Filter to calibration period (e.g., 2004-2007)

        The calibration period is typically defined to START after warmup ends,
        so no additional warmup skip is needed within the period itself.

        Args:
            runoff: Full simulation runoff array (mm/day)
            observations: Full observation array (mm/day)
            time_index: Time index for the data
            period: Tuple of (start_timestamp, end_timestamp)
            prefix: Metric name prefix (e.g., 'Calib', 'Eval')
            skip_warmup: Whether to skip warmup days from FULL simulation start

        Returns:
            Dictionary of prefixed metrics
        """
        try:
            # Apply warmup skip from the FULL simulation FIRST (matching calibration)
            # This ensures we compare the same data as during optimization
            if skip_warmup and len(runoff) > self.worker.warmup_days:
                runoff = runoff[self.worker.warmup_days:]
                observations = observations[self.worker.warmup_days:]
                time_index = time_index[self.worker.warmup_days:]
                self.logger.debug(
                    f"Skipped {self.worker.warmup_days} warmup days from full simulation. "
                    f"Data now starts at {time_index[0]}"
                )

            # Create pandas Series for easier time-based filtering
            sim_series = pd.Series(runoff, index=time_index)
            obs_series = pd.Series(observations, index=time_index)

            # Filter to period if specified
            if period[0] and period[1]:
                period_mask = (time_index >= period[0]) & (time_index <= period[1])
                sim_period = sim_series[period_mask]
                obs_period = obs_series[period_mask]

                self.logger.debug(
                    f"{prefix} period: {period[0]} to {period[1]}, "
                    f"{len(sim_period)} points"
                )
            else:
                # No period specified, use all data (already had warmup skipped if requested)
                sim_period = sim_series
                obs_period = obs_series

            # Align and remove NaN
            common_idx = sim_period.index.intersection(obs_period.index)
            if len(common_idx) == 0:
                self.logger.warning(f"No common indices for {prefix} period")
                return {}

            sim_aligned = sim_period.loc[common_idx].values
            obs_aligned = obs_period.loc[common_idx].values

            # Remove NaN values
            valid_mask = ~(np.isnan(sim_aligned) | np.isnan(obs_aligned))
            sim_valid = sim_aligned[valid_mask]
            obs_valid = obs_aligned[valid_mask]

            if len(sim_valid) < 10:
                self.logger.warning(f"Insufficient valid points for {prefix} metrics: {len(sim_valid)}")
                return {}

            # Calculate metrics using the centralized metrics module
            metrics_result = calculate_all_metrics(
                pd.Series(obs_valid),
                pd.Series(sim_valid)
            )

            # Add prefix to metric names
            prefixed_metrics = {}
            for key, value in metrics_result.items():
                prefixed_metrics[f"{prefix}_{key}"] = value

            return prefixed_metrics

        except Exception as e:
            self.logger.error(f"Error calculating {prefix} metrics: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {}
