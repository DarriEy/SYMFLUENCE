# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Final Evaluation Runner

Orchestrates final model evaluation after optimization.
"""

import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .file_manager_updater import FileManagerUpdater
from .model_decisions_updater import ModelDecisionsUpdater
from .results_saver import FinalResultsSaver


class FinalEvaluationRunner:
    """
    Orchestrates final model evaluation after optimization.

    Coordinates:
    - File manager updates for full period
    - Model decisions updates (optional)
    - Parameter application
    - Model execution
    - Metric calculation
    - Result saving and restoration
    """

    def __init__(
        self,
        config: Dict[str, Any],
        results_dir: Path,
        settings_dir: Path,
        file_manager_path: Path,
        experiment_id: str,
        domain_name: str,
        calibration_target: Any,
        worker: Any,
        run_model_callback: Callable[[Path], bool],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize final evaluation runner.

        Args:
            config: Configuration dictionary
            results_dir: Results directory path
            settings_dir: Settings directory path
            file_manager_path: Path to file manager
            experiment_id: Experiment identifier
            domain_name: Domain name
            calibration_target: Calibration target instance
            worker: Worker instance for parameter application
            run_model_callback: Callback to run the model
            logger: Optional logger instance
        """
        self.config = config
        self.results_dir = results_dir
        self.settings_dir = settings_dir
        self.calibration_target = calibration_target
        self.worker = worker
        self.run_model_callback = run_model_callback
        self.logger = logger or logging.getLogger(__name__)

        # Initialize sub-components
        self.file_manager_updater = FileManagerUpdater(
            file_manager_path,
            config,
            logger
        )
        self.model_decisions_updater = ModelDecisionsUpdater(
            settings_dir,
            logger
        )
        self.results_saver = FinalResultsSaver(
            results_dir,
            experiment_id,
            domain_name,
            logger
        )

    def run(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Run final evaluation with best parameters over full period.

        This evaluates the calibrated model on both calibration and evaluation periods,
        providing comprehensive performance metrics.

        Args:
            best_params: Best parameters from optimization

        Returns:
            Dictionary with final metrics for both periods, or None if failed
        """
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)
        self.logger.info("Running model with best parameters over full simulation period...")

        try:
            # Update file manager for full period
            self.file_manager_updater.update_for_full_period()

            # Apply best parameters directly
            if not self._apply_best_parameters(best_params):
                self.logger.error("Failed to apply best parameters for final evaluation")
                return None

            # Setup output directory
            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            # Update file manager output path
            self.file_manager_updater.update_output_path(final_output_dir)

            # Run model
            if not self.run_model_callback(final_output_dir):
                self.logger.error("Model run failed during final evaluation")
                return None

            # Calculate metrics for both periods (calibration_only=False)
            metrics = self.calibration_target.calculate_metrics(
                final_output_dir,
                calibration_only=False
            )

            if not metrics:
                self.logger.error("Failed to calculate final evaluation metrics")
                return None

            # Extract period-specific metrics
            calib_metrics = self.results_saver.extract_period_metrics(metrics, 'Calib')
            eval_metrics = self.results_saver.extract_period_metrics(metrics, 'Eval')

            # Log detailed results
            self.results_saver.log_results(calib_metrics, eval_metrics)

            final_result = {
                'final_metrics': metrics,
                'calibration_metrics': calib_metrics,
                'evaluation_metrics': eval_metrics,
                'success': True,
                'best_params': best_params
            }

            return final_result

        except (ValueError, RuntimeError, IOError) as e:
            self.logger.error(f"Error in final evaluation: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
        finally:
            # Restore optimization settings
            self.model_decisions_updater.restore_for_optimization()
            self.file_manager_updater.restore_calibration_period()

    def _apply_best_parameters(self, best_params: Dict[str, float]) -> bool:
        """
        Apply best parameters for final evaluation.

        Args:
            best_params: Best parameters dictionary

        Returns:
            True if successful
        """
        try:
            return self.worker.apply_parameters(
                best_params,
                self.settings_dir,
                config=self.config
            )
        except (ValueError, RuntimeError, IOError) as e:
            self.logger.error(f"Error applying parameters for final evaluation: {e}")
            return False

    def save_results(
        self,
        final_result: Dict[str, Any],
        algorithm: str
    ) -> Optional[Path]:
        """
        Save final evaluation results.

        Args:
            final_result: Final evaluation results
            algorithm: Algorithm name

        Returns:
            Path to saved file
        """
        return self.results_saver.save_results(final_result, algorithm)
