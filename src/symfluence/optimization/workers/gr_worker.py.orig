#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GR Worker

Worker implementation for GR model optimization.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import xarray as xr
import numpy as np

from .base_worker import BaseWorker, WorkerTask, WorkerResult
from ..registry import OptimizerRegistry
from symfluence.models.gr.runner import GRRunner
from symfluence.evaluation.metrics import kge, nse, rmse, mae
from symfluence.core.constants import UnitConversion

logger = logging.getLogger(__name__)


@OptimizerRegistry.register_worker('GR')
class GRWorker(BaseWorker):
    """
    Parallel worker for GR model calibration.

    Handles isolated model execution during optimization, where each worker
    instance can run GR4J with a different parameter set concurrently.
    Unlike file-based models (SUMMA, FUSE), GR parameters are passed directly
    to the R runtime, so no file modification is needed in apply_parameters().

    GR4J Parameters:
        - X1: Production store capacity (mm) [100-1500]
        - X2: Groundwater exchange coefficient (mm) [-5 to 5]
        - X3: Routing store capacity (mm) [1-500]
        - X4: Unit hydrograph time base (days) [0.5-10]

    Output:
        - Simulated streamflow as NetCDF or CSV
        - Performance metrics (KGE, NSE, RMSE, MAE)

    Note:
        GR models require R and rpy2. The worker delegates to GRRunner
        which manages the R/Python interface.
    """

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters for GR model run.

        Unlike file-based models (SUMMA, FUSE), GR parameters are passed directly
        to the R runtime via rpy2, so no file modification is needed. This method
        is a no-op for GR models.

        Args:
            params: Dictionary of parameter names to values
                   (e.g., {'X1': 350.0, 'X2': 0.5, 'X3': 100.0, 'X4': 2.0})
            settings_dir: Path to model settings directory (unused for GR)
            **kwargs: Additional arguments (unused)

        Returns:
            Always returns True since no file operations are performed

        Note:
            Parameters are applied in run_model() method when calling GRRunner.
            This design allows thread-safe parallel execution without file conflicts.
        """
        return True

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Execute GR model with specified parameters.

        Creates a GRRunner instance and executes the GR4J model with parameters
        provided in kwargs. The runner handles the R/Python interface via rpy2
        and executes the airGR package functions.

        Args:
            config: Configuration dictionary containing model settings
            settings_dir: Path to settings directory for this worker instance
            output_dir: Path where model output should be saved
            **kwargs: Additional arguments including:
                     - params: Dictionary of GR parameters (X1, X2, X3, X4)

        Returns:
            True if model execution successful, False otherwise

        Raises:
            Exception: If R execution fails or output files not generated

        Note:
            - Requires R with airGR package installed
            - Parameters are passed directly to R runtime (no file modification)
            - Output directory is overridden for worker isolation
            - Each worker gets its own output directory to avoid conflicts
        """
        try:
            # Get parameters from kwargs (passed by _evaluate_once)
            # WorkerTask passes params as a direct argument to run_model in newer BaseWorker
            # but let's be safe and check both kwargs and params if it were passed explicitly
            params = kwargs.get('params')
            
            # If not in kwargs, it might be in task (if we're calling it from evaluate)
            # Actually, BaseWorker._evaluate_once calls run_model(task.config, task.settings_dir, task.output_dir, **task.additional_data)
            # So params is NOT passed by default in BaseWorker unless it's in additional_data.
            # We need to ensure params are passed.
            if params:
                self.logger.info(f"Worker received params: {params}")
            else:
                self.logger.warning("Worker run_model received NO params!")
            
            # Create a runner instance
            # We use the config provided in the task and pass settings_dir for isolation
            runner = GRRunner(config, self.logger, settings_dir=settings_dir)
            
            # Override output directory to the one provided for this worker
            runner.output_dir = output_dir
            runner.output_path = output_dir
            
            # Execute GR
            success_path = runner.run_gr(params=params)
            
            return success_path is not None
        except Exception as e:
            self.logger.error(f"Error running GR model in worker: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from GR model output.

        Uses GRStreamflowTarget to extract simulated streamflow, align with
        observations, and calculate performance metrics (KGE, NSE, RMSE, MAE).

        Args:
            output_dir: Path to directory containing GR model output
            config: Configuration dictionary with observation settings
            **kwargs: Additional arguments (unused)

        Returns:
            Dictionary of metric names to values, e.g.:
            {'kge': 0.75, 'nse': 0.72, 'rmse': 10.5, 'mae': 8.2}
            Returns {'kge': penalty_score} if calculation fails

        Note:
            - Automatically filters to calibration period specified in config
            - Handles unit conversions (mm/day to cms if needed)
            - Returns penalty score (-9999) if metrics cannot be calculated
            - Primary metric (KGE) used for optimization objective
        """
        try:
            from ..calibration_targets import GRStreamflowTarget
            
            # Resolve project directory
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            domain_name = config.get('DOMAIN_NAME')
            project_dir = data_dir / f"domain_{domain_name}"
            
            # Initialize target
            target = GRStreamflowTarget(config, project_dir, self.logger)
            
            # Calculate metrics (handles period filtering and unit conversion internally)
            metrics = target.calculate_metrics(
                output_dir,
                calibration_only=True
            )
            
            if metrics:
                return metrics
            else:
                self.logger.warning("GR calibration target returned empty metrics")
                return {'kge': self.penalty_score}

        except Exception as e:
            self.logger.error(f"Error calculating GR metrics via target: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return {'kge': self.penalty_score}

    def _get_observed_streamflow(self, config: Dict[str, Any]):
        """DEPRECATED: Use GRStreamflowTarget instead."""
        pass

    def _get_catchment_area(self, config: Dict[str, Any]) -> float:
        """DEPRECATED: Use GRStreamflowTarget instead."""
        return 1000.0

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for parallel execution."""
        return _evaluate_gr_parameters_worker(task_data)


def _evaluate_gr_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI execution.
    Naming matches convention for dynamic resolution in BaseModelOptimizer.
    """
    worker = GRWorker(config=task_data.get('config'))
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
