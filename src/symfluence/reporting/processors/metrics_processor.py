"""
Processor for calculating and managing hydrological performance metrics.

This module centralizes metrics calculation logic, supporting different
evaluation periods (calibration, validation) and multiple data formats.
"""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from typing import Dict, Optional, Tuple, Any
import logging

from symfluence.reporting.core.plot_utils import calculate_metrics, align_timeseries


class MetricsProcessor:
    """
    Handles calculation of performance metrics for model evaluation.

    Supports:
    - Standard metrics (RMSE, KGE, NSE, MAE, etc.)
    - Multiple evaluation periods (calibration, validation, full)
    - Data alignment and spinup removal
    - Consistent handling of missing values
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the metrics processor.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def calculate_for_period(
        self,
        obs: pd.Series,
        sim: pd.Series,
        period_start: Optional[pd.Timestamp] = None,
        period_end: Optional[pd.Timestamp] = None,
        spinup_days: Optional[int] = None,
        spinup_percent: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate metrics for a specific time period.

        Args:
            obs: Observed time series
            sim: Simulated time series
            period_start: Start date for evaluation (None = use all data)
            period_end: End date for evaluation (None = use all data)
            spinup_days: Number of spinup days to remove from beginning
            spinup_percent: Percentage of spinup to remove from beginning

        Returns:
            Dictionary of metric names and values

        Example:
            >>> processor = MetricsProcessor()
            >>> obs = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('2020-01-01', periods=5))
            >>> sim = pd.Series([1.1, 1.9, 3.2, 3.8, 5.1], index=pd.date_range('2020-01-01', periods=5))
            >>> metrics = processor.calculate_for_period(obs, sim)  # doctest: +SKIP
        """
        # Align time series
        obs_aligned, sim_aligned = align_timeseries(
            obs, sim, spinup_days=spinup_days, spinup_percent=spinup_percent
        )

        if obs_aligned.empty or sim_aligned.empty:
            self.logger.warning("No overlapping data after alignment")
            return self._empty_metrics()

        # Filter by period if specified
        if period_start is not None:
            obs_aligned = obs_aligned[obs_aligned.index >= period_start]
            sim_aligned = sim_aligned[sim_aligned.index >= period_start]

        if period_end is not None:
            obs_aligned = obs_aligned[obs_aligned.index <= period_end]
            sim_aligned = sim_aligned[sim_aligned.index <= period_end]

        if obs_aligned.empty or sim_aligned.empty:
            self.logger.warning("No data in specified period")
            return self._empty_metrics()

        # Calculate metrics
        try:
            metrics = calculate_metrics(obs_aligned.values, sim_aligned.values)
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return self._empty_metrics()

    def calculate_for_calibration_validation(
        self,
        obs: pd.Series,
        sim: pd.Series,
        calib_start: pd.Timestamp,
        calib_end: pd.Timestamp,
        valid_start: pd.Timestamp,
        valid_end: pd.Timestamp,
        spinup_days: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate metrics for both calibration and validation periods.

        Args:
            obs: Observed time series
            sim: Simulated time series
            calib_start: Calibration period start
            calib_end: Calibration period end
            valid_start: Validation period start
            valid_end: Validation period end
            spinup_days: Number of spinup days to remove

        Returns:
            Tuple of (calibration_metrics, validation_metrics)

        Example:
            >>> processor = MetricsProcessor()
            >>> # ... setup obs, sim ...
            >>> calib_metrics, valid_metrics = processor.calculate_for_calibration_validation(
            ...     obs, sim,
            ...     calib_start=pd.Timestamp('2015-01-01'),
            ...     calib_end=pd.Timestamp('2017-12-31'),
            ...     valid_start=pd.Timestamp('2018-01-01'),
            ...     valid_end=pd.Timestamp('2020-12-31')
            ... )  # doctest: +SKIP
        """
        # Calculate calibration metrics
        calib_metrics = self.calculate_for_period(
            obs, sim,
            period_start=calib_start,
            period_end=calib_end,
            spinup_days=spinup_days
        )

        # Calculate validation metrics
        valid_metrics = self.calculate_for_period(
            obs, sim,
            period_start=valid_start,
            period_end=valid_end,
            spinup_days=None  # No spinup for validation period
        )

        return calib_metrics, valid_metrics

    def calculate_for_multiple_models(
        self,
        obs: pd.Series,
        simulations: Dict[str, pd.Series],
        spinup_days: Optional[int] = None,
        spinup_percent: Optional[float] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for multiple model simulations against same observations.

        Args:
            obs: Observed time series
            simulations: Dictionary mapping model names to simulated time series
            spinup_days: Number of spinup days to remove
            spinup_percent: Percentage of spinup to remove

        Returns:
            Dictionary mapping model names to their metrics

        Example:
            >>> processor = MetricsProcessor()
            >>> obs = pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3))
            >>> sims = {
            ...     'Model A': pd.Series([1.1, 2.1, 3.1], index=pd.date_range('2020-01-01', periods=3)),
            ...     'Model B': pd.Series([0.9, 1.9, 2.9], index=pd.date_range('2020-01-01', periods=3))
            ... }
            >>> all_metrics = processor.calculate_for_multiple_models(obs, sims)  # doctest: +SKIP
        """
        results = {}

        for model_name, sim in simulations.items():
            try:
                metrics = self.calculate_for_period(
                    obs, sim,
                    spinup_days=spinup_days,
                    spinup_percent=spinup_percent
                )
                results[model_name] = metrics
            except Exception as e:
                self.logger.error(f"Error calculating metrics for {model_name}: {str(e)}")
                results[model_name] = self._empty_metrics()

        return results

    def rank_models_by_metric(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metric_name: str = 'NSE',
        ascending: bool = False
    ) -> list:
        """
        Rank models by a specific metric.

        Args:
            metrics_dict: Dictionary mapping model names to metrics
            metric_name: Name of metric to rank by (default: 'NSE')
            ascending: If True, rank ascending (lower is better); if False, descending (higher is better)

        Returns:
            List of (model_name, metric_value) tuples sorted by metric

        Example:
            >>> metrics = {
            ...     'Model A': {'NSE': 0.85, 'RMSE': 1.2},
            ...     'Model B': {'NSE': 0.92, 'RMSE': 0.8}
            ... }
            >>> processor = MetricsProcessor()
            >>> ranked = processor.rank_models_by_metric(metrics, metric_name='NSE')
            >>> ranked[0][0]  # Best model name
            'Model B'
        """
        # Extract metric values
        model_metrics = []
        for model_name, metrics in metrics_dict.items():
            if metric_name in metrics and not np.isnan(metrics[metric_name]):
                model_metrics.append((model_name, metrics[metric_name]))

        # Sort
        model_metrics.sort(key=lambda x: x[1], reverse=not ascending)

        return model_metrics

    def create_metrics_summary(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        precision: int = 3
    ) -> pd.DataFrame:
        """
        Create a DataFrame summary of metrics for multiple models.

        Args:
            metrics_dict: Dictionary mapping model names to metrics
            precision: Number of decimal places

        Returns:
            DataFrame with models as rows and metrics as columns

        Example:
            >>> metrics = {
            ...     'Model A': {'NSE': 0.85, 'RMSE': 1.2},
            ...     'Model B': {'NSE': 0.92, 'RMSE': 0.8}
            ... }
            >>> processor = MetricsProcessor()
            >>> df = processor.create_metrics_summary(metrics)  # doctest: +SKIP
        """
        df = pd.DataFrame(metrics_dict).T
        df = df.round(precision)
        return df

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        """Return dictionary with NaN for all standard metrics."""
        return {
            'RMSE': np.nan,
            'KGE': np.nan,
            'KGEp': np.nan,
            'NSE': np.nan,
            'MAE': np.nan,
            'KGEnp': np.nan
        }
