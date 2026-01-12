#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RHESSys Calibration Targets

Provides calibration target classes for RHESSys ecosystem-hydrological model.
Handles RHESSys output formats (CSV results, basin.daily) with daily resampling
of observations to match RHESSys output frequency.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from symfluence.evaluation.evaluators import StreamflowEvaluator
from symfluence.evaluation.output_file_locator import OutputFileLocator
from symfluence.optimization.registry import OptimizerRegistry

if TYPE_CHECKING:
    pass


@OptimizerRegistry.register_calibration_target('RHESSys', 'streamflow')
class RHESSysStreamflowTarget(StreamflowEvaluator):
    """Streamflow calibration target for RHESSys ecosystem-hydrological model outputs.

    RHESSys is a spatially-distributed, dynamic ecosystem-hydrological model
    that simulates both water and energy balance. This target handles:
    1. CSV results: Formatted discharge data with streamflow_cms column
    2. basin.daily: RHESSys native text format with whitespace separation

    Key Features:
        - Automatic column detection for streamflow (streamflow_cms, discharge, Q)
        - Whitespace-separated basin.daily parsing
        - Date construction from year/month/day columns
        - Daily resampling of observations (RHESSys outputs daily values)

    Output Characteristics:
        - Format: Multiple formats (CSV, basin.daily, native RHESSys)
        - Frequency: Daily discharge values
        - Units: m³/s (cubic meters per second)
        - Spatial: Basin-scale outlet discharge
    """

    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load and resample observed streamflow to daily frequency.

        RHESSys outputs daily discharge values, so observations must be
        aggregated to daily frequency for valid comparison.

        Returns:
            Optional[pd.Series]: Daily-aggregated streamflow (m³/s) or None if load fails
        """
        try:
            obs_path = self.get_observed_data_path()
            obs_series = self._load_observed_data_from_path(obs_path)

            if obs_series is not None:
                # Resample to daily frequency (mean) to match RHESSys output frequency
                obs_daily = obs_series.resample('D').mean()
                self.logger.info(f"Resampled observations from {len(obs_series)} to {len(obs_daily)} daily values")
                return obs_daily

            return obs_series

        except Exception as e:
            self.logger.error(f"Error loading observed data: {str(e)}")
            return None

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Locate RHESSys streamflow output files (CSV or basin.daily).

        Args:
            sim_dir: Directory containing RHESSys simulation outputs

        Returns:
            List[Path]: Paths to RHESSys streamflow output file(s)
        """
        locator = OutputFileLocator(self.logger)
        return locator.find_rhessys_output(sim_dir, 'streamflow')

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract RHESSys streamflow with automatic format detection.

        Detects output format and dispatches to appropriate extraction method:
        - .csv files: Calls _extract_from_csv()
        - Other files: Calls _extract_from_basin_daily()

        Args:
            sim_files: List of RHESSys simulation output file(s)
            **kwargs: Additional parameters (unused)

        Returns:
            pd.Series: Daily streamflow time series (m³/s)

        Raises:
            ValueError: If no simulation files provided
        """
        if not sim_files:
            raise ValueError("No simulation files provided")

        sim_file = sim_files[0]
        self.logger.info(f"Extracting simulated streamflow from: {sim_file}")

        if sim_file.suffix == '.csv':
            return self._extract_from_csv(sim_file)
        else:
            return self._extract_from_basin_daily(sim_file)

    def _extract_from_csv(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from RHESSys CSV results file.

        Column Detection Priority:
        1. streamflow_cms: Standard RHESSys export (m³/s)
        2. discharge: Alternative naming
        3. streamflow: Generic naming
        4. First numeric column (fallback)

        Args:
            sim_file: Path to RHESSys results CSV file

        Returns:
            pd.Series: Daily streamflow time series (m³/s)

        Raises:
            ValueError: If no streamflow column found
        """
        df = pd.read_csv(sim_file, index_col=0, parse_dates=True)

        if 'streamflow_cms' in df.columns:
            q_sim = df['streamflow_cms']
        elif 'discharge' in df.columns:
            q_sim = df['discharge']
        elif 'streamflow' in df.columns:
            q_sim = df['streamflow']
        else:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                q_sim = df[numeric_cols[0]]
            else:
                raise ValueError(f"No streamflow column found in {sim_file}")

        q_sim.name = 'streamflow_cms'
        return q_sim

    def _extract_from_basin_daily(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from RHESSys basin.daily native output format.

        Column Detection Priority:
        1. Explicit columns: streamflow, Qout, discharge, streamflow_m3s, Q
        2. Contains 'stream' or 'flow': Case-insensitive pattern matching
        3. Column 'q': Exact match (common short notation)
        4. First numeric column: Fallback

        Args:
            sim_file: Path to RHESSys basin.daily file

        Returns:
            pd.Series: Daily streamflow time series (m³/s)

        Raises:
            ValueError: If no streamflow column found
        """
        # RHESSys basin daily format varies - try whitespace-separated
        df = pd.read_csv(sim_file, sep=r'\s+', comment='#')

        # Construct date from year/month/day columns
        if all(col in df.columns for col in ['year', 'month', 'day']):
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
            df.set_index('date', inplace=True)
        elif 'DATE' in df.columns:
            df['date'] = pd.to_datetime(df['DATE'])
            df.set_index('date', inplace=True)

        # Find streamflow column
        q_col = None
        for col in ['streamflow', 'Qout', 'discharge', 'streamflow_m3s', 'Q']:
            if col in df.columns:
                q_col = col
                break

        if q_col is None:
            # Look for any column containing 'stream' or 'flow'
            for col in df.columns:
                if 'stream' in col.lower() or 'flow' in col.lower() or col.lower() == 'q':
                    q_col = col
                    break

        if q_col is None:
            raise ValueError(f"No streamflow column found in {sim_file}")

        q_sim = df[q_col]
        q_sim.name = 'streamflow_cms'
        return q_sim

    def needs_routing(self) -> bool:
        """RHESSys handles its own routing internally."""
        return False


__all__ = ['RHESSysStreamflowTarget']
