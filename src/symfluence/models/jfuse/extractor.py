"""
jFUSE Result Extractor.

Provides utilities for extracting and analyzing jFUSE model results
beyond standard streamflow extraction.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import logging

import numpy as np
import pandas as pd
import xarray as xr


class JFUSEResultExtractor:
    """
    Utility class for extracting jFUSE simulation results.

    Provides methods for extracting:
    - Streamflow timeseries
    - State variables (if saved)
    - Model performance metrics
    - Parameter values used in simulation
    """

    def __init__(
        self,
        output_dir: Path,
        domain_name: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize result extractor.

        Args:
            output_dir: Directory containing jFUSE output files
            domain_name: Name of the domain
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.domain_name = domain_name
        self.logger = logger or logging.getLogger(__name__)

    def extract_streamflow(
        self,
        convert_to_cms: bool = False,
        catchment_area_km2: Optional[float] = None
    ) -> Optional[pd.DataFrame]:
        """
        Extract simulated streamflow.

        Args:
            convert_to_cms: Convert from mm/day to m3/s
            catchment_area_km2: Catchment area for unit conversion

        Returns:
            DataFrame with datetime index and streamflow column
        """
        # Try NetCDF first
        nc_file = self.output_dir / f"{self.domain_name}_jfuse_output.nc"
        if nc_file.exists():
            ds = xr.open_dataset(nc_file)
            streamflow_cms = ds['streamflow'].values  # Already in m3/s
            time = pd.to_datetime(ds.time.values)

            # Also get runoff if available (in mm/day)
            runoff_mm_day = ds['runoff'].values if 'runoff' in ds else None
            ds.close()

            df = pd.DataFrame({
                'streamflow_cms': streamflow_cms
            }, index=time)

            # Add runoff in mm/day if available
            if runoff_mm_day is not None:
                df['streamflow_mm_day'] = runoff_mm_day

            df.index.name = 'datetime'
            return df

        # Try CSV
        csv_file = self.output_dir / f"{self.domain_name}_jfuse_output.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file, index_col='datetime', parse_dates=True)
            return df

        self.logger.error(f"No output file found in {self.output_dir}")
        return None

    def extract_distributed_runoff(self) -> Optional[xr.Dataset]:
        """
        Extract distributed runoff (for routing).

        Returns:
            xarray Dataset with runoff per HRU
        """
        patterns = [
            f"{self.domain_name}_*_runs_def.nc",
            "*_runs_def.nc",
            f"{self.domain_name}_jfuse_output_distributed.nc"
        ]

        for pattern in patterns:
            matches = list(self.output_dir.glob(pattern))
            if matches:
                return xr.open_dataset(matches[0])

        self.logger.error("No distributed output file found")
        return None

    def extract_states(self) -> Optional[pd.DataFrame]:
        """
        Extract saved state variables (if available).

        Returns:
            DataFrame with state variables or None
        """
        states_file = self.output_dir / f"{self.domain_name}_jfuse_states.nc"
        if not states_file.exists():
            self.logger.info("No state file found (states may not have been saved)")
            return None

        ds = xr.open_dataset(states_file)

        # Extract state variables (jFUSE uses S1, S2, snow, etc.)
        states = {}
        for var in ['S1', 'S2', 'snow', 'soil_moisture']:
            if var in ds:
                states[var] = ds[var].values

        if not states:
            return None

        time = pd.to_datetime(ds.time.values)
        df = pd.DataFrame(states, index=time)
        df.index.name = 'datetime'

        ds.close()
        return df

    def calculate_metrics(
        self,
        observations: pd.Series,
        warmup_days: int = 365
    ) -> Dict[str, float]:
        """
        Calculate performance metrics against observations.

        Args:
            observations: Observed streamflow series (same units as simulation)
            warmup_days: Days to exclude from metric calculation

        Returns:
            Dictionary with metric values
        """
        from symfluence.evaluation.metrics import kge, nse, pbias, rmse

        # Load simulation
        sim_df = self.extract_streamflow()
        if sim_df is None:
            return {'error': 'No simulation found'}

        # Use cms for metrics
        sim_series = sim_df['streamflow_cms']

        # Skip warmup
        if len(sim_series) > warmup_days:
            sim_series = sim_series.iloc[warmup_days:]

        # Align time series
        common_idx = sim_series.index.intersection(observations.index)
        if len(common_idx) < 10:
            return {'error': f'Insufficient overlap ({len(common_idx)} points)'}

        sim_aligned = sim_series.loc[common_idx].values
        obs_aligned = observations.loc[common_idx].values

        # Remove NaN
        valid_mask = ~(np.isnan(sim_aligned) | np.isnan(obs_aligned))
        sim_aligned = sim_aligned[valid_mask]
        obs_aligned = obs_aligned[valid_mask]

        if len(sim_aligned) == 0:
            return {'error': 'No valid data pairs'}

        # Calculate metrics
        metrics = {
            'kge': float(kge(obs_aligned, sim_aligned, transfo=1)),
            'nse': float(nse(obs_aligned, sim_aligned, transfo=1)),
            'pbias': float(pbias(obs_aligned, sim_aligned)),
            'rmse': float(rmse(obs_aligned, sim_aligned)),
            'n_points': len(sim_aligned),
        }

        # KGE components
        r = np.corrcoef(obs_aligned, sim_aligned)[0, 1]
        alpha = np.std(sim_aligned) / (np.std(obs_aligned) + 1e-10)
        beta = np.mean(sim_aligned) / (np.mean(obs_aligned) + 1e-10)

        metrics['kge_r'] = float(r)
        metrics['kge_alpha'] = float(alpha)
        metrics['kge_beta'] = float(beta)

        return metrics

    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the simulation.

        Returns:
            Dictionary with summary information
        """
        sim_df = self.extract_streamflow()
        if sim_df is None:
            return {'error': 'No simulation found'}

        streamflow = sim_df['streamflow_cms']

        summary = {
            'start_date': str(streamflow.index.min()),
            'end_date': str(streamflow.index.max()),
            'n_timesteps': len(streamflow),
            'mean_streamflow_cms': float(streamflow.mean()),
            'max_streamflow_cms': float(streamflow.max()),
            'min_streamflow_cms': float(streamflow.min()),
            'std_streamflow_cms': float(streamflow.std()),
        }

        # Add mm/day stats if available
        if 'streamflow_mm_day' in sim_df.columns:
            runoff = sim_df['streamflow_mm_day']
            summary['mean_runoff_mm_day'] = float(runoff.mean())
            summary['total_runoff_mm'] = float(runoff.sum())

        # Check for missing values
        n_missing = streamflow.isna().sum()
        if n_missing > 0:
            summary['n_missing'] = int(n_missing)
            summary['pct_missing'] = float(n_missing / len(streamflow) * 100)

        return summary

    def compare_runs(
        self,
        other_dir: Path,
        other_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare results with another jFUSE run.

        Args:
            other_dir: Directory of the other run
            other_domain: Domain name for other run (defaults to same)

        Returns:
            Dictionary with comparison statistics
        """
        other_extractor = JFUSEResultExtractor(
            other_dir,
            other_domain or self.domain_name,
            self.logger
        )

        # Extract both
        this_df = self.extract_streamflow()
        other_df = other_extractor.extract_streamflow()

        if this_df is None or other_df is None:
            return {'error': 'Could not load both simulations'}

        # Align
        common_idx = this_df.index.intersection(other_df.index)
        if len(common_idx) == 0:
            return {'error': 'No overlapping time period'}

        this_q = this_df.loc[common_idx, 'streamflow_cms'].values
        other_q = other_df.loc[common_idx, 'streamflow_cms'].values

        # Calculate differences
        diff = this_q - other_q
        rel_diff = diff / (other_q + 1e-10) * 100

        comparison = {
            'n_common_timesteps': len(common_idx),
            'mean_diff_cms': float(np.mean(diff)),
            'max_diff_cms': float(np.max(np.abs(diff))),
            'rmse_cms': float(np.sqrt(np.mean(diff ** 2))),
            'mean_rel_diff_pct': float(np.mean(rel_diff)),
            'correlation': float(np.corrcoef(this_q, other_q)[0, 1]),
        }

        return comparison

    def compare_model_structures(
        self,
        structure_dirs: Dict[str, Path]
    ) -> pd.DataFrame:
        """
        Compare results from different jFUSE model structures.

        Args:
            structure_dirs: Dictionary mapping structure name to output directory

        Returns:
            DataFrame with comparison metrics for each structure
        """
        results = []

        for structure_name, output_dir in structure_dirs.items():
            extractor = JFUSEResultExtractor(output_dir, self.domain_name, self.logger)
            summary = extractor.get_simulation_summary()

            if 'error' not in summary:
                summary['structure'] = structure_name
                results.append(summary)

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results).set_index('structure')
