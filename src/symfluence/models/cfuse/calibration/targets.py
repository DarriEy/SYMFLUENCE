"""
cFUSE Calibration Targets.

Provides calibration targets (evaluators) for cFUSE model calibration.
These targets load observations from cFUSE-specific paths where the
preprocessor stores daily-resampled observations.
"""

from pathlib import Path

import pandas as pd

from symfluence.evaluation.evaluators.streamflow import StreamflowEvaluator
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_calibration_target('CFUSE', 'streamflow')
class CFUSEStreamflowTarget(StreamflowEvaluator):
    """
    Streamflow calibration target for cFUSE model.

    This target loads observations from the cFUSE-specific path where
    the preprocessor stores daily-resampled observations. This ensures
    the final evaluation uses the same observation data as the
    optimization worker.

    Key difference from base StreamflowEvaluator:
    - Loads observations from forcing/CFUSE_input/ instead of
      observations/streamflow/preprocessed/
    - The cFUSE preprocessor resamples hourly observations to daily
      to match the cFUSE model's daily timestep
    """

    def get_observed_data_path(self) -> Path:
        """Get path to cFUSE-specific daily observations.

        The cFUSE preprocessor creates a daily-resampled version of
        observations at: forcing/CFUSE_input/{domain}_observations.csv

        Returns:
            Path to cFUSE daily observations file
        """
        # Try cFUSE-specific path first
        cfuse_obs_path = (
            self.project_forcing_dir / "CFUSE_input" /
            f"{self.domain_name}_observations.csv"
        )

        if cfuse_obs_path.exists():
            return cfuse_obs_path

        # Fall back to default path if cFUSE path doesn't exist
        self.logger.warning(
            f"cFUSE observations not found at {cfuse_obs_path}, "
            "falling back to default path"
        )
        return super().get_observed_data_path()

    def extract_simulated_data(self, sim_files, **kwargs) -> pd.Series:  # type: ignore[override]
        """Extract simulated streamflow from cFUSE output.

        cFUSE outputs are stored as CSV and NetCDF with streamflow_cms column.
        This method looks for cFUSE-specific output files.

        Args:
            sim_files: List of simulation output files

        Returns:
            Pandas Series with simulated streamflow (m³/s)
        """
        # Try to find cFUSE output CSV
        output_dir = sim_files[0].parent if sim_files else None
        if output_dir is None:
            return super().extract_simulated_data(sim_files, **kwargs)

        # Look for cFUSE CSV output
        cfuse_csv = output_dir / f"{self.domain_name}_cfuse_output.csv"
        if cfuse_csv.exists():
            try:
                df = pd.read_csv(cfuse_csv)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
                return df['streamflow_cms']
            except Exception as e:  # noqa: BLE001 — calibration resilience
                self.logger.warning(f"Error reading cFUSE CSV output: {e}")

        # Look for cFUSE NetCDF output
        cfuse_nc = output_dir / f"{self.domain_name}_cfuse_output.nc"
        if cfuse_nc.exists():
            try:
                import xarray as xr
                ds = xr.open_dataset(cfuse_nc)
                sim_data = ds['streamflow'].to_pandas()
                ds.close()
                return sim_data  # type: ignore[return-value]
            except Exception as e:  # noqa: BLE001 — calibration resilience
                self.logger.warning(f"Error reading cFUSE NetCDF output: {e}")

        # Fall back to parent implementation
        return super().extract_simulated_data(sim_files, **kwargs)


# Alias for backward compatibility
CFUSECalibrationTarget = CFUSEStreamflowTarget
