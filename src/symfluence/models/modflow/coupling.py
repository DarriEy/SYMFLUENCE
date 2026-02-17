"""
SUMMA → MODFLOW Coupling

Extracts recharge from SUMMA output (scalarSoilDrainage), converts
to MODFLOW 6 recharge time-series format, and combines MODFLOW
baseflow with SUMMA surface runoff for downstream routing.

Unit conversions:
    SUMMA scalarSoilDrainage: m/s → m/d (× 86400)
    SUMMA scalarSurfaceRunoff: m/s → m3/s (× area_m2)
    MODFLOW drain discharge: m3/d → m3/s (÷ 86400)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SUMMAToMODFLOWCoupler:
    """Couples SUMMA land surface output to MODFLOW 6 groundwater model.

    Workflow:
        1. Extract scalarSoilDrainage from SUMMA output → recharge
        2. Write MODFLOW 6 time-series recharge file
        3. After MODFLOW runs, extract drain discharge (baseflow)
        4. Combine surface runoff + baseflow → total streamflow
    """

    # SUMMA m/s → m/d: (1 m/s) × (86400 s/d) = 86400 m/d
    M_S_TO_M_D = 86400.0

    def __init__(self, config_dict: dict, logger_instance=None):
        self.config_dict = config_dict
        self.logger = logger_instance or logger

    def _open_summa_variable(self, summa_output_dir: Path, variable: str):
        """Open the SUMMA NetCDF file containing the requested variable.

        Opens each file individually to avoid NaN-padding issues that
        occur when open_mfdataset merges files with different time
        dimensions (e.g. hourly timestep file + daily output file).

        Returns:
            (xr.Dataset, list[Path]) — opened dataset and files used
        """
        import xarray as xr

        summa_output_dir = Path(summa_output_dir)
        nc_files = sorted(summa_output_dir.glob("*_output_*.nc"))
        if not nc_files:
            nc_files = sorted(summa_output_dir.glob("*.nc"))

        if not nc_files:
            raise FileNotFoundError(
                f"No SUMMA output NetCDF files found in {summa_output_dir}"
            )

        # Find files that contain the variable and open only those
        matching_files = []
        for nc_file in nc_files:
            ds_check = xr.open_dataset(nc_file)
            has_var = variable in ds_check
            ds_check.close()
            if has_var:
                matching_files.append(nc_file)

        if not matching_files:
            # Gather all available variables for error message
            all_vars = set()
            for nc_file in nc_files:
                ds_check = xr.open_dataset(nc_file)
                all_vars.update(ds_check.data_vars)
                ds_check.close()
            raise ValueError(
                f"Variable '{variable}' not found in SUMMA output. "
                f"Available: {sorted(all_vars)}"
            )

        if len(matching_files) == 1:
            ds = xr.open_dataset(matching_files[0])
        else:
            ds = xr.open_mfdataset(matching_files, combine='by_coords')

        self.logger.info(
            f"Reading {variable} from {len(matching_files)} file(s)"
        )
        return ds, matching_files

    def extract_recharge_from_summa(
        self,
        summa_output_dir: Path,
        variable: str = 'scalarSoilDrainage',
    ) -> pd.Series:
        """Extract recharge time series from SUMMA NetCDF output.

        Args:
            summa_output_dir: Path to SUMMA output directory
            variable: SUMMA variable name for soil drainage

        Returns:
            Recharge time series in m/d with datetime index
        """
        ds, _ = self._open_summa_variable(summa_output_dir, variable)

        data = ds[variable]

        # Squeeze non-time dimensions (single HRU)
        for dim in list(data.dims):
            if dim != 'time' and data.sizes[dim] == 1:
                data = data.squeeze(dim)

        times = pd.to_datetime(ds['time'].values)
        values = data.values

        ds.close()

        # Convert m/s → m/d
        recharge_m_d = values * self.M_S_TO_M_D

        series = pd.Series(recharge_m_d, index=times, name='recharge_m_d')

        # Resample to daily if sub-daily (preserves NaN correctly)
        if len(series) > 1:
            dt = (series.index[1] - series.index[0]).total_seconds()
            if dt < 86400:
                series = series.resample('D').mean()
                self.logger.info("Resampled sub-daily SUMMA output to daily")

        # Replace NaN and ensure non-negative after resampling
        series = series.fillna(0.0).clip(lower=0.0)

        self.logger.info(
            f"Extracted recharge: {len(series)} timesteps, "
            f"mean={series.mean():.6f} m/d, total={series.sum():.3f} m"
        )

        return series

    def write_modflow_recharge_rch(
        self,
        recharge_series: pd.Series,
        rch_path: Path,
        n_stress_periods: int = 0,
    ) -> Path:
        """Write MODFLOW 6 RCH package with per-period recharge.

        Uses READASARRAYS CONSTANT format — one PERIOD block per stress
        period.  Periods beyond the recharge series reuse the last value
        (MODFLOW default behaviour when a period block is omitted).

        Args:
            recharge_series: Recharge in m/d with datetime index
            rch_path: Path to write the gwf.rch file
            n_stress_periods: Total stress periods in TDIS (0 = use series length)

        Returns:
            Path to written file
        """
        rch_path = Path(rch_path)

        lines = [
            "BEGIN OPTIONS",
            "  READASARRAYS",
            "END OPTIONS",
            "",
        ]

        values = recharge_series.values
        for i, val in enumerate(values):
            if np.isnan(val):
                val = 0.0
            lines.append(f"BEGIN PERIOD {i + 1}")
            lines.append("  RECHARGE")
            lines.append(f"    CONSTANT {val:.8e}")
            lines.append(f"END PERIOD {i + 1}")
            lines.append("")

        rch_path.write_text("\n".join(lines))

        self.logger.info(
            f"Wrote MODFLOW per-period recharge: {len(recharge_series)} periods to {rch_path}"
        )

        return rch_path

    def extract_surface_runoff(
        self,
        summa_output_dir: Path,
        variable: str = 'scalarSurfaceRunoff',
    ) -> pd.Series:
        """Extract surface runoff from SUMMA output.

        Args:
            summa_output_dir: Path to SUMMA output directory
            variable: SUMMA variable name for surface runoff

        Returns:
            Surface runoff in m/s with datetime index
        """
        ds, _ = self._open_summa_variable(summa_output_dir, variable)

        data = ds[variable]
        for dim in list(data.dims):
            if dim != 'time' and data.sizes[dim] == 1:
                data = data.squeeze(dim)

        times = pd.to_datetime(ds['time'].values)
        values = data.values
        ds.close()

        series = pd.Series(values, index=times, name='surface_runoff_m_s')

        # Resample to daily if sub-daily
        if len(series) > 1:
            dt = (series.index[1] - series.index[0]).total_seconds()
            if dt < 86400:
                series = series.resample('D').mean()

        series = series.fillna(0.0)
        return series

    def combine_flows(
        self,
        surface_runoff_kg_m2_s: pd.Series,
        drain_discharge_m3_d: pd.Series,
        catchment_area_m2: float,
    ) -> pd.Series:
        """Combine surface runoff and MODFLOW baseflow into total streamflow.

        Args:
            surface_runoff_kg_m2_s: SUMMA surface runoff (kg/m2/s)
            drain_discharge_m3_d: MODFLOW drain discharge (m3/d)
            catchment_area_m2: Catchment area in m2

        Returns:
            Combined streamflow in m3/s
        """
        # Convert surface runoff: m/s → m3/s
        # (m/s) × area_m2 = m3/s
        surface_m3s = surface_runoff_kg_m2_s * catchment_area_m2

        # Convert drain discharge: m3/d → m3/s
        baseflow_m3s = drain_discharge_m3_d / 86400.0

        # Normalize both indices to date-only for alignment
        surface_m3s.index = surface_m3s.index.normalize()
        baseflow_m3s.index = baseflow_m3s.index.normalize()

        # Align indices
        common_idx = surface_m3s.index.intersection(baseflow_m3s.index)
        if len(common_idx) == 0:
            self.logger.warning(
                "No overlapping dates between surface runoff and baseflow. "
                "Returning surface runoff only."
            )
            return surface_m3s.rename('total_streamflow_m3s')

        surface_aligned = surface_m3s.loc[common_idx]
        baseflow_aligned = baseflow_m3s.loc[common_idx]

        total = surface_aligned + baseflow_aligned
        total.name = 'total_streamflow_m3s'

        self.logger.info(
            f"Combined flows: surface={surface_aligned.mean():.4f} m3/s, "
            f"baseflow={baseflow_aligned.mean():.4f} m3/s, "
            f"total={total.mean():.4f} m3/s"
        )

        return total
