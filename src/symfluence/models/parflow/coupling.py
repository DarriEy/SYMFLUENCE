"""
SUMMA -> ParFlow Coupling

Extracts recharge from SUMMA output (scalarSoilDrainage), converts
to ParFlow units (m/hr), writes recharge as ParFlow flux input, and
combines ParFlow subsurface flow with SUMMA surface runoff.

Unit conversions:
    SUMMA scalarSoilDrainage: kg/m2/s -> m/hr (x 3600 / 1000 = x 3.6)
    SUMMA scalarSurfaceRunoff: kg/m2/s -> m3/s (x area_m2 / 1000)
    ParFlow subsurface drainage: m3/hr -> m3/s (/ 3600)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SUMMAToParFlowCoupler:
    """Couples SUMMA land surface output to ParFlow groundwater model.

    Workflow:
        1. Extract scalarSoilDrainage from SUMMA output -> recharge (m/hr)
        2. Write ParFlow recharge flux file (.pfb or time-series)
        3. After ParFlow runs, extract subsurface drainage
        4. Combine surface runoff + subsurface drainage -> total streamflow
    """

    # SUMMA kg/m2/s -> m/hr: (1 kg/m2/s) x (3600 s/hr) / (1000 kg/m3) = 3.6 m/hr
    KG_M2_S_TO_M_HR = 3600.0 / 1000.0  # 3.6

    def __init__(self, config_dict: dict, logger_instance=None):
        self.config_dict = config_dict
        self.logger = logger_instance or logger

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
            Recharge time series in m/hr with datetime index
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

        self.logger.info(f"Reading {variable} from {len(nc_files)} SUMMA output file(s)")

        ds = xr.open_mfdataset(nc_files, combine='by_coords')

        if variable not in ds:
            available = list(ds.data_vars)
            ds.close()
            raise ValueError(
                f"Variable '{variable}' not found in SUMMA output. "
                f"Available: {available}"
            )

        data = ds[variable]

        # Squeeze non-time dimensions (single HRU)
        for dim in list(data.dims):
            if dim != 'time' and data.sizes[dim] == 1:
                data = data.squeeze(dim)

        times = pd.to_datetime(ds['time'].values)
        values = data.values

        ds.close()

        # Convert kg/m2/s -> m/hr
        recharge_m_hr = values * self.KG_M2_S_TO_M_HR

        # Ensure non-negative
        recharge_m_hr = np.maximum(recharge_m_hr, 0.0)

        series = pd.Series(recharge_m_hr, index=times, name='recharge_m_hr')

        # Resample to hourly if sub-hourly
        if len(series) > 1:
            dt = (series.index[1] - series.index[0]).total_seconds()
            if dt < 3600:
                series = series.resample('h').mean()
                self.logger.info("Resampled sub-hourly SUMMA output to hourly")

        self.logger.info(
            f"Extracted recharge: {len(series)} timesteps, "
            f"mean={series.mean():.6f} m/hr, total={series.sum():.3f} m"
        )

        return series

    def write_parflow_recharge(
        self,
        recharge_series: pd.Series,
        output_path: Path,
    ) -> Path:
        """Write recharge as a ParFlow-compatible flux time series file.

        Writes a simple CSV that can be read by a ParFlow preprocessor
        or used to generate .pfb flux files. For coupled runs, the
        preprocessor should update the ParFlow .pfidb to reference these
        recharge values.

        Args:
            recharge_series: Recharge in m/hr with datetime index
            output_path: Path to write the recharge file

        Returns:
            Path to written file
        """
        output_path = Path(output_path)

        # Write as CSV with simulation hours and recharge values
        start_time = recharge_series.index[0]
        rows = []
        for dt, val in recharge_series.items():
            sim_hour = (dt - start_time).total_seconds() / 3600.0
            rows.append(f"{sim_hour:.1f},{val:.8e}")

        header = "# ParFlow recharge flux (m/hr)\n# sim_hour,recharge_m_hr\n"
        output_path.write_text(header + "\n".join(rows) + "\n")

        self.logger.info(
            f"Wrote ParFlow recharge file: {len(recharge_series)} values to {output_path}"
        )

        return output_path

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
            Surface runoff in kg/m2/s with datetime index
        """
        import xarray as xr

        summa_output_dir = Path(summa_output_dir)
        nc_files = sorted(summa_output_dir.glob("*_output_*.nc"))
        if not nc_files:
            nc_files = sorted(summa_output_dir.glob("*.nc"))

        if not nc_files:
            raise FileNotFoundError(
                f"No SUMMA output files found in {summa_output_dir}"
            )

        ds = xr.open_mfdataset(nc_files, combine='by_coords')

        if variable not in ds:
            ds.close()
            raise ValueError(f"Variable '{variable}' not found in SUMMA output")

        data = ds[variable]
        for dim in list(data.dims):
            if dim != 'time' and data.sizes[dim] == 1:
                data = data.squeeze(dim)

        times = pd.to_datetime(ds['time'].values)
        values = data.values
        ds.close()

        series = pd.Series(values, index=times, name='surface_runoff_kg_m2_s')

        # Resample to hourly if sub-hourly
        if len(series) > 1:
            dt = (series.index[1] - series.index[0]).total_seconds()
            if dt < 3600:
                series = series.resample('h').mean()

        return series

    def combine_flows(
        self,
        surface_runoff_kg_m2_s: pd.Series,
        subsurface_drainage_m3_hr: pd.Series,
        catchment_area_m2: float,
    ) -> pd.Series:
        """Combine surface runoff and ParFlow subsurface flow into total streamflow.

        Args:
            surface_runoff_kg_m2_s: SUMMA surface runoff (kg/m2/s)
            subsurface_drainage_m3_hr: ParFlow subsurface drainage (m3/hr)
            catchment_area_m2: Catchment area in m2

        Returns:
            Combined streamflow in m3/s
        """
        # Convert surface runoff: kg/m2/s -> m3/s
        # (kg/m2/s) x area_m2 / 1000 kg/m3 = m3/s
        surface_m3s = surface_runoff_kg_m2_s * catchment_area_m2 / 1000.0

        # Convert subsurface drainage: m3/hr -> m3/s
        baseflow_m3s = subsurface_drainage_m3_hr / 3600.0

        # Align indices
        common_idx = surface_m3s.index.intersection(baseflow_m3s.index)
        if len(common_idx) == 0:
            self.logger.warning(
                "No overlapping dates between surface runoff and subsurface flow. "
                "Returning surface runoff only."
            )
            return surface_m3s.rename('total_streamflow_m3s')

        surface_aligned = surface_m3s.loc[common_idx]
        baseflow_aligned = baseflow_m3s.loc[common_idx]

        total = surface_aligned + baseflow_aligned
        total.name = 'total_streamflow_m3s'

        self.logger.info(
            f"Combined flows: surface={surface_aligned.mean():.4f} m3/s, "
            f"subsurface={baseflow_aligned.mean():.4f} m3/s, "
            f"total={total.mean():.4f} m3/s"
        )

        return total
