"""
SUMMA → HydroGeoSphere Coupling

Extracts recharge from SUMMA output (scalarSoilDrainage), converts
to HGS recharge time-series format, and combines HGS hydrograph
with SUMMA surface runoff for downstream routing.

Unit conversions:
    SUMMA scalarSoilDrainage: m/s → m/s (direct, HGS uses m/s)
    SUMMA scalarSurfaceRunoff: kg/m2/s → m3/s (× area_m2 / 1000)
    HGS hydrograph outlet: m3/s (direct)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SUMMAToHGSCoupler:
    """Couples SUMMA land surface output to HydroGeoSphere model.

    Workflow:
        1. Extract scalarSoilDrainage from SUMMA output → recharge
        2. Write HGS recharge time-series file
        3. After HGS runs, extract hydrograph (baseflow)
        4. Combine surface runoff + hydrograph → total streamflow
    """

    def __init__(self, config_dict: dict, logger_instance=None):
        self.config_dict = config_dict
        self.logger = logger_instance or logger

    def extract_recharge_from_summa(
        self,
        summa_output_dir: Path,
        variable: str = 'scalarSoilDrainage',
    ) -> pd.Series:
        """Extract recharge time series from SUMMA NetCDF output.

        Returns:
            Recharge time series in m/s with datetime index
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

        for dim in list(data.dims):
            if dim != 'time' and data.sizes[dim] == 1:
                data = data.squeeze(dim)

        times = pd.to_datetime(ds['time'].values)
        values = data.values
        ds.close()

        # HGS uses m/s directly — no conversion needed
        recharge_m_s = np.nan_to_num(values, nan=0.0)
        recharge_m_s = np.maximum(recharge_m_s, 0.0)

        series = pd.Series(recharge_m_s, index=times, name='recharge_m_s')

        if len(series) > 1:
            dt = (series.index[1] - series.index[0]).total_seconds()
            if dt < 86400:
                series = series.resample('D').mean()
                self.logger.info("Resampled sub-daily SUMMA output to daily")

        self.logger.info(
            f"Extracted recharge: {len(series)} timesteps, "
            f"mean={series.mean():.8f} m/s"
        )

        return series

    def write_hgs_recharge(
        self,
        recharge_series: pd.Series,
        recharge_path: Path,
    ) -> Path:
        """Write HGS recharge time-series file.

        HGS format: time(seconds) flux(m/s), one per line.

        Args:
            recharge_series: Recharge in m/s with datetime index
            recharge_path: Path to write the recharge file

        Returns:
            Path to written file
        """
        recharge_path = Path(recharge_path)

        lines = []
        start_time = recharge_series.index[0]
        for dt, val in recharge_series.items():
            t_seconds = (dt - start_time).total_seconds()
            if np.isnan(val):
                val = 0.0
            lines.append(f"{t_seconds:.1f} {val:.8e}")

        recharge_path.write_text("\n".join(lines) + "\n")

        self.logger.info(
            f"Wrote HGS recharge: {len(recharge_series)} timesteps to {recharge_path}"
        )

        return recharge_path

    def extract_surface_runoff(
        self,
        summa_output_dir: Path,
        variable: str = 'scalarSurfaceRunoff',
    ) -> pd.Series:
        """Extract surface runoff from SUMMA output.

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

        if len(series) > 1:
            dt = (series.index[1] - series.index[0]).total_seconds()
            if dt < 86400:
                series = series.resample('D').mean()

        return series

    def combine_flows(
        self,
        surface_runoff_kg_m2_s: pd.Series,
        hydrograph_m3_s: pd.Series,
        catchment_area_m2: float,
    ) -> pd.Series:
        """Combine surface runoff and HGS hydrograph into total streamflow.

        Args:
            surface_runoff_kg_m2_s: SUMMA surface runoff (kg/m2/s)
            hydrograph_m3_s: HGS hydrograph discharge (m3/s)
            catchment_area_m2: Catchment area in m2

        Returns:
            Combined streamflow in m3/s
        """
        # Convert surface runoff: kg/m2/s → m3/s
        # (kg/m2/s) × area_m2 / 1000 = m3/s
        surface_m3s = surface_runoff_kg_m2_s * catchment_area_m2 / 1000.0

        # HGS hydrograph is already in m3/s
        baseflow_m3s = hydrograph_m3_s

        # Normalize indices
        surface_m3s.index = surface_m3s.index.normalize()
        baseflow_m3s.index = baseflow_m3s.index.normalize()

        common_idx = surface_m3s.index.intersection(baseflow_m3s.index)
        if len(common_idx) == 0:
            self.logger.warning(
                "No overlapping dates between surface runoff and hydrograph. "
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
