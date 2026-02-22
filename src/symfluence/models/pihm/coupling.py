"""
SUMMA → PIHM Coupling

Extracts recharge from SUMMA output (scalarSoilDrainage), converts
to PIHM forcing format, and combines PIHM river flux with SUMMA
surface runoff for downstream routing.

Unit conversions:
    SUMMA scalarSoilDrainage: m/s (direct — PIHM uses m/s natively)
    SUMMA scalarSurfaceRunoff: kg/m2/s → m/s (÷ 1000)
    PIHM river flux: m3/s (direct)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SUMAToPIHMCoupler:
    """Couples SUMMA land surface output to PIHM groundwater model.

    Workflow:
        1. Extract scalarSoilDrainage from SUMMA output → recharge
        2. Write PIHM forcing file with recharge time series
        3. After PIHM runs, extract river flux (baseflow)
        4. Combine surface runoff + river flux → total streamflow
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

        Args:
            summa_output_dir: Path to SUMMA output directory
            variable: SUMMA variable name for soil drainage

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

        ds = xr.open_mfdataset(nc_files, combine='by_coords', data_vars='minimal', coords='minimal', compat='override')

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

        # PIHM uses m/s natively — no conversion needed
        recharge_m_s = np.nan_to_num(values, nan=0.0)
        recharge_m_s = np.maximum(recharge_m_s, 0.0)

        series = pd.Series(recharge_m_s, index=times, name='recharge_m_s')

        # Resample to daily if sub-daily
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

    def write_pihm_forcing(
        self,
        recharge_series: pd.Series,
        forc_path: Path,
    ) -> Path:
        """Write PIHM forcing file with recharge time series.

        Args:
            recharge_series: Recharge in m/s with datetime index
            forc_path: Path to write the forcing file

        Returns:
            Path to written file
        """
        forc_path = Path(forc_path)

        lines = [str(len(recharge_series))]
        for dt, val in recharge_series.items():
            epoch = int(dt.timestamp())
            if np.isnan(val):
                val = 0.0
            # Format: time recharge 0.0 0.0 0.0 0.0 0.0
            lines.append(f"{epoch} {val:.8e} 0.0 0.0 0.0 0.0 0.0")

        forc_path.write_text("\n".join(lines) + "\n")

        self.logger.info(
            f"Wrote PIHM forcing: {len(recharge_series)} timesteps to {forc_path}"
        )

        return forc_path

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

        ds = xr.open_mfdataset(nc_files, combine='by_coords', data_vars='minimal', coords='minimal', compat='override')

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
        river_flux_m3_s: pd.Series,
        catchment_area_m2: float,
    ) -> pd.Series:
        """Combine surface runoff and PIHM river flux into total streamflow.

        Args:
            surface_runoff_kg_m2_s: SUMMA surface runoff (kg/m2/s)
            river_flux_m3_s: PIHM river flux (m3/s)
            catchment_area_m2: Catchment area in m2

        Returns:
            Combined streamflow in m3/s
        """
        # Convert surface runoff: kg/m2/s → m/s (÷ 1000) → m3/s (× area)
        surface_m3s = surface_runoff_kg_m2_s * catchment_area_m2 / 1000.0

        # PIHM river flux is already in m3/s
        baseflow_m3s = river_flux_m3_s

        # Normalize indices for alignment
        surface_m3s.index = surface_m3s.index.normalize()
        baseflow_m3s.index = baseflow_m3s.index.normalize()

        common_idx = surface_m3s.index.intersection(baseflow_m3s.index)
        if len(common_idx) == 0:
            self.logger.warning(
                "No overlapping dates between surface runoff and river flux. "
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
