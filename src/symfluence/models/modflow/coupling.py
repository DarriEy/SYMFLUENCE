"""
SUMMA → MODFLOW Coupling

Extracts recharge from SUMMA output (scalarSoilDrainage), converts
to MODFLOW 6 recharge time-series format, and combines MODFLOW
baseflow with SUMMA surface runoff for downstream routing.

Unit conversions:
    SUMMA scalarSoilDrainage: kg/m2/s → m/d (× 86400 / 1000 = × 86.4)
    SUMMA scalarSurfaceRunoff: kg/m2/s → m3/s (× area_m2 / 1000)
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

    # SUMMA kg/m2/s → m/d: (1 kg/m2/s) × (86400 s/d) / (1000 kg/m3) = 86.4 m/d
    KG_M2_S_TO_M_D = 86400.0 / 1000.0  # 86.4

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
            Recharge time series in m/d with datetime index
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

        # Convert kg/m2/s → m/d
        recharge_m_d = values * self.KG_M2_S_TO_M_D

        # Ensure non-negative
        recharge_m_d = np.maximum(recharge_m_d, 0.0)

        series = pd.Series(recharge_m_d, index=times, name='recharge_m_d')

        # Resample to daily if sub-daily
        if len(series) > 1:
            dt = (series.index[1] - series.index[0]).total_seconds()
            if dt < 86400:
                series = series.resample('D').mean()
                self.logger.info("Resampled sub-daily SUMMA output to daily")

        self.logger.info(
            f"Extracted recharge: {len(series)} timesteps, "
            f"mean={series.mean():.6f} m/d, total={series.sum():.3f} m"
        )

        return series

    def write_modflow_recharge_ts(
        self,
        recharge_series: pd.Series,
        output_path: Path,
    ) -> Path:
        """Write MODFLOW 6 time-array-series (TAS6) recharge file.

        MODFLOW 6 TAS6 format:
            BEGIN ATTRIBUTES
              NAME rch_array
              METHOD LINEAR
            END ATTRIBUTES

            BEGIN TIME 0.0
              CONSTANT <recharge_value>
            END TIME 0.0

            BEGIN TIME 1.0
              CONSTANT <recharge_value>
            END TIME 1.0
            ...

        Args:
            recharge_series: Recharge in m/d with datetime index
            output_path: Path to write the .ts file

        Returns:
            Path to written file
        """
        output_path = Path(output_path)

        lines = [
            "BEGIN ATTRIBUTES",
            "  NAME rch_array",
            "  METHOD LINEAR",
            "END ATTRIBUTES",
            "",
        ]

        # Convert datetime index to simulation days (relative to start)
        start_time = recharge_series.index[0]
        for dt, val in recharge_series.items():
            sim_day = (dt - start_time).total_seconds() / 86400.0
            lines.append(f"BEGIN TIME {sim_day:.1f}")
            lines.append(f"  CONSTANT {val:.8e}")
            lines.append(f"END TIME {sim_day:.1f}")
            lines.append("")

        output_path.write_text("\n".join(lines))

        self.logger.info(
            f"Wrote MODFLOW recharge time-array-series: {len(recharge_series)} values to {output_path}"
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

        # Resample to daily if sub-daily
        if len(series) > 1:
            dt = (series.index[1] - series.index[0]).total_seconds()
            if dt < 86400:
                series = series.resample('D').mean()

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
        # Convert surface runoff: kg/m2/s → m3/s
        # (kg/m2/s) × area_m2 / 1000 kg/m3 = m3/s
        surface_m3s = surface_runoff_kg_m2_s * catchment_area_m2 / 1000.0

        # Convert drain discharge: m3/d → m3/s
        baseflow_m3s = drain_discharge_m3_d / 86400.0

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
