"""
CLM Result Extractor

Extracts multi-variable results from CLM5 history output.
Supports streamflow, ET, snow, and soil moisture extraction.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import xarray as xr

from symfluence.models.base.base_extractor import ModelResultExtractor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

# CLM variable groups
CLM_VARIABLES = {
    'streamflow': {
        'QRUNOFF': {'long_name': 'Total column runoff', 'units': 'mm/s'},
        'QOVER': {'long_name': 'Surface runoff', 'units': 'mm/s'},
        'QDRAI': {'long_name': 'Sub-surface drainage', 'units': 'mm/s'},
    },
    'evapotranspiration': {
        'EFLX_LH_TOT': {'long_name': 'Total latent heat flux', 'units': 'W/m2'},
        'QFLX_EVAP_TOT': {'long_name': 'Total evapotranspiration', 'units': 'mm/s'},
    },
    'snow': {
        'H2OSNO': {'long_name': 'Snow water equivalent', 'units': 'mm'},
        'SNOWDP': {'long_name': 'Snow depth', 'units': 'm'},
        'FSNO': {'long_name': 'Fraction of ground covered by snow', 'units': '-'},
    },
    'soil_moisture': {
        'SOILWATER_10CM': {'long_name': 'Soil liquid water in top 10cm', 'units': 'kg/m2'},
        'TWS': {'long_name': 'Total water storage', 'units': 'mm'},
    },
}


@ModelRegistry.register_result_extractor("CLM")
class CLMResultExtractor(ModelResultExtractor):
    """
    Extracts results from CLM5 history output.

    Supports extraction of streamflow, ET, snow, and soil moisture
    variables from *.clm2.h0.*.nc history files.
    """

    def __init__(self, model_name: str = 'CLM'):
        super().__init__(model_name)
        self.logger = logger

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for locating CLM outputs."""
        return {
            'streamflow': ['*.clm2.h0.*.nc'],
            'snow': ['*.clm2.h0.*.nc'],
            'evapotranspiration': ['*.clm2.h0.*.nc'],
            'soil_moisture': ['*.clm2.h0.*.nc'],
        }

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs,
    ) -> pd.Series:
        """Extract a variable time series from CLM output.

        Args:
            output_file: Path to output file or directory
            variable_type: Type ('streamflow', 'snow', 'evapotranspiration', 'soil_moisture')
            **kwargs: Additional args (catchment_area_km2, variable_name)

        Returns:
            Time series of extracted variable
        """
        output_path = Path(output_file)

        # If a directory is given, open all history files in it
        if output_path.is_dir():
            ds = self._open_history(output_path)
        else:
            ds = xr.open_dataset(output_path)

        if ds is None:
            raise ValueError(f"Could not open CLM output: {output_file}")

        # Determine which variable to extract
        var_name = kwargs.get('variable_name')
        if var_name is None:
            # Map variable_type to default CLM variable
            type_to_var = {
                'streamflow': 'QRUNOFF',
                'snow': 'H2OSNO',
                'evapotranspiration': 'QFLX_EVAP_TOT',
                'soil_moisture': 'SOILWATER_10CM',
            }
            var_name = type_to_var.get(variable_type, variable_type)

        if var_name not in ds:
            # Try fallback for streamflow
            if variable_type == 'streamflow' and 'QOVER' in ds and 'QDRAI' in ds:
                data = ds['QOVER'] + ds['QDRAI']
            else:
                available = list(ds.data_vars)
                ds.close()
                raise ValueError(
                    f"Variable '{var_name}' not in CLM output. Available: {available}"
                )
        else:
            data = ds[var_name]

        # Squeeze spatial dims
        for dim in list(data.dims):
            if dim != 'time' and data.sizes[dim] == 1:
                data = data.squeeze(dim)

        times = pd.to_datetime(ds['time'].values)
        values = data.values.flatten()
        ds.close()

        # Unit conversion for streamflow (mm/s → m3/s)
        catchment_area_km2 = kwargs.get('catchment_area_km2')
        if variable_type == 'streamflow' and catchment_area_km2:
            area_m2 = catchment_area_km2 * 1e6
            values = values * area_m2 / 1000.0

        return pd.Series(values, index=times, name=var_name)

    def extract_streamflow(
        self,
        output_dir: Path,
        catchment_area_km2: Optional[float] = None,
        warmup_days: int = 0,
    ) -> Optional[pd.Series]:
        """
        Extract streamflow from CLM output as m3/s.

        Args:
            output_dir: Directory containing CLM history files
            catchment_area_km2: Catchment area for unit conversion
            warmup_days: Days to skip for spinup

        Returns:
            Pandas Series of streamflow in m3/s
        """
        try:
            series = self.extract_variable(
                output_dir, 'streamflow',
                catchment_area_km2=catchment_area_km2,
            )
        except ValueError as e:
            self.logger.error(str(e))
            return None

        # Skip warmup
        if warmup_days > 0 and len(series) > warmup_days:
            series = series.iloc[warmup_days:]

        return series

    def get_available_variables(self, output_dir: Path) -> List[str]:
        """List variables available in CLM output."""
        ds = self._open_history(output_dir)
        if ds is None:
            return []
        variables = list(ds.data_vars)
        ds.close()
        return variables

    def _open_history(self, output_dir: Path) -> Optional[xr.Dataset]:
        """Open CLM history files from output directory."""
        output_dir = Path(output_dir)
        hist_files = sorted(output_dir.glob("*.clm2.h0.*.nc"))

        if not hist_files:
            for subdir in ['run', 'hist', 'results']:
                sub = output_dir / subdir
                if sub.exists():
                    hist_files = sorted(sub.glob("*.clm2.h0.*.nc"))
                    if hist_files:
                        break

        if not hist_files:
            self.logger.warning(f"No CLM history files in {output_dir}")
            return None

        return xr.open_mfdataset(hist_files, combine='by_coords')
