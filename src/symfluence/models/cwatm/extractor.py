# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""CWatM Result Extractor."""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.models.base import ModelResultExtractor


class CWatMResultExtractor(ModelResultExtractor):
    """CWatM-specific result extraction.

    CWatM output files use the naming convention
    ``{variable}_{aggregation}.nc`` or ``.tss`` for gauge time series.
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        return {
            'streamflow': ['discharge_daily.nc', '*discharge*.nc', 'discharge_daily.tss'],
            'et': ['totalET*_daily.nc', '*ET*.nc'],
            'snow': ['SnowCover_daily.nc', '*snow*.nc'],
            'soil_moisture': ['*storUpp*.nc', '*tws*.nc'],
            'groundwater': ['storGroundwater_daily.nc', '*groundwater*.nc'],
            'runoff': ['runoff_daily.nc', '*runoff*.nc'],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        variable_mapping = {
            'streamflow': ['discharge', 'Qsim', 'Q'],
            'et': ['totalET_WB', 'totalET', 'ET'],
            'snow': ['SnowCover', 'snow_cover', 'SWE'],
            'soil_moisture': ['tws', 'storUppTotal'],
            'groundwater': ['storGroundwater', 'baseflow'],
            'runoff': ['runoff', 'sum_directRunoff', 'totalRunoff'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(self, output_file: Path, variable_type: str, **kwargs) -> pd.Series:
        import xarray as xr

        var_names = self.get_variable_names(variable_type)
        try:
            ds = xr.open_dataset(output_file)
            found_var = None
            for var_name in var_names:
                if var_name in ds.data_vars:
                    found_var = var_name
                    break
            if found_var is None:
                available = list(ds.data_vars)
                ds.close()
                raise ValueError(
                    f"No suitable variable found in {output_file}. "
                    f"Tried: {var_names}. Available: {available}"
                )
            data = ds[found_var]
            spatial_dims = [d for d in data.dims if d not in ['time']]
            if spatial_dims:
                agg = self.get_spatial_aggregation_method(variable_type)
                if agg == 'max':
                    data = data.max(dim=spatial_dims)
                elif agg == 'sum':
                    data = data.sum(dim=spatial_dims)
                else:
                    data = data.mean(dim=spatial_dims)
            series = data.to_series()
            ds.close()
            return series
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(
                f"Error reading CWatM output file {output_file}: {e}"
            ) from e

    def extract_streamflow(
        self, output_dir: Path, catchment_area: Optional[float] = None, **kwargs
    ) -> pd.Series:
        # Try TSS file first (gauge time series)
        tss_files = list(output_dir.glob('discharge*.tss'))
        if tss_files:
            return self._read_tss(tss_files[0])

        # Fall back to NetCDF
        patterns = self.get_output_file_patterns()['streamflow']
        for pattern in patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                return self.extract_variable(matches[0], 'streamflow', **kwargs)

        raise FileNotFoundError(
            f"No CWatM discharge output found in {output_dir}"
        )

    def _read_tss(self, tss_path: Path) -> pd.Series:
        """Read a PCRaster TSS (time series) file."""
        lines = tss_path.read_text().strip().split('\n')
        # Skip header lines (start with non-numeric)
        data_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0].replace('.', '').replace('-', '').isdigit():
                data_lines.append(parts)
        if not data_lines:
            raise ValueError(f"No data found in TSS file {tss_path}")
        import numpy as np
        arr = np.array(data_lines, dtype=float)
        return pd.Series(arr[:, -1], name='discharge')

    def requires_unit_conversion(self, variable_type: str) -> bool:
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        if variable_type == 'streamflow':
            return 'max'
        elif variable_type in ['et', 'runoff']:
            return 'sum'
        return 'mean'
