# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Noah-MP Result Extractor.

noah-owp-modular writes a single ``output.nc`` with all water/energy
variables on the model time axis.  Variable naming follows the NOAH-MP
convention (uppercase Fortran-style identifiers).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.models.base import ModelResultExtractor


class NoahMPResultExtractor(ModelResultExtractor):
    """Result extraction for standalone Noah-MP (noah-owp-modular).

    Output file: single ``output.nc`` containing time series of water
    balance, energy balance, snow, and soil state variables.
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        return {
            'streamflow': ['output.nc'],
            'runoff': ['output.nc'],
            'et': ['output.nc'],
            'snow': ['output.nc'],
            'soil_moisture': ['output.nc'],
            'energy': ['output.nc'],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        variable_mapping = {
            'streamflow': ['SFCRNOFF', 'UGDRNOFF'],
            'runoff': ['SFCRNOFF', 'UGDRNOFF', 'QINSUR'],
            'et': ['ECAN', 'ETRAN', 'EDIR', 'LH'],
            'snow': ['SNEQV', 'SNOWH', 'QSNBOT', 'QSNSUB', 'QSNFRO'],
            'soil_moisture': ['SMC', 'SH2O', 'ZWT'],
            'energy': ['LH', 'FSH', 'GH', 'FSA', 'FIRA', 'TV', 'TG', 'T2M'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(self, output_file: Path, variable_type: str, **kwargs) -> pd.Series:
        import xarray as xr

        var_names = self.get_variable_names(variable_type)
        try:
            ds = xr.open_dataset(output_file)

            if variable_type in ('streamflow', 'runoff'):
                components = []
                for var_name in ['SFCRNOFF', 'UGDRNOFF']:
                    if var_name in ds.data_vars:
                        components.append(ds[var_name])
                if components:
                    data = sum(components)
                    series = data.to_series()
                    ds.close()
                    return series

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
                if agg == 'sum':
                    data = data.sum(dim=spatial_dims)
                elif agg == 'max':
                    data = data.max(dim=spatial_dims)
                else:
                    data = data.mean(dim=spatial_dims)
            series = data.to_series()
            ds.close()
            return series
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(
                f"Error reading Noah-MP output file {output_file}: {e}"
            ) from e

    def extract_streamflow(
        self, output_dir: Path, catchment_area: Optional[float] = None, **kwargs
    ) -> pd.Series:
        patterns = self.get_output_file_patterns()['streamflow']
        for pattern in patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                return self.extract_variable(matches[0], 'streamflow', **kwargs)

        raise FileNotFoundError(
            f"No Noah-MP output found in {output_dir}"
        )

    def requires_unit_conversion(self, variable_type: str) -> bool:
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        if variable_type in ('et', 'runoff', 'streamflow'):
            return 'sum'
        return 'mean'
