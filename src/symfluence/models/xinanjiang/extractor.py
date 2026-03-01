# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Xinanjiang Result Extractor.

Handles extraction of simulation results from Xinanjiang model outputs
for integration with the evaluation framework.
"""

from pathlib import Path
from typing import Dict, List, cast

import pandas as pd
import xarray as xr

from symfluence.models.base import ModelResultExtractor


class XinanjiangResultExtractor(ModelResultExtractor):
    """Xinanjiang-specific result extraction."""

    def __init__(self, model_name: str = 'XINANJIANG'):
        super().__init__('XINANJIANG')

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        return {
            'streamflow': [
                '*_xinanjiang_output.nc',
                '*_xinanjiang_output.csv',
            ],
            'runoff': [
                '*_xinanjiang_output.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        variable_mapping = {
            'streamflow': ['streamflow', 'discharge', 'Q'],
            'runoff': ['runoff', 'total_runoff'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        output_file = Path(output_file)
        var_names = self.get_variable_names(variable_type)

        if output_file.suffix == '.csv':
            return self._extract_from_csv(output_file, var_names)
        else:
            return self._extract_from_netcdf(output_file, var_names, variable_type, **kwargs)

    def _extract_from_csv(self, output_file: Path, var_names: List[str]) -> pd.Series:
        df = pd.read_csv(output_file, index_col='datetime', parse_dates=True)
        for var_name in var_names:
            if var_name in df.columns:
                return df[var_name]
            if var_name == 'streamflow' and 'streamflow_cms' in df.columns:
                return df['streamflow_cms']
        raise ValueError(f"Variable not found in {output_file}. Available: {list(df.columns)}")

    def _extract_from_netcdf(
        self, output_file: Path, var_names: List[str],
        variable_type: str, **kwargs
    ) -> pd.Series:
        with xr.open_dataset(output_file) as ds:
            for var_name in var_names:
                if var_name in ds.variables:
                    var = ds[var_name]
                    non_time_dims = [d for d in var.dims if d != 'time']
                    for dim in non_time_dims:
                        var = var.isel({dim: 0})
                    return cast(pd.Series, var.to_pandas())
            raise ValueError(
                f"Variable not found for '{variable_type}' in {output_file}. "
                f"Available: {list(ds.data_vars)}"
            )

    def requires_unit_conversion(self, variable_type: str) -> bool:
        return variable_type == 'runoff'

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        return 'selection'
