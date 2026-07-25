# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""PCR-GLOBWB Result Extractor."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.core.modeling.base import ModelResultExtractor


class PCRGLOBWBResultExtractor(ModelResultExtractor):
    """PCR-GLOBWB-specific result extraction.

    PCR-GLOBWB output files follow the naming convention
    ``{variable}_{aggregation}_output.nc``, e.g.
    ``discharge_dailyTot_output.nc``.
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        return {
            'streamflow': ['discharge_*_output.nc', '*discharge*.nc'],
            'et': ['totalEvaporation_*_output.nc', '*evaporation*.nc'],
            'snow': ['snowCoverSWE_*_output.nc', '*snow*.nc'],
            'soil_moisture': ['storUppTotal_*_output.nc', '*storUpp*.nc'],
            'groundwater': ['storGroundwater_*_output.nc'],
            'runoff': ['totalRunoff_*_output.nc', 'directRunoff_*_output.nc'],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        variable_mapping = {
            'streamflow': ['discharge', 'Qsim', 'Q'],
            'et': ['totalEvaporation', 'total_evaporation', 'ET'],
            'snow': ['snowCoverSWE', 'snow_cover_swe', 'SWE'],
            'soil_moisture': ['storUppTotal', 'stor_upp_total'],
            'groundwater': ['storGroundwater', 'stor_groundwater'],
            'runoff': ['totalRunoff', 'total_runoff', 'directRunoff'],
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
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"Error reading PCR-GLOBWB output file {output_file}: {e}"
            ) from e

    def extract_streamflow(
        self, output_dir: Path, catchment_area: Optional[float] = None, **kwargs
    ) -> pd.Series:
        patterns = self.get_output_file_patterns()['streamflow']
        output_file = None
        for pattern in patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break
        if output_file is None:
            raise FileNotFoundError(
                f"No PCR-GLOBWB discharge output file found in {output_dir}"
            )
        return self.extract_variable(output_file, 'streamflow', **kwargs)

    def requires_unit_conversion(self, variable_type: str) -> bool:
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        if variable_type == 'streamflow':
            return 'max'
        elif variable_type in ['et', 'runoff']:
            return 'sum'
        return 'mean'
