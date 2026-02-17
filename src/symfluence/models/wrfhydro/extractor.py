"""
WRF-Hydro Result Extractor.

Handles extraction of simulation results from WRF-Hydro model outputs.
WRF-Hydro outputs are in NetCDF format with CHRTOUT (channel) and
LDASOUT (land surface) file types.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from symfluence.models.base import ModelResultExtractor


class WRFHydroResultExtractor(ModelResultExtractor):
    """WRF-Hydro-specific result extraction.

    Handles WRF-Hydro model's output characteristics:
    - File formats: NetCDF (.nc)
    - CHRTOUT files: Channel discharge (streamflow)
    - LDASOUT files: Land surface variables (ET, soil moisture, SWE)
    - Variable naming: streamflow, ACSNOM, SOIL_M, SNEQV, etc.
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for WRF-Hydro outputs."""
        return {
            'streamflow': [
                '*CHRTOUT*.nc',
                '*CHANOBS*.nc',
            ],
            'et': [
                '*LDASOUT*.nc',
            ],
            'soil_moisture': [
                '*LDASOUT*.nc',
            ],
            'snow': [
                '*LDASOUT*.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get WRF-Hydro variable names for different types.

        WRF-Hydro output variable naming conventions:
        - CHRTOUT: streamflow (m3/s), q_lateral, velocity
        - LDASOUT: ACSNOM (accumulated snowmelt), SOIL_M (soil moisture),
                   SNEQV (SWE), ACCET (accumulated ET)
        """
        variable_mapping = {
            'streamflow': [
                'streamflow',
                'q_lateral',
                'qSfcLatRunoff',
                'qBucket',
            ],
            'et': [
                'ACCET',
                'ECAN',
                'ETRAN',
                'EDIR',
            ],
            'soil_moisture': [
                'SOIL_M',
                'SH2O',
                'SMC',
                'SMCWTD',
            ],
            'snow': [
                'SNEQV',
                'SNOWH',
                'ACSNOM',
                'FSNO',
            ],
            'precipitation': [
                'RAINRATE',
                'RAINNC',
                'RAINC',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from WRF-Hydro output.

        Args:
            output_file: Path to WRF-Hydro output file (NetCDF)
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - aggregate: Whether to aggregate spatially (default: True)

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found or file cannot be parsed
        """
        import xarray as xr

        var_names = self.get_variable_names(variable_type)
        aggregate = kwargs.get('aggregate', True)

        try:
            ds = xr.open_dataset(output_file)

            # Find the variable
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

            # Aggregate spatially if needed
            if aggregate and len(data.dims) > 1:
                spatial_dims = [d for d in data.dims if d not in ['time']]
                method = self.get_spatial_aggregation_method(variable_type)
                if method == 'sum':
                    data = data.sum(dim=spatial_dims)
                else:
                    data = data.mean(dim=spatial_dims)

            # Convert to pandas Series
            if 'time' in data.dims:
                series = data.to_series()
            else:
                series = pd.Series([float(data.values)], index=[pd.Timestamp.now()])

            ds.close()
            return series

        except Exception as e:
            raise ValueError(
                f"Error reading WRF-Hydro output file {output_file}: {str(e)}"
            )

    def extract_streamflow(
        self,
        output_dir: Path,
        catchment_area: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """
        Extract streamflow from WRF-Hydro CHRTOUT output.

        WRF-Hydro streamflow is already in m3/s, no area conversion needed.

        Args:
            output_dir: Directory containing WRF-Hydro outputs
            catchment_area: Not used (streamflow already in cms)
            **kwargs: Additional options

        Returns:
            Time series of streamflow in m3/s
        """
        import xarray as xr

        # Find CHRTOUT files
        patterns = self.get_output_file_patterns()['streamflow']
        output_files = []
        for pattern in patterns:
            output_files.extend(sorted(output_dir.glob(pattern)))

        if not output_files:
            raise FileNotFoundError(
                f"No WRF-Hydro CHRTOUT files found in {output_dir}"
            )

        all_series = []
        for fpath in output_files:
            ds = xr.open_dataset(fpath)
            for var in self.get_variable_names('streamflow'):
                if var in ds.data_vars:
                    flow = ds[var]
                    # Take outlet (last feature_id)
                    if 'feature_id' in flow.dims:
                        flow = flow.isel(feature_id=-1)
                    if 'time' in flow.dims:
                        all_series.append(flow.to_series())
                    break
            ds.close()

        if not all_series:
            raise ValueError("No streamflow data found in WRF-Hydro output")

        return pd.concat(all_series).sort_index()

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Check if variable requires unit conversion.

        WRF-Hydro streamflow is already in m3/s, no conversion needed.
        """
        # Streamflow from CHRTOUT is already in cms
        return variable_type not in ['streamflow']

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """Get spatial aggregation method for variable type."""
        if variable_type in ['et', 'precipitation']:
            return 'sum'
        elif variable_type in ['streamflow']:
            return 'sum'
        else:
            return 'mean'  # Average for states like SWE, soil moisture
