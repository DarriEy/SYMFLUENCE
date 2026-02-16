"""
mHM Result Extractor.

Handles extraction of simulation results from mHM model outputs.
mHM outputs are in NetCDF format with discharge and fluxes/states.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from symfluence.models.base import ModelResultExtractor


class MHMResultExtractor(ModelResultExtractor):
    """mHM-specific result extraction.

    Handles mHM model's output characteristics:
    - File formats: NetCDF (.nc)
    - Output files: discharge_*.nc, mHM_Fluxes_States_*.nc
    - Variable naming: Qsim, aET, SWC, SWE, etc.
    - Discharge is already in m3/s
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for mHM outputs."""
        return {
            'streamflow': [
                'discharge_*.nc',
                '*discharge*.nc',
                'Qsim*.nc',
            ],
            'et': [
                'mHM_Fluxes_States_*.nc',
                '*Fluxes*.nc',
            ],
            'snow': [
                'mHM_Fluxes_States_*.nc',
                '*Fluxes*.nc',
            ],
            'soil_moisture': [
                'mHM_Fluxes_States_*.nc',
                '*Fluxes*.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get mHM variable names for different types.

        mHM output variables:
        - Qsim: Simulated discharge [m3/s]
        - aET: Actual evapotranspiration [mm/day]
        - SWC: Soil water content [mm]
        - SWE: Snow water equivalent [mm]
        - recharge: Groundwater recharge [mm/day]
        - runoff_sealed: Sealed surface runoff [mm/day]
        - slowInterflow: Slow interflow [mm/day]
        - fastInterflow: Fast interflow [mm/day]
        - baseflow: Baseflow [mm/day]
        """
        variable_mapping = {
            'streamflow': [
                'Qsim',
                'Q',
                'discharge',
                'Qrouted',
            ],
            'et': [
                'aET',
                'PET',
                'evapotranspiration',
            ],
            'snow': [
                'SWE',
                'snow_depth',
                'snowpack',
            ],
            'soil_moisture': [
                'SWC',
                'SM',
                'soil_moisture',
            ],
            'recharge': [
                'recharge',
                'GW_recharge',
            ],
            'baseflow': [
                'baseflow',
                'Qbase',
            ],
            'interflow': [
                'slowInterflow',
                'fastInterflow',
            ],
            'runoff': [
                'runoff_sealed',
                'total_runoff',
            ],
            'precipitation': [
                'pre',
                'precipitation',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from mHM output.

        Args:
            output_file: Path to mHM output file (NetCDF)
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - gauge_id: Gauge ID for discharge extraction (default: first)
                - aggregate: Whether to aggregate spatially (default: True)

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found or file cannot be parsed
        """
        import xarray as xr

        var_names = self.get_variable_names(variable_type)
        aggregate = kwargs.get('aggregate', True)
        gauge_id = kwargs.get('gauge_id', None)

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

            # Handle gauge selection for discharge
            if variable_type == 'streamflow' and gauge_id is not None:
                gauge_dim = None
                for dim in data.dims:
                    if dim not in ['time']:
                        gauge_dim = dim
                        break
                if gauge_dim is not None:
                    data = data.sel({gauge_dim: gauge_id})

            # Aggregate spatially if needed
            if aggregate and len(data.dims) > 1:
                spatial_dims = [d for d in data.dims if d not in ['time']]
                if spatial_dims:
                    if variable_type in ['streamflow']:
                        # For discharge, take first gauge (already in m3/s)
                        data = data.isel({spatial_dims[0]: 0}) if spatial_dims else data
                    else:
                        # For other variables, take spatial mean
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
                f"Error reading mHM output file {output_file}: {str(e)}"
            )

    def extract_streamflow(
        self,
        output_dir: Path,
        catchment_area: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """
        Extract streamflow from mHM discharge output.

        mHM discharge is already in m3/s, so no area conversion is needed.

        Args:
            output_dir: Directory containing mHM outputs
            catchment_area: Not needed (mHM outputs m3/s directly)
            **kwargs: Additional options

        Returns:
            Time series of streamflow in m3/s
        """
        import xarray as xr

        # Find discharge file
        patterns = self.get_output_file_patterns()['streamflow']
        output_file = None
        for pattern in patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break

        if output_file is None:
            raise FileNotFoundError(
                f"No mHM discharge file found in {output_dir} matching {patterns}"
            )

        ds = xr.open_dataset(output_file)

        # Get discharge variable
        discharge = None
        for var in self.get_variable_names('streamflow'):
            if var in ds.data_vars:
                discharge = ds[var]
                break

        if discharge is None:
            ds.close()
            raise ValueError("Could not find discharge variable in mHM output")

        # Handle spatial dimensions (select first gauge)
        spatial_dims = [d for d in discharge.dims if d not in ['time']]
        if spatial_dims:
            discharge = discharge.isel({spatial_dims[0]: 0})

        # Convert to series - already in m3/s
        series = discharge.to_series()
        ds.close()

        return series

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Check if variable requires unit conversion.

        mHM discharge is already in m3/s, so no conversion needed.
        """
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """Get spatial aggregation method for variable type."""
        if variable_type in ['streamflow']:
            return 'first'  # Take first gauge (discharge is already routed)
        elif variable_type in ['et', 'recharge', 'runoff', 'baseflow', 'interflow']:
            return 'sum'  # Sum over area
        else:
            return 'mean'  # Average for states like SWE, soil moisture
