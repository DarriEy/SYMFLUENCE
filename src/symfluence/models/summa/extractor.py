"""
SUMMA Result Extractor.

Handles extraction of simulation results from SUMMA model outputs.
Encapsulates SUMMA-specific logic for variable names, units, spatial
dimensions, and file patterns.
"""

from pathlib import Path
from typing import List, Dict, Optional, cast
import pandas as pd
import xarray as xr

from symfluence.models.base import ModelResultExtractor


class SUMMAResultExtractor(ModelResultExtractor):
    """SUMMA-specific result extraction.

    Handles SUMMA's unique output characteristics:
    - Variable naming: averageRoutedRunoff, scalarTotalRunoff, etc.
    - Unit conversion: mass flux (kg m⁻² s⁻¹) → volume flux (m s⁻¹)
    - Spatial aggregation: HRU/GRU area-weighted sums
    - File patterns: *_timestep.nc, *_day.nc
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for SUMMA outputs."""
        return {
            'streamflow': ['*_timestep.nc', '*_day.nc', '*output*.nc'],
            'snow': ['*_day.nc', '*_timestep.nc'],
            'et': ['*_day.nc', '*_timestep.nc'],
            'soil_moisture': ['*_day.nc', '*_timestep.nc'],
            'groundwater': ['*_day.nc', '*_timestep.nc'],
            'tws': ['*_day.nc', '*_timestep.nc'],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get SUMMA variable names for different types."""
        variable_mapping = {
            'streamflow': ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff'],
            'snow_swe': ['scalarSWE', 'scalarSnowDepth'],
            'snow_sca': ['scalarSnowCover'],
            'et': ['scalarLatHeatTotal', 'basin__ET'],
            'soil_moisture': ['scalarSoilMoisture', 'mLayerVolFracLiq'],
            'groundwater': ['scalarAquiferStorage'],
            'tws': ['scalarTotalSoilWat', 'scalarAquiferStorage'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from SUMMA output with unit conversion and aggregation.

        Args:
            output_file: Path to SUMMA NetCDF output
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - project_dir: Path for finding attributes.nc
                - catchment_area: Optional catchment area override

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found in output
        """
        var_names = self.get_variable_names(variable_type)

        with xr.open_dataset(output_file) as ds:
            for var_name in var_names:
                if var_name in ds.variables:
                    var = ds[var_name]

                    # Apply unit conversion if needed
                    if variable_type == 'streamflow':
                        var = self._convert_streamflow_units(var, var_name)
                        var = self._aggregate_spatial_dimensions(var, kwargs.get('project_dir'))
                        # Apply catchment area scaling if needed
                        result = cast(pd.Series, var.to_pandas())
                        catchment_area = kwargs.get('catchment_area')
                        if catchment_area is not None:
                            result = result * catchment_area
                        return result
                    else:
                        # For other variables, handle spatial aggregation
                        var = self._aggregate_spatial_dimensions(var, kwargs.get('project_dir'))
                        return cast(pd.Series, var.to_pandas())

            raise ValueError(
                f"No suitable variable found for '{variable_type}' in {output_file}. "
                f"Tried: {var_names}"
            )

    def _convert_streamflow_units(self, var: xr.DataArray, var_name: str) -> xr.DataArray:
        """Convert SUMMA streamflow from mass flux to volume flux if needed.

        SUMMA may output runoff in mass flux (kg m⁻² s⁻¹) incorrectly
        labeled as volume flux (m s⁻¹). Detection:
        - Check units attribute for 'kg'
        - Check data magnitude: if mean > 1e-6 m/s, likely mislabeled

        Args:
            var: xarray DataArray
            var_name: Variable name

        Returns:
            Converted xarray DataArray
        """
        units = var.attrs.get('units', 'unknown')

        # Check for explicit mass flux units
        is_mass_flux = False
        if 'units' in var.attrs and 'kg' in units and 's-1' in units:
            is_mass_flux = True
        # Check for unreasonably high values (> 1e-6 m/s = > 86 mm/day mean)
        elif float(var.mean().item()) > 1e-6:
            is_mass_flux = True

        if is_mass_flux:
            # Divide by water density (1000 kg/m³)
            return var / 1000.0

        return var

    def _aggregate_spatial_dimensions(
        self,
        var: xr.DataArray,
        project_dir: Optional[Path]
    ) -> xr.DataArray:
        """Perform area-weighted spatial aggregation for distributed SUMMA.

        Args:
            var: xarray DataArray with potential HRU/GRU dimensions
            project_dir: Project directory for finding attributes.nc

        Returns:
            Spatially aggregated xarray DataArray
        """
        # Check if spatial aggregation is needed
        if len(var.shape) <= 1 or not any(d in var.dims for d in ['hru', 'gru']):
            return var

        # Try area-weighted aggregation
        if project_dir:
            attrs_file = project_dir / 'settings' / 'SUMMA' / 'attributes.nc'
            if attrs_file.exists():
                try:
                    with xr.open_dataset(attrs_file) as attrs:
                        # Handle HRU dimension
                        if 'hru' in var.dims and 'HRUarea' in attrs:
                            areas = attrs['HRUarea']
                            if areas.sizes['hru'] == var.sizes['hru']:
                                return (var * areas).sum(dim='hru')

                        # Handle GRU dimension
                        elif 'gru' in var.dims and 'GRUarea' in attrs:
                            areas = attrs['GRUarea']
                            if areas.sizes['gru'] == var.sizes['gru']:
                                return (var * areas).sum(dim='gru')

                        # Fallback: GRU dim with HRUarea (lumped 1:1 mapping)
                        elif 'gru' in var.dims and 'HRUarea' in attrs:
                            if attrs.sizes['hru'] == var.sizes['gru']:
                                areas = attrs['HRUarea']
                                return (var * areas.values).sum(dim='gru')
                except Exception:
                    pass  # Fall through to simple selection

        # Fallback: select first spatial unit
        if 'hru' in var.dims:
            return var.isel(hru=0)
        elif 'gru' in var.dims:
            return var.isel(gru=0)
        else:
            non_time_dims = [dim for dim in var.dims if dim != 'time']
            if non_time_dims:
                return var.isel({non_time_dims[0]: 0})

        return var

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Check if variable requires unit conversion."""
        return variable_type == 'streamflow'

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """Get spatial aggregation method."""
        return 'weighted'  # SUMMA uses area-weighted aggregation
