"""
NGEN Result Extractor.

Handles extraction of simulation results from NextGen framework outputs.
NGEN outputs can come from troute (routing) or catchment-level results.
"""

from pathlib import Path
from typing import cast, List, Dict
import pandas as pd
import xarray as xr

from symfluence.models.base import ModelResultExtractor


class NGENResultExtractor(ModelResultExtractor):
    """NGEN-specific result extraction.

    Handles NextGen framework's output characteristics:
    - Variable naming: depends on formulation (CFE, NOAH, etc.)
    - File patterns: nexus_data.nc, catchment_data.nc, troute outputs
    - Routing: Often uses troute for streamflow routing
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for NGEN outputs."""
        return {
            'streamflow': [
                '**/nex-troute-out.nc',  # Troute routed outputs
                '**/troute_*.nc',
                '**/nexus_data.nc',      # Nexus outputs
                '**/catchment_data.nc',  # Catchment outputs
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get NGEN variable names for different types."""
        variable_mapping = {
            'streamflow': [
                'streamflow',
                'discharge',
                'q_lateral',
                'water_out',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from NGEN output.

        Args:
            output_file: Path to NGEN NetCDF output
            variable_type: Type of variable to extract
            **kwargs: Additional options (may include 'config' for target nexus)

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found
        """
        var_names = self.get_variable_names(variable_type)

        with xr.open_dataset(output_file) as ds:
            for var_name in var_names:
                if var_name in ds.variables:
                    var = ds[var_name]

                    # Handle spatial dimensions if present
                    if len(var.shape) > 1:
                        var = self._select_outlet_nexus(var, ds, variable_type, **kwargs)

                    return cast(pd.Series, var.to_pandas())

            raise ValueError(
                f"No suitable variable found for '{variable_type}' in {output_file}. "
                f"Tried: {var_names}"
            )

    def _select_outlet_nexus(
        self,
        var: xr.DataArray,
        ds: xr.Dataset,
        variable_type: str,
        **kwargs
    ) -> xr.DataArray:
        """Select outlet nexus from spatial data.

        Attempts to find the outlet nexus using multiple strategies:
        1. CALIBRATION_NEXUS_ID from config if provided
        2. Nexus with highest mean streamflow (outlet typically has highest flow)
        3. Largest drainage area if available
        4. Fallback to first spatial unit

        Args:
            var: Variable with spatial dimensions
            ds: Full dataset for accessing coordinates/variables
            variable_type: Type of variable being extracted
            **kwargs: May contain 'config' dict with CALIBRATION_NEXUS_ID

        Returns:
            Variable with spatial dimension selected to outlet
        """
        spatial_dims = [d for d in var.dims if d != 'time']
        if not spatial_dims:
            return var

        spatial_dim = spatial_dims[0]  # Primary spatial dimension
        n_spatial = var.sizes[spatial_dim]

        # Strategy 1: Use CALIBRATION_NEXUS_ID from config if provided
        config = kwargs.get('config', {})
        target_id = self._get_target_nexus_id(config)

        if target_id:
            coord = ds.coords.get(spatial_dim)
            if coord is not None:
                coord_values = coord.values.astype(str)
                # Try different ID format patterns
                target_patterns = [
                    str(target_id),
                    f"cat-{target_id}",
                    f"nex-{target_id}",
                    f"catchment-{target_id}",
                    f"nexus-{target_id}"
                ]

                for pattern in target_patterns:
                    matches = [i for i, v in enumerate(coord_values) if pattern in v]
                    if matches:
                        return var.isel({spatial_dim: matches[0]})

        # Strategy 2: For streamflow, find nexus with highest mean flow
        # The outlet nexus typically has the highest accumulated streamflow
        if variable_type == 'streamflow' and n_spatial > 1:
            mean_flow = var.mean(dim='time')
            outlet_idx = int(mean_flow.argmax())
            return var.isel({spatial_dim: outlet_idx})

        # Strategy 3: Check for drainage area (largest = outlet)
        if 'drainage_area' in ds.variables:
            drainage_area = ds['drainage_area']
            outlet_idx = int(drainage_area.argmax())
            return var.isel({spatial_dim: outlet_idx})

        # Fallback: Select first spatial unit
        return var.isel({spatial_dim: 0})

    def _get_target_nexus_id(self, config):
        """Extract CALIBRATION_NEXUS_ID from config (dict or typed object)."""
        if isinstance(config, dict):
            return config.get('CALIBRATION_NEXUS_ID')

        # Handle typed config objects
        if hasattr(config, 'calibration_nexus_id'):
            return config.calibration_nexus_id

        # Check nested ngen config
        if hasattr(config, 'ngen'):
            ngen_config = getattr(config, 'ngen', None)
            if ngen_config and hasattr(ngen_config, 'calibration_nexus_id'):
                return ngen_config.calibration_nexus_id

        return None

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """NGEN outputs are typically in standard units."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """NGEN can have catchment or nexus level outputs."""
        return 'outlet_selection'
