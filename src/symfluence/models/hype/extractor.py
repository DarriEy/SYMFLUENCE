"""
HYPE Result Extractor.

Handles extraction of simulation results from HYPE model outputs.
HYPE outputs are primarily text-based (timeCOUT.txt) but may also
use mizuRoute for routing.
"""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from symfluence.models.base import ModelResultExtractor


class HYPEResultExtractor(ModelResultExtractor):
    """HYPE-specific result extraction.

    Handles HYPE's unique output characteristics:
    - File format: timeCOUT.txt (text file)
    - Variable naming: COUT (computed outflow)
    - Routing: Can use mizuRoute for additional routing
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for HYPE outputs."""
        return {
            'streamflow': [
                'timeCOUT.txt',
                'HYPE/timeCOUT.txt',
                '**/timeCOUT.txt',
                # Fallback to mizuRoute
                'mizuRoute/*.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get HYPE variable names for different types."""
        variable_mapping = {
            'streamflow': ['COUT', 'streamflow', 'discharge'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from HYPE output.

        Args:
            output_file: Path to HYPE output file (txt or nc)
            variable_type: Type of variable to extract
            **kwargs: Additional options

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable cannot be extracted
        """
        if output_file.suffix == '.txt':
            # HYPE text output
            return self._extract_from_text(output_file)
        elif output_file.suffix == '.nc':
            # mizuRoute NetCDF output
            import xarray as xr
            import numpy as np

            with xr.open_dataset(output_file) as ds:
                # Use mizuRoute variables
                for var_name in ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']:
                    if var_name in ds.variables:
                        var = ds[var_name]
                        if 'seg' in var.dims:
                            outlet_idx = np.argmax(var.mean(dim='time').values)
                            return var.isel(seg=outlet_idx).to_pandas()
                        elif 'reachID' in var.dims:
                            outlet_idx = np.argmax(var.mean(dim='time').values)
                            return var.isel(reachID=outlet_idx).to_pandas()

        raise ValueError(f"Could not extract {variable_type} from {output_file}")

    def _extract_from_text(self, output_file: Path) -> pd.Series:
        """Extract streamflow from HYPE timeCOUT.txt file.

        Args:
            output_file: Path to timeCOUT.txt

        Returns:
            Time series of streamflow
        """
        # HYPE timeCOUT.txt format is typically:
        # DATE COUT
        # or has header with subbasin IDs
        try:
            df = pd.read_csv(output_file, sep='\\s+', parse_dates=[0], index_col=0)
            # Take first column (or sum across all subbasins)
            if df.shape[1] == 1:
                return df.iloc[:, 0]
            else:
                # Sum across all subbasins
                return df.sum(axis=1)
        except Exception as e:
            raise ValueError(f"Failed to parse HYPE output file {output_file}: {e}")

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """HYPE outputs are in m³/s."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """HYPE aggregates internally or uses routing."""
        return 'sum'  # Sum across subbasins if multiple
