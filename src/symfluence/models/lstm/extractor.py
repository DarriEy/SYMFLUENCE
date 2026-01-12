"""
LSTM Result Extractor.

Handles extraction of simulation results from LSTM (Long Short-Term Memory) model outputs.
LSTM is a data-driven surrogate model that predicts streamflow and possibly other
hydrological variables based on meteorological forcings.
"""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import xarray as xr

from symfluence.models.base import ModelResultExtractor


class LSTMResultExtractor(ModelResultExtractor):
    """LSTM-specific result extraction.

    Handles LSTM model's output characteristics:
    - Variable naming: predicted_streamflow, predicted_SWE, etc.
    - File patterns: *_LSTM_output.nc
    - Spatial dimensions: May have distributed outputs (hru/gru dimension)
    - Data-driven: Outputs are predictions, not physical simulations
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for LSTM outputs."""
        return {
            'streamflow': [
                '*_LSTM_output.nc',
                '*_lstm_output.nc',
                '*LSTM*.nc',
            ],
            'snow': [
                '*_LSTM_output.nc',
                '*_lstm_output.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get LSTM variable names for different types.

        LSTM uses 'predicted_' prefix for output variables.
        """
        variable_mapping = {
            'streamflow': [
                'predicted_streamflow',
                'streamflow',
                'discharge',
                'q_predicted',
            ],
            'snow': [
                'predicted_SWE',
                'scalarSWE',
            ],
            'snow_swe': [
                'predicted_SWE',
                'scalarSWE',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from LSTM output.

        Args:
            output_file: Path to LSTM NetCDF output
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - hru_index: HRU index to extract for distributed mode (default: 0)
                - aggregate_spatial: Whether to aggregate spatial dimensions

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

                    # Handle spatial dimensions
                    if len(var.shape) > 1:
                        # Find spatial dimensions (anything that's not time)
                        spatial_dims = [d for d in var.dims if d != 'time']

                        if spatial_dims:
                            # Check if HRU index provided
                            hru_index = kwargs.get('hru_index', 0)

                            # Check if aggregation requested
                            aggregate = kwargs.get('aggregate_spatial', False)

                            if aggregate:
                                # Mean aggregation over spatial dimension
                                var = var.mean(dim=spatial_dims[0])
                            else:
                                # Select specific HRU
                                var = var.isel({spatial_dims[0]: hru_index})

                    return var.to_pandas()

            raise ValueError(
                f"No suitable variable found for '{variable_type}' in {output_file}. "
                f"Tried: {var_names}"
            )

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """LSTM outputs are typically in standard units (m³/s for streamflow)."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """LSTM can have distributed outputs."""
        return 'outlet_selection'  # Default to selecting specific HRU/outlet
