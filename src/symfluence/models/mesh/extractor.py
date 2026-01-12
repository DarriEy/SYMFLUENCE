"""
MESH Result Extractor.

Handles extraction of simulation results from MESH (Modélisation
Environmentale Communautaire - Surface and Hydrology) model outputs.
"""

from pathlib import Path
from typing import List, Dict
import pandas as pd
from datetime import datetime, timedelta

from symfluence.models.base import ModelResultExtractor


class MESHResultExtractor(ModelResultExtractor):
    """MESH-specific result extraction.

    Handles MESH's unique output characteristics:
    - File format: CSV files (MESH_output_streamflow.csv)
    - Variable naming: QOSIM* (simulated streamflow), ET*, SNOW*, etc.
    - Time format: Julian day (DAY) and YEAR columns
    - Spatial: Multiple subbasins (QOSIM1, QOSIM2, etc.)
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for MESH outputs."""
        return {
            'streamflow': [
                'MESH_output_streamflow.csv',
                'MESH_output_streamflow_ts.csv',
                'forcing/MESH_input/MESH_output_streamflow.csv',
            ],
            'et': [
                'MESH_output_balance.csv',
                'forcing/MESH_input/MESH_output_balance.csv',
            ],
            'snow': [
                'MESH_output_balance.csv',
                'forcing/MESH_input/MESH_output_balance.csv',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get MESH variable names for different types."""
        variable_mapping = {
            'streamflow': ['QOSIM1', 'QOSIM2', 'QOSIM'],  # Simulated streamflow
            'et': ['EVAP', 'ET', 'EVAPOTRANSPIRATION'],
            'snow': ['SNOW', 'SWE', 'SNOWPACK'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from MESH CSV output.

        Args:
            output_file: Path to MESH CSV output
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - subbasin_index: Which subbasin to extract (default: 0 for outlet)

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found
        """
        if output_file.suffix != '.csv':
            raise ValueError(f"MESH extractor only supports CSV files, got {output_file.suffix}")

        try:
            # Read MESH CSV file
            df = pd.read_csv(output_file, skipinitialspace=True)

            # Convert DAY and YEAR to datetime
            df['datetime'] = df.apply(self._julian_to_datetime, axis=1)

            # Find the variable column
            if variable_type == 'streamflow':
                return self._extract_streamflow(df, **kwargs)
            else:
                return self._extract_generic_variable(df, variable_type)

        except Exception as e:
            raise ValueError(f"Failed to extract {variable_type} from {output_file}: {e}")

    def _julian_to_datetime(self, row) -> datetime:
        """Convert Julian day and year to datetime.

        Args:
            row: DataFrame row with DAY and YEAR columns

        Returns:
            datetime object
        """
        year = int(row['YEAR'])
        day = int(row['DAY'])
        return datetime(year, 1, 1) + timedelta(days=day - 1)

    def _extract_streamflow(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        """Extract streamflow from MESH output.

        Args:
            df: DataFrame with MESH output
            **kwargs: Optional subbasin_index

        Returns:
            Streamflow time series
        """
        # Find QOSIM columns (simulated streamflow)
        streamflow_cols = [col for col in df.columns if col.startswith('QOSIM')]

        if not streamflow_cols:
            raise ValueError("No simulated streamflow columns (QOSIM*) found in MESH output")

        # Select subbasin (default to first - usually the outlet)
        subbasin_index = kwargs.get('subbasin_index', 0)
        if subbasin_index >= len(streamflow_cols):
            subbasin_index = 0

        selected_col = streamflow_cols[subbasin_index]

        # Create time series
        return pd.Series(
            df[selected_col].values,
            index=df['datetime'],
            name=selected_col
        )

    def _extract_generic_variable(self, df: pd.DataFrame, variable_type: str) -> pd.Series:
        """Extract generic variable from MESH output.

        Args:
            df: DataFrame with MESH output
            variable_type: Type of variable

        Returns:
            Variable time series
        """
        var_names = self.get_variable_names(variable_type)

        # Find matching column
        for var_name in var_names:
            matching_cols = [col for col in df.columns if var_name in col.upper()]
            if matching_cols:
                selected_col = matching_cols[0]
                return pd.Series(
                    df[selected_col].values,
                    index=df['datetime'],
                    name=selected_col
                )

        raise ValueError(
            f"No suitable variable found for '{variable_type}'. "
            f"Tried: {var_names}"
        )

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """MESH outputs are typically in standard units (m³/s for streamflow)."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> str:
        """MESH outputs by subbasin/GRU."""
        return 'selection'  # Select specific subbasin/GRU
