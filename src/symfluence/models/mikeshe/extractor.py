"""
MIKE-SHE Result Extractor.

Handles extraction of simulation results from MIKE-SHE model outputs.
MIKE-SHE outputs are in .dfs0 (DHI binary time series) or CSV format.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from symfluence.models.base import ModelResultExtractor


class MIKESHEResultExtractor(ModelResultExtractor):
    """MIKE-SHE-specific result extraction.

    Handles MIKE-SHE model's output characteristics:
    - File formats: .dfs0 (binary) or .csv (text export)
    - Variable naming: discharge, overland_flow, baseflow, etc.
    - Components: overland flow + drain flow + baseflow = total discharge
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for MIKE-SHE outputs."""
        return {
            'streamflow': [
                '*discharge*.csv',
                '*flow*.csv',
                '*discharge*.dfs0',
                '*.csv',
            ],
            'et': [
                '*evapotranspiration*.csv',
                '*et*.csv',
                '*evapotranspiration*.dfs0',
            ],
            'groundwater': [
                '*groundwater*.csv',
                '*head*.csv',
                '*groundwater*.dfs0',
            ],
            'soil_moisture': [
                '*soil_moisture*.csv',
                '*moisture*.csv',
                '*unsaturated*.dfs0',
            ],
            'snow': [
                '*snow*.csv',
                '*swe*.csv',
                '*snow*.dfs0',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get MIKE-SHE variable names for different types.

        MIKE-SHE CSV export columns typically use descriptive names:
        - discharge / total_flow: Total discharge [m3/s]
        - overland_flow: Surface runoff component [m3/s]
        - baseflow: Groundwater contribution [m3/s]
        - evapotranspiration: Actual ET [mm]
        """
        variable_mapping = {
            'streamflow': [
                'discharge',
                'total_flow',
                'q_total',
                'total_discharge',
                'overland_flow',
                'runoff',
            ],
            'overland_flow': [
                'overland_flow',
                'surface_runoff',
                'ol_flow',
            ],
            'baseflow': [
                'baseflow',
                'base_flow',
                'groundwater_discharge',
                'sz_drain',
            ],
            'et': [
                'evapotranspiration',
                'actual_et',
                'aet',
                'total_et',
            ],
            'groundwater': [
                'groundwater_head',
                'head',
                'water_table',
                'sz_head',
            ],
            'soil_moisture': [
                'soil_moisture',
                'moisture_content',
                'uz_moisture',
                'theta',
            ],
            'snow': [
                'snow_water_equivalent',
                'swe',
                'snow_depth',
                'snow_cover',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from MIKE-SHE output.

        Args:
            output_file: Path to MIKE-SHE output file (CSV or dfs0)
            variable_type: Type of variable to extract
            **kwargs: Additional options

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found or file cannot be parsed
        """
        var_names = self.get_variable_names(variable_type)

        try:
            if output_file.suffix == '.csv':
                df = pd.read_csv(output_file, parse_dates=[0])
                datetime_col = df.columns[0]
                df.set_index(datetime_col, inplace=True)

                # Find the matching column
                found_col = None
                for var_name in var_names:
                    for col in df.columns:
                        if var_name.lower() in col.lower():
                            found_col = col
                            break
                    if found_col:
                        break

                if found_col is None:
                    # Fall back to first numeric column
                    numeric_cols = df.select_dtypes(include='number').columns
                    if len(numeric_cols) > 0:
                        found_col = numeric_cols[0]
                    else:
                        available = list(df.columns)
                        raise ValueError(
                            f"No suitable variable found in {output_file}. "
                            f"Tried: {var_names}. Available: {available}"
                        )

                series = df[found_col]
                return series

            else:
                # dfs0 format - attempt basic text-based parsing
                df = pd.read_csv(output_file, sep=r'\s+', parse_dates=[0])
                datetime_col = df.columns[0]
                df.set_index(datetime_col, inplace=True)
                return df.iloc[:, 0]

        except Exception as e:
            raise ValueError(
                f"Error reading MIKE-SHE output file {output_file}: {str(e)}"
            )

    def extract_streamflow(
        self,
        output_dir: Path,
        catchment_area: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """
        Extract total streamflow from MIKE-SHE output.

        MIKE-SHE discharge output is typically already in m3/s.

        Args:
            output_dir: Directory containing MIKE-SHE outputs
            catchment_area: Catchment area in m2 (not typically needed)
            **kwargs: Additional options

        Returns:
            Time series of streamflow in m3/s
        """
        # Find output file
        patterns = self.get_output_file_patterns()['streamflow']
        output_file = None
        for pattern in patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break

        if output_file is None:
            raise FileNotFoundError(
                f"No MIKE-SHE output file found in {output_dir} matching {patterns}"
            )

        series = self.extract_variable(output_file, 'streamflow')
        return series

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Check if variable requires unit conversion.

        MIKE-SHE typically outputs discharge in m3/s, so no conversion
        is needed for streamflow.
        """
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """Get spatial aggregation method for variable type."""
        if variable_type in ['streamflow', 'overland_flow', 'baseflow']:
            return 'sum'
        else:
            return 'mean'
