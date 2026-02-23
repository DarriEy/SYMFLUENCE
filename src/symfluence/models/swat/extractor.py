"""
SWAT Result Extractor.

Handles extraction of simulation results from SWAT model outputs.
SWAT outputs are in fixed-width text format, primarily output.rch.
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.models.base import ModelResultExtractor


class SWATResultExtractor(ModelResultExtractor):
    """SWAT-specific result extraction.

    Handles SWAT model's output characteristics:
    - File formats: Fixed-width text (output.rch, output.sub, output.hru)
    - Variable naming: FLOW_OUTcms, SED_OUTtons, etc.
    - Spatial aggregation: Reaches to basin outlet
    """

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for SWAT outputs."""
        return {
            'streamflow': [
                'output.rch',
            ],
            'sediment': [
                'output.rch',
                'output.sed',
            ],
            'et': [
                'output.sub',
                'output.hru',
            ],
            'snow': [
                'output.sub',
                'output.sno',
            ],
            'soil_moisture': [
                'output.hru',
                'output.sol',
            ],
            'groundwater': [
                'output.gw',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get SWAT variable names for different types.

        SWAT output variables are in fixed-width columns:
        - FLOW_OUTcms: Streamflow at reach outlet [m3/s]
        - SED_OUTtons: Sediment yield [metric tons]
        - ET: Evapotranspiration [mm]
        - SW: Soil water content [mm]
        """
        variable_mapping = {
            'streamflow': [
                'FLOW_OUTcms',
                'FLOW_INcms',
            ],
            'sediment': [
                'SED_OUTtons',
                'SED_INtons',
            ],
            'et': [
                'ET',
                'PET',
                'LATQ',
            ],
            'snow': [
                'SNO_HRU',
                'SNOMELT',
            ],
            'soil_moisture': [
                'SW',
                'SW_END',
            ],
            'groundwater': [
                'GW_Q',
                'GW_QD',
                'REVAP',
            ],
            'precipitation': [
                'PRECIP',
                'PCP',
            ],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from SWAT output.

        Args:
            output_file: Path to SWAT output file (fixed-width text)
            variable_type: Type of variable to extract
            **kwargs: Additional options:
                - reach_id: Reach number for output.rch (default: 1)
                - start_date: Simulation start date for building time index

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable not found or file cannot be parsed
        """
        reach_id = kwargs.get('reach_id', 1)
        start_date = kwargs.get('start_date', '2000-01-01')

        if not output_file.exists():
            raise ValueError(f"SWAT output file not found: {output_file}")

        try:
            if output_file.name == 'output.rch':
                return self._parse_output_rch(
                    output_file, variable_type, reach_id, start_date
                )
            elif output_file.name == 'output.sub':
                return self._parse_output_sub(
                    output_file, variable_type, start_date
                )
            else:
                raise ValueError(
                    f"Unsupported SWAT output file: {output_file.name}"
                )
        except Exception as e:
            raise ValueError(
                f"Error reading SWAT output file {output_file}: {str(e)}"
            )

    def _parse_output_rch(
        self,
        output_file: Path,
        variable_type: str,
        reach_id: int,
        start_date: str
    ) -> pd.Series:
        """Parse output.rch and extract the requested variable."""
        with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Find header end
        header_end = 0
        header_line = None
        for i, line in enumerate(lines):
            if 'RCH' in line.upper() and ('FLOW' in line.upper() or 'MON' in line.upper()):
                header_line = line
                header_end = i + 1
                break
        if header_end == 0:
            header_end = 9

        # Determine column index for requested variable
        var_names = self.get_variable_names(variable_type)
        col_idx = 5  # Default: FLOW_OUTcms is typically column 5 (0-indexed)

        if header_line:
            header_parts = header_line.split()
            for var_name in var_names:
                for idx, part in enumerate(header_parts):
                    if var_name.lower() in part.lower():
                        col_idx = idx
                        break

        # Parse data lines
        values = []
        data_lines = lines[header_end:]

        for line in data_lines:
            parts = line.split()
            if len(parts) < col_idx + 1:
                continue
            try:
                rch = int(parts[0])
                if rch == reach_id:
                    values.append(float(parts[col_idx]))
            except (ValueError, IndexError):
                continue

        if not values:
            raise ValueError(
                f"No data found for reach {reach_id} in {output_file}"
            )

        dates = pd.date_range(
            start=pd.to_datetime(start_date),
            periods=len(values),
            freq='D'
        )
        return pd.Series(values, index=dates, name=variable_type)

    def _parse_output_sub(
        self,
        output_file: Path,
        variable_type: str,
        start_date: str
    ) -> pd.Series:
        """Parse output.sub and extract the requested variable."""
        with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        header_end = 0
        for i, line in enumerate(lines):
            if 'SUB' in line.upper() and 'MON' in line.upper():
                header_end = i + 1
                break
        if header_end == 0:
            header_end = 9

        # Parse data for sub-basin 1
        values = []
        for line in lines[header_end:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                sub = int(parts[0])
                if sub == 1:
                    values.append(float(parts[5]))
            except (ValueError, IndexError):
                continue

        if not values:
            raise ValueError(f"No data found in {output_file}")

        dates = pd.date_range(
            start=pd.to_datetime(start_date),
            periods=len(values),
            freq='D'
        )
        return pd.Series(values, index=dates, name=variable_type)

    def extract_streamflow(
        self,
        output_dir: Path,
        catchment_area: Optional[float] = None,
        **kwargs
    ) -> pd.Series:
        """
        Extract streamflow from SWAT output.rch.

        Args:
            output_dir: Directory containing SWAT outputs (TxtInOut)
            catchment_area: Not needed for SWAT (output already in m3/s)
            **kwargs: Additional options (reach_id, start_date)

        Returns:
            Time series of streamflow in m3/s
        """
        output_rch = output_dir / 'output.rch'
        if not output_rch.exists():
            raise FileNotFoundError(
                f"SWAT output.rch not found in {output_dir}"
            )

        reach_id = kwargs.get('reach_id', 1)
        start_date = kwargs.get('start_date', '2000-01-01')

        series = self._parse_output_rch(
            output_rch, 'streamflow', reach_id, start_date
        )

        # SWAT FLOW_OUTcms is already in m3/s
        return series

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Check if variable requires unit conversion.

        SWAT streamflow is already in m3/s (cms), so no conversion needed.
        """
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """Get spatial aggregation method for variable type."""
        if variable_type in ['streamflow']:
            return None  # Already aggregated at reach outlet
        elif variable_type in ['et', 'precipitation', 'sediment']:
            return 'sum'
        else:
            return 'mean'
