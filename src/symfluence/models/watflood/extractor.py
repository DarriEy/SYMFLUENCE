# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WATFLOOD Result Extractor.

Extracts results from WATFLOOD output files including .tb0 (time-bin)
format and CSV files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.models.base import ModelResultExtractor

logger = logging.getLogger(__name__)


class WATFLOODResultExtractor(ModelResultExtractor):
    """Extract results from WATFLOOD output files."""

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        return {
            'streamflow': ['spl*.tb0', 'resin*.tb0', '*.csv', 'streamflow*.csv'],
            'runoff': ['*.tb0'],
            'et': ['*.tb0'],
            'snow': ['*.tb0'],
        }

    # Internal mapping of variable types to their possible names in output files
    _variable_name_map: Dict[str, List[str]] = {
        'streamflow': ['QO', 'QSIM', 'flow', 'discharge'],
        'runoff': ['ROF', 'runoff'],
        'et': ['ET', 'evap'],
        'snow': ['SWE', 'snow_depth'],
    }

    def get_variable_names(self, variable_type: str) -> List[str]:
        return self._variable_name_map.get(variable_type, [variable_type])

    def extract_variable(
        self, output_file: Path, variable_type: str, **kwargs
    ) -> pd.Series:
        if not output_file.exists():
            raise ValueError(f"Output file not found: {output_file}")

        try:
            if output_file.suffix == '.csv':
                df = pd.read_csv(output_file, parse_dates=[0], index_col=0)
                var_names = self.get_variable_names(variable_type)
                for var in var_names:
                    for col in df.columns:
                        if var.lower() in col.lower():
                            return df[col]
            elif output_file.suffix == '.tb0':
                result = self._parse_tb0_file(output_file, variable_type)
                if result is not None:
                    return result
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ValueError(f"Error extracting {variable_type}: {e}") from e

        raise ValueError(f"Variable '{variable_type}' not found in {output_file}")

    def _parse_tb0_file(
        self, tb0_file: Path, variable_type: str
    ) -> Optional[pd.Series]:
        """Parse WATFLOOD .tb0 (time-bin) format.

        tb0 format has a header section followed by data lines with:
        year month day hour value(s)
        """
        try:
            lines = tb0_file.read_text(encoding='utf-8').strip().split('\n')

            # Skip header lines (lines starting with : or #)
            data_start = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith((':','#')) and stripped[0].isdigit():
                    data_start = i
                    break

            dates, values = [], []
            for line in lines[data_start:]:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        hour = int(parts[3])
                        value = float(parts[4])
                        dates.append(pd.Timestamp(year=year, month=month, day=day, hour=hour))
                        values.append(value)
                    except (ValueError, IndexError):
                        continue

            if dates:
                return pd.Series(values, index=dates, name=f'WATFLOOD_{variable_type}')
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Error parsing tb0 file: {e}")
        return None

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
            raise FileNotFoundError(f"No WATFLOOD output found in {output_dir}")

        result = self.extract_variable(output_file, 'streamflow')
        return result

    def requires_unit_conversion(self, variable_type: str) -> bool:
        return variable_type not in ['streamflow']

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        if variable_type in ['streamflow']:
            return 'sum'
        return 'mean'
