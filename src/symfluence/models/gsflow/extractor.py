# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GSFLOW Result Extractor.

Extracts results from GSFLOW output files including PRMS statvar
output and MODFLOW listing/head files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.models.base import ModelResultExtractor

logger = logging.getLogger(__name__)


class GSFLOWResultExtractor(ModelResultExtractor):
    """Extract results from GSFLOW output files."""

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Return file patterns for GSFLOW outputs."""
        return {
            'streamflow': ['statvar*', '*.csv', 'gsflow_*.csv'],
            'et': ['statvar*'],
            'soil_moisture': ['statvar*'],
            'heads': ['*.hds', '*.bhd'],
            'water_budget': ['*.lst', 'gsflow_*.out'],
        }

    _variable_name_map: Dict[str, List[str]] = {
        'streamflow': ['seg_outflow', 'basin_cfs', 'basin_cms', 'streamflow_cfs'],
        'et': ['hru_actet', 'potet', 'basin_actet'],
        'soil_moisture': ['soil_moist', 'soil_rechr', 'basin_soil_moist'],
        'heads': ['head'],
        'baseflow': ['gw_flow', 'sfr_flow'],
    }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Return GSFLOW variable names for a given type."""
        return self._variable_name_map.get(variable_type, [variable_type])

    def extract_variable(
        self, output_file: Path, variable_type: str, **kwargs
    ) -> pd.Series:
        """Extract a variable from GSFLOW output."""
        if not output_file.exists():
            raise ValueError(f"Output file not found: {output_file}")

        try:
            if output_file.suffix == '.csv':
                df = pd.read_csv(output_file, parse_dates=[0], index_col=0)
                var_names = self.get_variable_names(variable_type)
                for var in var_names:
                    if var in df.columns:
                        return df[var]
            else:
                result = self._parse_statvar_file(output_file, variable_type)
                if result is not None:
                    return result
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            raise ValueError(f"Error extracting {variable_type}: {e}") from e

        raise ValueError(f"Variable '{variable_type}' not found in {output_file}")

    def _parse_statvar_file(
        self, statvar_file: Path, variable_type: str
    ) -> Optional[pd.Series]:
        """Parse PRMS/GSFLOW statvar.dat format."""
        try:
            lines = statvar_file.read_text(encoding='utf-8').strip().split('\n')
            dates, values = [], []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 9:
                    try:
                        year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
                        streamflow = float(parts[7])
                        dates.append(pd.Timestamp(year=year, month=month, day=day))
                        values.append(streamflow * 0.0283168)  # cfs to cms
                    except (ValueError, IndexError):
                        continue
            if dates:
                return pd.Series(values, index=dates, name=f'GSFLOW_{variable_type}')
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Error parsing statvar: {e}")
        return None

    def extract_streamflow(
        self, output_dir: Path, catchment_area: Optional[float] = None, **kwargs
    ) -> pd.Series:
        """Extract streamflow from GSFLOW output."""
        patterns = self.get_output_file_patterns()['streamflow']
        output_file = None
        for pattern in patterns:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break

        if output_file is None:
            raise FileNotFoundError(f"No GSFLOW output found in {output_dir}")

        result = self.extract_variable(output_file, 'streamflow')
        return result

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """GSFLOW seg_outflow is converted to cms during extraction."""
        return variable_type not in ['streamflow']

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        if variable_type in ['streamflow']:
            return 'sum'
        return 'mean'
