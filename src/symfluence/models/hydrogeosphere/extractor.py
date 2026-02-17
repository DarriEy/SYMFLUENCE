"""
HydroGeoSphere Result Extractor

Extracts results from HGS output files:
- Hydrograph at outlet from *hydrograph*.dat files (time, Q in m3/s)
- Head from *head_pm*.dat files
- Water balance from *water_balance*.dat files

HGS output files are tab/space-delimited text with header lines.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from symfluence.models.base.base_extractor import ModelResultExtractor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_result_extractor("HYDROGEOSPHERE")
class HGSResultExtractor(ModelResultExtractor):
    """
    Extracts results from HGS output files.

    Supports extraction of:
    - Hydrograph discharge from *hydrograph*.dat files
    - Porous media head from *head_pm*.dat files
    - Water balance from *water_balance*.dat files
    """

    def __init__(self, model_name: str = 'HYDROGEOSPHERE'):
        super().__init__(model_name)
        self.logger = logger

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for locating HGS outputs."""
        return {
            'hydrograph': ['*hydrograph*.dat'],
            'head': ['*head_pm*.dat', '*head*.dat'],
            'water_balance': ['*water_balance*.dat'],
        }

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs,
    ) -> pd.Series:
        """Extract a variable time series from HGS output.

        Args:
            output_file: Path to output file or directory
            variable_type: Type ('hydrograph', 'head', 'baseflow')
            **kwargs: Additional args (start_date for time index)

        Returns:
            Time series of extracted variable
        """
        output_path = Path(output_file)

        if variable_type in ('hydrograph', 'baseflow'):
            return self._extract_hydrograph(output_path, **kwargs)
        elif variable_type == 'head':
            return self._extract_head(output_path, **kwargs)
        else:
            raise ValueError(f"Unknown variable type: {variable_type}")

    def _extract_hydrograph(
        self,
        output_path: Path,
        **kwargs,
    ) -> pd.Series:
        """Extract hydrograph discharge from HGS output.

        HGS hydrograph format: space-delimited with header,
        columns are typically: time(s) Q(m3/s)
        """
        if output_path.is_dir():
            files = sorted(output_path.glob("*hydrograph*Outlet*"))
            if not files:
                files = sorted(output_path.glob("*hydrograph*"))
            if not files:
                raise ValueError(f"No hydrograph files found in {output_path}")
            data_file = files[0]
        else:
            data_file = output_path

        start_date = kwargs.get('start_date', '2000-01-01')

        times_s = []
        values = []

        with open(data_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('!'):
                    continue
                # Skip header lines
                parts = line.split()
                try:
                    t = float(parts[0])
                    q = float(parts[1]) if len(parts) > 1 else 0.0
                    times_s.append(t)
                    values.append(abs(q))
                except (ValueError, IndexError):
                    continue

        if not values:
            self.logger.warning(f"No data extracted from {data_file}")
            return pd.Series(dtype=float, name='hydrograph_m3s')

        # Convert time in seconds to daily
        values_arr = np.array(values)
        times_arr = np.array(times_s)

        # Resample to daily
        n_days = int(times_arr[-1] / 86400) + 1 if len(times_arr) > 0 else 0
        daily_values = np.zeros(n_days)

        for i in range(n_days):
            t_start = i * 86400
            t_end = (i + 1) * 86400
            mask = (times_arr >= t_start) & (times_arr < t_end)
            if np.any(mask):
                daily_values[i] = np.mean(values_arr[mask])

        date_index = pd.date_range(
            start=start_date,
            periods=len(daily_values),
            freq='D',
        )

        return pd.Series(daily_values[:len(date_index)], index=date_index,
                         name='hydrograph_m3s')

    def _extract_head(
        self,
        output_path: Path,
        **kwargs,
    ) -> pd.Series:
        """Extract porous media head from HGS output."""
        if output_path.is_dir():
            files = sorted(output_path.glob("*head_pm*"))
            if not files:
                files = sorted(output_path.glob("*head*"))
            if not files:
                raise ValueError(f"No head files found in {output_path}")
            data_file = files[0]
        else:
            data_file = output_path

        start_date = kwargs.get('start_date', '2000-01-01')

        values = []

        with open(data_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('!'):
                    continue
                parts = line.split()
                try:
                    val = float(parts[-1])  # last column is typically head
                    values.append(val)
                except (ValueError, IndexError):
                    continue

        date_index = pd.date_range(
            start=start_date,
            periods=len(values),
            freq='D',
        )

        return pd.Series(values[:len(date_index)], index=date_index,
                         name='head_m')
