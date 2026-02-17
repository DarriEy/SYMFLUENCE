"""
PIHM Result Extractor

Extracts results from PIHM output files:
- River flux from .rivflx files (channel discharge, m3/s)
- Groundwater head from .gwhead files (m)
- Surface water depth from .surf files (m)

PIHM output files are tab-delimited text with a time column (epoch seconds).
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from symfluence.models.base.base_extractor import ModelResultExtractor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_result_extractor("PIHM")
class PIHMResultExtractor(ModelResultExtractor):
    """
    Extracts results from PIHM output files.

    Supports extraction of:
    - River flux (channel discharge) from .rivflx files
    - Groundwater head from .gwhead files
    - Surface water depth from .surf files
    """

    def __init__(self, model_name: str = 'PIHM'):
        super().__init__(model_name)
        self.logger = logger

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for locating PIHM outputs."""
        return {
            'river_flux': ['*.rivflx*'],
            'groundwater_head': ['*.gwhead*'],
            'surface': ['*.surf*'],
        }

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs,
    ) -> pd.Series:
        """Extract a variable time series from PIHM output.

        Args:
            output_file: Path to output file or directory
            variable_type: Type ('river_flux', 'groundwater_head', 'baseflow')
            **kwargs: Additional args (start_date for time index)

        Returns:
            Time series of extracted variable
        """
        output_path = Path(output_file)

        if variable_type in ('river_flux', 'baseflow'):
            return self._extract_river_flux(output_path, **kwargs)
        elif variable_type == 'groundwater_head':
            return self._extract_gwhead(output_path, **kwargs)
        else:
            raise ValueError(f"Unknown variable type: {variable_type}")

    def _extract_river_flux(
        self,
        output_path: Path,
        **kwargs,
    ) -> pd.Series:
        """Extract river flux time series from .rivflx file.

        PIHM .rivflx format: tab-delimited, first column is epoch time,
        remaining columns are flux values per river segment.
        For lumped mode there is one segment.
        """
        if output_path.is_dir():
            files = sorted(output_path.glob("*.rivflx*"))
            if not files:
                raise ValueError(f"No .rivflx files found in {output_path}")
            data_file = files[0]
        else:
            data_file = output_path

        start_date = kwargs.get('start_date', '2000-01-01')

        try:
            data = np.loadtxt(data_file, skiprows=0)
        except Exception:
            data = self._read_pihm_output(data_file)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_timesteps = data.shape[0]

        if data.shape[1] >= 2:
            # First column is time, second is flux
            values = np.abs(data[:, 1])
        else:
            values = np.abs(data[:, 0])

        date_index = pd.date_range(
            start=start_date,
            periods=n_timesteps,
            freq='D',
        )

        return pd.Series(values[:len(date_index)], index=date_index,
                         name='river_flux_m3s')

    def _extract_gwhead(
        self,
        output_path: Path,
        **kwargs,
    ) -> pd.Series:
        """Extract groundwater head time series from .gwhead file."""
        if output_path.is_dir():
            files = sorted(output_path.glob("*.gwhead*"))
            if not files:
                raise ValueError(f"No .gwhead files found in {output_path}")
            data_file = files[0]
        else:
            data_file = output_path

        start_date = kwargs.get('start_date', '2000-01-01')

        try:
            data = np.loadtxt(data_file, skiprows=0)
        except Exception:
            data = self._read_pihm_output(data_file)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_timesteps = data.shape[0]

        if data.shape[1] >= 2:
            values = data[:, 1]
        else:
            values = data[:, 0]

        date_index = pd.date_range(
            start=start_date,
            periods=n_timesteps,
            freq='D',
        )

        return pd.Series(values[:len(date_index)], index=date_index,
                         name='gwhead_m')

    def _read_pihm_output(self, filepath: Path) -> np.ndarray:
        """Read PIHM output file with flexible parsing."""
        rows = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    vals = [float(x) for x in line.split()]
                    rows.append(vals)
                except ValueError:
                    continue
        return np.array(rows) if rows else np.array([]).reshape(0, 0)
