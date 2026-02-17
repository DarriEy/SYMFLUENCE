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
            'river_flux': ['*.river.flx1.txt'],
            'groundwater_head': ['*.gw.txt'],
            'surface': ['*.surf.txt'],
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

    def _parse_pihm_txt(self, filepath: Path) -> pd.Series:
        """Parse MM-PIHM .txt output file.

        Format: "YYYY-MM-DD HH:MM"<tab>value
        The timestamp is quoted and tab-separated from the numeric value.
        """
        times = []
        values = []
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Split on tab: first part is quoted timestamp, second is value
                parts = line.split('\t')
                if len(parts) >= 2:
                    try:
                        ts_str = parts[0].strip('"')
                        val = float(parts[-1])
                        times.append(pd.Timestamp(ts_str))
                        values.append(val)
                    except (ValueError, IndexError):
                        continue
        if not times:
            return pd.Series(dtype=float)
        return pd.Series(values, index=pd.DatetimeIndex(times))

    def _extract_river_flux(
        self,
        output_path: Path,
        **kwargs,
    ) -> pd.Series:
        """Extract river flux time series from MM-PIHM output.

        Looks for .river.flx1.txt (downstream discharge) first.
        """
        if output_path.is_dir():
            # Try river flux file first
            files = sorted(output_path.glob("*.river.flx1.txt"))
            if not files:
                # Fallback to legacy rivflx pattern
                files = sorted(output_path.glob("*.rivflx*"))
            if not files:
                self.logger.warning(f"No river flux files in {output_path}")
                return pd.Series(dtype=float, name='river_flux_m3s')
            data_file = files[0]
        else:
            data_file = output_path

        series = self._parse_pihm_txt(data_file)
        if series.empty:
            return pd.Series(dtype=float, name='river_flux_m3s')

        series = series.abs()
        series.name = 'river_flux_m3s'
        return series

    def _extract_gwhead(
        self,
        output_path: Path,
        **kwargs,
    ) -> pd.Series:
        """Extract groundwater head time series from .gw.txt file."""
        if output_path.is_dir():
            files = sorted(output_path.glob("*.gw.txt"))
            if not files:
                files = sorted(output_path.glob("*.gwhead*"))
            if not files:
                self.logger.warning(f"No groundwater files in {output_path}")
                return pd.Series(dtype=float, name='gwhead_m')
            data_file = files[0]
        else:
            data_file = output_path

        series = self._parse_pihm_txt(data_file)
        series.name = 'gwhead_m'
        return series
