"""
CLMParFlow Result Extractor

Extracts results from ParFlow-CLM .pfb binary output files:
- Pressure head from *.out.press.*.pfb
- Saturation from *.out.satur.*.pfb
- Overland flow sum from *.out.overlandsum.*.pfb
- CLM output from *.out.clm_output.*.C.pfb (ET, soil temperature, etc.)

Reuses the ParFlow _read_pfb function for .pfb file reading.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from symfluence.models.base.base_extractor import ModelResultExtractor
from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


# Reuse ParFlow's .pfb reader
from symfluence.models.parflow.extractor import _read_pfb


@ModelRegistry.register_result_extractor("CLMPARFLOW")
class CLMParFlowResultExtractor(ModelResultExtractor):
    """
    Extracts results from ParFlow-CLM .pfb output files.

    Supports extraction of:
    - Pressure head from .pfb files -> water table depth
    - Saturation from .pfb files
    - Overland flow from .pfb files
    - CLM output (ET, soil temperature) from .C.pfb files
    """

    def __init__(self, model_name: str = 'CLMPARFLOW'):
        super().__init__(model_name)
        self.logger = logger

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for locating CLMParFlow outputs."""
        return {
            'pressure': ['*.out.press.*.pfb'],
            'saturation': ['*.out.satur.*.pfb'],
            'overland': ['*.out.overlandsum.*.pfb'],
            'clm_output': ['*.out.clm_output.*.C.pfb'],
        }

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs,
    ) -> pd.Series:
        """Extract a variable time series from CLMParFlow output.

        Args:
            output_file: Path to output file or directory
            variable_type: Type ('pressure', 'overland_flow', 'subsurface_drainage',
                          'evapotranspiration', 'soil_temperature')
            **kwargs: Additional args (timestep_hours, start_date, dx, dy for unit conversion)

        Returns:
            Time series of extracted variable
        """
        output_path = Path(output_file)

        if variable_type == 'pressure':
            return self._extract_pressure(output_path, **kwargs)
        elif variable_type in ('overland_flow', 'overland'):
            return self._extract_overland_flow(output_path, **kwargs)
        elif variable_type in ('subsurface_drainage', 'baseflow'):
            return self._extract_subsurface_drainage(output_path, **kwargs)
        elif variable_type == 'evapotranspiration':
            return self._extract_clm_variable(output_path, 'et', **kwargs)
        elif variable_type == 'soil_temperature':
            return self._extract_clm_variable(output_path, 'tsoil', **kwargs)
        else:
            raise ValueError(f"Unknown variable type: {variable_type}")

    def _find_pfb_files(
        self, output_path: Path, pattern: str
    ) -> List[Path]:
        """Find and sort PFB files by timestep number."""
        if output_path.is_dir():
            files = sorted(output_path.glob(pattern))
        else:
            files = [output_path]
        return files

    def _get_timestep_from_filename(self, filepath: Path) -> Optional[int]:
        """Extract timestep number from ParFlow output filename."""
        stem = filepath.stem
        parts = stem.split('.')
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            return None

    def _build_time_index(
        self, n_steps: int, **kwargs
    ) -> pd.DatetimeIndex:
        """Build time index from timestep count and config."""
        dump_interval = kwargs.get('dump_interval_hours',
                                   kwargs.get('timestep_hours', 1.0))
        start_date = kwargs.get('start_date', '2000-01-01')
        freq_hours = int(dump_interval)
        return pd.date_range(
            start=start_date,
            periods=n_steps,
            freq=f'{freq_hours}h',
        )

    def _extract_pressure(
        self, output_path: Path, **kwargs
    ) -> pd.Series:
        """Extract pressure head time series from .pfb files."""
        files = self._find_pfb_files(output_path, '*.out.press.*.pfb')
        if not files:
            raise ValueError(f"No pressure .pfb files found in {output_path}")

        values = []
        for f in files:
            ts = self._get_timestep_from_filename(f)
            if ts is not None and ts == 0:
                continue
            data = _read_pfb(f)
            values.append(data[-1, 0, 0])

        date_index = self._build_time_index(len(values), **kwargs)
        return pd.Series(values, index=date_index, name='pressure_head_m')

    def _extract_overland_flow(
        self, output_path: Path, **kwargs
    ) -> pd.Series:
        """Extract overland flow from .pfb files."""
        files = self._find_pfb_files(output_path, '*.out.overlandsum.*.pfb')
        if not files:
            self.logger.warning(
                "No overland flow .pfb files found. Returning zeros."
            )
            n = len(self._find_pfb_files(output_path, '*.out.press.*.pfb'))
            n = max(n - 1, 1)
            date_index = self._build_time_index(n, **kwargs)
            return pd.Series(0.0, index=date_index, name='overland_flow_m3s')

        values = []
        for f in files:
            ts = self._get_timestep_from_filename(f)
            if ts is not None and ts == 0:
                continue
            data = _read_pfb(f)
            total = float(np.sum(data))
            values.append(total)

        dump_interval_hours = kwargs.get('dump_interval_hours',
                                         kwargs.get('timestep_hours', 1.0))
        dump_interval_s = dump_interval_hours * 3600.0
        values_m3s = [v / dump_interval_s for v in values]

        date_index = self._build_time_index(len(values_m3s), **kwargs)
        return pd.Series(values_m3s, index=date_index, name='overland_flow_m3s')

    def _extract_subsurface_drainage(
        self, output_path: Path, **kwargs
    ) -> pd.Series:
        """Compute subsurface drainage from pressure gradient at bottom."""
        files = self._find_pfb_files(output_path, '*.out.press.*.pfb')
        if not files:
            raise ValueError(f"No pressure .pfb files found in {output_path}")

        k_sat = kwargs.get('k_sat', 5.0)
        dx = kwargs.get('dx', 1000.0)
        dy = kwargs.get('dy', 1000.0)
        dz = kwargs.get('dz', 2.0)

        values = []
        prev_pressure = None
        for f in files:
            ts = self._get_timestep_from_filename(f)
            if ts is not None and ts == 0:
                data = _read_pfb(f)
                prev_pressure = data[-1, 0, 0]
                continue

            data = _read_pfb(f)
            pressure = data[-1, 0, 0]

            if pressure > 0 and prev_pressure is not None:
                gradient = max(pressure / dz, 0.0)
                flux_m_hr = k_sat * gradient
                drainage_m3_hr = flux_m_hr * dx * dy
            else:
                drainage_m3_hr = 0.0

            values.append(drainage_m3_hr)
            prev_pressure = pressure

        date_index = self._build_time_index(len(values), **kwargs)
        return pd.Series(values, index=date_index, name='subsurface_drainage_m3hr')

    def _extract_clm_variable(
        self, output_path: Path, var_name: str, **kwargs
    ) -> pd.Series:
        """Extract a CLM output variable from .C.pfb files.

        CLM output files contain multiple variables stacked in the z-dimension.
        Common variables (z-index):
            0: latent heat flux (W/m2)
            1: sensible heat flux (W/m2)
            2: ground heat flux (W/m2)
            3: total ET (mm/s)
            4: ground evaporation (mm/s)
            5: soil temperature at surface (K)
        """
        files = self._find_pfb_files(output_path, '*.out.clm_output.*.C.pfb')
        if not files:
            self.logger.warning(f"No CLM output .C.pfb files found in {output_path}")
            n = len(self._find_pfb_files(output_path, '*.out.press.*.pfb'))
            n = max(n - 1, 1)
            date_index = self._build_time_index(n, **kwargs)
            return pd.Series(0.0, index=date_index, name=f'clm_{var_name}')

        var_z_index = {
            'latent_heat': 0,
            'sensible_heat': 1,
            'ground_heat': 2,
            'et': 3,
            'ground_evap': 4,
            'tsoil': 5,
        }

        z_idx = var_z_index.get(var_name, 3)  # Default to ET

        values = []
        for f in files:
            ts = self._get_timestep_from_filename(f)
            if ts is not None and ts == 0:
                continue
            try:
                data = _read_pfb(f)
                if data.shape[0] > z_idx:
                    values.append(float(data[z_idx, 0, 0]))
                else:
                    values.append(0.0)
            except Exception:
                values.append(0.0)

        date_index = self._build_time_index(len(values), **kwargs)
        return pd.Series(values, index=date_index, name=f'clm_{var_name}')
