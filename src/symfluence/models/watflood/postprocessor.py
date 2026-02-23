"""
WATFLOOD Post-Processor.

Extracts and processes WATFLOOD model outputs.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from symfluence.models.base import StandardModelPostprocessor

logger = logging.getLogger(__name__)


class WATFLOODPostProcessor(StandardModelPostprocessor):
    """Post-processor for WATFLOOD model outputs."""

    model_name = "WATFLOOD"
    output_file_pattern = "spl*.tb0"
    streamflow_variable = "QO"
    streamflow_unit = "cms"

    def extract_streamflow(self) -> Optional[Path]:
        """Extract streamflow from WATFLOOD output.

        Searches the simulation directory for WATFLOOD output files (.tb0, .csv)
        and saves extracted streamflow to the standard results CSV.

        Returns:
            Path to saved results CSV, or None if extraction fails.
        """
        try:
            self.logger.info("Extracting WATFLOOD streamflow results")
            output_dir = self.sim_dir

            for pattern in ['spl*.tb0', 'resin*.tb0', '*.csv', 'streamflow*.csv']:
                matches = list(output_dir.glob(pattern))
                if matches:
                    streamflow = self._extract_from_file(matches[0])
                    if streamflow is not None:
                        return self.save_streamflow_to_results(streamflow)

            self.logger.error(f"No WATFLOOD output found in {output_dir}")
            return None
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error extracting WATFLOOD streamflow: {e}")
            return None

    def _extract_from_file(self, output_file: Path) -> Optional[pd.Series]:
        """Extract streamflow from output file."""
        try:
            if output_file.suffix == '.csv':
                df = pd.read_csv(output_file, parse_dates=[0], index_col=0)
                for col in df.columns:
                    if any(v in col.lower() for v in ['qo', 'qsim', 'flow', 'discharge']):
                        return df[col]
            elif output_file.suffix == '.tb0':
                return self._parse_tb0(output_file)
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Error extracting streamflow: {e}")
        return None

    def _parse_tb0(self, tb0_file: Path) -> Optional[pd.Series]:
        """Parse WATFLOOD .tb0 format."""
        try:
            lines = tb0_file.read_text(encoding='utf-8').strip().split('\n')
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
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                        hour = int(parts[3])
                        value = float(parts[4])
                        dates.append(pd.Timestamp(year=year, month=month, day=day, hour=hour))
                        values.append(value)
                    except (ValueError, IndexError):
                        continue
            if dates:
                return pd.Series(values, index=dates, name='WATFLOOD_discharge_cms')
        except Exception as e:  # noqa: BLE001 — model execution resilience
            logger.error(f"Error parsing tb0: {e}")
        return None
