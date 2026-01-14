"""
SMHI Observation Handlers

Provides handlers for Swedish Meteorological and Hydrological Institute (SMHI) streamflow data.
"""
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, List

from symfluence.core.constants import ModelDefaults
from symfluence.core.exceptions import DataAcquisitionError
from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

@ObservationRegistry.register('SMHI_STREAMFLOW')
class SMHIStreamflowHandler(BaseObservationHandler):
    """
    Handles SMHI streamflow data acquisition and processing.
    """

    def acquire(self) -> Path:
        data_access = self._get_config_value(lambda: self.config.domain.data_access, default='cloud', dict_key='DATA_ACCESS').lower()
        station_id = self._get_config_value(lambda: self.config.evaluation.streamflow.station_id, dict_key='STATION_ID')

        if not station_id:
            self.logger.error("Missing STATION_ID in configuration for SMHI streamflow")
            raise ValueError("STATION_ID required for SMHI streamflow acquisition")

        raw_dir = self.project_dir / "observations" / "streamflow" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"smhi_{station_id}_raw.csv"

        if data_access == 'cloud':
            return self._download_from_smhi(station_id, raw_file)

        if raw_file.exists():
            return raw_file

        self.logger.warning(f"SMHI raw file not found: {raw_file}")
        return raw_file

    def _download_from_smhi(self, station_id: str, output_path: Path) -> Path:
        self.logger.info(f"Downloading SMHI streamflow data for station {station_id}")

        # SMHI uses parameter 2 for Discharge (Vattenföring)
        parameter_key = 2
        period = 'corrected-archive'
        url = f"https://opendata-download-hydroobs.smhi.se/api/version/latest/parameter/{parameter_key}/station/{station_id}/period/{period}/data.json"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            values = data.get('value', [])
            if not values:
                self.logger.error(f"No data available for SMHI station {station_id}")
                raise DataAcquisitionError(f"No data found for SMHI station {station_id}")

            df = pd.DataFrame(values)
            # SMHI uses milliseconds for date
            df['date'] = pd.to_datetime(df['date'] / 1000, unit='s')
            df = df.rename(columns={'value': 'discharge_m3s', 'quality': 'quality_code'})

            # Save to CSV in a format that ObservedDataProcessor can understand or that process() expects
            df.to_csv(output_path, index=False)

            self.logger.info(f"Successfully downloaded {len(df)} records to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to download SMHI data: {e}")
            raise DataAcquisitionError(f"Could not retrieve SMHI data for station {station_id}") from e

    def process(self, input_path: Path) -> Path:
        """
        Process SMHI data into standard SYMFLUENCE format.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"SMHI raw data file not found: {input_path}")

        self.logger.info(f"Processing SMHI streamflow data from {input_path}")

        df = pd.read_csv(input_path)

        # SMHI download format from _download_from_smhi: date, discharge_m3s, quality_code
        datetime_col = 'date'
        discharge_col = 'discharge_m3s'

        if datetime_col not in df.columns or discharge_col not in df.columns:
            # Fallback to general column finding if format changed
            datetime_col = self._find_col(df.columns, ['date', 'datetime'])
            discharge_col = self._find_col(df.columns, ['value', 'discharge', 'm3s'])

        if not datetime_col or not discharge_col:
            raise DataAcquisitionError(f"Could not identify required columns in SMHI data: {input_path}")

        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df[discharge_col] = pd.to_numeric(df[discharge_col], errors='coerce')
        df = df.dropna(subset=[datetime_col, discharge_col])

        df.set_index(datetime_col, inplace=True)
        df.sort_index(inplace=True)

        # Standardize naming
        df['discharge_cms'] = df[discharge_col]

        # Resample to target timestep
        resample_freq = self._get_resample_freq()
        resampled = df['discharge_cms'].resample(resample_freq).mean()
        resampled = resampled.interpolate(method='time', limit_direction='both', limit=30)

        # Save processed data
        output_dir = self.project_dir / "observations" / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"

        resampled.to_csv(output_file, header=True, index_label='datetime')

        self.logger.info(f"SMHI streamflow processing complete: {output_file}")
        return output_file

    def _find_col(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        for col in columns:
            if any(c.lower() in col.lower() for c in candidates):
                return col
        return None

    def _get_resample_freq(self) -> str:
        timestep_size = int(self._get_config_value(lambda: self.config.forcing.time_step_size, default=3600, dict_key='FORCING_TIME_STEP_SIZE'))
        if timestep_size <= 10800:
            return 'h'
        elif timestep_size == ModelDefaults.DEFAULT_TIMESTEP_DAILY:
            return 'D'
        else:
            return f'{timestep_size}s'
