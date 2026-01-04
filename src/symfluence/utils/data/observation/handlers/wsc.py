"""
WSC Observation Handlers

Provides handlers for Water Survey of Canada (WSC) streamflow data.
Supports both local HYDAT SQLite database extraction and web API acquisition.
"""
import logging
import requests
import pandas as pd
import numpy as np
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from symfluence.utils.common.constants import UnitConversion, ModelDefaults
from symfluence.utils.exceptions import DataAcquisitionError
from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

@ObservationRegistry.register('WSC_STREAMFLOW')
class WSCStreamflowHandler(BaseObservationHandler):
    """
    Handles WSC streamflow data acquisition and processing.
    """

    def acquire(self) -> Path:
        """
        Acquire WSC streamflow data.
        If DATA_ACCESS is 'cloud' and DOWNLOAD_WSC_DATA is True, attempts to use WSC GeoMet API.
        Otherwise, looks for local raw data or triggers HYDAT extraction.
        """
        data_access = self.config.get('DATA_ACCESS', 'local')
        download_enabled = self.config.get('DOWNLOAD_WSC_DATA', False)
        station_id = self.config.get('STATION_ID')
        
        if not station_id:
            self.logger.error("Missing STATION_ID in configuration for WSC streamflow")
            raise ValueError("STATION_ID required for WSC streamflow acquisition")

        raw_dir = self.project_dir / "observations" / "streamflow" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"wsc_{station_id}_raw.csv"

        # Cloud pathway: Use WSC GeoMet API
        if data_access == 'cloud' and download_enabled:
            return self._download_from_geomet(station_id, raw_file)
        
        # Local/Default pathway: Use HYDAT or existing raw files
        if download_enabled:
            # In legacy SYMFLUENCE, DOWNLOAD_WSC_DATA=True often meant trigger HYDAT extraction
            # We handle this in process() if the file doesn't exist
            self.logger.info(f"WSC local access: will attempt HYDAT extraction if {raw_file} not found")
            return raw_file
        else:
            # Look for existing raw file
            raw_name = self.config.get('STREAMFLOW_RAW_NAME')
            if raw_name and raw_name != 'default':
                custom_raw = raw_dir / raw_name
                if custom_raw.exists():
                    return custom_raw
            
            if raw_file.exists():
                return raw_file
            
            self.logger.warning(f"WSC raw file not found: {raw_file}")
            return raw_file

    def _download_from_geomet(self, station_id: str, output_path: Path) -> Path:
        """
        Download daily mean discharge from WSC GeoMet API.
        """
        self.logger.info(f"Downloading WSC streamflow data for station {station_id} via GeoMet API")
        
        # GeoMet API for daily mean discharge
        base_url = "https://api.weather.gc.ca/collections/hydrometric-daily-mean/items"
        
        # We fetch all available data for the station
        params = {
            'STATION_NUMBER': station_id,
            'f': 'json',
            'limit': 10000  # Adjust as needed, daily records for 30 years is ~11k
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            features = data.get('features', [])
            
            if not features:
                raise DataAcquisitionError(f"No data found for WSC station {station_id} in GeoMet API")

            # Convert features to flat list for DataFrame
            rows = []
            for feat in features:
                props = feat.get('properties', {})
                rows.append(props)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            
            self.logger.info(f"Successfully downloaded {len(df)} records to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to download WSC data from GeoMet: {e}")
            raise DataAcquisitionError(f"Could not retrieve WSC data for station {station_id}") from e

    def process(self, input_path: Path) -> Path:
        """
        Process WSC data (GeoMet JSON-to-CSV or legacy raw CSV) into standard SYMFLUENCE format.
        """
        if not input_path.exists():
            # Special case: check if we should try HYDAT extraction as a fallback
            hydat_path = self.config.get('HYDAT_PATH')
            if hydat_path:
                return self._process_from_hydat()
            raise FileNotFoundError(f"WSC raw data file not found: {input_path}")

        self.logger.info(f"Processing WSC streamflow data from {input_path}")
        
        # Load the data
        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            # Try with '#' comments if it's a legacy RDB-like file
            df = pd.read_csv(input_path, comment='#')

        # Identify columns
        # GeoMet uses 'DATE' and 'VALUE' (Discharge)
        # Local files might use 'datetime' or 'Value'
        datetime_col = self._find_col(df.columns, ['date', 'datetime', 'ISO 8601 UTC', 'Timestamp'])
        discharge_col = self._find_col(df.columns, ['value', 'discharge', 'flow', 'discharge_cms'])

        if not datetime_col or not discharge_col:
            raise DataAcquisitionError(f"Could not identify required columns in WSC data: {input_path}")

        # Clean and convert
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df[discharge_col] = pd.to_numeric(df[discharge_col], errors='coerce')
        df = df.dropna(subset=[datetime_col, discharge_col])
        
        df.set_index(datetime_col, inplace=True)
        df.sort_index(inplace=True)

        # Standardize naming
        df['discharge_cms'] = df[discharge_col]  # WSC is already in cms (m3/s)

        # Resample to target timestep
        resample_freq = self._get_resample_freq()
        resampled = df['discharge_cms'].resample(resample_freq).mean()
        resampled = resampled.interpolate(method='time', limit_direction='both', limit=30)

        # Save processed data
        output_dir = self.project_dir / "observations" / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"
        
        resampled.to_csv(output_file, header=True, index_label='datetime')
        
        self.logger.info(f"WSC streamflow processing complete: {output_file}")
        return output_file

    def _process_from_hydat(self) -> Path:
        """
        Legacy fallback: Extract from local HYDAT database.
        """
        import sqlite3
        station_id = self.config.get('STATION_ID')
        hydat_path = self.config.get('HYDAT_PATH')
        if hydat_path == 'default':
            hydat_path = str(self.project_dir.parent.parent / 'geospatial-data' / 'hydat' / 'Hydat.sqlite3')
        
        if not Path(hydat_path).exists():
            raise FileNotFoundError(f"HYDAT database not found at: {hydat_path}")

        self.logger.info(f"Extracting WSC data from HYDAT: {hydat_path}")
        
        conn = sqlite3.connect(hydat_path)
        query = "SELECT * FROM DLY_FLOWS WHERE STATION_NUMBER = ?"
        df_raw = pd.read_sql_query(query, conn, params=(station_id,))
        conn.close()

        if df_raw.empty:
            raise DataAcquisitionError(f"No data for station {station_id} in HYDAT")

        # Reshape HYDAT format (FLOW1...FLOW31) to time series
        ts_data = []
        for _, row in df_raw.iterrows():
            year, month = int(row['YEAR']), int(row['MONTH'])
            for day in range(1, 32):
                col = f'FLOW{day}'
                if col in row and not pd.isna(row[col]):
                    try:
                        date = f"{year}-{month:02d}-{day:02d}"
                        ts_data.append({'datetime': date, 'discharge_cms': row[col]})
                    except ValueError:
                        continue # Invalid date (e.g., Feb 30)

        df = pd.DataFrame(ts_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        output_dir = self.project_dir / "observations" / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"
        
        resample_freq = self._get_resample_freq()
        resampled = df['discharge_cms'].resample(resample_freq).mean()
        resampled.to_csv(output_file, header=True, index_label='datetime')
        
        return output_file

    def _find_col(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        for col in columns:
            if any(c.lower() in col.lower() for c in candidates):
                return col
        return None

    def _get_resample_freq(self) -> str:
        timestep_size = int(self.config.get('FORCING_TIME_STEP_SIZE', 3600))
        if timestep_size <= 10800:
            return 'h'
        elif timestep_size == ModelDefaults.DEFAULT_TIMESTEP_DAILY:
            return 'D'
        else:
            return f'{timestep_size}s'
