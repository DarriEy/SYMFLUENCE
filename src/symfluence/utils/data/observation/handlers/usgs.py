"""
USGS Observation Handlers

Provides handlers for USGS streamflow and groundwater data.
"""
import io
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from symfluence.utils.common.constants import UnitConversion
from symfluence.utils.exceptions import DataAcquisitionError
from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

@ObservationRegistry.register('USGS_STREAMFLOW')
class USGSStreamflowHandler(BaseObservationHandler):
    """
    Handles USGS streamflow (discharge) data acquisition and processing.
    """

    def acquire(self) -> Path:
        """
        Acquire USGS streamflow data from the NWIS API or locate local raw file.
        """
        station_id = self.config.get('STATION_ID')
        if not station_id:
            self.logger.debug("STATION_ID not found, skipping USGS streamflow acquisition")
            return self.project_dir / "observations" / "streamflow" / "raw_data"

        download_enabled = self.config.get('DOWNLOAD_USGS_DATA', False)

        # Ensure station ID is properly formatted (usually 8+ digits)
        station_id_str = str(station_id)
        if station_id_str.isdigit() and len(station_id_str) < 8:
            station_id_str = station_id_str.zfill(8)

        raw_dir = self.project_dir / "observations" / "streamflow" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"usgs_{station_id_str}_raw.rdb"

        if download_enabled:
            return self._download_data(station_id_str, raw_file)
        else:
            # Look for existing raw file
            raw_name = self.config.get('STREAMFLOW_RAW_NAME')
            if raw_name:
                custom_raw = raw_dir / raw_name
                if custom_raw.exists():
                    return custom_raw
            
            if raw_file.exists():
                return raw_file
            
            self.logger.warning(f"USGS raw file not found and download disabled: {raw_file}")
            return raw_file

    def _download_data(self, station_id: str, output_path: Path) -> Path:
        """Fetch discharge data from USGS API."""
        self.logger.info(f"Downloading USGS streamflow data for station {station_id}")
        
        # Use experiment time range or defaults
        start_date = self.start_date.strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        parameter_cd = "00060"  # Discharge in cfs

        base_url = "https://nwis.waterservices.usgs.gov/nwis/iv/"
        url = f"{base_url}?site={station_id}&format=rdb&parameterCd={parameter_cd}&startDT={start_date}&endDT={end_date}"
        
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            if 'No data available' in response.text:
                raise DataAcquisitionError(f"USGS API returned no data for station {station_id}")

            with open(output_path, 'w') as f:
                f.write(response.text)
            
            self.logger.info(f"Successfully downloaded USGS data to {output_path}")
            return output_path

        except Exception as e:
            # Attempt fallback strategy (Waterservices without nwis prefix)
            fallback_url = url.replace("nwis.waterservices.usgs.gov", "waterservices.usgs.gov")
            self.logger.info(f"Primary USGS download failed, trying fallback: {fallback_url}")
            try:
                response = requests.get(fallback_url, timeout=60)
                response.raise_for_status()
                with open(output_path, 'w') as f:
                    f.write(response.text)
                return output_path
            except Exception as e2:
                self.logger.error(f"Failed to download USGS data: {e2}")
                raise DataAcquisitionError(f"Could not retrieve USGS data for station {station_id}") from e2

    def process(self, input_path: Path) -> Path:
        """
        Process USGS RDB format into standard SYMFLUENCE streamflow CSV.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"USGS raw data file not found: {input_path}")

        self.logger.info(f"Processing USGS streamflow data from {input_path}")
        
        # 1. Parse RDB file - find first non-comment line
        header_line_num = -1
        with open(input_path, 'r') as f:
            for i, line in enumerate(f):
                if not line.startswith('#'):
                    header_line_num = i
                    break
        
        if header_line_num == -1:
            raise DataAcquisitionError(f"No data found in USGS RDB file: {input_path}")

        df = pd.read_csv(
            input_path,
            sep='\t',
            skiprows=header_line_num,
            low_memory=False
        )
        
        # USGS RDB has a second line with format info (e.g., 5s, 15s)
        # If the first row of data contains such strings, skip it
        if not df.empty and df.iloc[0].astype(str).str.contains('s$|d$|n$').any():
            df = df.iloc[1:].reset_index(drop=True)


        # 2. Identify columns
        datetime_col = self._find_col(df.columns, ['datetime', 'date_time', 'dateTime', 'Timestamp'])
        discharge_col = self._find_col(df.columns, ['00060', 'discharge', 'flow', 'value'])

        if not datetime_col or not discharge_col:
            raise DataAcquisitionError(f"Could not identify required columns in USGS data: {input_path}")

        # 3. Clean and convert
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
        df[discharge_col] = pd.to_numeric(df[discharge_col], errors='coerce')
        df = df.dropna(subset=[datetime_col, discharge_col])
        
        df.set_index(datetime_col, inplace=True)
        df.sort_index(inplace=True)

        # Convert cfs to cms
        df['discharge_cms'] = df[discharge_col] * UnitConversion.CFS_TO_CMS

        # 4. Resample to target timestep
        resample_freq = self._get_resample_freq()
        resampled = df['discharge_cms'].resample(resample_freq).mean()
        
        # Interpolate small gaps
        resampled = resampled.interpolate(method='time', limit_direction='both', limit=30)

        # 5. Save processed data
        output_dir = self.project_dir / "observations" / "streamflow" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_streamflow_processed.csv"
        
        resampled.to_csv(output_file, header=True, index_label='datetime')
        
        self.logger.info(f"USGS streamflow processing complete: {output_file}")
        return output_file

    def _find_col(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        for col in columns:
            if any(c.lower() in col.lower() for c in candidates):
                return col
        return None

    def _get_resample_freq(self) -> str:
        timestep_size = int(self.config.get('FORCING_TIME_STEP_SIZE', 3600))
        if timestep_size == 3600 or timestep_size == 10800:
            return 'h'
        elif timestep_size == 86400:
            return 'D'
        else:
            return f'{timestep_size}s'

@ObservationRegistry.register('USGS_GW')
class USGSGroundwaterHandler(BaseObservationHandler):
    """
    Handles USGS groundwater level data.
    """

    def acquire(self) -> Path:
        """Download USGS groundwater data."""
        download_enabled = self.config.get('DOWNLOAD_USGS_GW', False)
        if isinstance(download_enabled, str):
            download_enabled = download_enabled.lower() == 'true'
            
        if not download_enabled:
            self.logger.info("USGS groundwater download disabled")
            return self.project_dir / "observations" / "groundwater" / "raw_data"

        station_id = self.config.get('USGS_STATION') or self.config.get('STATION_ID')
        if not station_id:
            raise ValueError("USGS_STATION or STATION_ID required for USGS groundwater acquisition")

        station_numeric = str(station_id).split('-')[-1] if '-' in str(station_id) else str(station_id)
        
        raw_dir = self.project_dir / "observations" / "groundwater" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / f"usgs_gw_{station_numeric}_raw.json"

        # Try gwlevels endpoint first
        url = f"https://waterservices.usgs.gov/nwis/gwlevels/?format=json&sites={station_numeric}&agencyCd=USGS&siteStatus=all"
        
        self.logger.info(f"Downloading USGS groundwater data: {url}")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Check if we got data
            data = response.json()
            if 'value' in data and 'timeSeries' in data['value'] and data['value']['timeSeries']:
                with open(raw_file, 'w') as f:
                    f.write(response.text)
                return raw_file
            
            # If gwlevels empty, try instantaneous values (iv)
            self.logger.info("gwlevels empty, trying 'iv' endpoint...")
            url_iv = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={station_numeric}&agencyCd=USGS&parameterCd=72019&siteStatus=all"
            response = requests.get(url_iv, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            if 'value' in data and 'timeSeries' in data['value'] and data['value']['timeSeries']:
                with open(raw_file, 'w') as f:
                    f.write(response.text)
                return raw_file

            # If iv empty, try daily values (dv)
            self.logger.info("iv empty, trying 'dv' endpoint...")
            url_dv = f"https://waterservices.usgs.gov/nwis/dv/?format=json&sites={station_numeric}&agencyCd=USGS&parameterCd=72019&siteStatus=all"
            response = requests.get(url_dv, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            if 'value' in data and 'timeSeries' in data['value'] and data['value']['timeSeries']:
                with open(raw_file, 'w') as f:
                    f.write(response.text)
                return raw_file
            
            # If all fail to provide timeSeries
            self.logger.warning(f"No groundwater data found for station {station_numeric} in any endpoint (gwlevels, iv, dv).")
            raise DataAcquisitionError(f"No USGS groundwater data available for station {station_numeric}")
            
        except Exception as e:
            self.logger.error(f"Failed to download USGS groundwater data: {e}")
            raise DataAcquisitionError(f"USGS groundwater acquisition failed: {e}")

    def process(self, input_path: Path) -> Path:
        """Process USGS groundwater JSON."""
        import json
        if not input_path.exists():
            raise FileNotFoundError(f"USGS GW raw file not found: {input_path}")

        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load USGS GW JSON: {e}")
            raise DataAcquisitionError(f"Invalid JSON in USGS GW file: {input_path}") from e

        # Robust check for data structure
        if 'value' not in data or 'timeSeries' not in data['value']:
            self.logger.warning(f"No valid timeSeries found in USGS GW data: {input_path}")
            raise DataAcquisitionError("No timeSeries structure in USGS GW JSON")
            
        time_series = data['value']['timeSeries']
        if not time_series:
            self.logger.warning("USGS GW timeSeries list is empty")
            raise DataAcquisitionError("Empty timeSeries in USGS GW JSON")

        dates, values, units = [], [], []
        for ts in time_series:
            # Check for groundwater level parameter (72019) or name matches
            param_code = ts.get('variable', {}).get('parameterCode', '')
            param_name = ts.get('variable', {}).get('variableName', '').lower()
            
            # 72019 is the standard parameter code for depth to water level
            is_gw_level = '72019' in param_code or 'depth to water level' in param_name or 'water level' in param_name
            
            if not is_gw_level:
                continue
            
            unit_code = ts.get('variable', {}).get('unit', {}).get('unitCode', 'unknown')
            
            # Extract values from the first available block
            for values_container in ts.get('values', []):
                val_list = values_container.get('value', [])
                for val_obj in val_list:
                    try:
                        dates.append(val_obj['dateTime'])
                        values.append(float(val_obj['value']))
                        units.append(unit_code)
                    except (KeyError, ValueError):
                        continue

        if not dates:
            self.logger.warning("No valid groundwater level records found in USGS JSON")
            raise DataAcquisitionError("No valid groundwater level records extracted from USGS JSON")

        df = pd.DataFrame({
            'datetime': pd.to_datetime(dates),
            'groundwater_level': values,
            'unit': units
        })
        
        # Standardize to meters
        def to_meters(row):
            val, unit = row['groundwater_level'], row['unit'].lower()
            if unit in ['ft', 'feet', 'foot']:
                return val * UnitConversion.FEET_TO_METERS
            return val

        df['groundwater_level'] = df.apply(to_meters, axis=1)
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)

        output_dir = self.project_dir / "observations" / "groundwater"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_groundwater_processed.csv"
        
        df[['groundwater_level']].to_csv(output_file)
        self.logger.info(f"USGS groundwater processing complete: {output_file}")
        return output_file
