"""
GGMN (Global Groundwater Monitoring Network) Observation Handler

Handles acquisition of groundwater level data from IGRAC's GGMN via WFS.
"""
import json
from pathlib import Path

import pandas as pd
import requests

from symfluence.core.exceptions import DataAcquisitionError, symfluence_error_handler

from ..base import BaseObservationHandler
from ..registry import ObservationRegistry


@ObservationRegistry.register('ggmn')
class GGMNHandler(BaseObservationHandler):
    """
    Handles GGMN groundwater data acquisition and processing.
    """

    obs_type = "groundwater"
    source_name = "IGRAC_GGMN"

    WFS_URL = "https://ggis.un-igrac.org/geoserver/ows"

    def acquire(self) -> Path:
        """
        Acquire GGMN data automatically.
        1. Query WFS for stations within the domain bounding box.
        2. For each station, fetch measurement data via the list API.
        3. Save CSV for each station.
        """
        self.logger.info("Starting automated GGMN data acquisition...")

        raw_dir = self.project_observations_dir / "groundwater" / "raw_data"
        raw_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = raw_dir / f"{self.domain_name}_ggmn_metadata.json"

        if not self.bbox:
             self.logger.warning("No bounding box found in config. Cannot query GGMN spatially.")
             return metadata_file

        min_lon = self.bbox['lon_min']
        min_lat = self.bbox['lat_min']
        max_lon = self.bbox['lon_max']
        max_lat = self.bbox['lat_max']

        cql_filter = f"BBOX(location, {min_lon}, {min_lat}, {max_lon}, {max_lat}) AND groundwater_level_data>0"

        params = {
            "service": "WFS",
            "version": "1.0.0",
            "request": "GetFeature",
            "typename": "groundwater:Groundwater_Well",
            "outputFormat": "application/json",
            "cql_filter": cql_filter
        }

        with symfluence_error_handler("GGMN data acquisition", self.logger, error_type=DataAcquisitionError):
            self.logger.info(f"Querying GGMN WFS for stations in bbox: [{min_lon}, {min_lat}, {max_lon}, {max_lat}]")
            response = requests.get(self.WFS_URL, params=params, timeout=60)
            response.raise_for_status()

            feature_collection = response.json()
            features = feature_collection.get('features', [])

            if not features:
                self.logger.warning("No GGMN stations found with data in this area.")
                return metadata_file

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(feature_collection, f, indent=2)

            self.logger.info(f"Found {len(features)} stations. Fetching measurements...")

            from bs4 import BeautifulSoup

            for feature in features:
                props = feature.get('properties', {})
                gid = str(props.get('id'))
                name = props.get('name')

                # Fetch measurements
                list_url = f"https://ggis.un-igrac.org/groundwater/record/{gid}/WellLevelMeasurement/list"
                try:
                    self.logger.debug(f"Fetching data for station {gid} ({name})...")
                    r_list = requests.get(list_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                    r_list.raise_for_status()

                    data_json = r_list.json()
                    measurements = data_json.get('data', [])

                    if not measurements:
                        self.logger.debug(f"No measurements returned for station {gid}")
                        continue

                    times, values = [], []
                    for item in measurements:
                        html_snippet = item.get('html', '')
                        if not html_snippet:
                            continue

                        soup = BeautifulSoup(html_snippet, 'html.parser')
                        time_input = soup.find('input', {'name': 'time'})
                        val_input = soup.find('input', {'name': 'value_value'})

                        if time_input and val_input:
                            times.append(time_input.get('value'))
                            values.append(val_input.get('value'))

                    if times:
                        df_st = pd.DataFrame({'datetime': times, 'groundwater_level': values})
                        df_st['groundwater_level'] = pd.to_numeric(df_st['groundwater_level'], errors='coerce')
                        df_st = df_st.dropna()

                        st_file = raw_dir / f"ggmn_{gid}_auto.csv"
                        df_st.to_csv(st_file, index=False)
                        self.logger.info(f"✓ Acquired {len(df_st)} records for station {gid}")

                except Exception as e_st:  # noqa: BLE001 — data acquisition resilience
                    self.logger.warning(f"Failed to fetch data for station {gid}: {e_st}")

            return metadata_file

    def process(self, input_path: Path) -> Path:
        """
        Process and average all acquired GGMN data.
        """
        raw_dir = self.project_observations_dir / "groundwater" / "raw_data"
        output_dir = self.project_observations_dir / "groundwater"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Look for all auto-acquired CSVs
        csv_files = list(raw_dir.glob("ggmn_*_auto.csv"))

        # Fallback to any manual CSVs if auto failed or user added some
        if not csv_files:
             csv_files = list(raw_dir.glob("*ggmn*.csv"))

        if not csv_files:
             raise DataAcquisitionError(f"No GGMN CSV files found in {raw_dir}")

        valid_dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, parse_dates=['datetime'], engine='python')
                df.set_index('datetime', inplace=True)
                df.sort_index(inplace=True)

                # Resample to daily mean
                df_daily = df['groundwater_level'].resample('D').mean()
                valid_dfs.append(df_daily)
            except Exception as e:  # noqa: BLE001 — data acquisition resilience
                self.logger.warning(f"Error processing {csv_file.name}: {e}")

        if not valid_dfs:
             raise DataAcquisitionError("No valid GGMN data could be processed from files.")

        self.logger.info(f"Aggregating data from {len(valid_dfs)} stations.")
        combined_df = pd.concat(valid_dfs, axis=1)
        mean_series = combined_df.mean(axis=1)

        # 4. Save processed average to path expected by evaluator
        # Evaluator expects: observations/groundwater/depth/processed/{domain_name}_gw_processed.csv
        out_dir = self.project_observations_dir / "groundwater" / "depth" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = out_dir / f"{self.domain_name}_gw_processed.csv"

        mean_series.to_csv(out_name, header=['groundwater_level'], index_label='datetime')

        self.logger.info(f"Saved averaged groundwater data to: {out_name}")
        return out_name
