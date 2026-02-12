"""
Gauge station metadata providers for WSC, USGS, SMHI, and LamaH-ICE.

Each provider fetches station metadata from its upstream source and returns
a uniform DataFrame with columns:
    station_id, name, lat, lon, river_name, network, country
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class GaugeProvider(ABC):
    """Base class for gauge station metadata providers."""

    network: str = ""
    country: str = ""

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Fetch station metadata and return a uniform DataFrame.

        Returns:
            DataFrame with columns:
                station_id (str), name (str), lat (float), lon (float),
                river_name (str), network (str), country (str)
        """

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=['station_id', 'name', 'lat', 'lon', 'river_name', 'network', 'country']
        )


class WSCProvider(GaugeProvider):
    """Water Survey of Canada — hydrometric stations via OGC API."""

    network = "WSC"
    country = "CA"

    # OGC API endpoint (paginated JSON)
    _BASE_URL = "https://api.weather.gc.ca/collections/hydrometric-stations/items"
    _LIMIT = 500  # items per page

    def fetch(self) -> pd.DataFrame:
        import urllib.request
        import json

        rows = []
        offset = 0
        while True:
            url = f"{self._BASE_URL}?f=json&limit={self._LIMIT}&offset={offset}"
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:  # nosec B310
                    data = json.loads(resp.read().decode())
            except Exception as exc:
                logger.warning(f"WSC fetch failed at offset {offset}: {exc}")
                break

            features = data.get('features', [])
            if not features:
                break

            for feat in features:
                props = feat.get('properties', {})
                geom = feat.get('geometry', {})
                coords = geom.get('coordinates', [None, None])
                rows.append({
                    'station_id': props.get('STATION_NUMBER', ''),
                    'name': props.get('STATION_NAME', ''),
                    'lat': coords[1] if len(coords) > 1 else None,
                    'lon': coords[0] if coords else None,
                    'river_name': props.get('STATION_NAME', '').split(' at ')[0]
                                  if ' at ' in props.get('STATION_NAME', '') else '',
                    'network': self.network,
                    'country': self.country,
                })

            returned = data.get('numberReturned', len(features))
            if returned < self._LIMIT:
                break
            offset += self._LIMIT

        if not rows:
            return self._empty_df()

        df = pd.DataFrame(rows)
        df = df.dropna(subset=['lat', 'lon'])
        logger.info(f"WSC: fetched {len(df)} stations")
        return df


class USGSProvider(GaugeProvider):
    """USGS NWIS — active streamflow sites via waterservices RDB format."""

    network = "USGS"
    country = "US"

    _URL = (
        "https://waterservices.usgs.gov/nwis/site/"
        "?format=rdb&siteType=ST&siteStatus=active"
        "&hasDataTypeCd=dv&parameterCd=00060"
        "&outputDataTypeCd=dv"
    )

    def fetch(self) -> pd.DataFrame:
        import urllib.request

        try:
            with urllib.request.urlopen(self._URL, timeout=60) as resp:  # nosec B310
                raw = resp.read().decode('utf-8')
        except Exception as exc:
            logger.warning(f"USGS fetch failed: {exc}")
            return self._empty_df()

        # Parse RDB: skip comment lines (#) and the format line (2nd non-comment)
        lines = [l for l in raw.splitlines() if not l.startswith('#')]
        if len(lines) < 2:
            return self._empty_df()

        # First line is header, second is column widths, rest is data
        header = lines[0].split('\t')
        data_lines = lines[2:]  # skip format spec line

        rows = []
        # Map expected columns
        col_map = {h.lower(): i for i, h in enumerate(header)}
        site_no_idx = col_map.get('site_no')
        name_idx = col_map.get('station_nm')
        lat_idx = col_map.get('dec_lat_va')
        lon_idx = col_map.get('dec_long_va')

        if any(idx is None for idx in [site_no_idx, name_idx, lat_idx, lon_idx]):
            logger.warning("USGS RDB: unexpected column layout")
            return self._empty_df()

        for line in data_lines:
            parts = line.split('\t')
            if len(parts) <= max(site_no_idx or 0, name_idx or 0, lat_idx or 0, lon_idx or 0):
                continue
            try:
                lat = float(parts[lat_idx])
                lon = float(parts[lon_idx])
            except (ValueError, TypeError):
                continue
            rows.append({
                'station_id': parts[site_no_idx],
                'name': parts[name_idx],
                'lat': lat,
                'lon': lon,
                'river_name': '',
                'network': self.network,
                'country': self.country,
            })

        if not rows:
            return self._empty_df()

        df = pd.DataFrame(rows)
        logger.info(f"USGS: fetched {len(df)} stations")
        return df


class SMHIProvider(GaugeProvider):
    """SMHI — Swedish hydrological observation stations (parameter 2 = discharge)."""

    network = "SMHI"
    country = "SE"

    _URL = "https://opendata-download-hydroobs.smhi.se/api/version/latest/parameter/2/station.json"

    def fetch(self) -> pd.DataFrame:
        import urllib.request
        import json

        try:
            with urllib.request.urlopen(self._URL, timeout=30) as resp:  # nosec B310
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.warning(f"SMHI fetch failed: {exc}")
            return self._empty_df()

        stations = data.get('station', [])
        rows = []
        for stn in stations:
            rows.append({
                'station_id': str(stn.get('key', '')),
                'name': stn.get('name', ''),
                'lat': stn.get('latitude'),
                'lon': stn.get('longitude'),
                'river_name': '',
                'network': self.network,
                'country': self.country,
            })

        if not rows:
            return self._empty_df()

        df = pd.DataFrame(rows)
        df = df.dropna(subset=['lat', 'lon'])
        logger.info(f"SMHI: fetched {len(df)} stations")
        return df


class LamaHICEProvider(GaugeProvider):
    """LamaH-ICE — gauge attributes from local CSV."""

    network = "LamaH-ICE"
    country = "IS"

    def __init__(self, lamah_ice_path: Optional[str] = None):
        self._path = lamah_ice_path

    def fetch(self) -> pd.DataFrame:
        csv_path = self._resolve_csv()
        if csv_path is None or not csv_path.exists():
            logger.info("LamaH-ICE CSV not found — skipping")
            return self._empty_df()

        try:
            df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
        except Exception as exc:
            logger.warning(f"LamaH-ICE CSV read failed: {exc}")
            return self._empty_df()

        # Expected columns: ID, gauge_name, gauge_lat, gauge_lon
        col_map = {c.lower(): c for c in df.columns}
        id_col = col_map.get('id', col_map.get('gauge_id'))
        name_col = col_map.get('gauge_name', col_map.get('name'))
        lat_col = col_map.get('gauge_lat', col_map.get('lat'))
        lon_col = col_map.get('gauge_lon', col_map.get('lon'))
        river_col = col_map.get('river_name', col_map.get('river'))

        if not all([id_col, lat_col, lon_col]):
            logger.warning(f"LamaH-ICE CSV: missing expected columns in {list(df.columns)}")
            return self._empty_df()

        out = pd.DataFrame({
            'station_id': df[id_col].astype(str),
            'name': df[name_col] if name_col else '',
            'lat': pd.to_numeric(df[lat_col], errors='coerce'),
            'lon': pd.to_numeric(df[lon_col], errors='coerce'),
            'river_name': df[river_col] if river_col else '',
            'network': self.network,
            'country': self.country,
        })
        out = out.dropna(subset=['lat', 'lon'])
        logger.info(f"LamaH-ICE: loaded {len(out)} stations")
        return out

    def _resolve_csv(self) -> Optional[Path]:
        import os
        if self._path:
            return Path(self._path) / 'D_gauges' / '1_attributes' / 'Gauge_attributes.csv'
        # Try environment variable
        env_path = os.environ.get('LAMAH_ICE_PATH')
        if env_path:
            return Path(env_path) / 'D_gauges' / '1_attributes' / 'Gauge_attributes.csv'
        return None
