"""
Gauge station cache and aggregation layer.

GaugeStationStore aggregates results from all gauge providers, caches them
as CSV files under ``{SYMFLUENCE_DATA_DIR}/cache/gauge_stations/`` with
a 30-day expiry, and provides viewport-filtered retrieval.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .gauge_provider import (
    GaugeProvider,
    WSCProvider,
    USGSProvider,
    SMHIProvider,
    LamaHICEProvider,
)

logger = logging.getLogger(__name__)

_CACHE_MAX_AGE_SECONDS = 30 * 24 * 3600  # 30 days
_COLUMNS = ['station_id', 'name', 'lat', 'lon', 'river_name', 'network', 'country']


class GaugeStationStore:
    """Aggregates gauge providers with CSV caching and viewport filtering.

    Usage::

        store = GaugeStationStore()
        df = store.load_all()                        # all networks
        df = store.load_all(networks=['WSC', 'USGS'])
        visible = store.get_in_viewport(-120, -110, 49, 52)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = Path(cache_dir) if cache_dir else self._default_cache_dir()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._providers: List[GaugeProvider] = [
            WSCProvider(),
            USGSProvider(),
            SMHIProvider(),
            LamaHICEProvider(),
        ]
        # In-memory combined frame (lazy)
        self._combined: Optional[pd.DataFrame] = None

    @staticmethod
    def _default_cache_dir() -> Path:
        data_dir = os.environ.get('SYMFLUENCE_DATA_DIR', '')
        if data_dir:
            return Path(data_dir) / 'cache' / 'gauge_stations'
        return Path.home() / '.symfluence' / 'cache' / 'gauge_stations'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self, networks: Optional[List[str]] = None) -> pd.DataFrame:
        """Load stations from all (or selected) networks, using cache.

        Args:
            networks: Optional list of network names to include.
                      If None, loads all available networks.

        Returns:
            Combined DataFrame of gauge station metadata.
        """
        if self._combined is None:
            frames = []
            for provider in self._providers:
                df = self._load_provider(provider)
                if df is not None and not df.empty:
                    frames.append(df)
            if frames:
                self._combined = pd.concat(frames, ignore_index=True)
            else:
                self._combined = pd.DataFrame(columns=_COLUMNS)

        if networks:
            return self._combined[self._combined['network'].isin(networks)].copy()
        return self._combined.copy()

    def get_in_viewport(
        self, lon_min: float, lon_max: float, lat_min: float, lat_max: float,
        networks: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return stations within the given geographic bounding box.

        Args:
            lon_min, lon_max, lat_min, lat_max: Viewport bounds in degrees.
            networks: Optional network filter.
        """
        df = self.load_all(networks=networks)
        if df.empty:
            return df
        mask = (
            (df['lon'] >= lon_min) & (df['lon'] <= lon_max) &
            (df['lat'] >= lat_min) & (df['lat'] <= lat_max)
        )
        return df[mask].copy()

    def invalidate_cache(self, network: Optional[str] = None):
        """Remove cached files, forcing a re-fetch."""
        self._combined = None
        if network:
            cache_file = self._cache_dir / f'{network}.csv'
            if cache_file.exists():
                cache_file.unlink()
        else:
            for f in self._cache_dir.glob('*.csv'):
                f.unlink()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_provider(self, provider: GaugeProvider) -> Optional[pd.DataFrame]:
        """Load from cache or fetch from upstream for a single provider."""
        cache_file = self._cache_dir / f'{provider.network}.csv'

        # Check cache freshness
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < _CACHE_MAX_AGE_SECONDS:
                try:
                    df = pd.read_csv(cache_file, dtype={'station_id': str})
                    logger.info(f"Cache hit for {provider.network}: {len(df)} stations")
                    return df
                except Exception as exc:
                    logger.warning(f"Corrupt cache for {provider.network}: {exc}")

        # Fetch fresh data
        try:
            df = provider.fetch()
        except Exception as exc:
            logger.warning(f"Provider {provider.network} failed: {exc}")
            # Fall back to stale cache if available
            if cache_file.exists():
                try:
                    return pd.read_csv(cache_file, dtype={'station_id': str})
                except Exception:
                    pass
            return None

        if df is not None and not df.empty:
            try:
                df.to_csv(cache_file, index=False)
            except Exception as exc:
                logger.warning(f"Failed to cache {provider.network}: {exc}")

        return df
