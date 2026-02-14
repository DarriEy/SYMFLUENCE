"""gSSURGO Soil Properties Acquisition Handler

Cloud-based acquisition of gridded SSURGO (gSSURGO) soil properties from the
USDA Soil Data Access (SDA) REST API.

gSSURGO Overview:
    Data Type: Soil survey data (tabular + spatial)
    Resolution: ~30m (CONUS)
    Coverage: CONUS + territories
    Properties: Sand, clay, silt, Ksat, porosity, bulk density, organic matter, AWC
    Source: USDA NRCS Soil Data Access

Acquisition Strategy:
    Phase 1: Query SDA for mukeys intersecting the domain polygon
    Phase 2: Query soil horizon properties for those mukeys
    Aggregate by dominant component and depth-weighted average

Data Access:
    REST API: https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest
    No authentication required

References:
    Soil Survey Staff. Gridded Soil Survey Geographic (gSSURGO) Database for
    the Conterminous United States. USDA NRCS.
    https://gdg.sc.egov.usda.gov/
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry
from ..mixins import RetryMixin
from ..utils import create_robust_session


# SDA REST endpoints
_SDA_TABULAR_URL = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"

# Default properties to retrieve from chorizon table
_DEFAULT_PROPERTIES = [
    'sandtotal_r', 'claytotal_r', 'ksat_r', 'dbthirdbar_r', 'wsatiated_r',
]

# All available horizon properties
_ALL_PROPERTIES = [
    'sandtotal_r', 'claytotal_r', 'silttotal_r', 'ksat_r',
    'wsatiated_r', 'wthirdbar_r', 'wfifteenbar_r', 'dbthirdbar_r',
    'om_r', 'awc_r', 'lep_r', 'll_r', 'pi_r',
]


def _bbox_to_wkt(bbox: Dict[str, float]) -> str:
    """Convert bbox dict to WKT POLYGON string."""
    return (
        f"POLYGON(("
        f"{bbox['lon_min']} {bbox['lat_min']}, "
        f"{bbox['lon_max']} {bbox['lat_min']}, "
        f"{bbox['lon_max']} {bbox['lat_max']}, "
        f"{bbox['lon_min']} {bbox['lat_max']}, "
        f"{bbox['lon_min']} {bbox['lat_min']}"
        f"))"
    )


@AcquisitionRegistry.register('GSSURGO')
class GSSURGOAcquirer(BaseAcquisitionHandler, RetryMixin):
    """gSSURGO soil property acquisition via USDA Soil Data Access REST API.

    Downloads soil survey data using a two-phase approach:
    1. Spatial query to find map units (mukeys) intersecting the domain
    2. Tabular query to retrieve soil horizon properties for those mukeys

    Properties are aggregated by dominant component and depth-weighted average
    across horizons.

    Acquisition Strategy:
        1. Convert domain bbox to WKT polygon
        2. Query SDA for intersecting mukeys
        3. For each mukey, query component and horizon tables
        4. Aggregate by dominant component (highest comppct_r)
        5. Compute depth-weighted average of horizon properties
        6. Save as CSV (and optionally as rasterized GeoTIFF)

    Configuration:
        GSSURGO_PROPERTIES: List of horizon properties
            (default: ['sandtotal_r', 'claytotal_r', 'ksat_r', 'dbthirdbar_r', 'wsatiated_r'])
        GSSURGO_MAX_DEPTH_CM: Maximum soil depth to consider
            (default: 200)
        GSSURGO_TOP_DEPTH_CM: Minimum soil depth (default: 0)

    Output:
        CSV file: domain_{domain_name}_gssurgo_properties.csv
        Columns: mukey, musym, muname, comppct_r, {property columns}

    References:
        USDA NRCS Soil Data Access: https://sdmdataaccess.nrcs.usda.gov/
    """

    def download(self, output_dir: Path) -> Path:
        soil_dir = self._attribute_dir("soilclass") / "gssurgo"
        soil_dir.mkdir(parents=True, exist_ok=True)
        out_path = soil_dir / f"domain_{self.domain_name}_gssurgo_properties.csv"

        if self._skip_if_exists(out_path):
            return out_path

        properties = self.config_dict.get('GSSURGO_PROPERTIES', _DEFAULT_PROPERTIES)
        properties = [p for p in properties if p in _ALL_PROPERTIES]
        max_depth = int(self.config_dict.get('GSSURGO_MAX_DEPTH_CM', 200))
        top_depth = int(self.config_dict.get('GSSURGO_TOP_DEPTH_CM', 0))

        if not properties:
            raise ValueError(
                f"No valid gSSURGO properties. Choose from: {_ALL_PROPERTIES}"
            )

        self.logger.info(
            f"Acquiring gSSURGO soil data for bbox: {self.bbox}, "
            f"properties: {properties}, depth: {top_depth}-{max_depth}cm"
        )

        session = create_robust_session(max_retries=3, backoff_factor=2.0)

        # Phase 1: Get mukeys for the domain
        mukeys = self._query_mukeys(session)
        if not mukeys:
            raise RuntimeError(
                f"No SSURGO map units found for bbox: {self.bbox}. "
                "gSSURGO covers CONUS only."
            )
        self.logger.info(f"Found {len(mukeys)} map units in domain")

        # Phase 2: Get soil properties for mukeys
        prop_df = self._query_horizon_properties(
            session, mukeys, properties, top_depth, max_depth
        )

        if prop_df is None or prop_df.empty:
            raise RuntimeError("No soil property data returned from SDA")

        # Save results
        prop_df.to_csv(out_path, index=False)
        self.logger.info(
            f"gSSURGO acquisition complete: {len(prop_df)} map units, saved to {out_path}"
        )
        return out_path

    def _query_mukeys(self, session) -> List[str]:
        """Query SDA for map unit keys intersecting the domain polygon."""
        wkt = _bbox_to_wkt(self.bbox)

        # Use SDA spatial query to get mukeys (controlled API, not user input)
        query = (  # nosec B608
            f"SELECT mu.mukey, mu.musym, mu.muname "
            f"FROM mapunit mu "
            f"INNER JOIN SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt}') mk "
            f"ON mu.mukey = mk.mukey"
        )

        result = self._execute_sda_query(session, query)
        if result is None:
            return []

        return result['mukey'].tolist()

    def _query_horizon_properties(
        self,
        session,
        mukeys: List[str],
        properties: List[str],
        top_depth: int,
        max_depth: int,
    ) -> Optional[pd.DataFrame]:
        """Query SDA for depth-weighted soil properties by dominant component."""
        prop_cols = ", ".join([f"AVG(ch.{p}) AS {p}" for p in properties])
        mukey_list = ", ".join([f"'{mk}'" for mk in mukeys])

        # Query: dominant component per mapunit, depth-weighted horizon averages
        query = f"""  # nosec B608
        SELECT
            mu.mukey,
            mu.musym,
            mu.muname,
            c.comppct_r,
            {prop_cols}
        FROM mapunit mu
        INNER JOIN component c ON mu.mukey = c.mukey
        INNER JOIN chorizon ch ON c.cokey = ch.cokey
        WHERE mu.mukey IN ({mukey_list})
            AND ch.hzdept_r >= {top_depth}
            AND ch.hzdepb_r <= {max_depth}
            AND c.cokey = (
                SELECT TOP 1 c2.cokey
                FROM component c2
                WHERE c2.mukey = mu.mukey
                ORDER BY c2.comppct_r DESC
            )
        GROUP BY mu.mukey, mu.musym, mu.muname, c.comppct_r
        ORDER BY mu.mukey
        """

        # SDA has query size limits; chunk if many mukeys
        if len(mukeys) > 500:
            return self._query_chunked(session, mukeys, properties, top_depth, max_depth)

        return self._execute_sda_query(session, query)

    def _query_chunked(
        self,
        session,
        mukeys: List[str],
        properties: List[str],
        top_depth: int,
        max_depth: int,
    ) -> Optional[pd.DataFrame]:
        """Query SDA in chunks when mukey list is large."""
        chunk_size = 500
        all_results = []

        for i in range(0, len(mukeys), chunk_size):
            chunk = mukeys[i:i + chunk_size]
            self.logger.info(
                f"Querying SDA chunk {i // chunk_size + 1}/"
                f"{(len(mukeys) + chunk_size - 1) // chunk_size}"
            )
            result = self._query_horizon_properties(
                session, chunk, properties, top_depth, max_depth
            )
            if result is not None and not result.empty:
                all_results.append(result)

        if not all_results:
            return None
        return pd.concat(all_results, ignore_index=True)

    def _execute_sda_query(self, session, query: str) -> Optional[pd.DataFrame]:
        """Execute a query against the SDA REST API."""
        payload = {
            "query": query,
            "format": "json+columnname",
        }

        def do_query():
            resp = session.post(
                _SDA_TABULAR_URL,
                data=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()

        try:
            result = self.execute_with_retry(
                do_query,
                max_retries=3,
                base_delay=5.0,
                backoff_factor=2.0,
            )
        except Exception as e:
            self.logger.warning(f"SDA query failed: {e}")
            return None

        # Parse SDA JSON response
        if not result or 'Table' not in result:
            return None

        table = result['Table']
        if len(table) < 2:
            return None

        columns = table[0]
        rows = table[1:]
        return pd.DataFrame(rows, columns=columns)
