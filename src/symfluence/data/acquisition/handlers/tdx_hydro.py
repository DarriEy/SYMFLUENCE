"""TDX-Hydro / GEOGLOWS V2 Acquisition Handler

Downloads global river network and catchment data from the GEOGLOWS V2
dataset hosted on public S3. Data is organized by Virtual Processing Units
(VPUs) with Hive-style partitioning.

Workflow:
    1. Download VPU boundary index (GeoPackage, cached locally)
    2. Spatial join pour point to determine VPU ID(s)
    3. Download catchment parquet + stream GeoPackage for matching VPU(s)

Primary Source:
    GEOGLOWS V2 on AWS S3: https://geoglows-v2.s3.amazonaws.com/
    ~7M global river segments organized by ~125 VPUs
    Format: GeoParquet (catchments), GeoPackage (streams)

Column Convention (matches subsetter TDX type):
    streamID, LINKNO, DSLINKNO, USLINKNO1, USLINKNO2

References:
    - Sanchez Lozano et al. (2021). A Streamflow-Based Global River Network
      and Its Mapping to HydroATLAS
    - https://data.geoglows.org/
"""

from pathlib import Path
import requests

from ..base import BaseAcquisitionHandler
from ..registry import AcquisitionRegistry
from ..mixins import RetryMixin
from ..utils import create_robust_session


# GEOGLOWS V2 S3 base URL (us-west-2, public)
_GEOGLOWS_S3_BASE = "https://geoglows-v2.s3.us-west-2.amazonaws.com"

# VPU boundary index (GeoPackage with VPU geometries)
_VPU_BOUNDARIES_URL = f"{_GEOGLOWS_S3_BASE}/hydrography-global/vpu-boundaries.gpkg"

# VPU data URL templates (Hive-style partitioning)
_CATCHMENTS_URL_TEMPLATE = f"{_GEOGLOWS_S3_BASE}/hydrography/vpu={{vpu}}/catchments_{{vpu}}.parquet"
_STREAMS_URL_TEMPLATE = f"{_GEOGLOWS_S3_BASE}/hydrography/vpu={{vpu}}/streams_{{vpu}}.gpkg"


@AcquisitionRegistry.register('TDX_HYDRO')
@AcquisitionRegistry.register('GEOGLOWS')
class TDXHydroAcquirer(BaseAcquisitionHandler, RetryMixin):
    """TDX-Hydro / GEOGLOWS V2 acquisition via public S3.

    Downloads global river catchments and networks from the GEOGLOWS V2
    dataset. Uses a VPU boundary index for spatial lookup, then fetches
    the corresponding regional data files.

    Config Keys:
        TDX_SOURCE: 'geoglows' (default) or 'nga' (full NGA archive, manual only)

    Output Files:
        tdx_catchments_{vpu}.parquet — catchment polygons
        tdx_streams_{vpu}.gpkg — river network lines
    """

    def download(self, output_dir: Path) -> Path:
        """Download TDX-Hydro/GEOGLOWS data for the domain.

        Args:
            output_dir: Base output directory

        Returns:
            Path to the directory containing downloaded geofabric files
        """
        import geopandas as gpd
        from shapely.geometry import Point

        source = self._get_config_value(lambda: None, default='geoglows', dict_key='TDX_SOURCE').lower()
        if source == 'nga':
            raise NotImplementedError(
                "NGA TDX-Hydro archive download is not supported for automatic "
                "acquisition due to data volume. Download manually and set "
                "SOURCE_GEOFABRIC_BASINS_PATH / SOURCE_GEOFABRIC_RIVERS_PATH."
            )

        geofabric_dir = self._attribute_dir("geofabric") / "tdx_hydro"
        geofabric_dir.mkdir(parents=True, exist_ok=True)

        # Determine pour point location
        lat, lon = self._get_pour_point_coords()
        self.logger.info(
            f"Downloading GEOGLOWS V2 data for pour point ({lat:.4f}, {lon:.4f})"
        )

        # Get VPU index and find matching VPU(s)
        vpu_index = self._get_vpu_index(geofabric_dir)
        pour_point = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        )
        matching_vpus = gpd.sjoin(
            vpu_index, pour_point, how="inner", predicate="contains"
        )

        if matching_vpus.empty:
            # Fallback: nearest VPU
            self.logger.warning(
                "Pour point not inside any VPU boundary, finding nearest VPU"
            )
            vpu_index_projected = vpu_index.to_crs("EPSG:3857")
            pour_projected = pour_point.to_crs("EPSG:3857")
            distances = vpu_index_projected.geometry.distance(
                pour_projected.geometry.iloc[0]
            )
            nearest_idx = distances.idxmin()
            matching_vpus = vpu_index.iloc[[nearest_idx]]

        vpu_ids = matching_vpus["VPUCode"].unique().tolist()
        self.logger.info(f"Matched VPU(s): {vpu_ids}")

        session = create_robust_session(max_retries=5, backoff_factor=2.0)

        all_catchment_files = []
        all_river_files = []

        for vpu_id in vpu_ids:
            cat_path = geofabric_dir / f"tdx_catchments_{vpu_id}.parquet"
            riv_path = geofabric_dir / f"tdx_streams_{vpu_id}.gpkg"

            if not self._skip_if_exists(cat_path):
                self._download_file(
                    session,
                    _CATCHMENTS_URL_TEMPLATE.format(vpu=vpu_id),
                    cat_path,
                    f"catchments VPU {vpu_id}"
                )
            if not self._skip_if_exists(riv_path):
                self._download_file(
                    session,
                    _STREAMS_URL_TEMPLATE.format(vpu=vpu_id),
                    riv_path,
                    f"streams VPU {vpu_id}"
                )

            all_catchment_files.append(cat_path)
            all_river_files.append(riv_path)

        # If multiple VPUs, merge into single files
        if len(vpu_ids) > 1:
            merged_cat = geofabric_dir / "tdx_catchments_merged.parquet"
            merged_riv = geofabric_dir / "tdx_streams_merged.gpkg"
            self._merge_parquets(all_catchment_files, merged_cat)
            self._merge_geopackages(all_river_files, merged_riv)

        self.logger.info(f"GEOGLOWS V2 data downloaded to: {geofabric_dir}")
        return geofabric_dir

    def _get_pour_point_coords(self) -> tuple:
        """Extract pour point lat/lon from config.

        Returns:
            Tuple of (lat, lon)
        """
        pour_point_str = self._get_config_value(lambda: self.config.domain.pour_point_coords, default=None)
        if pour_point_str:
            parts = str(pour_point_str).replace('/', ',').split(',')
            return float(parts[0].strip()), float(parts[1].strip())

        # Fallback: centroid of bounding box
        lat = (self.bbox['lat_min'] + self.bbox['lat_max']) / 2
        lon = (self.bbox['lon_min'] + self.bbox['lon_max']) / 2
        return lat, lon

    def _get_vpu_index(self, cache_dir: Path):
        """Download or load cached VPU boundary index.

        Args:
            cache_dir: Directory to cache the index file

        Returns:
            GeoDataFrame with VPU boundaries
        """
        import geopandas as gpd

        index_path = cache_dir / "vpu_boundaries.gpkg"
        if not index_path.exists():
            self.logger.info("Downloading GEOGLOWS VPU boundary index...")

            def do_download():
                session = create_robust_session(max_retries=3, backoff_factor=1.0)
                resp = session.get(_VPU_BOUNDARIES_URL, stream=True, timeout=300)
                resp.raise_for_status()
                part_path = index_path.with_suffix('.part')
                with open(part_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                part_path.rename(index_path)

            self.execute_with_retry(
                do_download,
                max_retries=3,
                base_delay=5,
                backoff_factor=2.0,
                retryable_exceptions=(
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    IOError,
                ),
            )

        return gpd.read_file(index_path)

    def _download_file(
        self, session, url: str, output_path: Path, description: str
    ):
        """Download a file with retry logic.

        Args:
            session: requests.Session
            url: URL to download from
            output_path: Local path to save to
            description: Human-readable description for logging
        """
        def do_download():
            self.logger.info(f"Downloading {description} from {url}")
            with session.get(url, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                part_path = output_path.with_suffix('.part')
                with open(part_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                part_path.rename(output_path)
            self.logger.info(f"Downloaded {description}: {output_path}")

        self.execute_with_retry(
            do_download,
            max_retries=3,
            base_delay=5,
            backoff_factor=2.0,
            retryable_exceptions=(
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                IOError,
            ),
        )

    def _merge_parquets(self, file_paths: list, output_path: Path):
        """Merge multiple GeoParquet files into one.

        Args:
            file_paths: List of parquet file paths
            output_path: Output merged file path
        """
        import geopandas as gpd
        import pandas as pd

        gdfs = [gpd.read_parquet(p) for p in file_paths if p.exists()]
        if gdfs:
            merged = pd.concat(gdfs, ignore_index=True)
            merged = gpd.GeoDataFrame(merged, crs=gdfs[0].crs)
            merged.to_parquet(output_path)
            self.logger.info(
                f"Merged {len(gdfs)} files into {output_path} "
                f"({len(merged)} features)"
            )

    def _merge_geopackages(self, file_paths: list, output_path: Path):
        """Merge multiple GeoPackage files into one.

        Args:
            file_paths: List of GeoPackage file paths
            output_path: Output merged file path
        """
        import geopandas as gpd
        import pandas as pd

        gdfs = [gpd.read_file(p) for p in file_paths if p.exists()]
        if gdfs:
            merged = pd.concat(gdfs, ignore_index=True)
            merged = gpd.GeoDataFrame(merged, crs=gdfs[0].crs)
            merged.to_file(output_path, driver="GPKG")
            self.logger.info(
                f"Merged {len(gdfs)} files into {output_path} "
                f"({len(merged)} features)"
            )
