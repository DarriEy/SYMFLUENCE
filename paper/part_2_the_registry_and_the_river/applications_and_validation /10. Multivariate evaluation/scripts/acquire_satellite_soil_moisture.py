"""
Acquire satellite soil moisture data for Bow at Banff catchment.

Downloads:
- SMAP L4 (~9km, NASA)
- ESA CCI SM (~25km, Copernicus CDS)

Both products are aggregated to the catchment boundary for comparison with SUMMA output.
"""

import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import xarray as xr
import geopandas as gpd

# Setup paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_multivar")
SMAP_DIR = DATA_DIR / "observations/soil_moisture/smap"
ESA_CCI_DIR = DATA_DIR / "observations/soil_moisture/esa_cci"
SHAPEFILE = DATA_DIR / "shapefiles/catchment/lumped/bow_tws_uncalibrated/Bow_at_Banff_multivar_HRUs_GRUS.shp"

# Domain parameters
BBOX = {
    "lat_min": 50.95,
    "lat_max": 51.73,
    "lon_min": -116.55,
    "lon_max": -115.53
}
START_DATE = datetime(2004, 1, 1)
END_DATE = datetime(2017, 12, 31)
DOMAIN_NAME = "Bow_at_Banff_multivar"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_smap_via_cmr():
    """
    Download SMAP L4 soil moisture using NASA CMR API.

    SMAP L4 product provides surface soil moisture at ~9km resolution.
    Uses NSIDC granule search via CMR.
    """
    import requests
    import netrc

    logger.info("Starting SMAP acquisition via NASA CMR...")
    SMAP_DIR.mkdir(parents=True, exist_ok=True)

    # Get Earthdata credentials from .netrc
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if not auth:
            logger.error("No Earthdata credentials found in ~/.netrc")
            return None
        user, _, password = auth
        logger.info(f"Found Earthdata credentials for user: {user}")
    except Exception as e:
        logger.error(f"Failed to read Earthdata credentials: {e}")
        return None

    # CMR search parameters for SMAP L4
    cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"

    # Use monthly data for less granules
    params = {
        "short_name": "SPL4SMGP",  # SMAP L4 Global 3-hourly
        "version": "008",
        "temporal": f"{START_DATE.strftime('%Y-%m-%d')},{END_DATE.strftime('%Y-%m-%d')}",
        "bounding_box": f"{BBOX['lon_min']},{BBOX['lat_min']},{BBOX['lon_max']},{BBOX['lat_max']}",
        "page_size": 10,  # Limited for testing
        "page_num": 1,
    }

    logger.info("Querying CMR for SMAP granules...")
    logger.info(f"Temporal: {params['temporal']}")
    logger.info(f"Bounding box: {params['bounding_box']}")

    session = requests.Session()
    session.auth = (user, password)

    try:
        response = session.get(cmr_url, params=params, timeout=60)
        response.raise_for_status()

        entries = response.json().get("feed", {}).get("entry", [])
        logger.info(f"Found {len(entries)} SMAP granules")

        if not entries:
            logger.warning("No SMAP granules found for the specified parameters")
            # Try alternate product
            params["short_name"] = "SPL4SMAU"  # Analysis Update product
            response = session.get(cmr_url, params=params, timeout=60)
            response.raise_for_status()
            entries = response.json().get("feed", {}).get("entry", [])
            logger.info(f"Found {len(entries)} SPL4SMAU granules")

        # Download first few granules as test
        downloaded = []
        for entry in entries[:3]:
            title = entry.get("title", "unknown")
            links = entry.get("links", [])

            data_links = [
                link.get("href") for link in links
                if link.get("href", "").endswith((".h5", ".nc"))
                and "data#" in link.get("rel", "")
            ]

            if data_links:
                url = data_links[0]
                filename = url.split("/")[-1]
                out_path = SMAP_DIR / filename

                if out_path.exists():
                    logger.info(f"Already exists: {filename}")
                    downloaded.append(out_path)
                    continue

                logger.info(f"Downloading: {filename}")
                with session.get(url, stream=True, timeout=600) as r:
                    r.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            f.write(chunk)
                downloaded.append(out_path)
                logger.info(f"Saved: {out_path}")

        return downloaded if downloaded else None

    except Exception as e:
        logger.error(f"SMAP acquisition failed: {e}")
        return None


def download_smap_via_appeears():
    """
    Alternative: Use NASA AppEEARS for area subsetting.
    AppEEARS provides spatial/temporal subsetting of SMAP data.

    Note: Requires AppEEARS API token.
    """
    logger.info("AppEEARS acquisition not implemented in standalone mode")
    logger.info("Consider using SYMFLUENCE's full acquisition pipeline instead")
    return None


def download_esa_cci_sm():
    """
    Download ESA CCI Soil Moisture via Copernicus CDS.

    ESA CCI SM provides merged active/passive microwave soil moisture
    at 0.25° (~25km) resolution, available from 1978-present.
    """
    try:
        import cdsapi
    except ImportError:
        logger.error("cdsapi not installed. Run: pip install cdsapi")
        return None

    logger.info("Starting ESA CCI SM acquisition via Copernicus CDS...")
    ESA_CCI_DIR.mkdir(parents=True, exist_ok=True)

    try:
        c = cdsapi.Client()
    except Exception as e:
        logger.error(f"CDS API client error: {e}")
        logger.info("Check ~/.cdsapirc configuration")
        return None

    downloaded = []

    # Download year by year (CDS rate limits)
    for year in range(START_DATE.year, END_DATE.year + 1):
        out_file = ESA_CCI_DIR / f"{DOMAIN_NAME}_ESA_CCI_SM_{year}.zip"

        if out_file.exists():
            logger.info(f"Already exists: {out_file.name}")
            downloaded.append(out_file)
            continue

        # Build request for one year
        request = {
            'variable': 'volumetric_surface_soil_moisture',
            'type_of_sensor': 'combined',
            'time_aggregation': 'day_average',
            'type_of_record': 'cdr',
            'version': 'v202312',
            'year': str(year),
            'month': [f"{m:02d}" for m in range(1, 13)],
            'day': [f"{d:02d}" for d in range(1, 32)],
            # Note: CDS doesn't support bbox for this dataset - global download
            # We'll subset after download
        }

        logger.info(f"Requesting ESA CCI SM for {year}...")

        try:
            c.retrieve(
                'satellite-soil-moisture',
                request,
                str(out_file)
            )
            downloaded.append(out_file)
            logger.info(f"Downloaded: {out_file.name}")
        except Exception as e:
            logger.warning(f"Failed to download {year}: {e}")
            continue

    return downloaded if downloaded else None


def process_esa_cci_to_catchment(nc_files, output_dir):
    """
    Process ESA CCI NetCDF files - extract and aggregate to catchment.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load catchment for masking
    gdf = gpd.read_file(SHAPEFILE)
    gdf_wgs = gdf.to_crs('EPSG:4326')

    all_data = []

    for nc_file in nc_files:
        try:
            ds = xr.open_dataset(nc_file)

            # Subset to bounding box (with buffer)
            buffer = 0.5  # degrees
            ds_subset = ds.sel(
                lat=slice(BBOX['lat_max'] + buffer, BBOX['lat_min'] - buffer),
                lon=slice(BBOX['lon_min'] - buffer, BBOX['lon_max'] + buffer)
            )

            if 'sm' in ds_subset:
                sm_var = 'sm'
            elif 'volumetric_surface_soil_moisture' in ds_subset:
                sm_var = 'volumetric_surface_soil_moisture'
            else:
                sm_var = list(ds_subset.data_vars)[0]

            # Compute spatial mean over catchment bbox
            sm_mean = ds_subset[sm_var].mean(dim=['lat', 'lon'])

            df = sm_mean.to_dataframe().reset_index()
            df = df.rename(columns={sm_var: 'sm'})
            all_data.append(df[['time', 'sm']])

            ds.close()

        except Exception as e:
            logger.warning(f"Error processing {nc_file}: {e}")
            continue

    if not all_data:
        return None

    # Combine all years
    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined = df_combined.sort_values('time')
    df_combined = df_combined.drop_duplicates(subset='time')
    df_combined.set_index('time', inplace=True)

    # Save processed data
    out_file = output_dir / f"{DOMAIN_NAME}_esa_cci_sm_processed.csv"
    df_combined.to_csv(out_file)
    logger.info(f"Saved processed ESA CCI SM: {out_file}")

    return out_file


def check_existing_data():
    """Check what satellite soil moisture data already exists."""
    logger.info("\n=== Checking existing satellite soil moisture data ===")

    # Check SMAP
    smap_files = list(SMAP_DIR.glob("*.h5")) + list(SMAP_DIR.glob("*.nc"))
    logger.info(f"SMAP files: {len(smap_files)}")

    # Check ESA CCI
    esa_files = list(ESA_CCI_DIR.glob("*.nc")) + list(ESA_CCI_DIR.glob("*.zip"))
    esa_extracted = list((ESA_CCI_DIR / "extracted").glob("*.nc")) if (ESA_CCI_DIR / "extracted").exists() else []
    logger.info(f"ESA CCI archives: {len(esa_files)}")
    logger.info(f"ESA CCI extracted: {len(esa_extracted)}")

    return {
        'smap': smap_files,
        'esa_cci': esa_files,
        'esa_cci_extracted': esa_extracted
    }


def main():
    """Main acquisition workflow."""
    logger.info("=" * 60)
    logger.info("Satellite Soil Moisture Acquisition for Bow at Banff")
    logger.info("=" * 60)
    logger.info(f"Domain: {DOMAIN_NAME}")
    logger.info(f"Bounding box: {BBOX}")
    logger.info(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    logger.info("")

    # Check existing data
    existing = check_existing_data()

    # Try SMAP acquisition
    if not existing['smap']:
        logger.info("\n=== Attempting SMAP acquisition ===")
        smap_result = download_smap_via_cmr()
        if smap_result:
            logger.info(f"SMAP acquisition successful: {len(smap_result)} files")
        else:
            logger.warning("SMAP acquisition failed or returned no data")
    else:
        logger.info(f"SMAP data already exists ({len(existing['smap'])} files)")

    # Try ESA CCI acquisition
    if not existing['esa_cci'] and not existing['esa_cci_extracted']:
        logger.info("\n=== Attempting ESA CCI SM acquisition ===")
        esa_result = download_esa_cci_sm()
        if esa_result:
            logger.info(f"ESA CCI acquisition successful: {len(esa_result)} files")
        else:
            logger.warning("ESA CCI acquisition failed or returned no data")
    else:
        logger.info("ESA CCI data already exists")

    logger.info("\n=== Acquisition complete ===")
    logger.info("Note: Full processing requires SYMFLUENCE pipeline for:")
    logger.info("  - SMAP granule merging and spatial aggregation")
    logger.info("  - ESA CCI extraction and catchment masking")
    logger.info("  - Quality filtering and gap filling")


if __name__ == "__main__":
    main()
