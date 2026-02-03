"""
Acquire MODIS MOD16A2 Evapotranspiration data for Bow at Banff catchment.

Uses NASA AppEEARS API for spatial subsetting and downloading 8-day ET composites.
"""

import logging
import time
import requests
import numpy as np
import pandas as pd
import xarray as xr
import netrc
from pathlib import Path
from datetime import datetime

# Setup paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_multivar")
ET_DIR = DATA_DIR / "observations/et/modis"
ET_DIR.mkdir(parents=True, exist_ok=True)

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

# AppEEARS API
APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_earthdata_credentials():
    """Get Earthdata credentials from .netrc."""
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if auth:
            return auth[0], auth[2]  # username, password
    except Exception as e:
        logger.warning(f"Could not read .netrc: {e}")
    return None, None


def appeears_login(username, password):
    """Login to AppEEARS and get token."""
    try:
        response = requests.post(
            f"{APPEEARS_BASE}/login",
            auth=(username, password),
            timeout=60
        )
        response.raise_for_status()
        token = response.json().get('token')
        logger.info("Successfully logged into AppEEARS")
        return token
    except Exception as e:
        logger.error(f"AppEEARS login failed: {e}")
        return None


def appeears_logout(token):
    """Logout from AppEEARS."""
    try:
        requests.post(
            f"{APPEEARS_BASE}/logout",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
    except Exception:
        pass


def list_available_products(token):
    """List available products in AppEEARS."""
    try:
        response = requests.get(
            f"{APPEEARS_BASE}/product",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60
        )
        response.raise_for_status()
        products = response.json()
        # Filter for MOD16
        mod16_products = [p for p in products if 'MOD16' in p.get('ProductAndVersion', '')]
        return mod16_products
    except Exception as e:
        logger.error(f"Failed to list products: {e}")
        return []


def submit_appeears_task(token, product="MOD16A2.061", variable="ET_500m"):
    """Submit an AppEEARS area task for MOD16 ET."""
    logger.info(f"Submitting AppEEARS task for {product} {variable}")

    # GeoJSON polygon for bounding box
    coordinates = [[
        [BBOX['lon_min'], BBOX['lat_min']],
        [BBOX['lon_max'], BBOX['lat_min']],
        [BBOX['lon_max'], BBOX['lat_max']],
        [BBOX['lon_min'], BBOX['lat_max']],
        [BBOX['lon_min'], BBOX['lat_min']]
    ]]

    task_name = f"SYMFLUENCE_{DOMAIN_NAME}_MOD16_ET_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    task_request = {
        "task_type": "area",
        "task_name": task_name,
        "params": {
            "dates": [{
                "startDate": START_DATE.strftime("%m-%d-%Y"),
                "endDate": END_DATE.strftime("%m-%d-%Y")
            }],
            "layers": [
                {"product": product, "layer": variable},
                {"product": product, "layer": "ET_QC_500m"}  # QC layer for filtering
            ],
            "geo": {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": coordinates
                    },
                    "properties": {}
                }]
            },
            "output": {
                "format": {"type": "netcdf4"},
                "projection": "geographic"
            }
        }
    }

    try:
        response = requests.post(
            f"{APPEEARS_BASE}/task",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json=task_request,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        task_id = result.get('task_id')
        logger.info(f"Submitted task: {task_id}")
        return task_id
    except Exception as e:
        logger.error(f"Failed to submit task: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text[:500]}")
        return None


def check_task_status(token, task_id):
    """Check status of an AppEEARS task."""
    try:
        response = requests.get(
            f"{APPEEARS_BASE}/task/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to check task status: {e}")
        return None


def wait_for_task(token, task_id, max_wait=7200, poll_interval=30):
    """Wait for an AppEEARS task to complete."""
    logger.info(f"Waiting for task {task_id} to complete...")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        status = check_task_status(token, task_id)
        if status is None:
            time.sleep(poll_interval)
            continue

        state = status.get('status', '')
        logger.info(f"Task status: {state}")

        if state == 'done':
            logger.info("Task completed successfully")
            return True
        elif state in ['error', 'expired']:
            logger.error(f"Task failed with status: {state}")
            return False

        time.sleep(poll_interval)

    logger.error(f"Task timed out after {max_wait}s")
    return False


def download_task_results(token, task_id, output_dir):
    """Download results from completed AppEEARS task."""
    logger.info(f"Downloading results for task {task_id}")

    try:
        # Get bundle info
        response = requests.get(
            f"{APPEEARS_BASE}/bundle/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60
        )
        response.raise_for_status()
        bundle = response.json()

        files = bundle.get('files', [])
        logger.info(f"Found {len(files)} files to download")

        downloaded = []
        for file_info in files:
            file_id = file_info.get('file_id')
            file_name = file_info.get('file_name', f"file_{file_id}")

            # Skip non-data files
            if not (file_name.endswith('.nc') or file_name.endswith('.tif')):
                continue

            output_path = output_dir / file_name

            if output_path.exists():
                logger.info(f"Already exists: {file_name}")
                downloaded.append(output_path)
                continue

            logger.info(f"Downloading: {file_name}")

            dl_response = requests.get(
                f"{APPEEARS_BASE}/bundle/{task_id}/{file_id}",
                headers={"Authorization": f"Bearer {token}"},
                stream=True,
                timeout=600
            )
            dl_response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in dl_response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)

            downloaded.append(output_path)
            logger.info(f"Downloaded: {output_path}")

        return downloaded

    except Exception as e:
        logger.error(f"Failed to download results: {e}")
        return []


def process_et_data(nc_files, output_dir):
    """Process downloaded ET data into a single timeseries."""
    logger.info("Processing ET data...")

    all_data = []

    for nc_file in nc_files:
        if not nc_file.suffix == '.nc':
            continue

        try:
            ds = xr.open_dataset(nc_file)

            # Find ET variable
            et_var = None
            for var in ds.data_vars:
                if 'et' in var.lower() and 'qc' not in var.lower():
                    et_var = var
                    break

            if et_var is None:
                logger.warning(f"No ET variable found in {nc_file.name}")
                continue

            et = ds[et_var]

            # Apply scale factor (MOD16A2 uses 0.1)
            if et.dtype in [np.int16, np.int32]:
                et = et.astype(float) * 0.1

            # Mask fill values
            et = et.where(et != 32767 * 0.1)
            et = et.where(et < 1000)  # Additional sanity check

            # Compute spatial mean
            if 'lat' in et.dims and 'lon' in et.dims:
                et_mean = et.mean(dim=['lat', 'lon'], skipna=True)
            elif 'y' in et.dims and 'x' in et.dims:
                et_mean = et.mean(dim=['y', 'x'], skipna=True)
            else:
                et_mean = et

            # Convert to mm/day (8-day composite to daily)
            et_daily = et_mean / 8.0

            df = et_daily.to_dataframe().reset_index()
            all_data.append(df)

            ds.close()

        except Exception as e:
            logger.warning(f"Error processing {nc_file}: {e}")
            continue

    if not all_data:
        logger.error("No data could be processed")
        return None

    # Combine all data
    df_combined = pd.concat(all_data, ignore_index=True)

    # Find time column
    time_col = None
    for col in df_combined.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            time_col = col
            break

    if time_col:
        df_combined[time_col] = pd.to_datetime(df_combined[time_col])
        df_combined = df_combined.sort_values(time_col)
        df_combined = df_combined.drop_duplicates(subset=time_col)
        df_combined = df_combined.set_index(time_col)

    # Find ET column
    et_cols = [c for c in df_combined.columns if c not in ['lat', 'lon', 'x', 'y', 'crs']]
    if et_cols:
        df_combined = df_combined[et_cols]
        df_combined.columns = ['et_mm_day']

    # Save processed data
    out_file = output_dir / f"{DOMAIN_NAME}_MOD16_ET_processed.csv"
    df_combined.to_csv(out_file)
    logger.info(f"Saved processed ET: {out_file}")

    # Print summary
    logger.info("ET data summary:")
    logger.info(f"  Period: {df_combined.index.min()} to {df_combined.index.max()}")
    logger.info(f"  N observations: {len(df_combined)}")
    logger.info(f"  Mean ET: {df_combined['et_mm_day'].mean():.2f} mm/day")
    logger.info(f"  Max ET: {df_combined['et_mm_day'].max():.2f} mm/day")

    return out_file


def check_existing_tasks(token, filter_product="MOD16"):
    """Check for existing AppEEARS tasks, optionally filtered by product."""
    try:
        response = requests.get(
            f"{APPEEARS_BASE}/task",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60
        )
        response.raise_for_status()
        tasks = response.json()

        # Filter for our domain and product
        our_tasks = [t for t in tasks
                     if DOMAIN_NAME in t.get('task_name', '')
                     and filter_product in t.get('task_name', '')]
        if our_tasks:
            logger.info(f"Found {len(our_tasks)} existing {filter_product} tasks for this domain")
            for t in our_tasks[:5]:
                logger.info(f"  {t['task_id']}: {t['task_name']} - {t['status']}")
        return our_tasks
    except Exception as e:
        logger.warning(f"Could not check existing tasks: {e}")
        return []


def main():
    """Main acquisition workflow."""
    logger.info("=" * 60)
    logger.info("MODIS MOD16 ET Acquisition for Bow at Banff")
    logger.info("=" * 60)
    logger.info(f"Domain: {DOMAIN_NAME}")
    logger.info(f"Bounding box: {BBOX}")
    logger.info(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    logger.info("")

    # Check for existing processed data
    processed_file = ET_DIR / f"{DOMAIN_NAME}_MOD16_ET_processed.csv"
    if processed_file.exists():
        logger.info(f"Processed ET data already exists: {processed_file}")
        df = pd.read_csv(processed_file, index_col=0, parse_dates=True)
        logger.info(f"  Period: {df.index.min()} to {df.index.max()}")
        logger.info(f"  N observations: {len(df)}")
        return

    # Get credentials
    username, password = get_earthdata_credentials()
    if not username:
        logger.error("Earthdata credentials not found in ~/.netrc")
        logger.info("Add entry: machine urs.earthdata.nasa.gov login <user> password <pass>")
        return

    logger.info(f"Using Earthdata credentials for user: {username}")

    # Login
    token = appeears_login(username, password)
    if not token:
        return

    try:
        # Check for existing MOD16 ET tasks
        existing_tasks = check_existing_tasks(token, filter_product="MOD16")

        # Look for completed MOD16 tasks we can download
        completed_tasks = [t for t in existing_tasks if t.get('status') == 'done']
        if completed_tasks:
            logger.info(f"Found {len(completed_tasks)} completed task(s)")
            task_id = completed_tasks[0]['task_id']
            logger.info(f"Using task: {task_id}")
        else:
            # Check for queued/processing tasks
            pending_tasks = [t for t in existing_tasks if t.get('status') in ['queued', 'processing']]
            if pending_tasks:
                task_id = pending_tasks[0]['task_id']
                logger.info(f"Found pending task: {task_id}")
            else:
                # Submit new task
                task_id = submit_appeears_task(token)
                if not task_id:
                    return

            # Wait for completion
            if not wait_for_task(token, task_id):
                logger.error("Task did not complete successfully")
                return

        # Download results
        downloaded = download_task_results(token, task_id, ET_DIR)
        if not downloaded:
            logger.error("No files downloaded")
            return

        # Process data
        nc_files = [f for f in downloaded if f.suffix == '.nc']
        if nc_files:
            process_et_data(nc_files, ET_DIR)

    finally:
        appeears_logout(token)

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
