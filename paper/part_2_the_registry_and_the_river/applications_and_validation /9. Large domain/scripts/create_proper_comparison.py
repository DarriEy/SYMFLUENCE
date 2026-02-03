#!/usr/bin/env python3
"""Create proper sim vs obs comparison using routed discharge at gauge locations.

The previous comparison had a critical bug: it matched gauges to HRUs using
centroid distance, which selected interior HRUs instead of outlet HRUs.
This meant comparing local HRU runoff to accumulated catchment discharge.

This script implements the CORRECT methodology:
1. Match gauge coordinates to the nearest river segment (reach)
2. Extract simulated routed discharge at that reach from mizuRoute output
3. Compare routed discharge to observed catchment discharge

Author: Claude (Anthropic)
Date: 2026-02-02
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False


# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
SYMFLUENCE_DATA = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data")
DOMAIN_DIR = SYMFLUENCE_DATA / "domain_Iceland_Multivar"
TOPOLOGY_FILE = DOMAIN_DIR / "settings" / "mizuRoute" / "topology.nc"
MIZU_OUTPUT_DIR = DOMAIN_DIR / "simulations" / "large_domain" / "mizuRoute"
LAMAHICE_DIR = SYMFLUENCE_DATA / "lamahice"


def load_topology():
    """Load mizuRoute topology with segment and HRU information."""
    if not TOPOLOGY_FILE.exists():
        raise FileNotFoundError(f"Topology file not found: {TOPOLOGY_FILE}")

    return xr.open_dataset(TOPOLOGY_FILE)


def load_river_network():
    """Load river network shapefile with segment geometries."""
    river_shp = DOMAIN_DIR / "shapefiles" / "river_network" / "Iceland_Multivar_riverNetwork_semidistributed.shp"
    if not river_shp.exists():
        return None
    return gpd.read_file(river_shp)


def match_gauges_to_segments(gauge_coords: pd.DataFrame, topology: xr.Dataset,
                              river_network: gpd.GeoDataFrame = None) -> pd.DataFrame:
    """Match gauge locations to the nearest river segment.

    This is the CORRECT approach: match to river network segments,
    not to interior HRUs.

    Parameters
    ----------
    gauge_coords : DataFrame with columns [gauge_id, lat, lon]
    topology : mizuRoute topology dataset
    river_network : Optional GeoDataFrame with segment geometries

    Returns
    -------
    DataFrame with [gauge_id, segment_id, distance_km]
    """

    matches = []

    # Get segment info from topology
    seg_ids = topology['segId'].values

    # If we have river network geometry, use it for precise matching
    if river_network is not None and HAS_GEOPANDAS:
        print("  Using river network geometry for segment matching")

        for _, row in gauge_coords.iterrows():
            gauge_point = Point(row['lon'], row['lat'])

            # Find nearest segment
            distances = river_network.geometry.distance(gauge_point)
            nearest_idx = distances.idxmin()
            nearest_seg = river_network.loc[nearest_idx]

            # Get segment ID (column name may vary)
            seg_id_col = 'LINKNO' if 'LINKNO' in river_network.columns else 'seg_id'
            seg_id = nearest_seg[seg_id_col] if seg_id_col in river_network.columns else nearest_idx

            matches.append({
                'gauge_id': row['gauge_id'],
                'segment_id': int(seg_id),
                'distance_km': distances[nearest_idx] * 111  # Approx deg to km
            })
    else:
        print("  Warning: No river network geometry, using topology centroids")
        # Fallback: would need segment coordinates from elsewhere

    return pd.DataFrame(matches)


def find_upstream_hrus(segment_id: int, topology: xr.Dataset) -> list:
    """Find all HRUs that drain to a given segment (including upstream).

    Uses the routing topology to trace all contributing HRUs.
    """
    seg_ids = topology['segId'].values
    down_seg_ids = topology['downSegId'].values
    hru_to_seg = topology['hruToSegId'].values
    hru_ids = topology['hruId'].values

    # Find all segments that drain to target segment (recursively)
    upstream_segs = set([segment_id])
    changed = True
    while changed:
        changed = False
        for i, down_id in enumerate(down_seg_ids):
            if down_id in upstream_segs and seg_ids[i] not in upstream_segs:
                upstream_segs.add(seg_ids[i])
                changed = True

    # Find all HRUs that drain to these segments
    upstream_hrus = []
    for i, seg_id in enumerate(hru_to_seg):
        if seg_id in upstream_segs:
            upstream_hrus.append(hru_ids[i])

    return upstream_hrus


def load_mizuroute_output(start_date: str = '2008-01-01',
                          end_date: str = '2010-12-31') -> xr.Dataset:
    """Load mizuRoute routed discharge output."""
    # Find output files
    output_files = list(MIZU_OUTPUT_DIR.glob("*.nc"))
    output_files = [f for f in output_files if 'log' not in f.name.lower()]

    if not output_files:
        raise FileNotFoundError(f"No mizuRoute output found in {MIZU_OUTPUT_DIR}")

    # Load and concatenate if multiple files
    ds = xr.open_mfdataset(output_files, combine='by_coords')

    # Select time period
    ds = ds.sel(time=slice(start_date, end_date))

    return ds


def extract_reach_discharge(mizu_ds: xr.Dataset, segment_id: int) -> pd.Series:
    """Extract routed discharge time series at a specific reach."""

    # Variable name depends on mizuRoute configuration
    var_names = ['IRFroutedRunoff', 'dlayRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']

    for var in var_names:
        if var in mizu_ds:
            # Find segment index
            seg_ids = mizu_ds['reachID'].values if 'reachID' in mizu_ds else mizu_ds['seg'].values
            seg_idx = np.where(seg_ids == segment_id)[0]

            if len(seg_idx) == 0:
                continue

            # Extract time series (m³/s)
            discharge = mizu_ds[var].isel(seg=seg_idx[0]).to_pandas()
            return discharge

    raise ValueError("Could not find discharge variable in mizuRoute output")


def load_observations(domain_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    """Load observed discharge for a LamaH-Ice domain."""
    obs_file = LAMAHICE_DIR / f"domain_{domain_id}" / "observations" / "streamflow" / "preprocessed" / f"{domain_id}_streamflow_processed.csv"

    if not obs_file.exists():
        return None

    obs = pd.read_csv(obs_file, parse_dates=['datetime'], index_col='datetime')
    obs = obs[start_date:end_date]

    return obs


def convert_to_specific_discharge(discharge_cms: pd.Series, area_km2: float) -> pd.Series:
    """Convert discharge (m³/s) to specific discharge (mm/day)."""
    # Q (mm/day) = Q (m³/s) * 86400 / (area_km² * 1e6) * 1000
    return discharge_cms * 86400 / (area_km2 * 1e6) * 1000


def create_comparison_data(output_file: Path = None):
    """Create proper sim vs obs comparison using routed discharge."""

    print("Creating proper validation comparison")
    print("=" * 60)

    # Load topology
    print("\nLoading topology...")
    topology = load_topology()
    print(f"  {len(topology['segId'])} segments, {len(topology['hruId'])} HRUs")

    # Load river network for matching
    print("\nLoading river network...")
    river_network = load_river_network()
    if river_network is not None:
        print(f"  {len(river_network)} river segments")

    # Load LamaH-Ice catchment info
    print("\nLoading LamaH-Ice catchment info...")
    matches_file = ANALYSIS_DIR / "lamahice_hru_matches.csv"
    if not matches_file.exists():
        raise FileNotFoundError(f"HRU matches file not found: {matches_file}")

    lamahice = pd.read_csv(matches_file)
    print(f"  {len(lamahice)} validation catchments")

    # Prepare gauge coordinates
    gauge_coords = lamahice[['lamahice_id', 'lamahice_lat', 'lamahice_lon']].copy()
    gauge_coords.columns = ['gauge_id', 'lat', 'lon']

    # Match gauges to river segments
    print("\nMatching gauges to river segments...")
    segment_matches = match_gauges_to_segments(gauge_coords, topology, river_network)

    if len(segment_matches) == 0:
        print("  Warning: No segment matches found, cannot proceed")
        return None

    print(f"  Matched {len(segment_matches)} gauges to segments")

    # Try to load mizuRoute output
    print("\nLoading mizuRoute output...")
    try:
        mizu_ds = load_mizuroute_output()
        print(f"  Time range: {mizu_ds.time.values[0]} to {mizu_ds.time.values[-1]}")
    except FileNotFoundError as e:
        print(f"  {e}")
        print("\n  *** mizuRoute output not available ***")
        print("  Run the simulation first, then re-run this script.")
        print("\n  Saving segment matches for future use...")

        # Save the segment matches
        segment_matches_file = ANALYSIS_DIR / "gauge_segment_matches.csv"
        segment_matches.to_csv(segment_matches_file, index=False)
        print(f"  Saved: {segment_matches_file}")

        topology.close()
        return None

    # Create comparison data
    print("\nExtracting routed discharge at gauge locations...")
    comparison_data = []

    for _, row in segment_matches.iterrows():
        gauge_id = row['gauge_id']
        segment_id = row['segment_id']

        # Get catchment area
        catchment_info = lamahice[lamahice['lamahice_id'] == gauge_id].iloc[0]
        area_km2 = catchment_info['lamahice_area_km2']

        # Extract simulated routed discharge
        try:
            sim_cms = extract_reach_discharge(mizu_ds, segment_id)
            sim_mm = convert_to_specific_discharge(sim_cms, area_km2)
        except Exception as e:
            print(f"  Warning: Could not extract discharge for segment {segment_id}: {e}")
            continue

        # Load observations
        obs = load_observations(gauge_id, '2008-01-01', '2010-12-31')
        if obs is None:
            continue

        obs_mm = convert_to_specific_discharge(obs['discharge_cms'], area_km2)

        # Align time series
        sim_daily = sim_mm.resample('D').mean()
        obs_daily = obs_mm.resample('D').mean()

        common_dates = sim_daily.index.intersection(obs_daily.index)

        for date in common_dates:
            comparison_data.append({
                'date': date,
                'domain_id': gauge_id,
                'segment_id': segment_id,
                'sim_mm': sim_daily[date],
                'obs_mm': obs_daily[date]
            })

    topology.close()
    mizu_ds.close()

    if not comparison_data:
        print("  No comparison data generated")
        return None

    df = pd.DataFrame(comparison_data)

    # Save
    if output_file is None:
        output_file = ANALYSIS_DIR / "routed_obs_comparison_data.csv"

    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    print(f"  {len(df)} data points")
    print(f"  {df['domain_id'].nunique()} catchments")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Create proper sim vs obs comparison using routed discharge"
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV file path")
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    create_comparison_data(args.output)


if __name__ == "__main__":
    main()
