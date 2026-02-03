#!/usr/bin/env python3
"""Create sim vs obs comparison using aggregated local runoff.

Since mizuRoute routing failed (topology error with multiple coastal outlets),
this script provides an alternative approach:
1. Match gauge locations to HRU catchments using the topology
2. Aggregate local runoff from all upstream HRUs
3. Compare aggregated runoff to observed discharge

Note: This is an approximation - it sums instantaneous runoff without
proper channel routing delays. However, at daily timesteps for small-medium
catchments, the difference should be minimal.

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
    print("Warning: xarray not available")

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
FUSE_OUTPUT_DIR = DOMAIN_DIR / "simulations" / "large_domain" / "FUSE"
LAMAHICE_DIR = SYMFLUENCE_DATA / "lamahice"


def load_topology():
    """Load mizuRoute topology with segment and HRU information."""
    if not TOPOLOGY_FILE.exists():
        raise FileNotFoundError(f"Topology file not found: {TOPOLOGY_FILE}")
    return xr.open_dataset(TOPOLOGY_FILE)


def load_fuse_output():
    """Load FUSE local runoff output."""
    # Find the runs_def.nc file
    output_file = FUSE_OUTPUT_DIR / "Iceland_Multivar_large_domain_runs_def.nc"

    if not output_file.exists():
        raise FileNotFoundError(f"FUSE output not found: {output_file}")

    return xr.open_dataset(output_file)


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


def match_gauges_to_segments(gauge_coords: pd.DataFrame, topology: xr.Dataset) -> pd.DataFrame:
    """Match gauge locations to the nearest river segment.

    Uses river network shapefile for precise matching.
    """
    river_shp = DOMAIN_DIR / "shapefiles" / "river_network" / "Iceland_Multivar_riverNetwork_semidistributed.shp"

    matches = []

    if river_shp.exists() and HAS_GEOPANDAS:
        print("  Using river network geometry for segment matching")
        river_network = gpd.read_file(river_shp)

        for _, row in gauge_coords.iterrows():
            gauge_point = Point(row['lon'], row['lat'])

            # Find nearest segment
            distances = river_network.geometry.distance(gauge_point)
            nearest_idx = distances.idxmin()
            nearest_seg = river_network.loc[nearest_idx]

            # Get segment ID
            seg_id_col = 'LINKNO' if 'LINKNO' in river_network.columns else 'seg_id'
            seg_id = nearest_seg[seg_id_col] if seg_id_col in river_network.columns else nearest_idx

            matches.append({
                'gauge_id': row['gauge_id'],
                'segment_id': int(seg_id),
                'distance_km': distances[nearest_idx] * 111
            })
    else:
        print("  Warning: No river network geometry, using HRU centroids")
        # Fallback: match to HRUs directly
        hru_shp = DOMAIN_DIR / "shapefiles" / "catchment" / "semidistributed" / "large_domain" / "Iceland_Multivar_HRUs_GRUs.shp"
        if hru_shp.exists() and HAS_GEOPANDAS:
            hrus = gpd.read_file(hru_shp)
            hru_centroids = hrus.geometry.centroid

            for _, row in gauge_coords.iterrows():
                gauge_point = Point(row['lon'], row['lat'])
                distances = hru_centroids.distance(gauge_point)
                nearest_idx = distances.idxmin()

                # Get HRU ID
                hru_id_col = 'HRU_ID' if 'HRU_ID' in hrus.columns else 'GRU_ID'
                hru_id = hrus.loc[nearest_idx, hru_id_col] if hru_id_col in hrus.columns else nearest_idx

                # Get segment from topology
                hru_to_seg = topology['hruToSegId'].values
                hru_ids = topology['hruId'].values
                idx = np.where(hru_ids == hru_id)[0]
                seg_id = hru_to_seg[idx[0]] if len(idx) > 0 else -1

                matches.append({
                    'gauge_id': row['gauge_id'],
                    'segment_id': int(seg_id),
                    'distance_km': distances[nearest_idx] * 111
                })

    return pd.DataFrame(matches)


def aggregate_runoff(fuse_ds: xr.Dataset, hru_ids: list,
                      hru_areas: dict = None) -> pd.Series:
    """Aggregate local runoff from multiple HRUs.

    Parameters
    ----------
    fuse_ds : FUSE output dataset
    hru_ids : List of HRU IDs to aggregate
    hru_areas : Optional dict of {hru_id: area_km2} for area-weighting

    Returns
    -------
    pd.Series of aggregated runoff (mm/day)
    """
    # Find runoff variable
    runoff_vars = ['WATEFROMSOIL', 'evapotrans', 'instRunoff', 'total_runoff',
                   'scalarTotalRunoff', 'q_instnt']

    runoff_var = None
    for var in runoff_vars:
        if var in fuse_ds:
            runoff_var = var
            break

    if runoff_var is None:
        # Try to compute from available variables
        if 'averageRoutedRunoff' in fuse_ds:
            runoff_var = 'averageRoutedRunoff'
        else:
            raise ValueError(f"No runoff variable found. Available: {list(fuse_ds.data_vars)}")

    print(f"  Using variable: {runoff_var}")

    # Get time coordinate
    time = fuse_ds['time'].values

    # Get HRU dimension
    hru_dim = None
    for dim in ['hru', 'gru', 'latitude', 'longitude']:
        if dim in fuse_ds.dims:
            hru_dim = dim
            break

    if hru_dim is None:
        raise ValueError(f"Could not find HRU dimension. Available: {list(fuse_ds.dims)}")

    # For gridded output (latitude, longitude), we need to handle differently
    if hru_dim in ['latitude', 'longitude']:
        print("  Note: FUSE output is gridded, aggregating all cells")
        # Sum all grid cells (simplified approach)
        runoff_data = fuse_ds[runoff_var].values
        if len(runoff_data.shape) == 3:  # time, lat, lon
            total_runoff = np.nanmean(runoff_data, axis=(1, 2))  # Mean across space
        else:
            total_runoff = np.nanmean(runoff_data, axis=1)
    else:
        # HRU-based output
        all_hru_ids = fuse_ds['hruId'].values if 'hruId' in fuse_ds else None

        if all_hru_ids is None:
            # Assume indices match
            indices = list(range(len(hru_ids)))
        else:
            indices = [np.where(all_hru_ids == h)[0][0] for h in hru_ids
                      if h in all_hru_ids]

        if len(indices) == 0:
            raise ValueError("No matching HRUs found in FUSE output")

        # Extract and aggregate
        runoff_data = fuse_ds[runoff_var].isel({hru_dim: indices}).values

        if hru_areas is not None:
            # Area-weighted mean
            areas = np.array([hru_areas.get(h, 1.0) for h in hru_ids])
            total_runoff = np.average(runoff_data, axis=1, weights=areas)
        else:
            total_runoff = np.mean(runoff_data, axis=1)

    return pd.Series(total_runoff, index=pd.to_datetime(time))


def load_observations(domain_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    """Load observed discharge for a LamaH-Ice domain."""
    obs_file = LAMAHICE_DIR / f"domain_{domain_id}" / "observations" / "streamflow" / "preprocessed" / f"{domain_id}_streamflow_processed.csv"

    if not obs_file.exists():
        # Try alternative path
        obs_file = LAMAHICE_DIR / "catchments" / f"{domain_id}" / "streamflow.csv"

    if not obs_file.exists():
        return None

    try:
        obs = pd.read_csv(obs_file, parse_dates=['datetime'], index_col='datetime')
        obs = obs[start_date:end_date]
        return obs
    except Exception as e:
        print(f"  Error loading observations for {domain_id}: {e}")
        return None


def convert_to_specific_discharge(discharge_cms: pd.Series, area_km2: float) -> pd.Series:
    """Convert discharge (m3/s) to specific discharge (mm/day)."""
    # Q (mm/day) = Q (m3/s) * 86400 / (area_km2 * 1e6) * 1000
    return discharge_cms * 86400 / (area_km2 * 1e6) * 1000


def compute_kge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Compute Kling-Gupta Efficiency."""
    if len(sim) == 0 or len(obs) == 0:
        return np.nan

    sim = np.array(sim)
    obs = np.array(obs)

    # Remove NaNs
    valid = ~(np.isnan(sim) | np.isnan(obs))
    sim = sim[valid]
    obs = obs[valid]

    if len(sim) < 10:
        return np.nan

    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)

    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge


def create_comparison_data(output_file: Path = None):
    """Create sim vs obs comparison using aggregated local runoff."""

    print("Creating validation comparison (aggregated local runoff)")
    print("=" * 60)
    print("Note: Using aggregated local runoff since mizuRoute routing failed")

    # Load topology
    print("\nLoading topology...")
    topology = load_topology()
    print(f"  {len(topology['segId'])} segments, {len(topology['hruId'])} HRUs")

    # Load FUSE output
    print("\nLoading FUSE output...")
    fuse_ds = load_fuse_output()
    print(f"  Time range: {fuse_ds.time.values[0]} to {fuse_ds.time.values[-1]}")
    print(f"  Variables: {list(fuse_ds.data_vars)[:5]}...")

    # Load LamaH-Ice catchment info
    print("\nLoading LamaH-Ice catchment info...")
    matches_file = ANALYSIS_DIR / "lamahice_hru_matches.csv"
    if not matches_file.exists():
        print(f"  Warning: HRU matches file not found: {matches_file}")
        print("  Creating placeholder results file...")

        # Create a simple summary of FUSE output
        summary = {
            'status': 'FUSE completed, awaiting LamaH-Ice matches',
            'fuse_output': str(FUSE_OUTPUT_DIR / "Iceland_Multivar_large_domain_runs_def.nc"),
            'variables': list(fuse_ds.data_vars),
            'time_range': f"{fuse_ds.time.values[0]} to {fuse_ds.time.values[-1]}",
            'dimensions': dict(fuse_ds.dims)
        }

        summary_file = ANALYSIS_DIR / "fuse_simulation_summary.txt"
        with open(summary_file, 'w') as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")

        print(f"  Saved summary: {summary_file}")
        topology.close()
        fuse_ds.close()
        return None

    lamahice = pd.read_csv(matches_file)
    print(f"  {len(lamahice)} validation catchments")

    # Prepare gauge coordinates
    gauge_coords = lamahice[['lamahice_id', 'lamahice_lat', 'lamahice_lon']].copy()
    gauge_coords.columns = ['gauge_id', 'lat', 'lon']

    # Match gauges to river segments
    print("\nMatching gauges to river segments...")
    segment_matches = match_gauges_to_segments(gauge_coords, topology)

    if len(segment_matches) == 0:
        print("  Warning: No segment matches found")
        return None

    print(f"  Matched {len(segment_matches)} gauges to segments")

    # Save segment matches
    segment_matches_file = ANALYSIS_DIR / "gauge_segment_matches.csv"
    segment_matches.to_csv(segment_matches_file, index=False)
    print(f"  Saved: {segment_matches_file}")

    # Create comparison data
    print("\nExtracting and aggregating runoff...")
    comparison_results = []

    for _, row in segment_matches.iterrows():
        gauge_id = row['gauge_id']
        segment_id = row['segment_id']

        # Get catchment area
        catchment_info = lamahice[lamahice['lamahice_id'] == gauge_id]
        if len(catchment_info) == 0:
            continue
        catchment_info = catchment_info.iloc[0]
        area_km2 = catchment_info['lamahice_area_km2']

        # Find upstream HRUs
        try:
            upstream_hrus = find_upstream_hrus(segment_id, topology)
            print(f"  Gauge {gauge_id}: segment {segment_id}, {len(upstream_hrus)} upstream HRUs")
        except Exception as e:
            print(f"  Warning: Could not find upstream HRUs for segment {segment_id}: {e}")
            continue

        if len(upstream_hrus) == 0:
            continue

        # Aggregate runoff
        try:
            sim_mm = aggregate_runoff(fuse_ds, upstream_hrus)
            sim_daily = sim_mm.resample('D').mean()
        except Exception as e:
            print(f"  Warning: Could not aggregate runoff: {e}")
            continue

        # Load observations
        obs = load_observations(gauge_id, '2008-01-01', '2010-12-31')
        if obs is None:
            # Store sim-only results
            comparison_results.append({
                'gauge_id': gauge_id,
                'segment_id': segment_id,
                'area_km2': area_km2,
                'n_upstream_hrus': len(upstream_hrus),
                'mean_sim_mm': float(sim_daily.mean()),
                'kge': np.nan,
                'has_obs': False
            })
            continue

        obs_mm = convert_to_specific_discharge(obs['discharge_cms'], area_km2)
        obs_daily = obs_mm.resample('D').mean()

        # Align time series
        common_dates = sim_daily.index.intersection(obs_daily.index)

        if len(common_dates) < 30:
            continue

        sim_aligned = sim_daily[common_dates].values
        obs_aligned = obs_daily[common_dates].values

        # Compute KGE
        kge = compute_kge(sim_aligned, obs_aligned)

        comparison_results.append({
            'gauge_id': gauge_id,
            'segment_id': segment_id,
            'area_km2': area_km2,
            'n_upstream_hrus': len(upstream_hrus),
            'mean_sim_mm': float(np.nanmean(sim_aligned)),
            'mean_obs_mm': float(np.nanmean(obs_aligned)),
            'kge': kge,
            'has_obs': True,
            'n_days': len(common_dates)
        })

    topology.close()
    fuse_ds.close()

    if not comparison_results:
        print("  No comparison data generated")
        return None

    df = pd.DataFrame(comparison_results)

    # Save
    if output_file is None:
        output_file = ANALYSIS_DIR / "aggregated_runoff_comparison.csv"

    df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    print(f"  {len(df)} catchments")
    print(f"  {df['has_obs'].sum()} with observations")

    if df['has_obs'].any():
        valid_kge = df[df['has_obs']]['kge'].dropna()
        print(f"  Median KGE: {valid_kge.median():.3f}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Create sim vs obs comparison using aggregated local runoff"
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV file path")
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    create_comparison_data(args.output)


if __name__ == "__main__":
    main()
