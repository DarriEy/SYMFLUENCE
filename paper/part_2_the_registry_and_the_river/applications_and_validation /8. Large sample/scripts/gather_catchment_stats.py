"""
Gather statistics for all 111 LamaH-Ice catchments.
Reads shapefiles for area, elevation, centroids, and streamflow CSVs for record length.
Outputs a summary CSV and prints summary statistics.
"""

import os
import warnings
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

BASE_DIR = "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/lamahice"
OUT_CSV = "/Users/darrieythorsson/compHydro/Papers/Article 2 - SYMFLUENCE/Applications and validation /8. Large sample/analysis/catchment_stats.csv"

# Collect all domain IDs
domain_ids = []
for name in sorted(os.listdir(BASE_DIR)):
    if name.startswith("domain_"):
        try:
            did = int(name.split("_")[1])
            domain_ids.append(did)
        except ValueError:
            pass

domain_ids.sort()
print(f"Found {len(domain_ids)} domains")

records = []

for did in domain_ids:
    domain_dir = os.path.join(BASE_DIR, f"domain_{did}")
    rec = {"domain_id": did}

    # --- Read shapefile ---
    shp_path = os.path.join(domain_dir, "shapefiles", "catchment",
                            f"{did}_HRUs_GRUs_wgs84.shp")
    if os.path.exists(shp_path):
        try:
            gdf = gpd.read_file(shp_path)

            # Area: use area_calc (km²); if multiple HRUs, sum them
            if "area_calc" in gdf.columns:
                rec["area_km2"] = gdf["area_calc"].sum()
            elif "HRU_area" in gdf.columns:
                rec["area_km2"] = gdf["HRU_area"].sum() / 1e6  # m² -> km²
            elif "GRU_area" in gdf.columns:
                rec["area_km2"] = gdf["GRU_area"].sum() / 1e6

            # Elevation
            if "elev_mean" in gdf.columns:
                # Area-weighted mean if multiple HRUs
                if len(gdf) > 1 and "area_calc" in gdf.columns:
                    weights = gdf["area_calc"] / gdf["area_calc"].sum()
                    rec["elev_mean_m"] = (gdf["elev_mean"] * weights).sum()
                else:
                    rec["elev_mean_m"] = gdf["elev_mean"].mean()

            # Centroid: use attribute fields if available, else compute from geometry
            if "center_lat" in gdf.columns and "center_lon" in gdf.columns:
                # Area-weighted centroid for multi-HRU catchments
                if len(gdf) > 1 and "area_calc" in gdf.columns:
                    weights = gdf["area_calc"] / gdf["area_calc"].sum()
                    rec["centroid_lat"] = (gdf["center_lat"] * weights).sum()
                    rec["centroid_lon"] = (gdf["center_lon"] * weights).sum()
                else:
                    rec["centroid_lat"] = gdf["center_lat"].mean()
                    rec["centroid_lon"] = gdf["center_lon"].mean()
            else:
                # Compute from geometry
                dissolved = gdf.dissolve()
                centroid = dissolved.geometry.centroid.iloc[0]
                rec["centroid_lat"] = centroid.y
                rec["centroid_lon"] = centroid.x

            # Number of HRUs
            rec["n_hrus"] = len(gdf)

            # Glacier fraction
            if "glac_fra" in gdf.columns:
                if len(gdf) > 1 and "area_calc" in gdf.columns:
                    weights = gdf["area_calc"] / gdf["area_calc"].sum()
                    rec["glac_fra"] = (gdf["glac_fra"] * weights).sum()
                else:
                    rec["glac_fra"] = gdf["glac_fra"].mean()

        except Exception as e:
            print(f"  WARNING: Could not read shapefile for domain {did}: {e}")
    else:
        print(f"  WARNING: Shapefile not found for domain {did}")

    # --- Read streamflow record ---
    sf_path = os.path.join(domain_dir, "observations", "streamflow", "raw_data",
                           f"{did}_streamflow.csv")
    if os.path.exists(sf_path):
        try:
            df_sf = pd.read_csv(sf_path, sep=";")
            df_sf["date"] = pd.to_datetime(
                df_sf[["YYYY", "MM", "DD"]].rename(
                    columns={"YYYY": "year", "MM": "month", "DD": "day"}
                )
            )
            rec["record_start"] = df_sf["date"].min().strftime("%Y-%m-%d")
            rec["record_end"] = df_sf["date"].max().strftime("%Y-%m-%d")
            rec["record_days"] = len(df_sf)
            rec["record_years"] = round(len(df_sf) / 365.25, 1)

            # Count valid (non-NaN) observations
            valid = df_sf["qobs"].notna().sum()
            rec["valid_obs"] = int(valid)
            rec["pct_valid"] = round(100.0 * valid / len(df_sf), 1)

        except Exception as e:
            print(f"  WARNING: Could not read streamflow for domain {did}: {e}")
    else:
        print(f"  WARNING: Streamflow file not found for domain {did}")

    records.append(rec)

# Build DataFrame and save
df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved {len(df)} catchment records to:\n  {OUT_CSV}\n")

# --- Summary statistics ---
print("=" * 70)
print("SUMMARY STATISTICS FOR LAMAH-ICE CATCHMENTS")
print("=" * 70)

print(f"\nTotal catchments: {len(df)}")

if "area_km2" in df.columns:
    a = df["area_km2"].dropna()
    print("\nCatchment area (km²):")
    print(f"  Min:    {a.min():.1f}")
    print(f"  Median: {a.median():.1f}")
    print(f"  Mean:   {a.mean():.1f}")
    print(f"  Max:    {a.max():.1f}")
    print(f"  Total:  {a.sum():.1f}")

if "elev_mean_m" in df.columns:
    e = df["elev_mean_m"].dropna()
    print("\nMean elevation (m):")
    print(f"  Min:    {e.min():.0f}")
    print(f"  Median: {e.median():.0f}")
    print(f"  Mean:   {e.mean():.0f}")
    print(f"  Max:    {e.max():.0f}")

if "centroid_lat" in df.columns:
    print("\nGeographic extent:")
    print(f"  Latitude:  {df['centroid_lat'].min():.4f} to {df['centroid_lat'].max():.4f}")
    print(f"  Longitude: {df['centroid_lon'].min():.4f} to {df['centroid_lon'].max():.4f}")

if "record_years" in df.columns:
    r = df["record_years"].dropna()
    print("\nStreamflow record length (years):")
    print(f"  Min:    {r.min():.1f}")
    print(f"  Median: {r.median():.1f}")
    print(f"  Mean:   {r.mean():.1f}")
    print(f"  Max:    {r.max():.1f}")

if "record_start" in df.columns:
    print("\nStreamflow record period:")
    print(f"  Earliest start: {df['record_start'].min()}")
    print(f"  Latest end:     {df['record_end'].max()}")

if "pct_valid" in df.columns:
    v = df["pct_valid"].dropna()
    print("\nValid observations (%):")
    print(f"  Min:    {v.min():.1f}")
    print(f"  Median: {v.median():.1f}")
    print(f"  Mean:   {v.mean():.1f}")

if "glac_fra" in df.columns:
    g = df["glac_fra"].dropna()
    glacierized = (g > 0).sum()
    print("\nGlacier coverage:")
    print(f"  Catchments with glaciers: {glacierized} / {len(g)}")
    print(f"  Max glacier fraction:     {g.max():.3f}")
    print(f"  Mean glacier fraction:    {g.mean():.3f}")

if "n_hrus" in df.columns:
    h = df["n_hrus"].dropna()
    print("\nHRUs per catchment:")
    print(f"  Min: {int(h.min())}, Median: {int(h.median())}, Max: {int(h.max())}, Total: {int(h.sum())}")

print("\n" + "=" * 70)
