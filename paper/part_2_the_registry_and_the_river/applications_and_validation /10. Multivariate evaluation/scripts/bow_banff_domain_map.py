#!/usr/bin/env python3
"""
Bow at Banff Domain Map
Creates a publication-quality map of the study domain with context.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import geopandas as gpd
from pathlib import Path
import contextily as ctx
from shapely.geometry import Point

# Set up paths
DATA_DIR = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_multivar")
OUTPUT_DIR = Path("/Users/darrieythorsson/compHydro/papers/article_2_symfluence/applications_and_validation /10. Multivariate evaluation/figures/bow_banff")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_shapefiles():
    """Load catchment and river network shapefiles."""
    # Catchment
    catchment_paths = [
        DATA_DIR / "shapefiles/catchment/lumped/bow_tws_uncalibrated/Bow_at_Banff_multivar_HRUs_GRUS.shp",
        Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_lumped_era5/shapefiles/catchment/lumped/run_1/Bow_at_Banff_lumped_era5_HRUs_GRUS.shp"),
    ]

    catchment = None
    for path in catchment_paths:
        if path.exists():
            catchment = gpd.read_file(path)
            print(f"Loaded catchment from: {path}")
            break

    # River network
    river_paths = [
        DATA_DIR / "shapefiles/river_network",
        Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/domain_Bow_at_Banff_lumped_era5/shapefiles/river_network"),
    ]

    rivers = None
    for path in river_paths:
        if path.exists():
            shp_files = list(path.glob("*.shp"))
            if shp_files:
                rivers = gpd.read_file(shp_files[0])
                print(f"Loaded rivers from: {shp_files[0]}")
                break

    return catchment, rivers


def create_domain_map(catchment, rivers=None, add_basemap=True):
    """Create publication-quality domain map."""
    fig, ax = plt.subplots(figsize=(10, 10))

    if catchment is None:
        print("No catchment data available")
        return fig

    # Ensure CRS is WGS84 for context
    if catchment.crs is None:
        catchment = catchment.set_crs(epsg=4326)
    catchment_wgs = catchment.to_crs(epsg=4326)

    # Get bounds
    bounds = catchment_wgs.total_bounds
    margin = 0.15
    xmin, ymin, xmax, ymax = bounds
    dx = xmax - xmin
    dy = ymax - ymin

    # Plot catchment
    catchment_web = catchment_wgs.to_crs(epsg=3857)  # Web Mercator for basemap
    catchment_web.plot(ax=ax, facecolor='#4A90D9', edgecolor='#1a1a1a',
                       linewidth=2, alpha=0.5, zorder=2)

    # Plot rivers if available
    if rivers is not None:
        rivers_wgs = rivers.to_crs(epsg=4326) if rivers.crs else rivers.set_crs(epsg=4326)
        rivers_web = rivers_wgs.to_crs(epsg=3857)
        rivers_web.plot(ax=ax, color='#1E90FF', linewidth=1.5, zorder=3, label='River Network')

    # Add pour point (Banff)
    pour_point = gpd.GeoDataFrame(
        {'name': ['Banff (WSC 05BB001)']},
        geometry=[Point(-115.5717, 51.1722)],
        crs='EPSG:4326'
    ).to_crs(epsg=3857)
    pour_point.plot(ax=ax, color='red', marker='^', markersize=150, zorder=5,
                    edgecolor='white', linewidth=1.5)

    # Add basemap
    if add_basemap:
        try:
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldTerrain, zoom=10, alpha=0.8)
        except Exception as e:
            print(f"Could not add basemap: {e}")
            try:
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=10, alpha=0.6)
            except:
                pass

    # Calculate area
    catchment_utm = catchment_wgs.to_crs(epsg=32611)  # UTM 11N
    area_km2 = catchment_utm.area.sum() / 1e6

    # Add scale bar (approximate)

    # Add labels and annotations
    ax.set_xlabel('Easting (m)', fontsize=11)
    ax.set_ylabel('Northing (m)', fontsize=11)
    ax.set_title('Bow River at Banff\nStudy Domain', fontsize=14, fontweight='bold', pad=15)

    # Add info box
    info_text = f"""Domain Info:
Area: {area_km2:.0f} km²
Outlet: Banff, Alberta
Gauge: WSC 05BB001
Elevation: ~1400-3400 m"""

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    # Legend
    legend_elements = [
        Patch(facecolor='#4A90D9', edgecolor='#1a1a1a', alpha=0.5, label='Catchment'),
        Line2D([0], [0], color='#1E90FF', linewidth=2, label='River Network'),
        Line2D([0], [0], marker='^', color='red', linestyle='None',
               markersize=10, markeredgecolor='white', label='Pour Point'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    # Add north arrow
    ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.88),
                xycoords='axes fraction', textcoords='axes fraction',
                fontsize=14, fontweight='bold', ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    plt.tight_layout()
    return fig


def create_inset_map(catchment):
    """Create map with regional inset showing location."""
    fig = plt.figure(figsize=(12, 10))

    # Main map
    ax_main = fig.add_axes([0.1, 0.1, 0.75, 0.85])

    if catchment is not None:
        catchment_wgs = catchment.to_crs(epsg=4326) if catchment.crs else catchment.set_crs(epsg=4326)
        catchment_web = catchment_wgs.to_crs(epsg=3857)

        catchment_web.plot(ax=ax_main, facecolor='#4A90D9', edgecolor='#1a1a1a',
                          linewidth=2, alpha=0.5, zorder=2)

        # Pour point
        pour_point = gpd.GeoDataFrame(
            geometry=[Point(-115.5717, 51.1722)], crs='EPSG:4326'
        ).to_crs(epsg=3857)
        pour_point.plot(ax=ax_main, color='red', marker='^', markersize=150,
                       zorder=5, edgecolor='white', linewidth=1.5)

        try:
            ctx.add_basemap(ax_main, source=ctx.providers.Esri.WorldTerrain, zoom=10, alpha=0.8)
        except:
            pass

    ax_main.set_title('Bow River at Banff - Study Domain', fontsize=14, fontweight='bold')
    ax_main.set_xlabel('Easting (m)')
    ax_main.set_ylabel('Northing (m)')

    # Inset map showing location in Western Canada
    ax_inset = fig.add_axes([0.65, 0.65, 0.25, 0.25])

    # Create simple inset with Canada context
    ax_inset.set_xlim(-130, -100)
    ax_inset.set_ylim(45, 60)

    # Plot study area location
    ax_inset.plot(-115.5717, 51.1722, 'r*', markersize=15, zorder=5)
    ax_inset.text(-115, 52.5, 'Study\nArea', fontsize=8, ha='center')

    # Add simple geographic context
    ax_inset.axhline(y=49, color='gray', linestyle='--', linewidth=0.5)  # US-Canada border
    ax_inset.text(-115, 48, 'USA', fontsize=8, ha='center', color='gray')
    ax_inset.text(-115, 54, 'Canada', fontsize=8, ha='center', color='gray')
    ax_inset.text(-120, 51, 'BC', fontsize=8, ha='center')
    ax_inset.text(-110, 51, 'AB', fontsize=8, ha='center')

    ax_inset.set_title('Location', fontsize=10)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.patch.set_facecolor('lightblue')
    ax_inset.patch.set_alpha(0.3)

    return fig


def main():
    """Main execution function."""
    print("=" * 50)
    print("Creating Bow at Banff Domain Maps")
    print("=" * 50)

    # Load data
    catchment, rivers = load_shapefiles()

    # Create main domain map
    print("\nCreating domain map...")
    fig1 = create_domain_map(catchment, rivers, add_basemap=True)

    output_path = OUTPUT_DIR / "bow_banff_domain_map.png"
    fig1.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    fig1.savefig(OUTPUT_DIR / "bow_banff_domain_map.pdf", bbox_inches='tight', facecolor='white')
    plt.close(fig1)

    # Create inset map
    print("\nCreating inset map...")
    fig2 = create_inset_map(catchment)

    output_path = OUTPUT_DIR / "bow_banff_domain_map_inset.png"
    fig2.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close(fig2)

    print("\n" + "=" * 50)
    print("Domain maps created successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
