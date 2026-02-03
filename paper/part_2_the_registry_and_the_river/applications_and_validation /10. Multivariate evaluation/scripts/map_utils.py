#!/usr/bin/env python3
"""
Shared map utilities for multivariate evaluation overview plots.
Provides consistent, publication-quality domain map styling.
"""

import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as pe


def plot_domain_map(gdf, ax, pour_point_coords, title,
                    catchment_color='#3498DB', edge_color='#1A5276',
                    show_scale=True, show_north=True, show_inset=True,
                    inset_extent=None, inset_marker_coords=None):
    """
    Create a publication-quality domain map.

    Parameters:
    -----------
    gdf : GeoDataFrame
        Catchment boundary shapefile
    ax : matplotlib.axes.Axes
        Axis to plot on
    pour_point_coords : tuple
        (latitude, longitude) of pour point
    title : str
        Map title (e.g., "Bow River at Banff")
    catchment_color : str
        Fill color for catchment
    edge_color : str
        Edge color for catchment
    show_scale : bool
        Whether to show scale bar
    show_north : bool
        Whether to show north arrow
    show_inset : bool
        Whether to show location inset map
    inset_extent : tuple
        (lon_min, lon_max, lat_min, lat_max) for inset map
    inset_marker_coords : tuple
        (lon, lat) for marker in inset (defaults to pour_point if None)
    """
    if gdf is None:
        ax.text(0.5, 0.5, 'Catchment data\nnot available',
                ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        return ax

    # Ensure WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    gdf_wgs = gdf.to_crs(epsg=4326)

    # Get bounds with margin
    bounds = gdf_wgs.total_bounds  # minx, miny, maxx, maxy
    dx = bounds[2] - bounds[0]
    dy = bounds[3] - bounds[1]
    margin = 0.15

    # Set axis limits
    ax.set_xlim(bounds[0] - dx * margin, bounds[2] + dx * margin)
    ax.set_ylim(bounds[1] - dy * margin, bounds[3] + dy * margin)

    # Plot catchment with shadow effect
    gdf_wgs.plot(ax=ax, facecolor=catchment_color, edgecolor=edge_color,
                 linewidth=2.5, alpha=0.6, zorder=2)

    # Add subtle gradient effect by plotting again with less alpha
    gdf_wgs.plot(ax=ax, facecolor='none', edgecolor=edge_color,
                 linewidth=1, alpha=0.9, zorder=3)

    # Pour point marker with glow effect
    lat, lon = pour_point_coords
    ax.plot(lon, lat, 'o', markersize=14, color='white', zorder=4)
    ax.plot(lon, lat, '^', markersize=10, color='#E74C3C',
            markeredgecolor='white', markeredgewidth=1.5, zorder=5)

    # Calculate area
    try:
        # Use appropriate UTM zone
        center_lon = (bounds[0] + bounds[2]) / 2
        utm_zone = int((center_lon + 180) / 6) + 1
        utm_crs = f"EPSG:{32600 + utm_zone}" if lat > 0 else f"EPSG:{32700 + utm_zone}"
        gdf_utm = gdf_wgs.to_crs(utm_crs)
        area_km2 = gdf_utm.area.sum() / 1e6
    except:
        area_km2 = None

    # Info box with improved styling
    info_lines = [title]
    if area_km2:
        info_lines.append(f"Area: {area_km2:.0f} km²")

    info_text = '\n'.join(info_lines)

    text_box = ax.text(0.03, 0.97, info_text, transform=ax.transAxes,
                       fontsize=10, fontweight='bold', verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor='#BDC3C7', alpha=0.95, linewidth=1.5),
                       zorder=10)

    # Scale bar
    if show_scale:
        _add_scale_bar(ax, bounds)

    # North arrow
    if show_north:
        _add_north_arrow(ax)

    # Location inset
    if show_inset and inset_extent:
        marker_coords = inset_marker_coords if inset_marker_coords else (lon, lat)
        _add_location_inset(ax, inset_extent, marker_coords)

    # Styling
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.tick_params(labelsize=9)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.3, zorder=1)

    # Clean up spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#7F8C8D')

    return ax


def _add_scale_bar(ax, bounds):
    """Add a scale bar to the map."""
    # Calculate appropriate scale bar length
    dx_deg = bounds[2] - bounds[0]
    center_lat = (bounds[1] + bounds[3]) / 2

    # Approximate km per degree at this latitude
    km_per_deg = 111.32 * np.cos(np.radians(center_lat))
    total_width_km = dx_deg * km_per_deg

    # Choose nice scale bar length
    nice_lengths = [1, 2, 5, 10, 20, 50, 100, 200]
    scale_km = nice_lengths[0]
    for length in nice_lengths:
        if length < total_width_km * 0.3:
            scale_km = length

    # Position scale bar
    x_start = bounds[0] + dx_deg * 0.05
    y_pos = bounds[1] + (bounds[3] - bounds[1]) * 0.08
    x_end = x_start + scale_km / km_per_deg

    # Draw scale bar
    ax.plot([x_start, x_end], [y_pos, y_pos], 'k-', linewidth=3, zorder=8)
    ax.plot([x_start, x_start], [y_pos - 0.003, y_pos + 0.003], 'k-', linewidth=2, zorder=8)
    ax.plot([x_end, x_end], [y_pos - 0.003, y_pos + 0.003], 'k-', linewidth=2, zorder=8)

    # Label
    ax.text((x_start + x_end) / 2, y_pos + 0.008, f'{scale_km} km',
            ha='center', va='bottom', fontsize=8, fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')], zorder=9)


def _add_north_arrow(ax):
    """Add a north arrow to the map."""
    # Position in upper right
    ax.annotate('N', xy=(0.95, 0.95), xytext=(0.95, 0.85),
                xycoords='axes fraction', textcoords='axes fraction',
                fontsize=12, fontweight='bold', ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2),
                color='#2C3E50', zorder=10)


def _add_location_inset(ax, extent, marker_coords):
    """Add a location inset map.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Main axis
    extent : tuple
        (lon_min, lon_max, lat_min, lat_max) for inset
    marker_coords : tuple
        (lon, lat) for marker
    """
    # Create inset axes
    ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower right',
                          borderpad=0.5)

    # Set extent
    lon_min, lon_max, lat_min, lat_max = extent
    ax_inset.set_xlim(lon_min, lon_max)
    ax_inset.set_ylim(lat_min, lat_max)

    # Simple background
    ax_inset.set_facecolor('#E8F4F8')

    # Add coastline approximation (simple box for land)
    # This is a placeholder - in production you'd use cartopy or similar

    # Mark study location
    lon, lat = marker_coords
    ax_inset.plot(lon, lat, 'r*', markersize=12, zorder=5,
                  markeredgecolor='white', markeredgewidth=0.5)

    # Styling
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for spine in ax_inset.spines.values():
        spine.set_linewidth(1)
        spine.set_color('#7F8C8D')

    return ax_inset


def get_region_inset_extent(region):
    """Get appropriate inset extent for different regions."""
    extents = {
        'western_canada': (-130, -100, 45, 62),
        'pacific_northwest': (-130, -115, 42, 52),
        'iceland': (-30, -10, 62, 68),
        'scandinavia': (0, 35, 55, 72),
        'alps': (5, 18, 43, 50),
    }
    return extents.get(region, (-180, 180, -90, 90))


# Color palettes for different domains
DOMAIN_COLORS = {
    'bow': {'catchment': '#3498DB', 'edge': '#1A5276'},      # Blue
    'paradise': {'catchment': '#27AE60', 'edge': '#1D6F42'}, # Green
    'iceland': {'catchment': '#9B59B6', 'edge': '#6C3483'},  # Purple
    'default': {'catchment': '#3498DB', 'edge': '#1A5276'},
}
