"""
Bokeh tile-map component with click-to-set pour point and shapefile overlays.

Uses Bokeh's built-in tile providers and TapTool for native Panel event
callbacks (no JavaScript bridges needed).
"""

import logging
from math import log, tan, pi

import panel as pn
import param

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

logger = logging.getLogger(__name__)

# Web Mercator helpers
_R = 6378137.0  # Earth radius (m)


def _lonlat_to_mercator(lon, lat):
    x = lon * (_R * pi / 180.0)
    y = _R * log(tan(pi / 4.0 + lat * pi / 360.0))
    return x, y


def _mercator_to_lonlat(x, y):
    from math import atan, exp, degrees
    lon = degrees(x / _R)
    lat = degrees(2.0 * atan(exp(y / _R)) - pi / 2.0)
    return lon, lat


class MapView(param.Parameterized):
    """Interactive Bokeh tile map for pour-point selection and layer overlays."""

    state = param.Parameter(doc="WorkflowState instance")

    # Layer visibility toggles
    show_basins = param.Boolean(default=True, doc="Show river basins")
    show_hrus = param.Boolean(default=False, doc="Show HRUs / catchments")
    show_rivers = param.Boolean(default=True, doc="Show river network")
    show_pour_point = param.Boolean(default=True, doc="Show pour point marker")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)

        # Bokeh data sources
        self._pour_source = ColumnDataSource(data=dict(x=[], y=[]))
        self._basin_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._hru_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._river_source = ColumnDataSource(data=dict(xs=[], ys=[]))

        # Build the figure
        self._fig = self._build_figure()

        # Sync pour point from state
        if state.pour_point_lat is not None and state.pour_point_lon is not None:
            self._update_pour_point_marker(state.pour_point_lat, state.pour_point_lon)

    def _build_figure(self):
        p = figure(
            x_range=(-20000000, 20000000),
            y_range=(-7000000, 12000000),
            x_axis_type='mercator',
            y_axis_type='mercator',
            tools='pan,wheel_zoom,box_zoom,reset,tap',
            active_scroll='wheel_zoom',
            sizing_mode='stretch_both',
            height=500,
        )
        # Bokeh 3.x: pass tile source name as string
        p.add_tile("CartoDB Positron")

        # Pour point marker
        self._pour_renderer = p.scatter(
            'x', 'y', source=self._pour_source,
            size=14, color='red', alpha=0.9,
            legend_label='Pour Point',
        )

        # Basin overlay (patches)
        self._basin_renderer = p.patches(
            'xs', 'ys', source=self._basin_source,
            fill_alpha=0.15, fill_color='steelblue',
            line_color='steelblue', line_width=1.5,
            legend_label='Basins',
        )

        # HRU overlay (patches)
        self._hru_renderer = p.patches(
            'xs', 'ys', source=self._hru_source,
            fill_alpha=0.1, fill_color='orange',
            line_color='orange', line_width=1.0,
            legend_label='HRUs',
        )

        # River network overlay (multi-line)
        self._river_renderer = p.multi_line(
            'xs', 'ys', source=self._river_source,
            line_color='dodgerblue', line_width=1.2,
            legend_label='Rivers',
        )

        p.legend.click_policy = 'hide'
        p.legend.location = 'top_left'

        # Tap callback for setting pour point
        p.on_event('tap', self._on_tap)

        return p

    def _on_tap(self, event):
        """Handle map tap - convert Web Mercator to lat/lon and update state."""
        lon, lat = _mercator_to_lonlat(event.x, event.y)
        self.state.pour_point_lat = round(lat, 6)
        self.state.pour_point_lon = round(lon, 6)
        self.state.config_dirty = True
        self._update_pour_point_marker(lat, lon)
        self.state.append_log(f"Pour point set: {lat:.6f} / {lon:.6f}\n")

    def _update_pour_point_marker(self, lat, lon):
        x, y = _lonlat_to_mercator(lon, lat)
        self._pour_source.data = dict(x=[x], y=[y])

    def load_layers(self):
        """Load shapefiles from project directory and render on map."""
        project_dir = self.state.project_dir
        if not project_dir:
            self.state.append_log("No project directory — load a config first.\n")
            return

        from pathlib import Path
        pdir = Path(project_dir)

        loaded = []

        # Basin shapefile
        basin_path = pdir / 'shapefiles' / 'river_basins' / 'river_basins.shp'
        if basin_path.exists():
            self._load_shapefile_to_source(basin_path, self._basin_source)
            loaded.append('basins')

        # HRU / catchment shapefile
        hru_path = pdir / 'shapefiles' / 'catchment' / 'catchment.shp'
        if hru_path.exists():
            self._load_shapefile_to_source(hru_path, self._hru_source)
            loaded.append('HRUs')

        # River network shapefile
        river_path = pdir / 'shapefiles' / 'river_network' / 'river_network.shp'
        if river_path.exists():
            self._load_line_shapefile(river_path, self._river_source)
            loaded.append('rivers')

        if loaded:
            self.state.append_log(f"Loaded layers: {', '.join(loaded)}\n")
            self._zoom_to_data()
        else:
            self.state.append_log("No shapefiles found in project directory.\n")

    def _load_shapefile_to_source(self, path, source):
        """Load a polygon shapefile into a ColumnDataSource (xs/ys in Web Mercator)."""
        try:
            import geopandas as gpd
            gdf = gpd.read_file(path).to_crs(epsg=3857)
            gdf['geometry'] = gdf['geometry'].simplify(tolerance=100)
            xs, ys = [], []
            for geom in gdf.geometry:
                if geom is None:
                    continue
                if geom.geom_type == 'Polygon':
                    xs.append(list(geom.exterior.coords.xy[0]))
                    ys.append(list(geom.exterior.coords.xy[1]))
                elif geom.geom_type == 'MultiPolygon':
                    for poly in geom.geoms:
                        xs.append(list(poly.exterior.coords.xy[0]))
                        ys.append(list(poly.exterior.coords.xy[1]))
            source.data = dict(xs=xs, ys=ys)
        except Exception as exc:
            logger.warning(f"Failed to load shapefile {path}: {exc}")

    def _load_line_shapefile(self, path, source):
        """Load a line shapefile into a ColumnDataSource."""
        try:
            import geopandas as gpd
            gdf = gpd.read_file(path).to_crs(epsg=3857)
            gdf['geometry'] = gdf['geometry'].simplify(tolerance=100)
            xs, ys = [], []
            for geom in gdf.geometry:
                if geom is None:
                    continue
                if geom.geom_type == 'LineString':
                    xs.append(list(geom.coords.xy[0]))
                    ys.append(list(geom.coords.xy[1]))
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        xs.append(list(line.coords.xy[0]))
                        ys.append(list(line.coords.xy[1]))
            source.data = dict(xs=xs, ys=ys)
        except Exception as exc:
            logger.warning(f"Failed to load line shapefile {path}: {exc}")

    def _zoom_to_data(self):
        """Auto-zoom to the extent of loaded basin data."""
        xs = self._basin_source.data.get('xs', [])
        ys = self._basin_source.data.get('ys', [])
        if not xs or not ys:
            return
        all_x = [c for coords in xs for c in coords]
        all_y = [c for coords in ys for c in coords]
        if all_x and all_y:
            pad = 0.05
            dx = max(all_x) - min(all_x)
            dy = max(all_y) - min(all_y)
            self._fig.x_range.start = min(all_x) - dx * pad
            self._fig.x_range.end = max(all_x) + dx * pad
            self._fig.y_range.start = min(all_y) - dy * pad
            self._fig.y_range.end = max(all_y) + dy * pad

    def panel(self):
        """Return the Panel component for embedding in the app layout."""
        load_btn = pn.widgets.Button(name='Load Layers', button_type='primary', width=120)
        load_btn.on_click(lambda e: self.load_layers())

        toggle_basins = pn.widgets.Checkbox(name='Basins', value=True)
        toggle_hrus = pn.widgets.Checkbox(name='HRUs', value=False)
        toggle_rivers = pn.widgets.Checkbox(name='Rivers', value=True)

        def _toggle_visibility(event, renderer, attr_name):
            renderer.visible = event.new

        toggle_basins.param.watch(lambda e: _toggle_visibility(e, self._basin_renderer, 'show_basins'), 'value')
        toggle_hrus.param.watch(lambda e: _toggle_visibility(e, self._hru_renderer, 'show_hrus'), 'value')
        toggle_rivers.param.watch(lambda e: _toggle_visibility(e, self._river_renderer, 'show_rivers'), 'value')

        self._hru_renderer.visible = False  # default hidden

        controls = pn.Row(
            load_btn, toggle_basins, toggle_hrus, toggle_rivers,
            sizing_mode='stretch_width',
        )
        return pn.Column(
            controls,
            pn.pane.Bokeh(self._fig, sizing_mode='stretch_both'),
            sizing_mode='stretch_both',
        )
