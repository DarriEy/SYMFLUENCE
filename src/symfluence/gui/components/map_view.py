"""
Bokeh tile-map component with click-to-set pour point, shapefile overlays,
raster (GeoTIFF) overlays, and bounding-box draw tool.

Uses Bokeh's built-in tile providers and TapTool for native Panel event
callbacks (no JavaScript bridges needed).
"""

import logging
import threading
import time
from math import log, tan, pi
from pathlib import Path

import panel as pn
import param

from bokeh.models import (
    BoxEditTool,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
)
from bokeh.plotting import figure

from ..utils.threading_utils import run_on_ui_thread
from .raster_layer import read_tiff_for_bokeh

logger = logging.getLogger(__name__)

# Web Mercator helpers
_R = 6378137.0  # Earth radius (m)

# Network colours for gauge markers
NETWORK_COLORS = {
    'WSC': '#e74c3c',       # red
    'USGS': '#3498db',      # blue
    'SMHI': '#2ecc71',      # green
    'LamaH-ICE': '#9b59b6', # purple
}
_DEFAULT_GAUGE_COLOR = '#95a5a6'


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
    show_gauges = param.Boolean(default=True, doc="Show gauge station markers")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)

        # Bokeh data sources
        self._pour_source = ColumnDataSource(data=dict(x=[], y=[]))
        self._basin_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._hru_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._river_source = ColumnDataSource(data=dict(xs=[], ys=[]))
        self._gauge_source = ColumnDataSource(data=dict(
            x=[], y=[], station_id=[], name=[], network=[], river=[],
            color=[], lat=[], lon=[],
        ))

        # Gauge data store (lazy)
        self._gauge_store = None
        self._gauge_loaded = False
        self._status_pane = None
        self._layers_loading = False
        self._gauges_loading = False

        # Guard: suppress tap-as-pour-point when a gauge was just clicked
        self._gauge_click_ts = 0.0

        # Raster layer tracking
        self._raster_renderers = {}
        self._raster_sources = {}
        self._raster_toggles_column = pn.Column(sizing_mode='stretch_width')

        # Bounding box source (Quad glyph for BoxEditTool)
        self._bbox_source = ColumnDataSource(data=dict(
            left=[], right=[], top=[], bottom=[],
        ))

        # Build the figure
        self._fig = self._build_figure()

        # Sync pour point from state
        if state.pour_point_lat is not None and state.pour_point_lon is not None:
            self._update_pour_point_marker(state.pour_point_lat, state.pour_point_lon)

        # Watch pour point changes for auto-zoom
        state.param.watch(self._on_pour_point_change, ['pour_point_lat', 'pour_point_lon'])

        # Auto-load shapefiles when project_dir changes (e.g. config loaded)
        state.param.watch(self._on_project_dir_change, ['project_dir'])

        # Watch bounding_box_coords from text field
        state.param.watch(self._on_bbox_text_change, ['bounding_box_coords'])

        # Bbox control buttons
        self._use_view_btn = pn.widgets.Button(
            name='Use Current View', button_type='default',
            sizing_mode='stretch_width', margin=(4, 5),
        )
        self._clear_bbox_btn = pn.widgets.Button(
            name='Clear Bounding Box', button_type='default',
            sizing_mode='stretch_width', margin=(4, 5),
        )
        self._use_view_btn.on_click(self._on_use_current_view)
        self._clear_bbox_btn.on_click(self._on_clear_bbox)

    def _build_figure(self):
        p = figure(
            x_range=(-20000000, 20000000),
            y_range=(-7000000, 12000000),
            x_axis_type='mercator',
            y_axis_type='mercator',
            tools='pan,wheel_zoom,box_zoom,reset,tap',
            active_scroll='wheel_zoom',
            sizing_mode='stretch_both',
            min_height=500,
        )
        p.toolbar.logo = None
        p.axis.visible = False
        p.grid.visible = False
        p.outline_line_color = None
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

        # Gauge station markers
        self._gauge_renderer = p.scatter(
            'x', 'y', source=self._gauge_source,
            size=8, color='color', alpha=0.75,
            legend_label='Gauges',
            nonselection_alpha=0.4,
        )

        # HoverTool for gauges
        gauge_hover = HoverTool(
            renderers=[self._gauge_renderer],
            tooltips=[
                ('Station', '@name'),
                ('ID', '@station_id'),
                ('Network', '@network'),
                ('River', '@river'),
                ('Lat/Lon', '@lat / @lon'),
            ],
        )
        p.add_tools(gauge_hover)

        # Bounding box overlay (Quad glyph for BoxEditTool)
        self._bbox_renderer = p.quad(
            left='left', right='right', top='top', bottom='bottom',
            source=self._bbox_source,
            fill_alpha=0.08, fill_color='green',
            line_color='green', line_width=2, line_dash='dashed',
            legend_label='Bounding Box',
        )
        box_edit = BoxEditTool(renderers=[self._bbox_renderer], num_objects=1)
        p.add_tools(box_edit)

        # Watch for bbox data changes from the draw tool
        self._bbox_source.on_change('data', self._on_bbox_source_change)

        p.legend.click_policy = 'hide'
        p.legend.location = 'top_left'
        p.legend.background_fill_alpha = 0.8
        p.legend.border_line_color = '#d7e2eb'
        p.legend.label_text_font_size = '11px'
        p.legend.spacing = 2
        p.legend.padding = 6

        # Tap callback for setting pour point
        p.on_event('tap', self._on_tap)

        # Selection callback for gauge clicks
        self._gauge_source.selected.on_change('indices', self._on_gauge_selected)

        return p

    # ------------------------------------------------------------------
    # Click / selection handlers
    # ------------------------------------------------------------------

    def _on_tap(self, event):
        """Handle map tap — convert Web Mercator to lat/lon and update state.

        Skipped when a gauge marker was just clicked (within 300ms) to
        prevent the tap from also setting a pour point.
        """
        # Suppress if a gauge was just selected (callbacks fire in
        # unpredictable order; the timestamp guard is more reliable than
        # checking selected.indices which gets cleared immediately).
        if time.monotonic() - self._gauge_click_ts < 0.3:
            return

        lon, lat = _mercator_to_lonlat(event.x, event.y)
        self.state.pour_point_lat = round(lat, 6)
        self.state.pour_point_lon = round(lon, 6)
        self.state.config_dirty = True
        self._update_pour_point_marker(lat, lon)
        self.state.append_log(f"Pour point set: {lat:.6f} / {lon:.6f}\n")

    def _on_gauge_selected(self, attr, old, new):
        """Handle gauge marker selection — populate state.selected_gauge."""
        if not new:
            return

        # Timestamp so _on_tap knows to skip
        self._gauge_click_ts = time.monotonic()

        idx = new[0]
        data = self._gauge_source.data
        gauge_info = {
            'station_id': data['station_id'][idx],
            'name': data['name'][idx],
            'lat': data['lat'][idx],
            'lon': data['lon'][idx],
            'network': data['network'][idx],
            'river': data['river'][idx],
        }
        self.state.selected_gauge = gauge_info
        self.state.append_log(
            f"Gauge selected: {gauge_info['name']} ({gauge_info['station_id']}, "
            f"{gauge_info['network']})\n"
        )
        # Clear selection so next tap can set pour point
        self._gauge_source.selected.indices = []

    def _on_project_dir_change(self, event):
        """Auto-load shapefiles when project_dir becomes available."""
        if event.new:
            self.load_layers()

    def _on_pour_point_change(self, *events):
        """Auto-zoom to ~1 degree window around pour point when it changes."""
        lat = self.state.pour_point_lat
        lon = self.state.pour_point_lon
        if lat is None or lon is None:
            return
        self._update_pour_point_marker(lat, lon)
        # Only auto-zoom if no basins are loaded (basins provide better zoom)
        if not self._basin_source.data.get('xs'):
            self._zoom_to_point(lat, lon)

    def _zoom_to_point(self, lat, lon, half_deg=0.5):
        """Zoom the map to a window around the given lat/lon."""
        x_min, y_min = _lonlat_to_mercator(lon - half_deg, lat - half_deg)
        x_max, y_max = _lonlat_to_mercator(lon + half_deg, lat + half_deg)
        self._fig.x_range.start = x_min
        self._fig.x_range.end = x_max
        self._fig.y_range.start = y_min
        self._fig.y_range.end = y_max

    def _update_pour_point_marker(self, lat, lon):
        x, y = _lonlat_to_mercator(lon, lat)
        self._pour_source.data = dict(x=[x], y=[y])

    # ------------------------------------------------------------------
    # Bounding box handling
    # ------------------------------------------------------------------

    def _on_bbox_source_change(self, attr, old, new):
        """Sync bbox draw-tool changes to state as north/west/south/east string."""
        left = new.get('left', [])
        right = new.get('right', [])
        top = new.get('top', [])
        bottom = new.get('bottom', [])
        if not left:
            return
        # Use the last drawn box (num_objects=1 but just in case)
        lon_min, lat_min = _mercator_to_lonlat(left[-1], bottom[-1])
        lon_max, lat_max = _mercator_to_lonlat(right[-1], top[-1])
        # Format: north/west/south/east (matches BOUNDING_BOX_COORDS config)
        bbox_str = f"{lat_max:.6f}/{lon_min:.6f}/{lat_min:.6f}/{lon_max:.6f}"
        # Avoid re-entrant loop: only update if actually different
        if self.state.bounding_box_coords != bbox_str:
            self.state.bounding_box_coords = bbox_str
            self.state.config_dirty = True
            self.state.append_log(f"Bounding box set: {bbox_str}\n")

    def _on_bbox_text_change(self, event):
        """Sync text-field bbox changes to the map rectangle."""
        self.update_bbox_from_string(event.new)

    def update_bbox_from_string(self, bbox_str):
        """Parse 'north/west/south/east' and draw rectangle on map."""
        if not bbox_str or not bbox_str.strip():
            return
        try:
            parts = bbox_str.split('/')
            if len(parts) != 4:
                return
            north, west, south, east = (float(p) for p in parts)
        except (ValueError, TypeError):
            return

        x_min, y_min = _lonlat_to_mercator(west, south)
        x_max, y_max = _lonlat_to_mercator(east, north)

        new_data = dict(left=[x_min], right=[x_max], top=[y_max], bottom=[y_min])
        # Avoid re-entrant loop: check if data is already matching
        current = self._bbox_source.data
        if (current.get('left') == new_data['left']
                and current.get('right') == new_data['right']):
            return
        self._bbox_source.data = new_data

    # ------------------------------------------------------------------
    # Raster overlay support
    # ------------------------------------------------------------------

    def add_raster_layer(self, tiff_path, name, palette='Viridis256', alpha=0.6):
        """Read a GeoTIFF and add it as an image overlay on the map.

        Args:
            tiff_path: Path to the .tif file.
            name: Display name for the layer toggle.
            palette: Bokeh palette name.
            alpha: Image transparency (0-1).
        """
        data = read_tiff_for_bokeh(tiff_path)
        if data is None:
            self.state.append_log(f"Could not load raster: {tiff_path}\n")
            return

        color_mapper = LinearColorMapper(
            palette=palette, low=data['vmin'], high=data['vmax'],
            nan_color='rgba(0,0,0,0)',
        )

        source = ColumnDataSource(data=dict(
            image=[data['image']],
            x=[data['x']],
            y=[data['y']],
            dw=[data['dw']],
            dh=[data['dh']],
        ))

        renderer = self._fig.image(
            image='image', x='x', y='y', dw='dw', dh='dh',
            source=source,
            color_mapper=color_mapper,
            alpha=alpha,
            level='image',
        )

        self._raster_renderers[name] = renderer
        self._raster_sources[name] = source

        # Add toggle checkbox
        toggle = pn.widgets.Checkbox(name=name, value=True, width=200)
        toggle.param.watch(
            lambda e, r=renderer: setattr(r, 'visible', e.new),
            'value',
        )
        self._raster_toggles_column.append(toggle)

        # Show the raster card now that layers exist
        if hasattr(self, '_raster_card') and self._raster_card is not None:
            self._raster_card.visible = True

        # Track in state
        if name not in self.state.raster_layers_loaded:
            self.state.raster_layers_loaded = self.state.raster_layers_loaded + [name]

        self.state.append_log(f"Raster layer added: {name}\n")

    def load_attribute_rasters(self):
        """Scan project attributes directory and load available rasters."""
        project_dir = self.state.project_dir
        if not project_dir:
            return

        pdir = Path(project_dir)
        attr_dirs = ['elevation', 'soilclass', 'landclass']
        palette_map = {
            'elevation': 'Viridis256',
            'soilclass': 'Spectral11',
            'landclass': 'Category20',
        }

        loaded = []
        for attr_name in attr_dirs:
            attr_dir = pdir / 'attributes' / attr_name
            if not attr_dir.is_dir():
                continue
            tif_files = sorted(attr_dir.glob('*.tif')) + sorted(attr_dir.glob('*.tiff'))
            for tif in tif_files:
                layer_name = f"{attr_name}/{tif.stem}"
                if layer_name in self._raster_renderers:
                    continue  # already loaded
                palette = palette_map.get(attr_name, 'Viridis256')

                def _add(p=tif, n=layer_name, pal=palette):
                    run_on_ui_thread(lambda: self.add_raster_layer(p, n, pal))

                _add()
                loaded.append(layer_name)

        if loaded:
            self.state.append_log(f"Loaded {len(loaded)} attribute raster(s)\n")

    # ------------------------------------------------------------------
    # Shapefile loading
    # ------------------------------------------------------------------

    def load_layers(self):
        """Load shapefiles from project directory and render on map."""
        project_dir = self.state.project_dir
        if not project_dir:
            self.state.append_log("No project directory — load a config first.\n")
            return
        if self._layers_loading:
            self.state.append_log("Map layers are already loading.\n")
            return

        pdir = Path(project_dir)
        disc_hint = self._get_discretization_hint()
        self._set_layers_loading(True)
        self.state.append_log("Loading map layers…\n")

        def _worker():
            loaded = []
            basin_data: dict[str, list[list[float]]] = dict(xs=[], ys=[])
            hru_data: dict[str, list[list[float]]] = dict(xs=[], ys=[])
            river_data: dict[str, list[list[float]]] = dict(xs=[], ys=[])
            try:
                basin_shp = self._find_shapefile(pdir / 'shapefiles' / 'river_basins', disc_hint)
                if basin_shp:
                    basin_data = self._read_polygon_shapefile(basin_shp)
                    if basin_data['xs']:
                        loaded.append('basins')

                hru_shp = self._find_shapefile(pdir / 'shapefiles' / 'catchment', disc_hint)
                if hru_shp:
                    hru_data = self._read_polygon_shapefile(hru_shp)
                    if hru_data['xs']:
                        loaded.append('HRUs')

                river_shp = self._find_shapefile(pdir / 'shapefiles' / 'river_network', disc_hint)
                if river_shp:
                    river_data = self._read_line_shapefile(river_shp)
                    if river_data['xs']:
                        loaded.append('rivers')

                run_on_ui_thread(
                    lambda b=basin_data, h=hru_data, r=river_data, names=loaded: self._apply_layer_data(
                        b, h, r, names
                    )
                )
            except Exception as exc:
                run_on_ui_thread(
                    lambda e=exc: self.state.append_log(f"Layer load error: {e}\n")
                )
            finally:
                run_on_ui_thread(lambda: self._set_layers_loading(False))

        threading.Thread(target=_worker, daemon=True).start()

    def _get_discretization_hint(self):
        """Return the current discretization method from config, if available."""
        try:
            cfg = self.state.typed_config
            if cfg and cfg.domain:
                method = getattr(cfg.domain, 'definition_method', None)
                if method:
                    return method.lower()
        except Exception:
            pass
        return None

    def _find_shapefile(self, directory, disc_hint=None):
        """Find a shapefile in *directory*, preferring one matching disc_hint.

        Falls back to the legacy exact filename (dirname.shp), then to any
        *.shp file found.
        """
        if not directory.is_dir():
            return None

        shp_files = sorted(directory.glob('*.shp'))

        # Fall back to recursive glob when flat glob finds nothing
        if not shp_files:
            shp_files = sorted(directory.glob('**/*.shp'))
        if not shp_files:
            return None

        # Legacy exact name (e.g. river_basins/river_basins.shp)
        legacy = directory / f'{directory.name}.shp'
        if legacy.exists():
            return legacy

        # Prefer file whose name contains the discretization hint
        if disc_hint:
            for f in shp_files:
                if disc_hint in f.stem.lower():
                    return f

        # Fall back to first available
        return shp_files[0]

    def _read_polygon_shapefile(self, path):
        """Read a polygon shapefile into Bokeh patch coordinates."""
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
            return dict(xs=xs, ys=ys)
        except Exception as exc:
            logger.warning(f"Failed to load shapefile {path}: {exc}")
            return dict(xs=[], ys=[])

    def _read_line_shapefile(self, path):
        """Read a line shapefile into Bokeh multi-line coordinates."""
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
            return dict(xs=xs, ys=ys)
        except Exception as exc:
            logger.warning(f"Failed to load line shapefile {path}: {exc}")
            return dict(xs=[], ys=[])

    def _apply_layer_data(self, basin_data, hru_data, river_data, loaded_names):
        """Apply loaded layer coordinates to map sources."""
        self._basin_source.data = basin_data
        self._hru_source.data = hru_data
        self._river_source.data = river_data
        if loaded_names:
            self.state.append_log(f"Loaded layers: {', '.join(loaded_names)}\n")
            self._zoom_to_data()
        else:
            self.state.append_log("No shapefiles found in project directory.\n")

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

    # ------------------------------------------------------------------
    # Gauge station loading
    # ------------------------------------------------------------------

    def load_gauges(self, networks=None):
        """Load gauge stations from configured providers and display on map."""
        if self._gauges_loading:
            self.state.append_log("Gauge stations are already loading.\n")
            return
        try:
            from symfluence.gui.data import GaugeStationStore
        except ImportError:
            self.state.append_log("Gauge data module not available.\n")
            return

        self._set_gauges_loading(True)
        self.state.append_log("Loading gauge stations…\n")

        if self._gauge_store is None:
            self._gauge_store = GaugeStationStore()

        def _fetch():
            try:
                df = self._gauge_store.load_all(networks=networks)
                if df.empty:
                    run_on_ui_thread(
                        lambda: self.state.append_log("No gauge stations found.\n")
                    )
                    return
                run_on_ui_thread(lambda d=df: self._push_gauges_to_map(d))
            except Exception as exc:
                run_on_ui_thread(
                    lambda e=exc: self.state.append_log(f"Gauge load error: {e}\n")
                )
            finally:
                run_on_ui_thread(lambda: self._set_gauges_loading(False))

        threading.Thread(target=_fetch, daemon=True).start()

    def _push_gauges_to_map(self, df):
        """Push a DataFrame of gauge stations to the Bokeh source."""
        xs, ys = [], []
        colors = []
        for _, row in df.iterrows():
            x, y = _lonlat_to_mercator(row['lon'], row['lat'])
            xs.append(x)
            ys.append(y)
            colors.append(NETWORK_COLORS.get(row['network'], _DEFAULT_GAUGE_COLOR))

        self._gauge_source.data = dict(
            x=xs, y=ys,
            station_id=df['station_id'].tolist(),
            name=df['name'].tolist(),
            network=df['network'].tolist(),
            river=df['river_name'].tolist(),
            color=colors,
            lat=df['lat'].tolist(),
            lon=df['lon'].tolist(),
        )
        self._gauge_loaded = True
        self.state.append_log(f"Loaded {len(df)} gauge stations.\n")

    def _update_gauge_visibility(self, networks):
        """Filter visible gauges by selected networks."""
        if not self._gauge_loaded or not self._gauge_store or self._gauges_loading:
            return
        try:
            if networks:
                df = self._gauge_store.load_all(networks=networks)
            else:
                df = self._gauge_store.load_all()
            self._push_gauges_to_map(df)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Bbox control button handlers
    # ------------------------------------------------------------------

    def _on_use_current_view(self, event):
        """Capture the current map viewport extent as the bounding box."""
        x_start = self._fig.x_range.start
        x_end = self._fig.x_range.end
        y_start = self._fig.y_range.start
        y_end = self._fig.y_range.end

        lon_min, lat_min = _mercator_to_lonlat(x_start, y_start)
        lon_max, lat_max = _mercator_to_lonlat(x_end, y_end)

        # Format: north/west/south/east
        bbox_str = f"{lat_max:.6f}/{lon_min:.6f}/{lat_min:.6f}/{lon_max:.6f}"
        self.state.bounding_box_coords = bbox_str
        self.state.config_dirty = True
        self.state.append_log(f"Bounding box set from viewport: {bbox_str}\n")

    def _on_clear_bbox(self, event):
        """Clear the bounding box from map and state."""
        self._bbox_source.data = dict(left=[], right=[], top=[], bottom=[])
        self.state.bounding_box_coords = ''
        self.state.config_dirty = True
        self.state.append_log("Bounding box cleared.\n")

    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------

    def _update_loading_ui(self):
        """Update status message."""
        if self._status_pane is not None:
            if self._layers_loading:
                self._status_pane.object = "Loading map layers..."
            elif self._gauges_loading:
                self._status_pane.object = "Loading gauges..."
            else:
                self._status_pane.object = ""

    def _set_layers_loading(self, loading):
        self._layers_loading = loading
        self._update_loading_ui()

    def _set_gauges_loading(self, loading):
        self._gauges_loading = loading
        self._update_loading_ui()

    # ------------------------------------------------------------------
    # Shapefile overlay (public API for DomainBrowser)
    # ------------------------------------------------------------------

    def add_shapefile_overlay(self, path, name, color='#e67e22'):
        """Read a shapefile and add it as an overlay on the map.

        Auto-detects polygon vs line geometry and adds the appropriate
        Bokeh renderer with a legend label.

        Args:
            path: Path to the .shp file.
            name: Display name for the legend.
            color: Line/fill colour.
        """
        try:
            import geopandas as gpd
            gdf = gpd.read_file(path).to_crs(epsg=3857)
        except Exception as exc:
            self.state.append_log(f"Could not read shapefile {path}: {exc}\n")
            return

        geom_type = self._detect_geometry_type(gdf)
        if geom_type == 'polygon':
            data = self._read_polygon_shapefile(path)
            if data['xs']:
                self._fig.patches(
                    'xs', 'ys', source=ColumnDataSource(data=data),
                    fill_alpha=0.12, fill_color=color,
                    line_color=color, line_width=1.2,
                    legend_label=name,
                )
                self.state.append_log(f"Shapefile overlay added: {name}\n")
        elif geom_type == 'line':
            data = self._read_line_shapefile(path)
            if data['xs']:
                self._fig.multi_line(
                    'xs', 'ys', source=ColumnDataSource(data=data),
                    line_color=color, line_width=1.5,
                    legend_label=name,
                )
                self.state.append_log(f"Shapefile overlay added: {name}\n")
        else:
            self.state.append_log(f"Unsupported geometry type in {path}\n")

    @staticmethod
    def _detect_geometry_type(gdf):
        """Detect whether a GeoDataFrame contains polygons or lines."""
        for geom in gdf.geometry:
            if geom is None:
                continue
            gt = geom.geom_type
            if gt in ('Polygon', 'MultiPolygon'):
                return 'polygon'
            if gt in ('LineString', 'MultiLineString'):
                return 'line'
            if gt in ('Point', 'MultiPoint'):
                return 'point'
        return 'unknown'

    def panel(self):
        """Return the Panel component for embedding in the app layout."""
        self._hru_renderer.visible = True
        self._status_pane = pn.pane.Str("", sizing_mode='stretch_width')
        self._update_loading_ui()

        self._raster_card = pn.Card(
            self._raster_toggles_column,
            title='Raster Layers',
            collapsed=True,
            visible=False,
            sizing_mode='stretch_width',
        )

        bbox_controls = pn.Row(
            self._use_view_btn,
            self._clear_bbox_btn,
            sizing_mode='stretch_width',
        )

        return pn.Column(
            self._status_pane,
            bbox_controls,
            pn.pane.Bokeh(self._fig, sizing_mode='stretch_both', min_height=500),
            self._raster_card,
            sizing_mode='stretch_both',
        )
