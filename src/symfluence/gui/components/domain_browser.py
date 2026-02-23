"""
Pull-out domain data store browser for the Map tab.

Provides a sliding drawer on the right edge of the map that lets users
browse the full domain directory tree and visualize files directly on
the map or in inline previews.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable

import panel as pn
import param

from ..utils.threading_utils import run_on_ui_thread

logger = logging.getLogger(__name__)

# Extensions hidden from the file tree (shapefile sidecars, etc.)
_HIDDEN_EXTENSIONS = {'.dbf', '.shx', '.prj', '.cpg', '.xml', '.aux'}

# Max files rendered per directory before "Show all" button
_MAX_FILES_PER_DIR = 50

# Max directory recursion depth
_MAX_DEPTH = 5

# Drawer width in pixels
_DRAWER_WIDTH = 320

# File type icon map
_ICONS = {
    '.tif': '\U0001F5FA',   # map
    '.tiff': '\U0001F5FA',
    '.shp': '\U0001F4CD',   # pin
    '.nc': '\U0001F4CA',    # chart
    '.nc4': '\U0001F4CA',
    '.csv': '\U0001F4C4',   # page
    '.png': '\U0001F5BC',   # frame
    '.jpg': '\U0001F5BC',
    '.jpeg': '\U0001F5BC',
    '.json': '\U0001F4CB',  # clipboard
    '.yaml': '\U0001F4CB',
    '.yml': '\U0001F4CB',
    '.txt': '\U0001F4C3',   # page curl
}
_DEFAULT_ICON = '\U0001F4C4'


def _format_size(nbytes):
    """Format byte count as human-readable string."""
    for unit in ('B', 'KB', 'MB', 'GB'):
        if nbytes < 1024:
            return f'{nbytes:.0f} {unit}' if unit == 'B' else f'{nbytes:.1f} {unit}'
        nbytes /= 1024
    return f'{nbytes:.1f} TB'


class DomainBrowser(param.Parameterized):
    """Pull-out file browser for the domain data store."""

    state = param.Parameter(doc="WorkflowState instance")
    map_view = param.Parameter(doc="MapView instance")

    def __init__(self, state, map_view, **kw):
        super().__init__(state=state, map_view=map_view, **kw)

        # Drawer open/closed state
        self._open = False

        # Toggle button (always visible on right edge)
        self._toggle_btn = pn.widgets.Button(
            name='\u276E',  # left chevron (open)
            width=28,
            height=80,
            button_type='light',
            styles={
                'border-radius': '6px 0 0 6px',
                'border': '1px solid #d7e2eb',
                'border-right': 'none',
                'background': '#eef4fb',
                'font-size': '14px',
                'padding': '0',
                'cursor': 'pointer',
                'align-self': 'center',
            },
            margin=(0, 0, 0, 0),
        )
        self._toggle_btn.on_click(self._on_toggle)

        # File tree container
        self._tree_column = pn.Column(
            sizing_mode='stretch_both',
            scroll=True,
        )

        # Preview area at bottom of drawer
        self._preview_pane = pn.Column(
            sizing_mode='stretch_width',
            height=250,
            scroll=True,
            styles={
                'border-top': '1px solid #d7e2eb',
                'background': '#f8fbff',
            },
        )
        self._preview_pane.visible = False

        # Header
        self._header = pn.pane.HTML(
            '<div style="font-weight:600; font-size:13px; color:#1b2f45; '
            'padding:8px 10px; border-bottom:1px solid #d7e2eb;">'
            'Domain Data Store</div>',
            sizing_mode='stretch_width',
        )

        # Drawer body (hidden initially)
        self._drawer_body = pn.Column(
            self._header,
            self._tree_column,
            self._preview_pane,
            width=_DRAWER_WIDTH,
            sizing_mode='stretch_height',
            visible=False,
            styles={
                'border-left': '1px solid #d7e2eb',
                'background': '#ffffff',
            },
        )

        # Watch state changes for auto-refresh
        state.param.watch(self._on_state_change, ['project_dir', 'workflow_status'])

        # Initial scan if project_dir already set
        if state.project_dir:
            self._refresh_tree_async()

    # ------------------------------------------------------------------
    # Toggle
    # ------------------------------------------------------------------

    def _on_toggle(self, event):
        """Toggle the drawer open/closed."""
        self._open = not self._open
        self._drawer_body.visible = self._open
        self._toggle_btn.name = '\u276F' if self._open else '\u276E'
        if self._open and not self._tree_column.objects:
            self._refresh_tree_async()

    # ------------------------------------------------------------------
    # State watchers
    # ------------------------------------------------------------------

    def _on_state_change(self, event):
        """Refresh tree when project_dir or workflow_status changes."""
        self._refresh_tree_async()

    # ------------------------------------------------------------------
    # File tree scanning
    # ------------------------------------------------------------------

    def _refresh_tree_async(self):
        """Scan the domain directory on a background thread."""
        project_dir = self.state.project_dir
        if not project_dir:
            run_on_ui_thread(self._show_no_project)
            return

        def _worker():
            try:
                root = Path(project_dir)
                if not root.is_dir():
                    run_on_ui_thread(
                        lambda: self._show_message(f'Directory not found: {root.name}')
                    )
                    return
                widgets = self._scan_directory(root, depth=0)
                run_on_ui_thread(lambda w=widgets: self._apply_tree(w))
            except Exception as exc:  # noqa: BLE001 — UI resilience
                logger.warning(f"Domain browser scan error: {exc}")
                run_on_ui_thread(
                    lambda e=exc: self._show_message(f'Scan error: {e}')
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _show_no_project(self):
        self._tree_column.clear()
        self._tree_column.append(
            pn.pane.HTML(
                '<span style="color:#6e7f91; font-size:12px; padding:10px;">'
                'No project loaded</span>',
            )
        )

    def _show_message(self, msg):
        self._tree_column.clear()
        self._tree_column.append(
            pn.pane.HTML(
                f'<span style="color:#6e7f91; font-size:12px; padding:10px;">'
                f'{msg}</span>',
            )
        )

    def _apply_tree(self, widgets):
        """Replace tree contents with scanned widgets."""
        self._tree_column.clear()
        for w in widgets:
            self._tree_column.append(w)

    def _scan_directory(self, path, depth):
        """Recursively scan a directory and return Panel widgets.

        Returns a list of Panel objects representing the directory contents.
        """
        if depth > _MAX_DEPTH:
            return []

        widgets = []
        try:
            entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            return []

        # Separate dirs and files
        dirs = [e for e in entries if e.is_dir() and not e.name.startswith('.')]
        files = [
            e for e in entries
            if e.is_file()
            and not e.name.startswith('.')
            and e.suffix.lower() not in _HIDDEN_EXTENSIONS
        ]

        # Subdirectories as collapsible cards
        for d in dirs:
            child_widgets = self._scan_directory(d, depth + 1)
            if not child_widgets:
                # Show empty directory indicator
                child_widgets = [
                    pn.pane.HTML(
                        '<span style="color:#90a4ae; font-size:11px; padding:4px 8px;">'
                        'Empty</span>',
                    )
                ]
            content = pn.Column(*child_widgets, sizing_mode='stretch_width')
            card = pn.Card(
                content,
                title=f'\U0001F4C1 {d.name}',
                collapsed=True,
                sizing_mode='stretch_width',
                header_background='#f8fbff',
                styles={
                    'margin': '1px 0',
                    'border': 'none',
                    'box-shadow': 'none',
                },
            )
            widgets.append(card)

        # Files
        truncated = False
        displayed_files = files[:_MAX_FILES_PER_DIR]
        if len(files) > _MAX_FILES_PER_DIR:
            truncated = True

        for f in displayed_files:
            row = self._build_file_row(f)
            widgets.append(row)

        if truncated:
            remaining = len(files) - _MAX_FILES_PER_DIR
            show_all_btn = pn.widgets.Button(
                name=f'Show all ({remaining} more)',
                button_type='light',
                width=200,
                margin=(2, 8),
                styles={'font-size': '11px'},
            )

            def _expand(event, all_files=files):
                # Replace the button with all remaining files
                idx = self._tree_column.objects.index(show_all_btn)
                self._tree_column.remove(show_all_btn)
                for extra_f in all_files[_MAX_FILES_PER_DIR:]:
                    row = self._build_file_row(extra_f)
                    self._tree_column.insert(idx, row)
                    idx += 1

            show_all_btn.on_click(_expand)
            widgets.append(show_all_btn)

        return widgets

    def _build_file_row(self, filepath):
        """Build a Row widget for a single file with icon, name, size, action buttons."""
        suffix = filepath.suffix.lower()
        icon = _ICONS.get(suffix, _DEFAULT_ICON)

        try:
            size_str = _format_size(filepath.stat().st_size)
        except OSError:
            size_str = '?'

        # Truncate long filenames
        display_name = filepath.name
        if len(display_name) > 28:
            display_name = display_name[:25] + '...'

        name_html = pn.pane.HTML(
            f'<span style="font-size:11px; color:#1b2f45; overflow:hidden; '
            f'text-overflow:ellipsis; white-space:nowrap;" title="{filepath.name}">'
            f'{icon} {display_name}</span>',
            width=160,
            margin=(2, 0),
        )
        size_html = pn.pane.HTML(
            f'<span style="font-size:10px; color:#90a4ae;">{size_str}</span>',
            width=50,
            margin=(2, 0),
        )

        buttons = self._action_buttons(filepath, suffix)

        row = pn.Row(
            name_html,
            size_html,
            *buttons,
            sizing_mode='stretch_width',
            margin=(1, 4),
            styles={
                'border-bottom': '1px solid #f0f4f8',
                'align-items': 'center',
            },
        )
        return row

    def _action_buttons(self, filepath, suffix):
        """Return action buttons appropriate for the file type."""
        buttons = []

        def _bind_path_action(button: Any, action: Callable[[Path], None]) -> None:
            def _handler(_event: Any, p: Path = filepath) -> None:
                action(p)

            button.on_click(_handler)

        if suffix in ('.tif', '.tiff'):
            btn = pn.widgets.Button(
                name='Map', button_type='primary', width=50, height=24,
                styles={'font-size': '10px', 'padding': '0 4px'},
                margin=(0, 2),
            )
            _bind_path_action(btn, self._action_map_raster)
            buttons.append(btn)

        elif suffix == '.shp':
            btn = pn.widgets.Button(
                name='Map', button_type='primary', width=50, height=24,
                styles={'font-size': '10px', 'padding': '0 4px'},
                margin=(0, 2),
            )
            _bind_path_action(btn, self._action_map_shapefile)
            buttons.append(btn)

        elif suffix in ('.nc', '.nc4'):
            view_btn = pn.widgets.Button(
                name='View', button_type='default', width=50, height=24,
                styles={'font-size': '10px', 'padding': '0 4px'},
                margin=(0, 2),
            )
            _bind_path_action(view_btn, self._action_view_netcdf)
            buttons.append(view_btn)

        elif suffix == '.csv':
            view_btn = pn.widgets.Button(
                name='View', button_type='default', width=50, height=24,
                styles={'font-size': '10px', 'padding': '0 4px'},
                margin=(0, 2),
            )
            _bind_path_action(view_btn, self._action_view_csv)
            buttons.append(view_btn)

        elif suffix in ('.png', '.jpg', '.jpeg'):
            view_btn = pn.widgets.Button(
                name='View', button_type='default', width=50, height=24,
                styles={'font-size': '10px', 'padding': '0 4px'},
                margin=(0, 2),
            )
            _bind_path_action(view_btn, self._action_view_image)
            buttons.append(view_btn)

        elif suffix == '.json':
            view_btn = pn.widgets.Button(
                name='View', button_type='default', width=50, height=24,
                styles={'font-size': '10px', 'padding': '0 4px'},
                margin=(0, 2),
            )
            _bind_path_action(view_btn, self._action_view_json)
            buttons.append(view_btn)

        return buttons

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _action_map_raster(self, path):
        """Add a raster layer to the map."""
        name = path.stem
        self.state.append_log(f"Loading raster: {path.name}...\n")

        def _worker():
            try:
                run_on_ui_thread(
                    lambda: self.map_view.add_raster_layer(str(path), name)
                )
            except Exception as exc:  # noqa: BLE001 — UI resilience
                run_on_ui_thread(
                    lambda e=exc: self.state.append_log(f"Raster load error: {e}\n")
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _action_map_shapefile(self, path):
        """Add a shapefile overlay to the map."""
        name = path.stem
        self.state.append_log(f"Loading shapefile: {path.name}...\n")

        def _worker():
            try:
                run_on_ui_thread(
                    lambda: self.map_view.add_shapefile_overlay(str(path), name)
                )
            except Exception as exc:  # noqa: BLE001 — UI resilience
                run_on_ui_thread(
                    lambda e=exc: self.state.append_log(f"Shapefile load error: {e}\n")
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _action_view_netcdf(self, path):
        """Show xarray metadata preview in the preview area."""
        self._show_preview_loading()

        def _worker():
            try:
                import xarray as xr
                ds = xr.open_dataset(path)
                lines = [f'<b>{path.name}</b><br>']
                lines.append(f'<b>Dimensions:</b> {dict(ds.dims)}<br>')
                lines.append('<b>Variables:</b><br>')
                for vname, var in ds.data_vars.items():
                    dtype_str = str(var.dtype)
                    dims_str = ', '.join(var.dims)
                    lines.append(
                        f'&nbsp;&nbsp;{vname} ({dims_str}) — {dtype_str}<br>'
                    )
                if ds.coords:
                    lines.append('<b>Coordinates:</b><br>')
                    for cname, coord in ds.coords.items():
                        lines.append(
                            f'&nbsp;&nbsp;{cname}: {coord.dtype}, size={coord.size}<br>'
                        )
                if ds.attrs:
                    lines.append('<b>Attributes:</b><br>')
                    for k, v in list(ds.attrs.items())[:10]:
                        lines.append(f'&nbsp;&nbsp;{k}: {v}<br>')
                ds.close()
                html = (
                    '<div style="font-size:11px; color:#1b2f45; padding:8px; '
                    'font-family:monospace;">' + ''.join(lines) + '</div>'
                )
                run_on_ui_thread(lambda h=html: self._set_preview(h))
            except Exception as exc:  # noqa: BLE001 — UI resilience
                run_on_ui_thread(
                    lambda e=exc: self._set_preview(
                        f'<span style="color:red; font-size:11px;">Error: {e}</span>'
                    )
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _action_view_csv(self, path):
        """Show first 20 rows of a CSV in the preview area."""
        self._show_preview_loading()

        def _worker():
            try:
                import pandas as pd
                df = pd.read_csv(path, nrows=20)
                table_html = df.to_html(
                    index=False,
                    max_cols=8,
                    classes='preview-table',
                )
                html = (
                    f'<div style="font-size:10px; overflow-x:auto; padding:4px;">'
                    f'<b>{path.name}</b> ({_format_size(path.stat().st_size)})<br>'
                    f'<style>.preview-table {{ font-size:10px; border-collapse:collapse; }}'
                    f'.preview-table td, .preview-table th {{ padding:2px 4px; '
                    f'border:1px solid #d7e2eb; }}</style>'
                    f'{table_html}</div>'
                )
                run_on_ui_thread(lambda h=html: self._set_preview(h))
            except Exception as exc:  # noqa: BLE001 — UI resilience
                run_on_ui_thread(
                    lambda e=exc: self._set_preview(
                        f'<span style="color:red; font-size:11px;">Error: {e}</span>'
                    )
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _action_view_image(self, path):
        """Show image thumbnail in the preview area."""
        self._preview_pane.visible = True
        self._preview_pane.clear()
        try:
            img = pn.pane.Image(
                str(path),
                width=_DRAWER_WIDTH - 20,
                sizing_mode='scale_width',
            )
            self._preview_pane.append(
                pn.pane.HTML(
                    f'<b style="font-size:11px; padding:4px 8px;">{path.name}</b>',
                )
            )
            self._preview_pane.append(img)
        except Exception as exc:  # noqa: BLE001 — UI resilience
            self._set_preview(
                f'<span style="color:red; font-size:11px;">Error: {exc}</span>'
            )

    def _action_view_json(self, path):
        """Show pretty-printed JSON in the preview area."""
        self._show_preview_loading()

        def _worker():
            try:
                text = path.read_text(encoding='utf-8')
                data = json.loads(text)
                pretty = json.dumps(data, indent=2)
                # Truncate if very large
                if len(pretty) > 5000:
                    pretty = pretty[:5000] + '\n... (truncated)'
                import html as html_mod
                escaped = html_mod.escape(pretty)
                html = (
                    f'<div style="font-size:10px; padding:8px; font-family:monospace; '
                    f'white-space:pre-wrap; color:#1b2f45;">'
                    f'<b>{path.name}</b><br>{escaped}</div>'
                )
                run_on_ui_thread(lambda h=html: self._set_preview(h))
            except Exception as exc:  # noqa: BLE001 — UI resilience
                run_on_ui_thread(
                    lambda e=exc: self._set_preview(
                        f'<span style="color:red; font-size:11px;">Error: {e}</span>'
                    )
                )

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Preview helpers
    # ------------------------------------------------------------------

    def _show_preview_loading(self):
        """Show loading indicator in preview area."""
        self._preview_pane.visible = True
        self._preview_pane.clear()
        self._preview_pane.append(
            pn.pane.HTML(
                '<span style="color:#6e7f91; font-size:11px; padding:10px;">'
                'Loading...</span>',
            )
        )

    def _set_preview(self, html_content):
        """Set the preview area content."""
        self._preview_pane.visible = True
        self._preview_pane.clear()
        self._preview_pane.append(
            pn.pane.HTML(html_content, sizing_mode='stretch_width')
        )

    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------

    def panel(self):
        """Return the Panel component: handle button + drawer body."""
        return pn.Row(
            self._toggle_btn,
            self._drawer_body,
            sizing_mode='stretch_height',
            styles={'flex-shrink': '0'},
            margin=0,
        )
