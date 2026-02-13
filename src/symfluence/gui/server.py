"""
Panel server launch logic.

Provides serve_app() which initializes Panel extensions, builds the
SymfluenceApp, and starts the Bokeh/Tornado server.
"""

import panel as pn

from .app import GUI_THEME_CSS, SymfluenceApp


def _serve_app(config_path=None, port=5006, show=True, demo=None):
    """
    Build and serve the SYMFLUENCE GUI as a Panel web application.

    Args:
        config_path: Optional path to a YAML config file to preload
        port: Port number for the web server (default 5006)
        show: Whether to auto-open a browser tab (default True)
        demo: Optional demo name (e.g. 'bow') to load on startup
    """
    pn.extension(
        'terminal', 'codeeditor', 'tabulator',
        sizing_mode='stretch_width',
        raw_css=[GUI_THEME_CSS],
    )

    app = SymfluenceApp(config_path=config_path, demo=demo)
    template = app.build()

    pn.serve(
        {'/': template},
        port=port,
        show=show,
        title='SYMFLUENCE',
        threaded=True,
    )
