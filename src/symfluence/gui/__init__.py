"""
SYMFLUENCE GUI Package.

Panel-based web application for interactive hydrological modeling workflows.
Provides map interaction, configuration editing, workflow execution, and results viewing.

Install dependencies: pip install "symfluence[gui]"
Launch: symfluence gui launch
"""


def serve_app(config_path=None, port=5006, show=True, demo=None):
    """
    Build and serve the SYMFLUENCE GUI as a Panel web application.

    Thin wrapper that imports the actual server module lazily so that
    ``from symfluence.gui import serve_app`` succeeds even when Panel
    is not installed (the ImportError is raised only when called).

    Args:
        config_path: Optional path to a YAML config file to preload.
        port: Server port (default 5006).
        show: Open a browser tab automatically.
        demo: Optional demo name (e.g. 'bow') to load a built-in config.
    """
    try:
        import panel  # noqa: F401
    except ImportError:
        raise ImportError(
            "Panel is required for the SYMFLUENCE GUI.\n"
            'Install with:  pip install "symfluence[gui]"'
        ) from None

    from .server import _serve_app
    _serve_app(config_path=config_path, port=port, show=show, demo=demo)


__all__ = ['serve_app']
