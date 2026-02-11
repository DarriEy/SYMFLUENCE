"""
SYMFLUENCE TUI Package.

Textual-based interactive terminal application for hydrological modeling workflows.
Provides domain browsing, run history, workflow execution, calibration monitoring,
and results comparison — all from an SSH terminal.

Install dependencies: pip install "symfluence[tui]"
Launch: symfluence tui launch
"""


def launch_tui(config_path=None, demo=None):
    """
    Build and launch the SYMFLUENCE TUI as an interactive terminal application.

    Thin wrapper that imports the actual app module lazily so that
    ``from symfluence.tui import launch_tui`` succeeds even when Textual
    is not installed (the ImportError is raised only when called).

    Args:
        config_path: Optional path to a YAML config file to preload.
        demo: Optional demo name (e.g. 'bow') to load a built-in config.
    """
    try:
        import textual  # noqa: F401
    except ImportError:
        raise ImportError(
            "Textual is required for the SYMFLUENCE TUI.\n"
            'Install with:  pip install "symfluence[tui]"'
        )

    from .app import SymfluenceTUI
    app = SymfluenceTUI(config_path=config_path, demo=demo)
    app.run()


__all__ = ['launch_tui']
