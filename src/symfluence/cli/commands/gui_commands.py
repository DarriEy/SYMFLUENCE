"""
GUI command handlers for SYMFLUENCE CLI.

This module implements the handler for launching the Panel-based GUI.
"""

from argparse import Namespace

from .base import BaseCommand, cli_exception_handler
from ..exit_codes import ExitCode


class GUICommands(BaseCommand):
    """Handlers for the gui command category."""

    @staticmethod
    @cli_exception_handler
    def launch(args: Namespace) -> int:
        """
        Execute: symfluence gui launch

        Starts the Panel web application server and opens a browser.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.gui import serve_app
        except ImportError as exc:
            BaseCommand._console.error(
                "GUI dependencies not installed. "
                'Install them with:  pip install "symfluence[gui]"'
            )
            BaseCommand._console.indent(f"({exc})")
            return ExitCode.DEPENDENCY_ERROR

        config_path = BaseCommand.get_arg(args, 'config', None)
        port = BaseCommand.get_arg(args, 'port', 5006)
        no_browser = BaseCommand.get_arg(args, 'no_browser', False)

        BaseCommand._console.info(f"Launching SYMFLUENCE GUI on port {port}...")
        if config_path:
            BaseCommand._console.indent(f"Preloading config: {config_path}")

        serve_app(
            config_path=config_path,
            port=port,
            show=not no_browser,
        )

        return ExitCode.SUCCESS
