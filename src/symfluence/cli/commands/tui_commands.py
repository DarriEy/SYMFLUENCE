"""
TUI command handlers for SYMFLUENCE CLI.

This module implements the handler for launching the Textual-based terminal UI.
"""

from argparse import Namespace

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler


class TUICommands(BaseCommand):
    """Handlers for the tui command category."""

    @staticmethod
    @cli_exception_handler
    def launch(args: Namespace) -> int:
        """
        Execute: symfluence tui launch

        Starts the interactive terminal UI application.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.tui import launch_tui
        except ImportError as exc:
            BaseCommand._console.error(
                "TUI dependencies not installed. "
                'Install them with:  pip install "symfluence[tui]"'
            )
            BaseCommand._console.indent(f"({exc})")
            return ExitCode.DEPENDENCY_ERROR

        config_path = BaseCommand.get_arg(args, 'config', None)
        demo = BaseCommand.get_arg(args, 'demo', None)

        BaseCommand._console.info("Launching SYMFLUENCE TUI...")
        if demo:
            BaseCommand._console.indent(f"Demo mode: {demo}")
        if config_path:
            BaseCommand._console.indent(f"Preloading config: {config_path}")

        launch_tui(
            config_path=config_path,
            demo=demo,
        )

        return ExitCode.SUCCESS
