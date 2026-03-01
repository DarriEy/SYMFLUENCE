# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Centralized console output for SYMFLUENCE CLI.

This module provides a unified interface for all CLI output, using the rich
library for colors, progress bars, and structured output. This replaces
direct print() calls throughout the codebase.
"""

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, List, Optional, TextIO

from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table


@dataclass
class ConsoleConfig:
    """Configuration for console output behavior."""

    use_colors: bool = True
    show_progress: bool = True
    quiet: bool = False
    output_stream: TextIO = field(default_factory=lambda: sys.stdout)
    error_stream: TextIO = field(default_factory=lambda: sys.stderr)


class Console:
    """
    Centralized console output for SYMFLUENCE CLI.

    Provides consistent output formatting with colors, progress bars,
    and structured data display. All CLI output should go through this class.

    Example:
        >>> console = Console()
        >>> console.info("Starting process...")
        >>> console.success("Process completed")
        >>> console.error("Something went wrong")
    """

    def __init__(self, config: Optional[ConsoleConfig] = None):
        """
        Initialize console with optional configuration.

        Args:
            config: Console configuration. Uses defaults if not provided.
        """
        self._config = config or ConsoleConfig()
        self._console = RichConsole(
            file=self._config.output_stream,
            force_terminal=self._config.use_colors if self._config.use_colors else None,
            no_color=not self._config.use_colors,
        )
        self._error_console = RichConsole(
            file=self._config.error_stream,
            force_terminal=self._config.use_colors if self._config.use_colors else None,
            no_color=not self._config.use_colors,
            stderr=True,
        )

    @property
    def is_quiet(self) -> bool:
        """Return True if console is in quiet mode."""
        return self._config.quiet

    def info(self, message: str) -> None:
        """
        Print informational message.

        Args:
            message: Message to print
        """
        if not self._config.quiet:
            self._console.print(message)

    def success(self, message: str) -> None:
        """
        Print success message with green [OK] prefix.

        Args:
            message: Success message to print
        """
        if not self._config.quiet:
            self._console.print(f"[green][OK][/green] {message}")

    def warning(self, message: str) -> None:
        """
        Print warning message with yellow [WARN] prefix.

        Args:
            message: Warning message to print
        """
        if not self._config.quiet:
            self._console.print(f"[yellow][WARN][/yellow] {message}")

    def error(self, message: str) -> None:
        """
        Print error message with red [ERROR] prefix to stderr.

        Args:
            message: Error message to print
        """
        self._error_console.print(f"[red][ERROR][/red] {message}")

    def debug(self, message: str) -> None:
        """
        Print debug message with dim styling.

        Args:
            message: Debug message to print
        """
        if not self._config.quiet:
            self._console.print(f"[dim][DEBUG] {message}[/dim]")

    def print(self, message: str, style: Optional[str] = None) -> None:
        """
        Print message with optional rich styling.

        Args:
            message: Message to print
            style: Optional rich style string
        """
        if not self._config.quiet:
            self._console.print(message, style=style)

    def rule(self, title: str = "") -> None:
        """
        Print a horizontal rule with optional title.

        Args:
            title: Optional title to display in the rule
        """
        if not self._config.quiet:
            self._console.rule(title)

    def panel(self, message: str, title: str = "", style: str = "blue") -> None:
        """
        Print message in a bordered panel.

        Args:
            message: Message to display in the panel
            title: Optional panel title
            style: Panel border style (color)
        """
        if not self._config.quiet:
            self._console.print(Panel(message, title=title, border_style=style))

    def table(
        self,
        columns: List[str],
        rows: List[List[Any]],
        title: Optional[str] = None,
    ) -> None:
        """
        Print data in a formatted table.

        Args:
            columns: Column headers
            rows: Table rows (list of lists)
            title: Optional table title
        """
        if self._config.quiet:
            return

        table = Table(title=title)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        self._console.print(table)

    @contextmanager
    def status(self, message: str) -> Generator[None, None, None]:
        """
        Display a status spinner while executing a block.

        Args:
            message: Status message to display

        Example:
            >>> with console.status("Installing..."):
            ...     install_package()
        """
        if self._config.quiet or not self._config.show_progress:
            yield
            return

        with self._console.status(message):
            yield

    def progress(
        self,
        description: str = "Processing",
        total: Optional[int] = None,
    ) -> Progress:
        """
        Create a progress bar context manager.

        Args:
            description: Description of the task
            total: Total number of steps (None for indeterminate)

        Returns:
            Progress context manager

        Example:
            >>> with console.progress("Downloading", total=100) as progress:
            ...     task = progress.add_task("files", total=100)
            ...     for i in range(100):
            ...         progress.update(task, advance=1)
        """
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ]

        if total is not None:
            columns.extend([
                BarColumn(),
                TaskProgressColumn(),
            ])

        columns.append(TimeElapsedColumn())

        return Progress(
            *columns,
            console=self._console,
            disable=self._config.quiet or not self._config.show_progress,
        )

    def newline(self) -> None:
        """Print an empty line."""
        if not self._config.quiet:
            self._console.print()

    def indent(self, message: str, level: int = 1) -> None:
        """
        Print an indented message.

        Args:
            message: Message to print
            level: Indentation level (number of 3-space indents)
        """
        if not self._config.quiet:
            indent_str = "   " * level
            self._console.print(f"{indent_str}{message}")


# Global console instance for convenience
console = Console()


def get_console() -> Console:
    """Get the global console instance."""
    return console


def set_console(new_console: Console) -> None:
    """
    Set the global console instance.

    Useful for testing or configuring console behavior.

    Args:
        new_console: Console instance to use globally
    """
    global console
    console = new_console
