"""
Auto-scrolling log panel with thread-safe write support.
"""

import logging
from typing import Callable, Optional

from textual.widgets import RichLog


class LogPanel(RichLog):
    """RichLog widget that auto-scrolls and supports thread-safe writes."""

    def __init__(self, **kwargs):
        super().__init__(highlight=True, markup=True, wrap=True, **kwargs)

    def write_line(self, text: str) -> None:
        """Write a line of text and auto-scroll. Thread-safe via call_from_thread."""
        self.write(text)


class TUILogHandler(logging.Handler):
    """Logging handler that routes records into a LogPanel widget.

    Attaches to the 'symfluence' logger. Uses app.call_from_thread()
    for thread-safe UI updates (analogous to GUILogHandler using
    pn.state.execute()).
    """

    def __init__(
        self,
        app,
        log_panel: LogPanel,
        level=logging.INFO,
        on_message: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(level)
        self._app = app
        self._log_panel = log_panel
        self._on_message = on_message
        self.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
        )

    def emit(self, record):
        try:
            msg = self.format(record)
            self._app.call_from_thread(self._log_panel.write_line, msg)
            if self._on_message:
                self._app.call_from_thread(self._on_message, record.getMessage())
        except Exception:
            self.handleError(record)
