# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Log viewer component.

Displays a terminal-style widget bound to WorkflowState.log_text
with auto-scroll and a clear button.

Uses periodic polling (every 300ms) to flush accumulated log text
to the Terminal widget. This approach is robust across threading
contexts -- it works regardless of whether log appends originate
from the Tornado event loop, daemon threads, or direct calls.
"""

import panel as pn
import param


class LogViewer(param.Parameterized):
    """Terminal widget that displays live log output."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._terminal = pn.widgets.Terminal(
            '',
            sizing_mode='stretch_both',
            min_height=180,
            options={
                'cursorBlink': False,
                'scrollback': 10000,
                'fontSize': 11,
                'fontFamily': '"SF Mono", "Fira Code", "Cascadia Code", Menlo, monospace',
                'lineHeight': 1.3,
                'theme': {
                    'background': '#1e1e2e',
                    'foreground': '#cdd6f4',
                    'cursor': '#f5e0dc',
                },
            },
        )
        self._flushed_len = 0
        self._periodic_cb = None

    def start_polling(self):
        """Begin periodic log flushing.  Call once the event loop is running."""
        if self._periodic_cb is None:
            self._periodic_cb = pn.state.add_periodic_callback(
                self._flush, period=300,
            )

    def _flush(self):
        """Write any new log_text content to the terminal widget."""
        text = self.state.log_text or ''
        if len(text) <= self._flushed_len:
            return
        delta = text[self._flushed_len:]
        self._flushed_len = len(text)
        try:
            self._terminal.write(delta)
        except Exception:  # noqa: BLE001 — UI resilience
            pass

    def panel(self):
        """Return the Panel layout."""
        clear_btn = pn.widgets.Button(
            name='Clear', button_type='light', width=70,
            styles={'font-size': '11px', 'border-radius': '4px'},
        )

        def _on_clear(event):
            self._terminal.clear()
            self.state.log_text = ""
            self._flushed_len = 0

        clear_btn.on_click(_on_clear)

        header_row = pn.Row(
            pn.pane.HTML(
                '<span style="color:#6e7f91; font-size:11px; font-weight:500;">'
                'Log output</span>',
                margin=(0, 5),
            ),
            pn.Spacer(sizing_mode='stretch_width'),
            clear_btn,
            sizing_mode='stretch_width',
            styles={'padding': '4px 6px', 'border-bottom': '1px solid #d7e2eb'},
        )

        return pn.Column(
            header_row,
            self._terminal,
            sizing_mode='stretch_both',
            styles={'border-radius': '0 0 8px 8px', 'overflow': 'hidden'},
        )
