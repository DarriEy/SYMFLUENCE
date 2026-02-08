"""
Log viewer component.

Displays a terminal-style widget bound to WorkflowState.log_text
with auto-scroll and a clear button.
"""

import param
import panel as pn


class LogViewer(param.Parameterized):
    """Terminal widget that displays live log output."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._terminal = pn.widgets.Terminal(
            '',
            sizing_mode='stretch_both',
            min_height=400,
            options={'cursorBlink': False, 'scrollback': 10000},
        )
        # Watch for log text changes
        self.state.param.watch(self._on_log_change, 'log_text')

    def _on_log_change(self, event):
        """When log_text grows, write the new content to the terminal."""
        new_text = event.new
        old_text = event.old or ''
        if len(new_text) > len(old_text):
            delta = new_text[len(old_text):]
            self._terminal.write(delta)

    def panel(self):
        """Return the Panel layout."""
        clear_btn = pn.widgets.Button(name='Clear', button_type='warning', width=80)

        def _on_clear(event):
            self._terminal.clear()
            self.state.log_text = ""

        clear_btn.on_click(_on_clear)

        return pn.Column(
            pn.Row(pn.pane.Str("## Logs", sizing_mode='stretch_width'), clear_btn),
            self._terminal,
            sizing_mode='stretch_both',
        )
