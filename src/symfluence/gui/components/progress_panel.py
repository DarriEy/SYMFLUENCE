"""
Step progress indicator for the SYMFLUENCE GUI.

Displays a compact horizontal progress bar with step dots that
update in real-time as workflow steps execute. Polled every 300ms
alongside the log viewer for responsive updates.
"""

import panel as pn
import param

from ..utils.threading_utils import STEP_LABELS

# Step status -> display properties
_STATUS_STYLES = {
    'pending':  ('○', '#b0bec5'),
    'running':  ('◉', '#ff9800'),
    'done':     ('●', '#4caf50'),
    'error':    ('✖', '#f44336'),
}


class ProgressPanel(param.Parameterized):
    """Compact step progress indicator with step dots and progress bar."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._progress_bar = pn.indicators.Progress(
            value=0, max=100, bar_color='primary',
            sizing_mode='stretch_width', height=6, margin=(4, 10),
        )
        self._steps_html = pn.pane.HTML(
            self._render_idle(),
            sizing_mode='stretch_width',
            margin=(0, 10),
        )
        self._status_text = pn.pane.HTML(
            '', sizing_mode='stretch_width', margin=(0, 10),
        )
        self._container = pn.Column(
            self._steps_html,
            self._progress_bar,
            self._status_text,
            sizing_mode='stretch_width',
            margin=(4, 0),
            visible=False,
        )
        self._periodic_cb = None

    def start_polling(self):
        """Begin periodic progress updates."""
        if self._periodic_cb is None:
            self._periodic_cb = pn.state.add_periodic_callback(
                self._refresh, period=300,
            )

    def _refresh(self):
        """Sync widgets from state."""
        statuses = self.state.step_statuses
        is_running = self.state.is_running

        if not statuses and not is_running:
            self._container.visible = False
            return

        self._container.visible = True
        self._steps_html.object = self._render_steps(statuses)

        total = self.state.steps_total or 1
        done = self.state.steps_done
        pct = int(100 * done / total)

        if is_running and done < total:
            self._progress_bar.active = True
            self._progress_bar.value = pct
            step_name = self.state.running_step or ''
            label = STEP_LABELS.get(step_name, step_name.replace('_', ' ').title())
            self._status_text.object = (
                f'<span style="color:#546e7a; font-size:12px;">'
                f'Step {done + 1}/{total} &mdash; {label}...</span>'
            )
        else:
            self._progress_bar.active = False
            self._progress_bar.value = pct
            # Check if any step errored
            errored = any(s == 'error' for _, s in statuses)
            if errored:
                self._status_text.object = (
                    '<span style="color:#f44336; font-size:12px; font-weight:600;">'
                    'Error during execution</span>'
                )
            elif done == total and total > 0:
                self._status_text.object = (
                    '<span style="color:#4caf50; font-size:12px; font-weight:600;">'
                    f'All {total} steps completed</span>'
                )
            else:
                self._status_text.object = ''

    def _render_steps(self, statuses):
        """Build HTML for the step dot indicators."""
        if not statuses:
            return self._render_idle()

        parts = []
        for name, status in statuses:
            icon, color = _STATUS_STYLES.get(status, _STATUS_STYLES['pending'])
            label = STEP_LABELS.get(name, name.replace('_', ' ').title())
            weight = 'font-weight:600;' if status == 'running' else ''
            parts.append(
                f'<span style="color:{color}; font-size:13px; margin-right:14px; {weight}">'
                f'{icon} {label}</span>'
            )
        return (
            '<div style="display:flex; flex-wrap:wrap; align-items:center; gap:2px;">'
            + ''.join(parts)
            + '</div>'
        )

    @staticmethod
    def _render_idle():
        return (
            '<span style="color:#90a4ae; font-size:12px;">'
            'Ready &mdash; click Initialize to begin</span>'
        )

    def panel(self):
        """Return the progress panel layout."""
        return self._container
