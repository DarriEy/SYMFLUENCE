"""
Calibration panel for configuring and launching iterative calibration.

Visible after model run completes (gui_phase >= 'model_ready').
Reuses the flatten-merge-reconstruct pattern from IterativeRunPanel.
"""

import logging

import panel as pn
import param

from ..utils.config_bridge import config_to_params, params_to_config_overrides
from ..utils.threading_utils import WorkflowThread

logger = logging.getLogger(__name__)

_WIDGET_KW = dict(sizing_mode='stretch_width', margin=(4, 5))
_BTN_KW = dict(sizing_mode='stretch_width', margin=(8, 5, 4, 5))

_ALGORITHMS = ['PSO', 'DDS', 'DE', 'SCE-UA', 'GLUE', 'Basin-hopping']
_METRICS = ['KGE', 'KGEp', 'KGEnp', 'NSE', 'RMSE', 'MAE', 'PBIAS']


class CalibrationPanel(param.Parameterized):
    """Sidebar panel for calibration configuration and execution."""

    state = param.Parameter(doc="WorkflowState instance")

    def __init__(self, state, **kw):
        super().__init__(state=state, **kw)
        self._wt = WorkflowThread(state)

        # Pre-populate from config
        defaults = {}
        if state.typed_config is not None:
            try:
                defaults = config_to_params(state.typed_config)
            except Exception:  # noqa: BLE001 — UI resilience
                pass

        self._algorithm = pn.widgets.Select(
            name='Algorithm',
            options=_ALGORITHMS,
            value=defaults.get('optimization_algorithm', 'PSO'),
            **_WIDGET_KW,
        )
        self._metric = pn.widgets.Select(
            name='Metric',
            options=_METRICS,
            value=defaults.get('optimization_metric', 'KGE'),
            **_WIDGET_KW,
        )
        self._iterations = pn.widgets.IntInput(
            name='Iterations',
            value=defaults.get('iterations', 1000),
            start=1,
            step=100,
            **_WIDGET_KW,
        )
        self._population_size = pn.widgets.IntInput(
            name='Population Size',
            value=defaults.get('population_size', 50),
            start=1,
            step=10,
            **_WIDGET_KW,
        )
        self._calibration_period = pn.widgets.TextInput(
            name='Calibration Period',
            placeholder='2003-01-01, 2005-12-31',
            value=defaults.get('calibration_period', ''),
            **_WIDGET_KW,
        )
        self._evaluation_period = pn.widgets.TextInput(
            name='Evaluation Period',
            placeholder='2006-01-01, 2007-12-31',
            value=defaults.get('evaluation_period', ''),
            **_WIDGET_KW,
        )
        self._experiment_id = pn.widgets.TextInput(
            name='Experiment ID',
            placeholder='run_pso_1000',
            value=defaults.get('experiment_id', ''),
            **_WIDGET_KW,
        )
        self._calibrate_btn = pn.widgets.Button(
            name='Run Calibration',
            button_type='primary',
            **_BTN_KW,
        )
        self._calibrate_btn.on_click(self._on_calibrate)

        # Phase and running-state sync
        state.param.watch(self._on_phase_change, ['gui_phase'])
        state.param.watch(self._sync_running, ['is_running'])

    # ------------------------------------------------------------------
    # Phase visibility
    # ------------------------------------------------------------------

    def _on_phase_change(self, event):
        phase = event.new
        self._panel_card.visible = phase in (
            'model_ready', 'calibrated', 'analyzed',
        )

    # ------------------------------------------------------------------
    # Running state
    # ------------------------------------------------------------------

    def _sync_running(self, event):
        self._calibrate_btn.disabled = bool(event.new)

    # ------------------------------------------------------------------
    # Action handler
    # ------------------------------------------------------------------

    def _on_calibrate(self, event):
        if self.state.typed_config is None:
            self.state.append_log("ERROR: No config loaded.\n")
            return

        # Auto-derive calibration period from experiment time range if not set
        # This avoids cold-start artifacts destroying metrics (e.g. SUMMA dumps
        # initial state as runoff in the first months, KGE can reach -20).
        calib_period = self._calibration_period.value.strip()
        eval_period = self._evaluation_period.value.strip()

        if not calib_period:
            try:
                import pandas as pd
                cfg = self.state.typed_config
                t_start = pd.Timestamp(cfg.domain.time_start)
                t_end = pd.Timestamp(cfg.domain.time_end)
                total_days = (t_end - t_start).days

                # Skip first 25% as spinup (minimum 90 days)
                spinup_days = max(90, int(total_days * 0.25))
                calib_start = t_start + pd.Timedelta(days=spinup_days)
                calib_period = f"{calib_start.strftime('%Y-%m-%d')}, {t_end.strftime('%Y-%m-%d')}"
                self._calibration_period.value = calib_period
                self.state.append_log(
                    f"Auto-set calibration period: {calib_period} "
                    f"(skipping {spinup_days}-day spinup)\n"
                )
            except Exception as exc:  # noqa: BLE001 — UI resilience
                logger.debug(f"Could not auto-derive calibration period: {exc}")

        # Build overrides via params_to_config_overrides
        overrides = params_to_config_overrides({
            'optimization_algorithm': self._algorithm.value,
            'optimization_metric': self._metric.value,
            'iterations': self._iterations.value,
            'population_size': self._population_size.value,
            'calibration_period': calib_period,
            'evaluation_period': eval_period,
            'experiment_id': self._experiment_id.value,
        })
        # Enable iterative optimization (required by optimization_manager)
        overrides['OPTIMIZATION_METHODS'] = 'iteration'

        # Flatten current config, merge overrides, reconstruct
        try:
            from symfluence.core.config.models import SymfluenceConfig
            current = self.state.typed_config.to_dict(flatten=True)
            current.update(overrides)
            self.state.typed_config = SymfluenceConfig(**current)
        except Exception as exc:  # noqa: BLE001 — UI resilience
            self.state.append_log(f"Config update failed: {exc}\n")
            return

        self.state.save_config()
        self.state.invalidate_symfluence()

        self.state.append_log(
            f"Starting calibration: {self._algorithm.value}, "
            f"{self._metric.value}, {self._iterations.value} iters, "
            f"experiment={self._experiment_id.value}\n"
        )
        self._wt.run_steps(['calibrate_model'], force_rerun=True)

    # ------------------------------------------------------------------
    # Panel layout
    # ------------------------------------------------------------------

    def panel(self):
        self._panel_card = pn.Card(
            self._algorithm,
            self._metric,
            self._iterations,
            self._population_size,
            self._calibration_period,
            self._evaluation_period,
            self._experiment_id,
            pn.layout.Divider(),
            self._calibrate_btn,
            title='Calibration',
            collapsed=True,
            visible=self.state.gui_phase in (
                'model_ready', 'calibrated', 'analyzed',
            ),
            sizing_mode='stretch_width',
            header_background='#eef4fb',
            styles={'margin-top': '8px'},
        )
        return self._panel_card
