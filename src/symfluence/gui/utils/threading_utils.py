"""
Background execution and log capture for the SYMFLUENCE GUI.

WorkflowThread runs SYMFLUENCE steps in a daemon thread so the Panel
Tornado event loop remains responsive.  GUILogHandler captures log
records and routes them to WorkflowState.log_text via pn.state.execute.
"""

import logging
import threading

import panel as pn


def run_on_ui_thread(callback):
    """Execute callback in the Panel UI context when available."""
    try:
        pn.state.execute(callback)
    except Exception:
        callback()


class GUILogHandler(logging.Handler):
    """
    Logging handler that routes records into WorkflowState.log_text.

    Attaches to the 'symfluence' logger and uses pn.state.execute()
    for thread-safe UI updates.
    """

    def __init__(self, state, level=logging.INFO):
        super().__init__(level)
        self.state = state
        self.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))

    def emit(self, record):
        try:
            msg = self.format(record) + '\n'
            run_on_ui_thread(lambda m=msg: self.state.append_log(m))
        except Exception:
            pass  # never crash the logger


class WorkflowThread:
    """
    Manages a daemon thread for running SYMFLUENCE workflow steps.

    Usage:
        wt = WorkflowThread(state)
        wt.run_steps(['setup_project', 'create_pour_point'])
    """

    def __init__(self, state):
        self.state = state
        self._thread = None
        self._log_handler = None

    @property
    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    def run_steps(self, step_names, force_rerun=False):
        """Launch step execution in a background thread."""
        step_label = step_names[0] if step_names else None
        if self.alive or not self.state.try_begin_run(step_label):
            self.state.append_log("A workflow is already running.\n")
            return

        self._thread = threading.Thread(
            target=self._execute,
            args=(step_names, force_rerun),
            daemon=True,
        )
        try:
            self._thread.start()
        except Exception:
            self.state.end_run()
            raise

    def run_workflow(self, force_rerun=False):
        """Launch full workflow in a background thread."""
        if self.alive or not self.state.try_begin_run('full_workflow'):
            self.state.append_log("A workflow is already running.\n")
            return

        self._thread = threading.Thread(
            target=self._execute_full,
            args=(force_rerun,),
            daemon=True,
        )
        try:
            self._thread.start()
        except Exception:
            self.state.end_run()
            raise

    def _attach_logger(self):
        """Attach GUILogHandler to the symfluence root logger."""
        self._log_handler = GUILogHandler(self.state)
        logging.getLogger('symfluence').addHandler(self._log_handler)

    def _detach_logger(self):
        """Remove GUILogHandler from the logger."""
        if self._log_handler:
            logging.getLogger('symfluence').removeHandler(self._log_handler)
            self._log_handler = None

    def _execute(self, step_names, force_rerun):
        """Thread target for individual steps."""
        self._attach_logger()
        try:
            sf = self.state.initialize_symfluence()
            if force_rerun:
                run_on_ui_thread(
                    lambda: self.state.append_log(
                        "Force re-run is always applied for individual steps in the current workflow engine.\n"
                    )
                )
            sf.run_individual_steps(step_names)
            run_on_ui_thread(lambda: self.state.append_log("Step(s) completed successfully.\n"))
        except Exception as exc:
            run_on_ui_thread(lambda e=exc: self.state.append_log(f"ERROR: {e}\n"))
        finally:
            self._detach_logger()
            run_on_ui_thread(self.state.end_run)
            run_on_ui_thread(self.state.refresh_status)
            # Signal results-producing steps so the Results tab auto-refreshes
            _RESULTS_STEPS = {'calibrate_model', 'run_benchmarking', 'postprocess_results'}
            if any(s in _RESULTS_STEPS for s in step_names):
                exp_id = (self.state.typed_config.domain.experiment_id
                          if self.state.typed_config else None)
                run_on_ui_thread(
                    lambda eid=exp_id: setattr(self.state, 'last_completed_run', eid)
                )

    def _execute_full(self, force_rerun):
        """Thread target for full workflow."""
        self._attach_logger()
        try:
            sf = self.state.initialize_symfluence()
            sf.run_workflow(force_run=force_rerun)
            run_on_ui_thread(lambda: self.state.append_log("Full workflow completed successfully.\n"))
        except Exception as exc:
            run_on_ui_thread(lambda e=exc: self.state.append_log(f"ERROR: {e}\n"))
        finally:
            self._detach_logger()
            run_on_ui_thread(self.state.end_run)
            run_on_ui_thread(self.state.refresh_status)
            # Full workflow includes calibration — signal for auto-refresh
            exp_id = (self.state.typed_config.domain.experiment_id
                      if self.state.typed_config else None)
            run_on_ui_thread(
                lambda eid=exp_id: setattr(self.state, 'last_completed_run', eid)
            )
