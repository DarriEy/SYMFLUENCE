"""
Background execution and log capture for the SYMFLUENCE GUI.

WorkflowThread runs SYMFLUENCE steps in a daemon thread so the Panel
Tornado event loop remains responsive.  GUILogHandler captures log
records and routes them to WorkflowState.log_text via pn.state.execute.
"""

import logging
import threading

import panel as pn


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
            pn.state.execute(lambda m=msg: self.state.append_log(m))
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
        if self.alive:
            self.state.append_log("A workflow is already running.\n")
            return

        self._thread = threading.Thread(
            target=self._execute,
            args=(step_names, force_rerun),
            daemon=True,
        )
        self._thread.start()

    def run_workflow(self, force_rerun=False):
        """Launch full workflow in a background thread."""
        if self.alive:
            self.state.append_log("A workflow is already running.\n")
            return

        self._thread = threading.Thread(
            target=self._execute_full,
            args=(force_rerun,),
            daemon=True,
        )
        self._thread.start()

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
            pn.state.execute(self._set_running, True, step_names[0] if step_names else None)
            sf = self.state.initialize_symfluence()
            sf.run_individual_steps(step_names)
            pn.state.execute(lambda: self.state.append_log("Step(s) completed successfully.\n"))
        except Exception as exc:
            pn.state.execute(lambda e=exc: self.state.append_log(f"ERROR: {e}\n"))
        finally:
            self._detach_logger()
            pn.state.execute(self._set_running, False, None)
            pn.state.execute(self.state.refresh_status)

    def _execute_full(self, force_rerun):
        """Thread target for full workflow."""
        self._attach_logger()
        try:
            pn.state.execute(self._set_running, True, 'full_workflow')
            sf = self.state.initialize_symfluence()
            sf.run_workflow(force_run=force_rerun)
            pn.state.execute(lambda: self.state.append_log("Full workflow completed successfully.\n"))
        except Exception as exc:
            pn.state.execute(lambda e=exc: self.state.append_log(f"ERROR: {e}\n"))
        finally:
            self._detach_logger()
            pn.state.execute(self._set_running, False, None)
            pn.state.execute(self.state.refresh_status)

    def _set_running(self, running, step_name):
        self.state.is_running = running
        self.state.running_step = step_name
