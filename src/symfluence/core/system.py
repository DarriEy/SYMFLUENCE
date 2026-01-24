"""
SYMFLUENCE Core System Module.

Provides the main SYMFLUENCE class that serves as the primary entry point
for hydrological modeling workflows. This module coordinates all manager
components and orchestrates the complete modeling pipeline from domain
definition through model calibration and analysis.

Example:
    >>> from symfluence import SYMFLUENCE
    >>> s = SYMFLUENCE("config.yaml")
    >>> s.run_workflow()
"""
try:
    from symfluence.symfluence_version import __version__
except ImportError:
    __version__ = "0+unknown"


from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Union

# Import core components
from symfluence.project.workflow_orchestrator import WorkflowOrchestrator
from symfluence.project.logging_manager import LoggingManager
from symfluence.project.manager_factory import LazyManagerDict
from symfluence.core.config.models import SymfluenceConfig


class SYMFLUENCE:
    """
    Enhanced SYMFLUENCE main class with comprehensive CLI support.

    This class serves as the central coordinator for all SYMFLUENCE operations,
    with enhanced CLI capabilities including individual step execution,
    pour point setup, SLURM job submission, and comprehensive workflow management.
    """

    def __init__(self, config_input: Union[Path, str, SymfluenceConfig], config_overrides: Dict[str, Any] = None, debug_mode: bool = False, visualize: bool = False, diagnostic: bool = False):
        """
        Initialize the SYMFLUENCE system with configuration and CLI options.

        Args:
            config_input: Path to the configuration file or a SymfluenceConfig instance
            config_overrides: Dictionary of configuration overrides from CLI
            debug_mode: Whether to enable debug mode
            visualize: Whether to enable visualization
            diagnostic: Whether to enable diagnostic plots for workflow validation
        """
        self.debug_mode = debug_mode
        self.visualize = visualize
        self.diagnostic = diagnostic
        self.config_overrides = config_overrides or {}

        # Handle different config input types
        if isinstance(config_input, SymfluenceConfig):
            self.typed_config = config_input
            # If overrides provided, we merge them into a flat dict and re-create the model
            if self.config_overrides:
                flat_config = self.typed_config.to_dict(flatten=True)
                flat_config.update(self.config_overrides)
                self.typed_config = SymfluenceConfig(**flat_config)
            self.config_path = getattr(config_input, '_source_file', None)
        else:
            self.config_path = Path(config_input)
            self.typed_config = self._load_typed_config()

        self.config = self.typed_config.to_dict(flatten=True)  # Backward compatibility

        # Ensure log level consistency with debug mode
        if self.debug_mode:
            self.config['LOG_LEVEL'] = 'DEBUG'

        # Initialize logging
        self.logging_manager = LoggingManager(self.config, debug_mode=debug_mode)
        self.logger = self.logging_manager.logger

        self.logger.info("SYMFLUENCE initialized")
        if self.config_path:
            self.logger.info(f"Config path: {self.config_path}")
        if self.config_overrides:
            self.logger.info(f"Configuration overrides applied: {list(self.config_overrides.keys())}")


        # Initialize managers (lazy loaded)
        self.managers = LazyManagerDict(self.typed_config, self.logger, self.visualize, self.diagnostic)

        # Initialize workflow orchestrator
        self.workflow_orchestrator = WorkflowOrchestrator(
            self.managers, self.config, self.logger, self.logging_manager
        )


    def _load_typed_config(self) -> SymfluenceConfig:
        """
        Load configuration using new hierarchical SymfluenceConfig.

        Returns:
            SymfluenceConfig: Fully validated hierarchical configuration
        """
        try:
            return SymfluenceConfig.from_file(
                self.config_path,
                overrides=self.config_overrides,
                use_env=True,
                validate=True
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

    def run_workflow(self) -> None:
        """Execute the complete SYMFLUENCE workflow (CLI wrapper)."""
        start = datetime.now()
        steps_completed: List[Any] = []
        errors: List[Any] = []
        warns: List[Any] = []

        try:
            self.logger.info("Starting complete SYMFLUENCE workflow execution")

            # Run the workflow
            self.workflow_orchestrator.run_workflow()

            # Collect status information
            status_info = self.workflow_orchestrator.get_workflow_status()
            steps_completed = [s for s in status_info['step_details'] if s['complete']]
            status = "completed" if status_info['total_steps'] == status_info['completed_steps'] else "partial"

            self.logger.info("Complete SYMFLUENCE workflow execution completed")

        except Exception as e:
            status = "failed"
            errors.append({"where": "run_workflow", "error": str(e)})
            self.logger.error(f"Workflow execution failed: {e}")
            # re-raise after summary so the CI can fail meaningfully if needed
            raise
        finally:
            end = datetime.now()
            elapsed_s = (end - start).total_seconds()
            # Call with the expected signature:
            self.logging_manager.create_run_summary(
                steps_completed=steps_completed,
                errors=errors,
                warnings=warns,
                execution_time=elapsed_s,
                status=status,
            )

    def run_individual_steps(self, step_names: List[str]) -> None:
        """
        Execute specific workflow steps by name.

        Allows selective execution of individual workflow steps rather than
        running the complete pipeline. Useful for debugging, testing, or
        re-running specific portions of the workflow.

        Args:
            step_names: List of step names to execute (e.g., ['setup_project', 'calibrate_model'])
        """
        start = datetime.now()
        steps_completed: List[Any] = []
        errors: List[Any] = []
        warns: List[Any] = []

        status = "completed"

        try:
            continue_on_error = self.config_overrides.get("continue_on_error", False)

            # Execute individual steps via orchestrator
            results = self.workflow_orchestrator.run_individual_steps(step_names, continue_on_error)

            # Process results for summary
            for res in results:
                if res['success']:
                    steps_completed.append({"cli": res['cli'], "fn": res['fn']})
                else:
                    errors.append({"step": res['cli'], "error": res['error']})
                    status = "partial" if steps_completed else "failed"

        finally:
            end = datetime.now()
            elapsed_s = (end - start).total_seconds()
            self.logging_manager.create_run_summary(
                steps_completed=steps_completed,
                errors=errors,
                warnings=warns,
                execution_time=elapsed_s,
                status=status,
            )
