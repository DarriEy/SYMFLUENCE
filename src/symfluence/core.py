"""
SYMFLUENCE Core Logic
"""
try:
    from symfluence.symfluence_version import __version__
except Exception:
    __version__ = "0+unknown"


from pathlib import Path
import yaml
from datetime import datetime
from typing import Dict, Any, List
import sys

# Import SYMFLUENCE components
from symfluence.utils.project.project_manager import ProjectManager
from symfluence.utils.project.workflow_orchestrator import WorkflowOrchestrator
from symfluence.utils.project.logging_manager import LoggingManager
from symfluence.utils.data.data_manager import DataManager
from symfluence.utils.geospatial.domain_manager import DomainManager
from symfluence.utils.models.model_manager import ModelManager
from symfluence.utils.evaluation.analysis_manager import AnalysisManager
from symfluence.utils.optimization.optimization_manager import OptimizationManager
from symfluence.utils.cli.cli_argument_manager import CLIArgumentManager
from symfluence.utils.config.config_loader import load_config


class SYMFLUENCE:
    """
    Enhanced SYMFLUENCE main class with comprehensive CLI support.
    
    This class serves as the central coordinator for all SYMFLUENCE operations,
    with enhanced CLI capabilities including individual step execution,
    pour point setup, SLURM job submission, and comprehensive workflow management.
    """
    
    def __init__(self, config_path: Path, config_overrides: Dict[str, Any] = None, debug_mode: bool = False):
        """
        Initialize the SYMFLUENCE system with configuration and CLI options.
        
        Args:
            config_path: Path to the configuration file
            config_overrides: Dictionary of configuration overrides from CLI
            debug_mode: Whether to enable debug mode
        """
        self.config_path = config_path
        self.debug_mode = debug_mode
        self.config_overrides = config_overrides or {}
        
        # Load and merge configuration
        self.config = self._load_and_merge_config()
        
        # Initialize logging
        self.logging_manager = LoggingManager(self.config, debug_mode=debug_mode)
        self.logger = self.logging_manager.logger

        self.logger.info(f"SYMFLUENCE initialized with config: {config_path}")
        if self.config_overrides:
            self.logger.info(f"Configuration overrides applied: {list(self.config_overrides.keys())}")


        # Initialize managers
        self.managers = self._initialize_managers()
        
        # Initialize workflow orchestrator
        self.workflow_orchestrator = WorkflowOrchestrator(
            self.managers, self.config, self.logger, self.logging_manager
        )
        
    
    def _load_and_merge_config(self) -> Dict[str, Any]:
        try:
            return load_config(self.config_path, self.config_overrides, validate=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
    
    def _initialize_managers(self) -> Dict[str, Any]:
        """Initialize all manager components."""
        try:
            return {
                'project': ProjectManager(self.config, self.logger),
                'domain': DomainManager(self.config, self.logger),
                'data': DataManager(self.config, self.logger),
                'model': ModelManager(self.config, self.logger),
                'analysis': AnalysisManager(self.config, self.logger),
                'optimization': OptimizationManager(self.config, self.logger),
            }
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to initialize managers: {str(e)}")
            raise RuntimeError(f"Manager initialization failed: {str(e)}")
    
    def run_workflow(self) -> None:
        """Execute the complete SYMFLUENCE workflow (CLI wrapper)."""
        start = datetime.now()
        steps_completed = []
        errors = []
        warns = []

        try:
            self.logger.info("Starting complete SYMFLUENCE workflow execution")

            # Run the workflow; if your orchestrator exposes steps executed, collect them
            self.workflow_orchestrator.run_workflow()
            steps_completed = getattr(self.workflow_orchestrator, "steps_executed", []) or []

            status = getattr(self.workflow_orchestrator, "get_workflow_status", lambda: "completed")()
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
        """Execute specific workflow steps (CLI wrapper)."""
        start = datetime.now()
        steps_completed = []
        errors = []
        warns = []

        # Resolve workflow steps from orchestrator
        workflow_steps = self.workflow_orchestrator.define_workflow_steps()
        cli_to_step = {step.cli_name: step for step in workflow_steps}

        status = "completed"
        try:
            self.logger.info(f"Starting individual step execution: {', '.join(step_names)}")

            for cli_name in step_names:
                step = cli_to_step.get(cli_name)
                if not step:
                    self.logger.warning(f"Step '{cli_name}' not recognized; skipping")
                    continue

                self.logger.info(f"Executing step: {cli_name} -> {step.name}")
                # Force execution; skip completion checks in CLI wrapper
                try:
                    step.func()
                    steps_completed.append({"cli": cli_name, "fn": step.name})
                    self.logger.info(f"✓ Completed step: {cli_name}")
                except Exception as e:
                    status = "partial" if steps_completed else "failed"
                    errors.append({"step": cli_name, "error": str(e)})
                    self.logger.error(f"Step '{cli_name}' failed: {e}")
                    if not self.config_overrides.get("continue_on_error", False):
                        raise
                    # else: continue to the next step
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
