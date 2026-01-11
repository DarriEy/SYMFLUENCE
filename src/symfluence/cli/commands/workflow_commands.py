"""
Workflow command handlers for SYMFLUENCE CLI.

This module implements handlers for the workflow command category.
"""

import sys
from argparse import Namespace
from pathlib import Path
from typing import List

from .base import BaseCommand
from ..exit_codes import ExitCode


class WorkflowCommands(BaseCommand):
    """Handlers for workflow category commands."""

    # Workflow step definitions (from original CLIArgumentManager)
    WORKFLOW_STEPS = {
        'setup_project': 'Initialize project directory structure and shapefiles',
        'create_pour_point': 'Create pour point shapefile from coordinates',
        'acquire_attributes': 'Download and process geospatial attributes (soil, land class, etc.)',
        'define_domain': 'Define hydrological domain boundaries and river basins',
        'discretize_domain': 'Discretize domain into HRUs or other modeling units',
        'process_observed_data': 'Process observational data (streamflow, etc.)',
        'acquire_forcings': 'Acquire meteorological forcing data',
        'model_agnostic_preprocessing': 'Run model-agnostic preprocessing of forcing and attribute data',
        'model_specific_preprocessing': 'Setup model-specific input files and configuration',
        'run_model': 'Execute the hydrological model simulation',
        'calibrate_model': 'Run model calibration and parameter optimization',
        'run_emulation': 'Run emulation-based optimization if configured',
        'run_benchmarking': 'Run benchmarking analysis against observations',
        'run_decision_analysis': 'Run decision analysis for model comparison',
        'run_sensitivity_analysis': 'Run sensitivity analysis on model parameters',
        'postprocess_results': 'Postprocess and finalize model results'
    }

    @staticmethod
    def run(args: Namespace) -> int:
        """
        Execute: symfluence workflow run

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Import here to avoid circular dependencies
            from symfluence.core import SYMFLUENCE

            config_path = BaseCommand.get_config_path(args)

            # Validate config exists
            if not BaseCommand.validate_config(config_path, required=True):
                return ExitCode.CONFIG_ERROR

            # Initialize SYMFLUENCE instance
            BaseCommand._console.info("Starting full workflow execution...")
            symfluence = SYMFLUENCE(
                config_path,
                debug_mode=getattr(args, 'debug', False),
                visualize=getattr(args, 'visualise', False)
            )

            # Execute full workflow
            symfluence.run_workflow()

            BaseCommand._console.success("Workflow execution completed successfully")
            return ExitCode.SUCCESS

        except Exception as e:
            BaseCommand._console.error(f"Workflow execution failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.WORKFLOW_ERROR

    @staticmethod
    def run_step(args: Namespace) -> int:
        """
        Execute: symfluence workflow step STEP_NAME

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.core import SYMFLUENCE

            config_path = BaseCommand.get_config_path(args)

            if not BaseCommand.validate_config(config_path, required=True):
                return ExitCode.CONFIG_ERROR

            BaseCommand._console.info(f"Executing step: {args.step_name}")
            BaseCommand._console.indent(WorkflowCommands.WORKFLOW_STEPS.get(args.step_name, ''))

            symfluence = SYMFLUENCE(
                config_path,
                debug_mode=getattr(args, 'debug', False),
                visualize=getattr(args, 'visualise', False)
            )

            # Run single step
            symfluence.run_individual_steps([args.step_name])

            BaseCommand._console.success(f"Step '{args.step_name}' completed successfully")
            return ExitCode.SUCCESS

        except Exception as e:
            BaseCommand._console.error(f"Step execution failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.WORKFLOW_ERROR

    @staticmethod
    def run_steps(args: Namespace) -> int:
        """
        Execute: symfluence workflow steps STEP1 STEP2 ...

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.core import SYMFLUENCE

            config_path = BaseCommand.get_config_path(args)

            if not BaseCommand.validate_config(config_path, required=True):
                return ExitCode.CONFIG_ERROR

            BaseCommand._console.info(f"Executing {len(args.step_names)} steps:")
            for step_name in args.step_names:
                BaseCommand._console.indent(f"{step_name}: {WorkflowCommands.WORKFLOW_STEPS.get(step_name, '')}")

            symfluence = SYMFLUENCE(
                config_path,
                debug_mode=getattr(args, 'debug', False),
                visualize=getattr(args, 'visualise', False)
            )

            # Run multiple steps in order
            symfluence.run_individual_steps(args.step_names)

            BaseCommand._console.success(f"All {len(args.step_names)} steps completed successfully")
            return ExitCode.SUCCESS

        except Exception as e:
            BaseCommand._console.error(f"Step execution failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.WORKFLOW_ERROR

    @staticmethod
    def status(args: Namespace) -> int:
        """
        Execute: symfluence workflow status

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.core import SYMFLUENCE

            config_path = BaseCommand.get_config_path(args)

            if not BaseCommand.validate_config(config_path, required=True):
                return ExitCode.CONFIG_ERROR

            symfluence = SYMFLUENCE(
                config_path,
                debug_mode=getattr(args, 'debug', False),
                visualize=getattr(args, 'visualise', False)
            )

            # Show workflow status
            BaseCommand._console.info("Workflow Status:")
            BaseCommand._console.rule()

            # Call the status method from SYMFLUENCE if it exists
            if hasattr(symfluence, 'get_workflow_status'):
                status_info = symfluence.get_workflow_status()
                BaseCommand._console.info(status_info)
            else:
                BaseCommand._console.info("Workflow status tracking not yet implemented")

            return ExitCode.SUCCESS

        except Exception as e:
            BaseCommand._console.error(f"Failed to get workflow status: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.GENERAL_ERROR

    @staticmethod
    def validate(args: Namespace) -> int:
        """
        Execute: symfluence workflow validate

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            config_path = BaseCommand.get_config_path(args)

            # Load and validate config using typed system
            config = BaseCommand.load_typed_config(config_path, required=True)
            if config is None:
                return ExitCode.CONFIG_ERROR

            BaseCommand._console.success("Configuration validated successfully")
            return ExitCode.SUCCESS

        except Exception as e:
            BaseCommand._console.error(f"Validation failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.VALIDATION_ERROR

    @staticmethod
    def list_steps(args: Namespace) -> int:
        """
        Execute: symfluence workflow list-steps

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        BaseCommand._console.info("Available workflow steps:")
        BaseCommand._console.rule()
        for i, (step_name, description) in enumerate(WorkflowCommands.WORKFLOW_STEPS.items(), 1):
            BaseCommand._console.info(f"{i:2}. {step_name:30s} - {description}")
        BaseCommand._console.rule()
        BaseCommand._console.info(f"Total: {len(WorkflowCommands.WORKFLOW_STEPS)} steps")
        return ExitCode.SUCCESS

    @staticmethod
    def resume(args: Namespace) -> int:
        """
        Execute: symfluence workflow resume STEP_NAME

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.core import SYMFLUENCE

            config_path = BaseCommand.get_config_path(args)

            if not BaseCommand.validate_config(config_path, required=True):
                return ExitCode.CONFIG_ERROR

            # Get all steps from the resume point onwards
            step_list = list(WorkflowCommands.WORKFLOW_STEPS.keys())
            if args.step_name not in step_list:
                BaseCommand._console.error(f"Unknown step: {args.step_name}")
                return ExitCode.USAGE_ERROR

            resume_index = step_list.index(args.step_name)
            steps_to_run = step_list[resume_index:]

            BaseCommand._console.info(f"Resuming workflow from: {args.step_name}")
            BaseCommand._console.indent(f"Will execute {len(steps_to_run)} steps:")
            for step in steps_to_run:
                BaseCommand._console.indent(f"- {step}")

            symfluence = SYMFLUENCE(
                config_path,
                debug_mode=getattr(args, 'debug', False),
                visualize=getattr(args, 'visualise', False)
            )

            # Run steps from resume point
            symfluence.run_individual_steps(steps_to_run)

            BaseCommand._console.success(f"Workflow resumed and completed from '{args.step_name}'")
            return ExitCode.SUCCESS

        except Exception as e:
            BaseCommand._console.error(f"Resume failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.WORKFLOW_ERROR

    @staticmethod
    def clean(args: Namespace) -> int:
        """
        Execute: symfluence workflow clean

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            config_path = BaseCommand.get_config_path(args)

            if not BaseCommand.validate_config(config_path, required=True):
                return ExitCode.CONFIG_ERROR

            level = args.level
            dry_run = getattr(args, 'dry_run', False)

            BaseCommand._console.info(f"Cleaning {level} files...")
            if dry_run:
                BaseCommand._console.indent("(DRY RUN - no files will be deleted)")

            # Import cleaning logic from cli_argument_manager if it exists
            # For now, provide placeholder implementation
            from symfluence.core import SYMFLUENCE
            symfluence = SYMFLUENCE(config_path, debug_mode=getattr(args, 'debug', False))

            if hasattr(symfluence, 'clean_workflow_files'):
                symfluence.clean_workflow_files(level=level, dry_run=dry_run)
            else:
                BaseCommand._console.info("Cleaning functionality not yet implemented in SYMFLUENCE core")
                BaseCommand._console.info(f"Would clean {level} files from workflow directories")

            if not dry_run:
                BaseCommand._console.success(f"Cleaned {level} files")
            return ExitCode.SUCCESS

        except Exception as e:
            BaseCommand._console.error(f"Clean failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.GENERAL_ERROR

    @staticmethod
    def execute(args: Namespace) -> int:
        """
        Main execution dispatcher (required by BaseCommand).

        This method is not used directly since we use func= in argument parser.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code
        """
        # This method exists to satisfy the abstract base class
        # In practice, specific methods (run, run_step, etc.) are called directly
        if hasattr(args, 'func'):
            return args.func(args)
        else:
            BaseCommand._console.error("No workflow action specified")
            return ExitCode.USAGE_ERROR
