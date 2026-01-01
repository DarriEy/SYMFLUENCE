"""
Tool executor for the SYMFLUENCE AI agent.

This module executes CLI commands/tools called by the LLM and returns
structured results. It integrates with the existing CLI manager to avoid
code duplication.
"""

import sys
import io
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class ToolResult:
    """
    Structured result from tool execution.

    Attributes:
        success: Whether the tool executed successfully
        output: stdout/result output from the tool
        error: Error message if execution failed
        exit_code: Exit code (0 for success, non-zero for failure)
    """
    success: bool
    output: str
    error: Optional[str]
    exit_code: int

    def to_string(self) -> str:
        """
        Format the result as a string for LLM consumption.

        Returns:
            Formatted result string with status and output/error
        """
        if self.success:
            return f"✓ Success\n\n{self.output}" if self.output else "✓ Success"
        else:
            error_msg = f"✗ Failed (exit code: {self.exit_code})\n\n"
            if self.error:
                error_msg += f"Error: {self.error}\n"
            if self.output:
                error_msg += f"\nOutput:\n{self.output}"
            return error_msg


class ToolExecutor:
    """
    Executes tools called by the LLM agent.

    This class integrates with the existing CLI argument manager to execute
    workflow steps, binary management, configuration operations, etc.
    """

    def __init__(self, cli_manager):
        """
        Initialize the tool executor.

        Args:
            cli_manager: Instance of CLIArgumentManager
        """
        self.cli_manager = cli_manager

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool and return structured result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments for the tool

        Returns:
            ToolResult with execution status and output
        """
        try:
            # Workflow step execution
            if tool_name in self.cli_manager.workflow_steps:
                return self._execute_workflow_step(tool_name, arguments)

            # Binary management operations
            elif tool_name in ['install_executables', 'validate_binaries', 'run_doctor', 'show_tools_info']:
                return self._execute_binary_operation(tool_name, arguments)

            # Configuration operations
            elif tool_name in ['list_config_templates', 'update_config', 'validate_environment', 'validate_config_file']:
                return self._execute_config_operation(tool_name, arguments)

            # Workflow management
            elif tool_name in ['show_workflow_status', 'list_workflow_steps', 'resume_from_step', 'clean_workflow_files', 'dry_run_workflow']:
                return self._execute_workflow_management(tool_name, arguments)

            # Pour point setup
            elif tool_name == 'setup_pour_point_workflow':
                return self._execute_pour_point_setup(arguments)

            # SLURM operations
            elif tool_name in ['submit_slurm_job', 'monitor_slurm_job']:
                return self._execute_slurm_operation(tool_name, arguments)

            # Meta operations
            elif tool_name in ['show_help', 'list_available_tools', 'explain_workflow']:
                return self._execute_meta_operation(tool_name, arguments)

            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown tool: {tool_name}",
                    exit_code=1
                )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution failed: {str(e)}",
                exit_code=1
            )

    def _execute_workflow_step(self, step_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a workflow step using SYMFLUENCE instance.

        Args:
            step_name: Name of the workflow step
            arguments: Must include 'config_path'

        Returns:
            ToolResult with execution status
        """
        try:
            config_path = arguments.get('config_path')
            if not config_path:
                return ToolResult(
                    success=False,
                    output="",
                    error="config_path argument is required for workflow steps",
                    exit_code=1
                )

            # Verify config file exists
            if not Path(config_path).exists():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Configuration file not found: {config_path}",
                    exit_code=2
                )

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                # Import SYMFLUENCE and execute step
                from symfluence.core import SYMFLUENCE

                debug_mode = arguments.get('debug', False)
                symfluence = SYMFLUENCE(config_path, debug_mode=debug_mode)
                symfluence.run_individual_steps([step_name])

                output = captured_output.getvalue()
                return ToolResult(
                    success=True,
                    output=output or f"Successfully completed {step_name}",
                    error=None,
                    exit_code=0
                )

            finally:
                sys.stdout = old_stdout

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
                exit_code=1
            )

    def _execute_binary_operation(self, operation: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute binary management operations.

        Args:
            operation: Operation name (install_executables, validate_binaries, etc.)
            arguments: Operation-specific arguments

        Returns:
            ToolResult with execution status
        """
        try:
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                if operation == 'install_executables':
                    tools = arguments.get('tools', [])
                    force_install = arguments.get('force_install', False)

                    # Use binary manager
                    if tools:
                        for tool in tools:
                            self.cli_manager.binary_manager.get_executable(tool, force=force_install)
                    else:
                        # Install all tools
                        for tool_name in self.cli_manager.binary_manager.external_tools.keys():
                            self.cli_manager.binary_manager.get_executable(tool_name, force=force_install)

                elif operation == 'validate_binaries':
                    self.cli_manager.binary_manager.validate_binaries()

                elif operation == 'run_doctor':
                    self.cli_manager.binary_manager.run_doctor()

                elif operation == 'show_tools_info':
                    self.cli_manager.binary_manager.show_tools_info()

                output = captured_output.getvalue()
                return ToolResult(
                    success=True,
                    output=output or f"Successfully completed {operation}",
                    error=None,
                    exit_code=0
                )

            finally:
                sys.stdout = old_stdout

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
                exit_code=1
            )

    def _execute_config_operation(self, operation: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute configuration management operations.

        Args:
            operation: Operation name
            arguments: Operation-specific arguments

        Returns:
            ToolResult with execution status
        """
        try:
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                if operation == 'list_config_templates':
                    self.cli_manager.list_templates()

                elif operation == 'update_config':
                    config_file = arguments.get('config_file')
                    if not config_file:
                        raise ValueError("config_file argument required")
                    self.cli_manager.update_config(config_file)

                elif operation == 'validate_environment':
                    self.cli_manager.validate_environment()

                elif operation == 'validate_config_file':
                    config_file = arguments.get('config_file')
                    if not config_file:
                        raise ValueError("config_file argument required")
                    # Validate by trying to load it
                    from symfluence.core import SYMFLUENCE
                    SYMFLUENCE(config_file, debug_mode=False)
                    print(f"✓ Configuration file is valid: {config_file}")

                output = captured_output.getvalue()
                return ToolResult(
                    success=True,
                    output=output or f"Successfully completed {operation}",
                    error=None,
                    exit_code=0
                )

            finally:
                sys.stdout = old_stdout

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
                exit_code=1
            )

    def _execute_workflow_management(self, operation: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute workflow management operations.

        Args:
            operation: Operation name
            arguments: Operation-specific arguments

        Returns:
            ToolResult with execution status
        """
        try:
            if operation == 'list_workflow_steps':
                steps_info = "Available workflow steps:\n\n"
                for step_name, step_info in self.cli_manager.workflow_steps.items():
                    steps_info += f"• {step_name}\n  {step_info['description']}\n\n"

                return ToolResult(
                    success=True,
                    output=steps_info,
                    error=None,
                    exit_code=0
                )

            # For operations that need a SYMFLUENCE instance, similar to workflow steps
            elif operation in ['show_workflow_status', 'resume_from_step', 'clean_workflow_files']:
                config_path = arguments.get('config_path')
                if not config_path:
                    return ToolResult(
                        success=False,
                        output="",
                        error="config_path argument is required",
                        exit_code=1
                    )

                old_stdout = sys.stdout
                sys.stdout = captured_output = io.StringIO()

                try:
                    from symfluence.core import SYMFLUENCE
                    symfluence = SYMFLUENCE(config_path, debug_mode=False)

                    if operation == 'show_workflow_status':
                        ops = {"workflow_status": True}
                        self.cli_manager.print_status_information(symfluence, ops)

                    elif operation == 'resume_from_step':
                        step_name = arguments.get('step_name')
                        if not step_name:
                            raise ValueError("step_name argument required")
                        # Get all steps from specified step onwards
                        steps = list(self.cli_manager.workflow_steps.keys())
                        if step_name not in steps:
                            raise ValueError(f"Unknown step: {step_name}")
                        start_idx = steps.index(step_name)
                        resume_steps = steps[start_idx:]
                        symfluence.run_individual_steps(resume_steps)

                    elif operation == 'clean_workflow_files':
                        level = arguments.get('clean_level', 'intermediate')
                        dry_run = arguments.get('dry_run', False)
                        self.cli_manager.clean_workflow_files(level, symfluence, dry_run)

                    output = captured_output.getvalue()
                    return ToolResult(
                        success=True,
                        output=output or f"Successfully completed {operation}",
                        error=None,
                        exit_code=0
                    )

                finally:
                    sys.stdout = old_stdout

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
                exit_code=1
            )

    def _execute_pour_point_setup(self, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute pour point workflow setup.

        Args:
            arguments: Must include latitude, longitude, domain_name, domain_definition_method

        Returns:
            ToolResult with execution status
        """
        try:
            required = ['latitude', 'longitude', 'domain_name', 'domain_definition_method']
            for arg in required:
                if arg not in arguments:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Missing required argument: {arg}",
                        exit_code=1
                    )

            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                coordinates = (arguments['latitude'], arguments['longitude'])
                bounding_box = arguments.get('bounding_box')
                bounding_box_coords = None
                if bounding_box:
                    bounding_box_coords = (
                        bounding_box['lat_max'],
                        bounding_box['lon_min'],
                        bounding_box['lat_min'],
                        bounding_box['lon_max']
                    )

                self.cli_manager.setup_pour_point_workflow(
                    coordinates=coordinates,
                    domain_def_method=arguments['domain_definition_method'],
                    domain_name=arguments['domain_name'],
                    bounding_box_coords=bounding_box_coords,
                    symfluence_code_dir=None
                )

                output = captured_output.getvalue()
                return ToolResult(
                    success=True,
                    output=output or f"Successfully set up pour point workflow for {arguments['domain_name']}",
                    error=None,
                    exit_code=0
                )

            finally:
                sys.stdout = old_stdout

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
                exit_code=1
            )

    def _execute_slurm_operation(self, operation: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute SLURM job operations.

        Args:
            operation: Operation name
            arguments: Operation-specific arguments

        Returns:
            ToolResult with execution status
        """
        return ToolResult(
            success=False,
            output="",
            error="SLURM operations not yet implemented in agent mode",
            exit_code=1
        )

    def _execute_meta_operation(self, operation: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute meta operations (help, list tools, etc.).

        Args:
            operation: Operation name
            arguments: Operation-specific arguments

        Returns:
            ToolResult with execution status
        """
        try:
            if operation == 'show_help':
                from . import system_prompts
                return ToolResult(
                    success=True,
                    output=system_prompts.HELP_MESSAGE,
                    error=None,
                    exit_code=0
                )

            elif operation == 'list_available_tools':
                tools_info = "Available Tools:\n\n"
                tools_info += "Workflow Steps:\n"
                for step_name, step_info in self.cli_manager.workflow_steps.items():
                    tools_info += f"  • {step_name}: {step_info['description']}\n"

                tools_info += "\nBinary Management:\n"
                tools_info += "  • install_executables: Install external modeling tools\n"
                tools_info += "  • validate_binaries: Validate installed tools\n"
                tools_info += "  • run_doctor: System diagnostics\n"
                tools_info += "  • show_tools_info: Show installed tools info\n"

                return ToolResult(
                    success=True,
                    output=tools_info,
                    error=None,
                    exit_code=0
                )

            elif operation == 'explain_workflow':
                explanation = """
SYMFLUENCE Workflow Explanation:

The typical workflow consists of these sequential steps:

1. setup_project - Initialize project directory structure
2. acquire_attributes - Download geospatial data (soil, land cover, etc.)
3. acquire_forcings - Download meteorological forcing data
4. define_domain - Define hydrological domain boundaries
5. discretize_domain - Discretize into modeling units (HRUs)
6. model_agnostic_preprocessing - Preprocess data
7. model_specific_preprocessing - Setup model-specific inputs
8. run_model - Execute the model simulation
9. postprocess_results - Analyze and visualize results

Optional steps:
  - calibrate_model: Parameter calibration
  - run_benchmarking: Compare against observations
  - run_sensitivity_analysis: Parameter sensitivity analysis
"""
                return ToolResult(
                    success=True,
                    output=explanation,
                    error=None,
                    exit_code=0
                )

        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=str(e),
                exit_code=1
            )
