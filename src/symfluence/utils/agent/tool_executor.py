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

    This class integrates with the modern CLI command architecture to execute
    workflow steps, binary management, configuration operations, etc.
    """

    def __init__(self, tool_registry=None):
        """
        Initialize the tool executor.

        Args:
            tool_registry: Optional instance of ToolRegistry for tool discovery
        """
        self.tool_registry = tool_registry
        self._sf_cache: Dict[str, Any] = {}  # Cache for SYMFLUENCE instances

    def _get_symfluence_instance(self, config_path: str, debug_mode: bool = False) -> Any:
        """
        Get or create a SYMFLUENCE instance for the given config path.

        Args:
            config_path: Path to the configuration file
            debug_mode: Enable debug output

        Returns:
            SYMFLUENCE instance
        """
        config_key = str(Path(config_path).absolute())
        
        # Check cache (ignore debug_mode for cache hits, but update if requested)
        if config_key in self._sf_cache:
            sf = self._sf_cache[config_key]
            if debug_mode:
                sf.debug_mode = True
            return sf
            
        # Create new instance
        from symfluence.core import SYMFLUENCE
        sf = SYMFLUENCE(config_path, debug_mode=debug_mode)
        self._sf_cache[config_key] = sf
        return sf

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool and return structured result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments for the tool

        Returns:
            ToolResult with execution status and output
        """
        from symfluence.utils.cli.commands.workflow_commands import WorkflowCommands
        
        try:
            # Workflow step execution
            if tool_name in WorkflowCommands.WORKFLOW_STEPS:
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
        Execute a workflow step using WorkflowCommands.

        Args:
            step_name: Name of the workflow step
            arguments: Must include 'config_path'

        Returns:
            ToolResult with execution status
        """
        from symfluence.utils.cli.commands.workflow_commands import WorkflowCommands
        from argparse import Namespace
        
        try:
            config_path = arguments.get('config_path')
            if not config_path:
                return ToolResult(
                    success=False,
                    output="",
                    error="config_path argument is required for workflow steps",
                    exit_code=1
                )

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                args = Namespace(
                    config=config_path,
                    step_name=step_name,
                    debug=arguments.get('debug', False),
                    visualise=arguments.get('visualise', False),
                    force_rerun=arguments.get('force_rerun', False)
                )
                
                exit_code = WorkflowCommands.run_step(args)

                output = captured_output.getvalue()
                return ToolResult(
                    success=exit_code == 0,
                    output=output,
                    error=None if exit_code == 0 else f"Step {step_name} failed",
                    exit_code=exit_code
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
        Execute binary management operations using BinaryCommands.

        Args:
            operation: Operation name
            arguments: Operation-specific arguments

        Returns:
            ToolResult with execution status
        """
        from symfluence.utils.cli.commands.binary_commands import BinaryCommands
        from argparse import Namespace
        
        try:
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                args = Namespace(
                    debug=arguments.get('debug', False),
                    verbose=arguments.get('verbose', True)
                )
                
                exit_code = 1
                if operation == 'install_executables':
                    args.tools = arguments.get('tools', [])
                    args.force = arguments.get('force_install', False)
                    exit_code = BinaryCommands.install(args)
                elif operation == 'validate_binaries':
                    exit_code = BinaryCommands.validate(args)
                elif operation == 'run_doctor':
                    exit_code = BinaryCommands.doctor(args)
                elif operation == 'show_tools_info':
                    exit_code = BinaryCommands.info(args)

                output = captured_output.getvalue()
                return ToolResult(
                    success=exit_code == 0,
                    output=output,
                    error=None if exit_code == 0 else f"Binary operation {operation} failed",
                    exit_code=exit_code
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
        Execute configuration management operations using ConfigCommands.

        Args:
            operation: Operation name
            arguments: Operation-specific arguments

        Returns:
            ToolResult with execution status
        """
        from symfluence.utils.cli.commands.config_commands import ConfigCommands
        from argparse import Namespace
        
        try:
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                args = Namespace(
                    debug=arguments.get('debug', False)
                )
                
                exit_code = 1
                if operation == 'list_config_templates':
                    exit_code = ConfigCommands.list_templates(args)
                elif operation == 'update_config':
                    args.config_file = arguments.get('config_file')
                    args.interactive = arguments.get('interactive', False)
                    exit_code = ConfigCommands.update(args)
                elif operation == 'validate_environment':
                    exit_code = ConfigCommands.validate_env(args)
                elif operation == 'validate_config_file':
                    args.config = arguments.get('config_file')
                    exit_code = ConfigCommands.validate(args)

                output = captured_output.getvalue()
                return ToolResult(
                    success=exit_code == 0,
                    output=output,
                    error=None if exit_code == 0 else f"Config operation {operation} failed",
                    exit_code=exit_code
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
        Execute workflow management operations using WorkflowCommands.

        Args:
            operation: Operation name
            arguments: Operation-specific arguments

        Returns:
            ToolResult with execution status
        """
        from symfluence.utils.cli.commands.workflow_commands import WorkflowCommands
        from argparse import Namespace
        
        try:
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                args = Namespace(
                    debug=arguments.get('debug', False),
                    config=arguments.get('config_path'),
                    visualise=False
                )
                
                exit_code = 1
                if operation == 'list_workflow_steps':
                    exit_code = WorkflowCommands.list_steps(args)
                elif operation == 'show_workflow_status':
                    exit_code = WorkflowCommands.status(args)
                elif operation == 'resume_from_step':
                    args.step_name = arguments.get('step_name')
                    args.force_rerun = arguments.get('force_rerun', False)
                    exit_code = WorkflowCommands.resume(args)
                elif operation == 'clean_workflow_files':
                    args.level = arguments.get('clean_level', 'intermediate')
                    args.dry_run = arguments.get('dry_run', False)
                    exit_code = WorkflowCommands.clean(args)

                output = captured_output.getvalue()
                return ToolResult(
                    success=exit_code == 0,
                    output=output,
                    error=None if exit_code == 0 else f"Workflow operation {operation} failed",
                    exit_code=exit_code
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
        Execute pour point workflow setup using ProjectCommands.

        Args:
            arguments: Must include latitude, longitude, domain_name, domain_definition_method

        Returns:
            ToolResult with execution status
        """
        from symfluence.utils.cli.commands.project_commands import ProjectCommands
        from argparse import Namespace
        
        try:
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()

            try:
                coords = f"{arguments['latitude']}/{arguments['longitude']}"
                bbox = None
                if arguments.get('bounding_box'):
                    b = arguments['bounding_box']
                    bbox = f"{b['lat_max']}/{b['lon_min']}/{b['lat_min']}/{b['lon_max']}"
                
                args = Namespace(
                    debug=arguments.get('debug', False),
                    coordinates=coords,
                    domain_name=arguments['domain_name'],
                    domain_def=arguments['domain_definition_method'],
                    bounding_box_coords=bbox,
                    experiment_id=arguments.get('experiment_id')
                )
                
                exit_code = ProjectCommands.pour_point(args)

                output = captured_output.getvalue()
                return ToolResult(
                    success=exit_code == 0,
                    output=output,
                    error=None if exit_code == 0 else "Pour point setup failed",
                    exit_code=exit_code
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
                if not self.tool_registry:
                    return ToolResult(
                        success=False,
                        output="",
                        error="Tool registry not available",
                        exit_code=1
                    )
                
                tools_by_category = self.tool_registry.get_tools_by_category()
                tools_info = "Available Tools:\n\n"
                
                for category, tools in tools_by_category.items():
                    tools_info += f"{category}:\n"
                    for tool in tools:
                        name = tool["function"]["name"]
                        desc = tool["function"]["description"]
                        # Use first line of description to keep it concise
                        short_desc = desc.split('.')[0] if '.' in desc else desc
                        tools_info += f"  • {name}: {short_desc}\n"
                    tools_info += "\n"

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

            elif operation == 'list_available_tools':
                if not self.tool_registry:
                    return ToolResult(
                        success=False,
                        output="",
                        error="Tool registry not available",
                        exit_code=1
                    )
                
                tools_by_category = self.tool_registry.get_tools_by_category()
                tools_info = "Available Tools:\n\n"
                
                for category, tools in tools_by_category.items():
                    tools_info += f"{category}:\n"
                    for tool in tools:
                        name = tool["function"]["name"]
                        desc = tool["function"]["description"]
                        # Use first line of description to keep it concise
                        short_desc = desc.split('.')[0] if '.' in desc else desc
                        tools_info += f"  • {name}: {short_desc}\n"
                    tools_info += "\n"

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
