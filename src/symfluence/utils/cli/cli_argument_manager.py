#!/usr/bin/env python3
"""
try:
    from symfluence.symfluence_version import __version__
except Exception:
    __version__ = "0+unknown"

SYMFLUENCE CLI Argument Manager

This utility provides comprehensive command-line interface functionality for the SYMFLUENCE
hydrological modeling platform. It handles argument parsing, validation, and workflow
step execution control.

Features:
- Individual workflow step execution
- Pour point coordinate setup
- Flexible configuration management
- Debug and logging controls
- Workflow validation and status reporting
- External tool installation and validation
- SLURM job submission support

Usage:
    from symfluence.utils.cli.cli_argument_manager import CLIArgumentManager
    
    cli_manager = CLIArgumentManager()
    args = cli_manager.parse_arguments()
    plan = cli_manager.get_execution_plan(args)
"""

try:
    from symfluence_version import __version__
except Exception:  # fallback for odd contexts
    __version__ = "0+unknown"

import argparse
import os
import shutil
import subprocess
import sys
import re
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .external_tools_config import get_external_tools_definitions
from .binary_manager import BinaryManager
from .job_scheduler import JobScheduler
from .notebook_service import NotebookService

class CLIArgumentManager:
    """
    Manages command-line arguments and workflow execution options for SYMFLUENCE.
    
    This class provides a comprehensive CLI interface that allows users to:
    - Run individual workflow steps
    - Set up pour point configurations
    - Control workflow execution behavior
    - Manage configuration and debugging options
    
    The argument manager integrates with the existing SYMFLUENCE workflow orchestrator
    to provide granular control over workflow execution.
    """
    
    def __init__(self):
        """Initialize the CLI argument manager."""
        self.parser = None
        self.workflow_steps = self._define_workflow_steps()
        self.domain_definition_methods = ['lumped', 'point', 'subset', 'delineate']
        
        # Initialize sub-managers
        self.binary_manager = BinaryManager()
        self.job_scheduler = JobScheduler()
        self.notebook_service = NotebookService()
        
        # For backward compatibility within this class if needed
        self.external_tools = self.binary_manager.external_tools
        
        self._setup_parser()
    
    # ============================================================================
    # WORKFLOW STEP DEFINITIONS
    # ============================================================================
    
    def _define_workflow_steps(self) -> Dict[str, Dict[str, str]]:
        """
        Define available workflow steps that can be run individually.
        
        Returns:
            Dictionary mapping step names to their descriptions and manager methods
        """
        return {
            'setup_project': {
                'description': 'Initialize project directory structure and shapefiles',
                'manager': 'project',
                'method': 'setup_project',
                'function_name': 'setup_project'
            },
            'create_pour_point': {
                'description': 'Create pour point shapefile from coordinates',
                'manager': 'project',
                'method': 'create_pour_point',
                'function_name': 'create_pour_point'
            },
            'acquire_attributes': {
                'description': 'Download and process geospatial attributes (soil, land class, etc.)',
                'manager': 'data',
                'method': 'acquire_attributes',
                'function_name': 'acquire_attributes'
            },
            'define_domain': {
                'description': 'Define hydrological domain boundaries and river basins',
                'manager': 'domain',
                'method': 'define_domain',
                'function_name': 'define_domain'
            },
            'discretize_domain': {
                'description': 'Discretize domain into HRUs or other modeling units',
                'manager': 'domain',
                'method': 'discretize_domain',
                'function_name': 'discretize_domain'
            },
            'process_observed_data': {
                'description': 'Process observational data (streamflow, etc.)',
                'manager': 'data',
                'method': 'process_observed_data',
                'function_name': 'process_observed_data'
            },
            'acquire_forcings': {
                'description': 'Acquire meteorological forcing data',
                'manager': 'data',
                'method': 'acquire_forcings',
                'function_name': 'acquire_forcings'
            },
            'model_agnostic_preprocessing': {
                'description': 'Run model-agnostic preprocessing of forcing and attribute data',
                'manager': 'data',
                'method': 'model_agnostic_preprocessing',
                'function_name': 'model_agnostic_preprocessing'
            },
            'model_specific_preprocessing': {
                'description': 'Setup model-specific input files and configuration',
                'manager': 'model',
                'method': 'model_specific_preprocessing',
                'function_name': 'model_specific_preprocessing'
            },
            'run_model': {
                'description': 'Execute the hydrological model simulation',
                'manager': 'model',
                'method': 'run_model',
                'function_name': 'run_model'
            },
            'calibrate_model': {
                'description': 'Run model calibration and parameter optimization',
                'manager': 'optimization',
                'method': 'run_calibration',
                'function_name': 'run_calibration'
            },
            'run_emulation': {
                'description': 'Run emulation-based optimization if configured',
                'manager': 'optimization',
                'method': 'run_emulation',
                'function_name': 'run_emulation'
            },
            'run_benchmarking': {
                'description': 'Run benchmarking analysis against observations',
                'manager': 'analysis',
                'method': 'run_benchmarking',
                'function_name': 'run_benchmarking'
            },
            'run_decision_analysis': {
                'description': 'Run decision analysis for model comparison',
                'manager': 'analysis',
                'method': 'run_decision_analysis',
                'function_name': 'run_decision_analysis'
            },
            'run_sensitivity_analysis': {
                'description': 'Run sensitivity analysis on model parameters',
                'manager': 'analysis',
                'method': 'run_sensitivity_analysis',
                'function_name': 'run_sensitivity_analysis'
            },
            'postprocess_results': {
                'description': 'Postprocess and finalize model results',
                'manager': 'model',
                'method': 'postprocess_results',
                'function_name': 'postprocess_results'
            }
        }
    
    # ============================================================================
    # ARGUMENT PARSER SETUP
    # ============================================================================
    
    def _setup_parser(self) -> None:
        """Set up the argument parser with all CLI options."""
        self.parser = argparse.ArgumentParser(
            description='SYMFLUENCE - SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexii for Computational Exploration',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_examples_text()
        )
        
        # Configuration options
        config_group = self.parser.add_argument_group('Configuration Options')
        config_group.add_argument(
            '--config',
            type=str,
            default='./0_config_files/config_template.yaml',
            help='Path to YAML configuration file (default: ./0_config_files/config_template.yaml)'
        )
        config_group.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug output and detailed logging'
        )
        config_group.add_argument(
            '--version',
            action='version',
            version=f'SYMFLUENCE {__version__}',
        )
        
        # Configuration Management
        config_mgmt_group = self.parser.add_argument_group('Configuration Management')
        config_mgmt_group.add_argument(
            '--list_templates',
            action='store_true',
            help='List all available configuration templates'
        )
        config_mgmt_group.add_argument(
            '--update_config',
            type=str,
            metavar='CONFIG_FILE',
            help='Update an existing configuration file with new settings'
        )
        config_mgmt_group.add_argument(
            '--validate_environment',
            action='store_true',
            help='Validate system environment and dependencies'
        )
        
        # Binary/Executable Management
        binary_mgmt_group = self.parser.add_argument_group('Binary Management')
        binary_mgmt_group.add_argument(
            '--get_executables',
            nargs='*',
            metavar='TOOL_NAME',
            help='Clone and install external tool repositories (summa, mizuroute, fuse, taudem, gistool, datatool). ' +
                 'Use without arguments to install all tools, or specify specific tools.'
        )
        binary_mgmt_group.add_argument(
            '--validate_binaries',
            action='store_true',
            help='Validate that external tool binaries exist and are functional'
        )
        binary_mgmt_group.add_argument(
            '--force_install',
            action='store_true',
            help='Force reinstallation of tools even if they already exist'
        )
        binary_mgmt_group.add_argument(
            '--doctor',
            action='store_true',
            help='Run system diagnostics: check binaries, toolchain, and system libraries'
        )
        binary_mgmt_group.add_argument(
            '--tools_info',
            action='store_true',
            help='Display installed tools information from toolchain metadata'
        )
        
        # Workflow execution options
        workflow_group = self.parser.add_argument_group('Workflow Execution')
        workflow_group.add_argument(
            '--run_workflow',
            action='store_true',
            help='Run the complete SYMFLUENCE workflow (default behavior if no individual steps specified)'
        )
        workflow_group.add_argument(
            '--force_rerun',
            action='store_true',
            help='Force rerun of all steps, overwriting existing outputs'
        )
        workflow_group.add_argument(
            '--stop_on_error',
            action='store_true',
            default=True,
            help='Stop workflow execution on first error (default: True)'
        )
        workflow_group.add_argument(
            '--continue_on_error',
            action='store_true',
            help='Continue workflow execution even if errors occur'
        )
        
        # Workflow Management
        workflow_mgmt_group = self.parser.add_argument_group('Workflow Management')
        workflow_mgmt_group.add_argument(
            '--workflow_status',
            action='store_true',
            help='Show detailed workflow status with step completion and file checks'
        )
        workflow_mgmt_group.add_argument(
            '--resume_from',
            type=str,
            metavar='STEP_NAME',
            help='Resume workflow execution from a specific step'
        )
        workflow_mgmt_group.add_argument(
            '--clean',
            action='store_true',
            help='Clean intermediate files and outputs'
        )
        workflow_mgmt_group.add_argument(
            '--clean_level',
            type=str,
            choices=['intermediate', 'outputs', 'all'],
            default='intermediate',
            help='Level of cleaning: intermediate files only, outputs, or all (default: intermediate)'
        )
        
        # Individual workflow steps
        steps_group = self.parser.add_argument_group('Individual Workflow Steps')
        for step_name, step_info in self.workflow_steps.items():
            steps_group.add_argument(
                f'--{step_name}',
                action='store_true',
                help=step_info['description']
            )
        
        # Pour point setup
        pourpoint_group = self.parser.add_argument_group('Pour Point Setup')
        pourpoint_group.add_argument(
            '--pour_point',
            type=str,
            metavar='LAT/LON',
            help='Set up SYMFLUENCE for a pour point coordinate (format: lat/lon, e.g., 51.1722/-115.5717)'
        )
        pourpoint_group.add_argument(
            '--domain_def',
            type=str,
            choices=self.domain_definition_methods,
            help=f'Domain definition method when using --pour_point. Options: {", ".join(self.domain_definition_methods)}'
        )
        pourpoint_group.add_argument(
            '--domain_name',
            type=str,
            help='Domain name when using --pour_point (required)'
        )
        pourpoint_group.add_argument(
            '--experiment_id',
            type=str,
            help='Override experiment ID in configuration'
        )
        pourpoint_group.add_argument(
            '--bounding_box_coords',
            type=str,
            metavar='LAT_MAX/LON_MIN/LAT_MIN/LON_MAX',
            help='Bounding box coordinates (format: lat_max/lon_min/lat_min/lon_max, e.g., 51.76/-116.55/50.95/-115.5). Default: 1 degree buffer around pour point'
        )
        
        # Analysis and status options
        status_group = self.parser.add_argument_group('Status and Analysis')
        status_group.add_argument(
            '--status',
            action='store_true',
            help='Show current workflow status and exit'
        )
        status_group.add_argument(
            '--list_steps',
            action='store_true',
            help='List all available workflow steps and exit'
        )
        status_group.add_argument(
            '--validate_config',
            action='store_true',
            help='Validate configuration file and exit'
        )
        status_group.add_argument(
            '--dry_run',
            action='store_true',
            help='Show what would be executed without actually running'
        )
        
        # SLURM group
        slurm_group = self.parser.add_argument_group('SLURM Job Submission')
        slurm_group.add_argument(
            '--submit_job',
            action='store_true',
            help='Submit the execution plan as a SLURM job instead of running locally'
        )
        slurm_group.add_argument(
            '--job_name',
            type=str,
            help='SLURM job name (default: auto-generated from domain and steps)'
        )
        slurm_group.add_argument(
            '--job_time',
            type=str,
            default='48:00:00',
            help='SLURM job time limit (default: 48:00:00)'
        )
        slurm_group.add_argument(
            '--job_nodes',
            type=int,
            default=1,
            help='Number of nodes for SLURM job (default: 1)'
        )
        slurm_group.add_argument(
            '--job_ntasks',
            type=int,
            default=1,
            help='Number of tasks for SLURM job (default: 1 for workflow, auto for calibration)'
        )
        slurm_group.add_argument(
            '--job_memory',
            type=str,
            default='50G',
            help='Memory requirement for SLURM job (default: 50G)'
        )
        slurm_group.add_argument(
            '--job_account',
            type=str,
            help='SLURM account to charge job to (required for most systems)'
        )
        slurm_group.add_argument(
            '--job_partition',
            type=str,
            help='SLURM partition/queue to submit to'
        )
        slurm_group.add_argument(
            '--job_modules',
            type=str,
            default='symfluence_modules',
            help='Module to restore in SLURM job (default: symfluence_modules)'
        )
        slurm_group.add_argument(
            '--conda_env',
            type=str,
            default='symfluence',
            help='Conda environment to activate (default: symfluence)'
        )
        slurm_group.add_argument(
            '--submit_and_wait',
            action='store_true',
            help='Submit job and wait for completion (monitors job status)'
        )
        slurm_group.add_argument(
            '--slurm_template',
            type=str,
            help='Custom SLURM script template file to use'
        )

        # Examples group
        examples_group = self.parser.add_argument_group('Examples')
        examples_group.add_argument(
            '--example_notebook',
            type=str,
            metavar='ID',
            help='Open an example notebook (e.g., 1a, 3b) in Jupyter using the root venv'
        )
    
    def _get_examples_text(self) -> str:
        """Generate examples text for help output including new binary management commands."""
        return """
Examples:
# Basic workflow execution
python SYMFLUENCE.py
python SYMFLUENCE.py --config /path/to/config.yaml

# Individual workflow steps
python SYMFLUENCE.py --calibrate_model
python SYMFLUENCE.py --setup_project --create_pour_point --define_domain

# Pour point setup
python SYMFLUENCE.py --pour_point 51.1722/-115.5717 --domain_def delineate --domain_name "MyWatershed"
python SYMFLUENCE.py --pour_point 51.1722/-115.5717 --domain_def delineate --domain_name "Test" --bounding_box_coords 52.0/-116.0/51.0/-115.0

# Configuration management
python SYMFLUENCE.py --list_templates
python SYMFLUENCE.py --update_config my_config.yaml
python SYMFLUENCE.py --validate_environment

# Binary/executable management
python SYMFLUENCE.py --get_executables
python SYMFLUENCE.py --get_executables summa mizuroute
python SYMFLUENCE.py --validate_binaries
python SYMFLUENCE.py --get_executables --force_install

# Workflow management
python SYMFLUENCE.py --workflow_status
python SYMFLUENCE.py --resume_from define_domain
python SYMFLUENCE.py --clean --clean_level intermediate
python SYMFLUENCE.py --clean --clean_level all --dry_run

# Status and validation
python SYMFLUENCE.py --status
python SYMFLUENCE.py --list_steps
python SYMFLUENCE.py --validate_config

# Advanced options
python SYMFLUENCE.py --debug --force_rerun
python SYMFLUENCE.py --dry_run
python SYMFLUENCE.py --continue_on_error

For more information, visit: https://github.com/DarriEy/SYMFLUENCE
    """
    
    # ============================================================================
    # ARGUMENT PARSING AND VALIDATION
    # ============================================================================
    
    def parse_arguments(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command line arguments.
        
        Args:
            args: Optional list of arguments to parse (for testing)
            
        Returns:
            Parsed arguments namespace
        """
        return self.parser.parse_args(args)
    
    def validate_arguments(self, args: argparse.Namespace) -> Tuple[bool, List[str]]:
        """
        Validate parsed arguments for logical consistency.
        
        Args:
            args: Parsed arguments namespace
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check pour point format
        if args.pour_point:
            if not self._validate_coordinates(args.pour_point):
                errors.append(f"Invalid pour point format: {args.pour_point}. Expected format: lat/lon (e.g., 51.1722/-115.5717)")
            
            if not args.domain_def:
                errors.append("--domain_def is required when using --pour_point")
            
            if not args.domain_name:
                errors.append("--domain_name is required when using --pour_point")
            
            # Validate bounding box if provided
            if hasattr(args, 'bounding_box_coords') and args.bounding_box_coords and not self._validate_bounding_box(args.bounding_box_coords):
                errors.append(f"Invalid bounding box format: {args.bounding_box_coords}. Expected format: lat_max/lon_min/lat_min/lon_max")
        
        # Validate resume_from step name
        if args.resume_from:
            if args.resume_from not in self.workflow_steps:
                errors.append(f"Invalid step name for --resume_from: {args.resume_from}. Available steps: {', '.join(self.workflow_steps.keys())}")
        
        # Validate update_config file exists
        if args.update_config:
            config_path = Path(args.update_config)
            if not config_path.exists():
                errors.append(f"Configuration file not found for --update_config: {config_path}")
        
        # Check conflicting options
        if args.stop_on_error and args.continue_on_error:
            errors.append("Cannot specify both --stop_on_error and --continue_on_error")
        
        # Check if operations that don't need config files are being run
        binary_management_ops = (
            (hasattr(args, 'get_executables') and args.get_executables is not None) or
            getattr(args, 'validate_binaries', False) or
            getattr(args, 'doctor', False) or
            getattr(args, 'tools_info', False)
        )
        
        standalone_management_ops = (
            args.list_templates or
            args.validate_environment or
            args.update_config
        )
        
        status_only_ops = (
            args.list_steps or
            (args.validate_config and not args.pour_point)
        )
        
        # Only validate config file if we actually need it
        needs_config_file = not (binary_management_ops or standalone_management_ops or status_only_ops)
        
        if needs_config_file and not args.pour_point:
            config_path = Path(args.config)
            if not config_path.exists():
                errors.append(f"Configuration file not found: {config_path}")
        
        return len(errors) == 0, errors
    
    def _validate_coordinates(self, coord_string: str) -> bool:
        """
        Validate coordinate string format.
        
        Args:
            coord_string: Coordinate string in format "lat/lon"
            
        Returns:
            True if valid format, False otherwise
        """
        try:
            parts = coord_string.split('/')
            if len(parts) != 2:
                return False
            
            lat, lon = float(parts[0]), float(parts[1])
            
            # Basic range validation
            if not (-90 <= lat <= 90):
                return False
            if not (-180 <= lon <= 180):
                return False
            
            return True
        except (ValueError, IndexError):
            return False
    
    def _validate_bounding_box(self, bbox_string: str) -> bool:
        """
        Validate bounding box coordinate string format.
        
        Args:
            bbox_string: Bounding box string in format "lat_max/lon_min/lat_min/lon_max"
            
        Returns:
            True if valid format, False otherwise
        """
        try:
            parts = bbox_string.split('/')
            if len(parts) != 4:
                return False
            
            lat_max, lon_min, lat_min, lon_max = map(float, parts)
            
            # Basic range and logic validation
            if not (-90 <= lat_min <= lat_max <= 90):
                return False
            if not (-180 <= lon_min <= lon_max <= 180):
                return False
            
            return True
        except (ValueError, IndexError):
            return False
    
    # ============================================================================
    # EXECUTION PLAN GENERATION
    # ============================================================================
    
    def get_execution_plan(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Determine what should be executed based on parsed arguments.
        
        Args:
            args: Parsed arguments namespace
            
        Returns:
            Dictionary describing the execution plan
        """
        plan = {
            'mode': 'workflow',
            'steps': [],
            'config_overrides': {},
            'settings': {
                'force_rerun': args.force_rerun,
                'stop_on_error': args.stop_on_error and not args.continue_on_error,
                'debug': args.debug,
                'dry_run': args.dry_run
            }
        }

        # NEW: example notebook open (early return; does not affect any other flags)
        if getattr(args, 'example_notebook', None):
            plan['mode'] = 'example_notebook'
            plan['example_notebook'] = args.example_notebook.strip()
            return plan
        
        # Handle binary management operations
        if (hasattr(args, 'get_executables') and args.get_executables is not None) or \
           getattr(args, 'validate_binaries', False) or \
           getattr(args, 'doctor', False) or \
           getattr(args, 'tools_info', False):
            plan['mode'] = 'binary_management'
            plan['binary_operations'] = {
                'get_executables': getattr(args, 'get_executables', None),
                'validate_binaries': getattr(args, 'validate_binaries', False),
                'force_install': getattr(args, 'force_install', False),
                'doctor': getattr(args, 'doctor', False),
                'tools_info': getattr(args, 'tools_info', False)
            }
            return plan
        
        # Handle management operations
        if (args.list_templates or args.update_config or args.validate_environment or
                args.workflow_status or args.resume_from or args.clean):
            plan['mode'] = 'management'
            plan['management_operations'] = {
                'list_templates': args.list_templates,
                'update_config': args.update_config,
                'validate_environment': args.validate_environment,
                'workflow_status': args.workflow_status,
                'resume_from': args.resume_from,
                'clean': args.clean,
                'clean_level': args.clean_level
            }
            return plan
        
        # Handle status-only operations
        if args.status or args.list_steps or args.validate_config:
            plan['mode'] = 'status_only'
            plan['status_operations'] = {
                'show_status': args.status,
                'list_steps': args.list_steps,
                'validate_config': args.validate_config
            }
            return plan
        
        # Handle pour point setup
        if args.pour_point:
            plan['mode'] = 'pour_point_setup'
            plan['pour_point'] = {
                'coordinates': args.pour_point,
                'domain_definition_method': args.domain_def,
                'domain_name': args.domain_name,
                'bounding_box_coords': args.bounding_box_coords
            }
            return plan
        
        # Handle individual workflow steps
        requested_steps = [
            step_name for step_name in self.workflow_steps.keys()
            if getattr(args, step_name, False)
        ]
        
        if requested_steps:
            plan['mode'] = 'individual_steps'
            plan['steps'] = requested_steps
        else:
            plan['mode'] = 'workflow'
            plan['steps'] = list(self.workflow_steps.keys())
        
        # Handle experiment ID override
        if args.experiment_id:
            plan['config_overrides']['EXPERIMENT_ID'] = args.experiment_id
        
        # Handle SLURM job submission
        if args.submit_job:
            plan['submit_job'] = True
            plan['slurm_options'] = {
                'job_name': args.job_name,
                'job_time': args.job_time,
                'job_nodes': args.job_nodes,
                'job_ntasks': args.job_ntasks,
                'job_memory': args.job_memory,
                'job_account': args.job_account,
                'job_partition': args.job_partition,
                'job_modules': args.job_modules,
                'conda_env': args.conda_env,
                'submit_and_wait': args.submit_and_wait,
                'slurm_template': args.slurm_template
            }
        
        return plan

    def launch_example_notebook(
        self,
        example_id: str,
        repo_root=None,
        venv_candidates=None,
        prefer_lab: bool = True
    ) -> int:
        """Launch an example notebook bound to the repo's root venv."""
        return self.notebook_service.launch_example_notebook(
            example_id=example_id,
            repo_root=repo_root,
            venv_candidates=venv_candidates,
            prefer_lab=prefer_lab
        )
    def apply_config_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply configuration overrides from CLI arguments.
        
        Args:
            config: Original configuration dictionary
            overrides: Override values from CLI
            
        Returns:
            Updated configuration dictionary
        """
        updated_config = config.copy()
        updated_config.update(overrides)
        return updated_config
    
    # ============================================================================
    # EXTERNAL TOOL MANAGEMENT
    # ============================================================================
    
    def get_executables(self, *args, **kwargs):
        """Clone and install external tool repositories."""
        return self.binary_manager.get_executables(*args, **kwargs)

    def validate_binaries(self, *args, **kwargs):
        """Validate that required binary executables exist."""
        return self.binary_manager.validate_binaries(*args, **kwargs)

    def _check_dependencies(self, *args, **kwargs):
        return self.binary_manager._check_dependencies(*args, **kwargs)

    def _verify_installation(self, *args, **kwargs):
        return self.binary_manager._verify_installation(*args, **kwargs)

    def _resolve_tool_dependencies(self, *args, **kwargs):
        return self.binary_manager._resolve_tool_dependencies(*args, **kwargs)

    def _print_installation_summary(self, *args, **kwargs):
        return self.binary_manager._print_installation_summary(*args, **kwargs)

    def _ensure_valid_config_paths(self, *args, **kwargs):
        return self.binary_manager._ensure_valid_config_paths(*args, **kwargs)

    def handle_binary_management(self, *args, **kwargs):
        """Handle binary management operations."""
        return self.binary_manager.handle_binary_management(*args, **kwargs)
    def print_status_information(self, symfluence_instance, operations: Dict[str, bool]) -> None:
        """
        Print various status information based on requested operations.
        
        Args:
            symfluence_instance: SYMFLUENCE system instance
            operations: Dictionary of status operations to perform
        """
        if operations.get('list_steps'):
            self._print_workflow_steps()
        
        if operations.get('validate_config'):
            self._print_config_validation(symfluence_instance)
        
        if operations.get('show_status'):
            self._print_workflow_status(symfluence_instance)
    
    def _print_workflow_steps(self) -> None:
        """Print all available workflow steps."""
        print("\n📋 Available Workflow Steps:")
        print("=" * 50)
        for step_name, step_info in self.workflow_steps.items():
            print(f"--{step_name:<25} {step_info['description']}")
        print("\n💡 Use these flags to run individual steps, e.g.:")
        print("  python SYMFLUENCE.py --setup_project --create_pour_point")
        print()
    
    def _print_config_validation(self, symfluence_instance) -> None:
        """Print configuration validation results."""
        print("\n🔍 Configuration Validation:")
        print("=" * 30)
        
        if hasattr(symfluence_instance, 'workflow_orchestrator'):
            is_valid = symfluence_instance.workflow_orchestrator.validate_workflow_prerequisites()
            if is_valid:
                print("✅ Configuration is valid")
            else:
                print("❌ Configuration validation failed")
                print("Check logs for detailed error information")
        else:
            print("⚠️  Configuration validation not available")
    
    def _print_workflow_status(self, symfluence_instance) -> None:
        """Print current workflow status."""
        print("\n📊 Workflow Status:")
        print("=" * 20)
        
        if hasattr(symfluence_instance, 'get_status'):
            status = symfluence_instance.get_status()
            print(f"🏞️  Domain: {status.get('domain', 'Unknown')}")
            print(f"🧪 Experiment: {status.get('experiment', 'Unknown')}")
            print(f"⚙️  Config Valid: {'✅' if status.get('config_valid', False) else '❌'}")
            print(f"🔧 Managers Initialized: {'✅' if status.get('managers_initialized', False) else '❌'}")
            print(f"📁 Config Path: {status.get('config_path', 'Unknown')}")
            print(f"🐛 Debug Mode: {'✅' if status.get('debug_mode', False) else '❌'}")
            
            if 'workflow_status' in status:
                workflow_status = status['workflow_status']
                print(f"🔄 Workflow Status: {workflow_status}")
        else:
            print("⚠️  Status information not available")
    
    # ============================================================================
    # POUR POINT WORKFLOW
    # ============================================================================
    
    def setup_pour_point_workflow(self, coordinates: str, domain_def_method: str, domain_name: str,
                                   bounding_box_coords: Optional[str] = None,
                                   symfluence_code_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a configuration setup for pour point workflow.
        
        This method:
        1. Loads the config template from SYMFLUENCE_CODE_DIR/0_config_files/config_template.yaml
        2. Updates key settings (pour point, domain name, domain definition method, bounding box)
        3. Saves as config_{domain_name}.yaml
        4. Returns configuration details for workflow execution
        
        Args:
            coordinates: Pour point coordinates in "lat/lon" format
            domain_def_method: Domain definition method to use
            domain_name: Name for the domain/watershed
            bounding_box_coords: Optional bounding box, defaults to 1-degree buffer around pour point
            symfluence_code_dir: Path to SYMFLUENCE code directory
            
        Returns:
            Dictionary with pour point workflow configuration including 'config_file' path
        """
        try:
            print(f"\n🎯 Setting up pour point workflow:")
            print(f"   📍 Coordinates: {coordinates}")
            print(f"   🗺️  Domain Definition Method: {domain_def_method}")
            print(f"   🏞️  Domain Name: {domain_name}")
            
            lat, lon = map(float, coordinates.split('/'))
            
            if not bounding_box_coords:
                lat_max = lat + 0.5
                lat_min = lat - 0.5
                lon_max = lon + 0.5
                lon_min = lon - 0.5
                bounding_box_coords = f"{lat_max}/{lon_min}/{lat_min}/{lon_max}"
                print(f"   📦 Auto-calculated bounding box (1° buffer): {bounding_box_coords}")
            else:
                print(f"   📦 User-provided bounding box: {bounding_box_coords}")
            
            template_path = None
            
            possible_locations = [
                Path("./0_config_files/config_template.yaml"),
                Path("../0_config_files/config_template.yaml"),
                Path("../../0_config_files/config_template.yaml"),
            ]
            
            if symfluence_code_dir:
                possible_locations.insert(0, Path(symfluence_code_dir) / "0_config_files" / "config_template.yaml")
            
            for location in possible_locations:
                if location.exists():
                    template_path = location
                    break
            
            if not template_path:
                raise FileNotFoundError(
                    f"Config template not found. Tried locations: {[str(p) for p in possible_locations]}\n"
                    f"Please ensure you're running from the SYMFLUENCE directory or specify --config with a template path."
                )
            
            print(f"   📄 Loading template from: {template_path}")
            
            with open(template_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config = self._ensure_valid_config_paths(config, template_path)
            
            config['DOMAIN_NAME'] = domain_name
            config['POUR_POINT_COORDS'] = coordinates
            config['DOMAIN_DEFINITION_METHOD'] = domain_def_method
            config['BOUNDING_BOX_COORDS'] = bounding_box_coords
            
            if 'EXPERIMENT_ID' not in config or config['EXPERIMENT_ID'] == 'run_1':
                config['EXPERIMENT_ID'] = 'pour_point_setup'
            
            output_config_path = Path(f"0_config_files/config_{domain_name}.yaml")
            with open(output_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"   💾 Created config file: {output_config_path}")
            print(f"   ✅ Pour point workflow setup complete!")
            print(f"\n🚀 Next steps:")
            print(f"   1. Review the generated config file: {output_config_path}")
            print(f"   2. SYMFLUENCE will now use this config to run the pour point workflow")
            print(f"   3. Essential steps (setup_project, create_pour_point, define_domain, discretize_domain) will be executed")
            
            return {
                'config_file': str(output_config_path.resolve()),
                'coordinates': coordinates,
                'domain_name': domain_name,
                'domain_definition_method': domain_def_method,
                'bounding_box_coords': bounding_box_coords,
                'template_used': str(template_path),
                'setup_steps': [
                    'setup_project',
                    'create_pour_point',
                    'define_domain',
                    'discretize_domain'
                ]
            }
        
        except Exception as e:
            print(f"❌ Error setting up pour point workflow: {str(e)}")
            raise
    
    # ============================================================================
    # SLURM JOB SUBMISSION
    # ============================================================================
    
    def submit_slurm_job(self, *args, **kwargs):
        """Submit a SLURM job for the execution plan."""
        return self.job_scheduler.submit_slurm_job(*args, **kwargs)

    def handle_slurm_job_submission(self, *args, **kwargs):
        """Handle SLURM job submission workflow."""
        return self.job_scheduler.handle_slurm_job_submission(*args, **kwargs)

    def detect_environment(self, *args, **kwargs):
        """Detect whether we're running on HPC or a personal computer."""
        return self.job_scheduler.detect_environment(*args, **kwargs)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_cli_manager() -> CLIArgumentManager:
    """
    Factory function to create a CLI argument manager instance.
    
    Returns:
        Configured CLIArgumentManager instance
    """
    return CLIArgumentManager()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """Test the CLI argument manager independently."""
    cli_manager = CLIArgumentManager()
    
    test_args = ['--calibrate_model', '--debug']
    args = cli_manager.parse_arguments(test_args)
    
    plan = cli_manager.get_execution_plan(args)
    
    print("Test execution plan:")
    print(f"Mode: {plan['mode']}")
    print(f"Steps: {plan['steps']}")
    print(f"Settings: {plan['settings']}")