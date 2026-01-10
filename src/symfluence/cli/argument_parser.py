"""
SYMFLUENCE CLI Argument Parser

This module provides the main CLI parser with subcommand structure for SYMFLUENCE.
"""

import argparse
from typing import Optional, List

try:
    from symfluence.symfluence_version import __version__
except Exception:
    __version__ = "0+unknown"

# Workflow steps available for individual execution
WORKFLOW_STEPS = [
    'setup_project',
    'create_pour_point',
    'acquire_attributes',
    'define_domain',
    'discretize_domain',
    'process_observed_data',
    'acquire_forcings',
    'model_agnostic_preprocessing',
    'model_specific_preprocessing',
    'run_model',
    'calibrate_model',
    'run_emulation',
    'run_benchmarking',
    'run_decision_analysis',
    'run_sensitivity_analysis',
    'postprocess_results'
]

# Domain definition methods
DOMAIN_DEFINITION_METHODS = ['lumped', 'point', 'subset', 'delineate']

# Available tools for binary installation
EXTERNAL_TOOLS = ['summa', 'mizuroute', 'fuse', 'hype', 'mesh', 'taudem', 'gistool', 'datatool', 'rhessys', 'ngen', 'ngiab', 'sundials']

# Hydrological models
MODELS = ['SUMMA', 'FUSE', 'GR', 'HYPE', 'MESH']


class CLIParser:
    """Main CLI parser with subcommand architecture."""

    def __init__(self):
        """Initialize the CLI parser."""
        self.common_parser = self._create_common_parser()
        self.parser = self._create_parser()

    def _create_common_parser(self) -> argparse.ArgumentParser:
        """Create a parent parser with common arguments."""
        # Use SUPPRESS to avoid overwriting global flags with subcommand defaults
        parser = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
        
        # Global options available to all commands
        parser.add_argument('--config', type=str,
                          help='Path to configuration file (default: ./config.yaml)')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug output')
        parser.add_argument('--visualise', '--visualize', action='store_true', dest='visualise',
                          help='Enable visualization during execution')
        parser.add_argument('--dry-run', action='store_true', dest='dry_run',
                          help='Show what would be executed without running')
        return parser

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main parser with global options and subparsers."""
        parser = argparse.ArgumentParser(
            prog='symfluence',
            description='SYMFLUENCE - Hydrological Modeling Framework',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[self.common_parser],
            epilog="""
Examples:
  symfluence workflow run --config my_config.yaml
  symfluence workflow step calibrate_model
  symfluence project init fuse-provo --scaffold
  symfluence binary install summa mizuroute
  symfluence binary doctor
  symfluence project pour-point 51.1722/-115.5717 --domain-name Bow --definition delineate

For more help on a specific command:
  symfluence <category> --help
  symfluence <category> <action> --help
"""
        )

        parser.add_argument('--version', action='version',
                          version=f'SYMFLUENCE {__version__}')

        # Create subparsers for command categories
        subparsers = parser.add_subparsers(
            dest='category',
            required=True,
            help='Command category',
            metavar='<category>'
        )

        # Register all category commands
        self._register_workflow_commands(subparsers)
        self._register_project_commands(subparsers)
        self._register_binary_commands(subparsers)
        self._register_config_commands(subparsers)
        self._register_job_commands(subparsers)
        self._register_example_commands(subparsers)
        self._register_agent_commands(subparsers)

        return parser

    def _register_workflow_commands(self, subparsers):
        """Register workflow category commands."""
        from .commands import WorkflowCommands

        workflow_parser = subparsers.add_parser(
            'workflow',
            help='Workflow execution and management',
            description='Execute and manage SYMFLUENCE workflows'
        )
        workflow_subparsers = workflow_parser.add_subparsers(
            dest='action',
            required=True,
            help='Workflow action',
            metavar='<action>'
        )

        # workflow run
        run_parser = workflow_subparsers.add_parser(
            'run',
            help='Run complete workflow',
            parents=[self.common_parser]
        )
        run_parser.add_argument('--force-rerun', action='store_true', dest='force_rerun',
                              help='Force rerun of all steps')
        run_parser.add_argument('--continue-on-error', action='store_true', dest='continue_on_error',
                              help='Continue executing steps even if errors occur')
        run_parser.set_defaults(func=WorkflowCommands.run)

        # workflow step
        step_parser = workflow_subparsers.add_parser(
            'step',
            help='Run a single workflow step',
            parents=[self.common_parser]
        )
        step_parser.add_argument('step_name', choices=WORKFLOW_STEPS, metavar='STEP_NAME',
                               help=f'Step to execute. Choices: {", ".join(WORKFLOW_STEPS)}')
        step_parser.add_argument('--force-rerun', action='store_true', dest='force_rerun',
                               help='Force rerun of this step')
        step_parser.set_defaults(func=WorkflowCommands.run_step)

        # workflow steps (multiple)
        steps_parser = workflow_subparsers.add_parser(
            'steps',
            help='Run multiple workflow steps',
            parents=[self.common_parser]
        )
        steps_parser.add_argument('step_names', nargs='+', choices=WORKFLOW_STEPS, metavar='STEP_NAME',
                                help=f'Steps to execute in order. Choices: {", ".join(WORKFLOW_STEPS)}')
        steps_parser.add_argument('--force-rerun', action='store_true', dest='force_rerun',
                                help='Force rerun of these steps')
        steps_parser.set_defaults(func=WorkflowCommands.run_steps)

        # workflow status
        status_parser = workflow_subparsers.add_parser(
            'status',
            help='Show workflow execution status',
            parents=[self.common_parser]
        )
        status_parser.set_defaults(func=WorkflowCommands.status)

        # workflow validate
        validate_parser = workflow_subparsers.add_parser(
            'validate',
            help='Validate configuration file',
            parents=[self.common_parser]
        )
        validate_parser.set_defaults(func=WorkflowCommands.validate)

        # workflow list-steps
        list_steps_parser = workflow_subparsers.add_parser(
            'list-steps',
            help='List available workflow steps'
        )
        list_steps_parser.set_defaults(func=WorkflowCommands.list_steps)

        # workflow resume
        resume_parser = workflow_subparsers.add_parser(
            'resume',
            help='Resume workflow from a specific step',
            parents=[self.common_parser]
        )
        resume_parser.add_argument('step_name', choices=WORKFLOW_STEPS, metavar='STEP_NAME',
                                 help=f'Step to resume from. Choices: {", ".join(WORKFLOW_STEPS)}')
        resume_parser.add_argument('--force-rerun', action='store_true', dest='force_rerun',
                                 help='Force rerun from this step')
        resume_parser.set_defaults(func=WorkflowCommands.resume)

        # workflow clean
        clean_parser = workflow_subparsers.add_parser(
            'clean',
            help='Clean intermediate or output files',
            parents=[self.common_parser]
        )
        clean_parser.add_argument('--level', choices=['intermediate', 'outputs', 'all'],
                                default='intermediate',
                                help='Cleaning level (default: intermediate)')
        clean_parser.set_defaults(func=WorkflowCommands.clean)

    def _register_project_commands(self, subparsers):
        """Register project category commands."""
        from .commands import ProjectCommands

        project_parser = subparsers.add_parser(
            'project',
            help='Project initialization and setup',
            description='Initialize projects and configure pour points'
        )
        project_subparsers = project_parser.add_subparsers(
            dest='action',
            required=True,
            help='Project action',
            metavar='<action>'
        )

        # project init
        init_parser = project_subparsers.add_parser(
            'init',
            help='Initialize a new project'
        )
        init_parser.add_argument('preset', nargs='?', default=None,
                               help='Preset name to use (optional)')
        init_parser.add_argument('--domain', type=str,
                               help='Domain name')
        init_parser.add_argument('--model', choices=MODELS,
                               help=f'Hydrological model. Choices: {", ".join(MODELS)}')
        init_parser.add_argument('--start-date', dest='start_date', type=str,
                               help='Start date (YYYY-MM-DD)')
        init_parser.add_argument('--end-date', dest='end_date', type=str,
                               help='End date (YYYY-MM-DD)')
        init_parser.add_argument('--forcing', type=str,
                               help='Forcing dataset')
        init_parser.add_argument('--discretization', type=str,
                               help='Discretization method')
        init_parser.add_argument('--definition-method', dest='definition_method', type=str,
                               help='Domain definition method')
        init_parser.add_argument('--output-dir', dest='output_dir', type=str,
                               default='./',
                               help='Output directory for config file (default: ./)')
        init_parser.add_argument('--scaffold', action='store_true',
                               help='Create full directory structure')
        init_parser.add_argument('--minimal', action='store_true',
                               help='Create minimal configuration')
        init_parser.add_argument('--comprehensive', action='store_true',
                               help='Create comprehensive configuration (default)')
        init_parser.set_defaults(func=ProjectCommands.init)

        # project pour-point
        pour_point_parser = project_subparsers.add_parser(
            'pour-point',
            help='Set up pour point workflow'
        )
        pour_point_parser.add_argument('coordinates', type=str,
                                      help='Pour point coordinates in format lat/lon (e.g., 51.1722/-115.5717)')
        pour_point_parser.add_argument('--domain-name', dest='domain_name', type=str, required=True,
                                      help='Domain name (required)')
        pour_point_parser.add_argument('--definition', dest='domain_def',
                                      choices=DOMAIN_DEFINITION_METHODS, required=True,
                                      help=f'Domain definition method. Choices: {", ".join(DOMAIN_DEFINITION_METHODS)}')
        pour_point_parser.add_argument('--bounding-box', dest='bounding_box_coords', type=str,
                                      help='Bounding box in format lat_max/lon_min/lat_min/lon_max')
        pour_point_parser.add_argument('--experiment-id', dest='experiment_id', type=str,
                                      help='Override experiment ID')
        pour_point_parser.set_defaults(func=ProjectCommands.pour_point)

        # project list-presets
        list_presets_parser = project_subparsers.add_parser(
            'list-presets',
            help='List available initialization presets'
        )
        list_presets_parser.set_defaults(func=ProjectCommands.list_presets)

        # project show-preset
        show_preset_parser = project_subparsers.add_parser(
            'show-preset',
            help='Show details of a specific preset'
        )
        show_preset_parser.add_argument('preset_name', type=str,
                                       help='Name of preset to display')
        show_preset_parser.set_defaults(func=ProjectCommands.show_preset)

    def _register_binary_commands(self, subparsers):
        """Register binary/tool management commands."""
        from .commands import BinaryCommands

        binary_parser = subparsers.add_parser(
            'binary',
            help='External tool management',
            description='Install, validate, and manage external tools'
        )
        binary_subparsers = binary_parser.add_subparsers(
            dest='action',
            required=True,
            help='Binary action',
            metavar='<action>'
        )

        # binary install
        install_parser = binary_subparsers.add_parser(
            'install',
            help='Install external tools'
        )
        install_parser.add_argument('tools', nargs='*', metavar='TOOL',
                                  help=f'Tools to install. If not specified, installs all. Choices: {", ".join(EXTERNAL_TOOLS)}')
        install_parser.add_argument('--force', action='store_true',
                                  help='Force reinstall even if already installed')
        install_parser.set_defaults(func=BinaryCommands.install)

        # binary validate
        validate_parser = binary_subparsers.add_parser(
            'validate',
            help='Validate installed binaries'
        )
        validate_parser.add_argument('--verbose', action='store_true',
                                   help='Show detailed validation output')
        validate_parser.set_defaults(func=BinaryCommands.validate)

        # binary doctor
        doctor_parser = binary_subparsers.add_parser(
            'doctor',
            help='Run system diagnostics'
        )
        doctor_parser.set_defaults(func=BinaryCommands.doctor)

        # binary info
        info_parser = binary_subparsers.add_parser(
            'info',
            help='Display information about installed tools'
        )
        info_parser.set_defaults(func=BinaryCommands.info)

    def _register_config_commands(self, subparsers):
        """Register configuration management commands."""
        from .commands import ConfigCommands

        config_parser = subparsers.add_parser(
            'config',
            help='Configuration management',
            description='Manage and validate configuration files'
        )
        config_subparsers = config_parser.add_subparsers(
            dest='action',
            required=True,
            help='Config action',
            metavar='<action>'
        )

        # config list-templates
        list_templates_parser = config_subparsers.add_parser(
            'list-templates',
            help='List available configuration templates'
        )
        list_templates_parser.set_defaults(func=ConfigCommands.list_templates)

        # config update
        update_parser = config_subparsers.add_parser(
            'update',
            help='Update an existing configuration file'
        )
        update_parser.add_argument('config_file', type=str,
                                  help='Configuration file to update')
        update_parser.add_argument('--interactive', action='store_true',
                                  help='Interactive update mode')
        update_parser.set_defaults(func=ConfigCommands.update)

        # config validate
        validate_parser = config_subparsers.add_parser(
            'validate',
            help='Validate configuration file syntax'
        )
        validate_parser.set_defaults(func=ConfigCommands.validate)

        # config validate-env
        validate_env_parser = config_subparsers.add_parser(
            'validate-env',
            help='Validate system environment'
        )
        validate_env_parser.set_defaults(func=ConfigCommands.validate_env)

    def _register_job_commands(self, subparsers):
        """Register SLURM job submission commands."""
        from .commands import JobCommands

        job_parser = subparsers.add_parser(
            'job',
            help='SLURM job submission',
            description='Submit workflow commands as SLURM jobs'
        )
        job_subparsers = job_parser.add_subparsers(
            dest='action',
            required=True,
            help='Job action',
            metavar='<action>'
        )

        # job submit
        submit_parser = job_subparsers.add_parser(
            'submit',
            help='Submit workflow as SLURM job'
        )
        submit_parser.add_argument('--name', dest='job_name', type=str,
                                  help='SLURM job name')
        submit_parser.add_argument('--time', dest='job_time', type=str, default='48:00:00',
                                  help='Time limit (default: 48:00:00)')
        submit_parser.add_argument('--nodes', dest='job_nodes', type=int, default=1,
                                  help='Number of nodes (default: 1)')
        submit_parser.add_argument('--tasks', dest='job_ntasks', type=int, default=1,
                                  help='Number of tasks (default: 1)')
        submit_parser.add_argument('--memory', dest='job_memory', type=str, default='50G',
                                  help='Memory requirement (default: 50G)')
        submit_parser.add_argument('--account', dest='job_account', type=str,
                                  help='Account to charge')
        submit_parser.add_argument('--partition', dest='job_partition', type=str,
                                  help='Partition/queue name')
        submit_parser.add_argument('--modules', dest='job_modules', type=str, default='symfluence_modules',
                                  help='Module to restore (default: symfluence_modules)')
        submit_parser.add_argument('--conda-env', dest='conda_env', type=str, default='symfluence',
                                  help='Conda environment (default: symfluence)')
        submit_parser.add_argument('--wait', dest='submit_and_wait', action='store_true',
                                  help='Submit and monitor job until completion')
        submit_parser.add_argument('--template', dest='slurm_template', type=str,
                                  help='Custom SLURM template file')
        submit_parser.add_argument('workflow_args', nargs=argparse.REMAINDER,
                                  help='Workflow command and arguments to submit')
        submit_parser.set_defaults(func=JobCommands.submit)

    def _register_example_commands(self, subparsers):
        """Register example notebook commands."""
        from .commands import ExampleCommands

        example_parser = subparsers.add_parser(
            'example',
            help='Example notebooks',
            description='Launch and manage example Jupyter notebooks'
        )
        example_subparsers = example_parser.add_subparsers(
            dest='action',
            required=True,
            help='Example action',
            metavar='<action>'
        )

        # example launch
        launch_parser = example_subparsers.add_parser(
            'launch',
            help='Launch an example notebook'
        )
        launch_parser.add_argument('example_id', type=str,
                                  help='Example ID (e.g., 1a, 2b, 3c)')
        launch_parser.add_argument('--lab', action='store_true',
                                  help='Launch in JupyterLab (default)')
        launch_parser.add_argument('--notebook', action='store_true',
                                  help='Launch in classic Jupyter Notebook')
        launch_parser.set_defaults(func=ExampleCommands.launch)

        # example list
        list_parser = example_subparsers.add_parser(
            'list',
            help='List available example notebooks'
        )
        list_parser.set_defaults(func=ExampleCommands.list_examples)

    def _register_agent_commands(self, subparsers):
        """Register AI agent commands."""
        from .commands import AgentCommands

        agent_parser = subparsers.add_parser(
            'agent',
            help='AI agent interface',
            description='Interactive AI agent for SYMFLUENCE'
        )
        agent_subparsers = agent_parser.add_subparsers(
            dest='action',
            required=True,
            help='Agent action',
            metavar='<action>'
        )

        # agent start
        start_parser = agent_subparsers.add_parser(
            'start',
            help='Start interactive agent mode'
        )
        start_parser.add_argument('--verbose', action='store_true',
                                help='Show verbose agent output')
        start_parser.set_defaults(func=AgentCommands.start)

        # agent run
        run_parser = agent_subparsers.add_parser(
            'run',
            help='Execute a single agent prompt'
        )
        run_parser.add_argument('prompt', type=str,
                              help='Prompt to execute')
        run_parser.add_argument('--verbose', action='store_true',
                              help='Show verbose agent output')
        run_parser.set_defaults(func=AgentCommands.run)

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Args:
            args: List of argument strings (for testing). If None, uses sys.argv.

        Returns:
            Parsed arguments namespace
        """
        return self.parser.parse_args(args)
