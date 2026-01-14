"""
Project command handlers for SYMFLUENCE CLI.

This module implements handlers for the project command category,
including initialization and pour point setup.
"""

from argparse import Namespace
from pathlib import Path

from .base import BaseCommand
from ..exit_codes import ExitCode
from ..validators import validate_coordinates, validate_bounding_box


class ProjectCommands(BaseCommand):
    """Handlers for project category commands."""

    @staticmethod
    def init(args: Namespace) -> int:
        """
        Execute: symfluence project init [PRESET]

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Import initialization manager
            from symfluence.cli.services import InitializationManager

            init_manager = InitializationManager()

            # Build initialization operations dict
            preset_name = args.preset if args.preset else None

            cli_overrides = {
                'domain': args.domain,
                'model': args.model,
                'start_date': args.start_date,
                'end_date': args.end_date,
                'forcing': args.forcing,
                'discretization': args.discretization,
                'definition_method': args.definition_method,
            }

            # Remove None values
            cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

            output_dir = args.output_dir if args.output_dir else './0_config_files/'
            scaffold = args.scaffold
            minimal = args.minimal
            comprehensive = args.comprehensive if hasattr(args, 'comprehensive') else True

            BaseCommand._console.info("Initializing SYMFLUENCE project...")

            # Call initialization manager
            config = init_manager.generate_config(
                preset_name=preset_name,
                cli_overrides=cli_overrides,
                minimal=minimal,
                comprehensive=comprehensive
            )

            # 2. Determine output path
            domain_name = config.get("DOMAIN_NAME", "unnamed_project")
            output_dir_path = Path(output_dir)
            output_file = output_dir_path / f"config_{domain_name}.yaml"

            # 3. Write config file
            written_path = init_manager.write_config(config, output_file)
            BaseCommand._console.success(f"✅ Created config file: {written_path}")

            # 4. Create scaffold if requested
            if scaffold:
                BaseCommand._console.info("Creating project scaffold...")
                domain_dir = init_manager.create_scaffold(config)
                BaseCommand._console.success(f"✅ Created project structure at: {domain_dir}")
            else:
                BaseCommand._console.info(f"📁 To create project structure, run: symfluence setup_project --config {written_path}")

            return ExitCode.SUCCESS

        except ValueError as e:
            BaseCommand._console.error(str(e))
            return ExitCode.USAGE_ERROR
        except Exception as e:
            BaseCommand._console.error(f"Initialization failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.GENERAL_ERROR

    @staticmethod
    def pour_point(args: Namespace) -> int:
        """
        Execute: symfluence project pour-point LAT/LON

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Validate coordinates using Result pattern
            coord_result = validate_coordinates(args.coordinates)
            if coord_result.is_err:
                error = coord_result.first_error()
                BaseCommand._console.error(f"Invalid coordinates: {error.message if error else 'validation failed'}")
                return ExitCode.VALIDATION_ERROR

            # Validate bounding box if provided
            if hasattr(args, 'bounding_box_coords') and args.bounding_box_coords:
                bbox_result = validate_bounding_box(args.bounding_box_coords)
                if bbox_result.is_err:
                    error = bbox_result.first_error()
                    BaseCommand._console.error(f"Invalid bounding box: {error.message if error else 'validation failed'}")
                    return ExitCode.VALIDATION_ERROR

            BaseCommand._console.info("Setting up pour point workflow...")
            BaseCommand._console.indent(f"Coordinates: {args.coordinates}")
            BaseCommand._console.indent(f"Domain name: {args.domain_name}")
            BaseCommand._console.indent(f"Definition method: {args.domain_def}")

            from symfluence.project.pour_point_workflow import setup_pour_point_workflow

            # Get output directory from args or use default
            output_dir = Path(getattr(args, 'output_dir', '.'))

            result = setup_pour_point_workflow(
                coordinates=args.coordinates,
                domain_def_method=args.domain_def,
                domain_name=args.domain_name,
                bounding_box_coords=getattr(args, 'bounding_box_coords', None),
                output_dir=output_dir,
            )

            BaseCommand._console.success("Pour point workflow setup completed")
            BaseCommand._console.indent(f"Config file: {result.config_file}")
            if result.used_auto_bounding_box:
                BaseCommand._console.indent(f"Auto-generated bounding box: {result.bounding_box_coords}")
            return ExitCode.SUCCESS

        except Exception as e:
            BaseCommand._console.error(f"Pour point setup failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.GENERAL_ERROR

    @staticmethod
    def list_presets(args: Namespace) -> int:
        """
        Execute: symfluence project list-presets

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.cli.services import InitializationManager

            init_manager = InitializationManager()

            # InitializationManager handles all output formatting
            init_manager.list_presets()

            return ExitCode.SUCCESS

        except Exception as e:
            BaseCommand._console.error(f"Failed to list presets: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.GENERAL_ERROR

    @staticmethod
    def show_preset(args: Namespace) -> int:
        """
        Execute: symfluence project show-preset PRESET_NAME

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.cli.services import InitializationManager

            init_manager = InitializationManager()

            preset_name = args.preset_name

            # InitializationManager handles all output formatting
            preset_info = init_manager.show_preset(preset_name)

            if preset_info:
                return ExitCode.SUCCESS
            else:
                # Error already printed by manager if invalid
                return ExitCode.FILE_NOT_FOUND

        except Exception as e:
            BaseCommand._console.error(f"Failed to show preset: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return ExitCode.GENERAL_ERROR

    @staticmethod
    def execute(args: Namespace) -> int:
        """
        Main execution dispatcher (required by BaseCommand).

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code
        """
        if hasattr(args, 'func'):
            return args.func(args)
        else:
            BaseCommand._console.error("No project action specified")
            return ExitCode.USAGE_ERROR
