"""
Project command handlers for SYMFLUENCE CLI.

This module implements handlers for the project command category,
including initialization and pour point setup.
"""

import sys
from argparse import Namespace
from pathlib import Path

from .base import BaseCommand
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
            from symfluence.cli.initialization_manager import InitializationManager

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

            BaseCommand.print_info("🌱 Initializing SYMFLUENCE project...")

            # Call initialization manager
            success = init_manager.generate_config(
                preset_name=preset_name,
                cli_overrides=cli_overrides,
                output_dir=output_dir,
                scaffold=scaffold,
                minimal=minimal,
                comprehensive=comprehensive
            )

            if success:
                BaseCommand.print_success("Project initialized successfully")
                return 0
            else:
                BaseCommand.print_error("Project initialization failed")
                return 1

        except Exception as e:
            BaseCommand.print_error(f"Initialization failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return 1

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
            # Validate coordinates
            is_valid, error_msg = validate_coordinates(args.coordinates)
            if not is_valid:
                BaseCommand.print_error(f"Invalid coordinates: {error_msg}")
                return 1

            # Validate bounding box if provided
            if hasattr(args, 'bounding_box_coords') and args.bounding_box_coords:
                is_valid, error_msg = validate_bounding_box(args.bounding_box_coords)
                if not is_valid:
                    BaseCommand.print_error(f"Invalid bounding box: {error_msg}")
                    return 1

            BaseCommand.print_info("📍 Setting up pour point workflow...")
            BaseCommand.print_info(f"   Coordinates: {args.coordinates}")
            BaseCommand.print_info(f"   Domain name: {args.domain_name}")
            BaseCommand.print_info(f"   Definition method: {args.domain_def}")

            # TODO: Refactor pour point setup to not depend on old CLIArgumentManager
            # For now, import from the archived version
            import sys
            import importlib.util
            from pathlib import Path

            old_cli_path = Path(__file__).parent.parent / "cli_argument_manager.py.old"
            spec = importlib.util.spec_from_file_location("cli_argument_manager_old", old_cli_path)
            cli_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cli_module)

            cli_manager = cli_module.CLIArgumentManager()

            # Call the pour point setup
            cli_manager.setup_pour_point_workflow(
                coordinates=args.coordinates,
                domain_def_method=args.domain_def,
                domain_name=args.domain_name,
                bounding_box_coords=getattr(args, 'bounding_box_coords', None),
                symfluence_code_dir=None,
                experiment_id=getattr(args, 'experiment_id', None)
            )

            BaseCommand.print_success("Pour point workflow setup completed")
            return 0

        except Exception as e:
            BaseCommand.print_error(f"Pour point setup failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return 1

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
            from symfluence.cli.initialization_manager import InitializationManager

            init_manager = InitializationManager()

            BaseCommand.print_info("Available initialization presets:")
            BaseCommand.print_info("=" * 70)

            # List presets
            presets = init_manager.list_presets()
            if presets:
                for i, preset_info in enumerate(presets, 1):
                    if isinstance(preset_info, dict):
                        name = preset_info.get('name', 'Unknown')
                        description = preset_info.get('description', 'No description')
                        BaseCommand.print_info(f"{i:2}. {name:20s} - {description}")
                    else:
                        BaseCommand.print_info(f"{i:2}. {preset_info}")
                BaseCommand.print_info("=" * 70)
                BaseCommand.print_info(f"Total: {len(presets)} presets")
            else:
                BaseCommand.print_info("No presets found")

            return 0

        except Exception as e:
            BaseCommand.print_error(f"Failed to list presets: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return 1

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
            from symfluence.cli.initialization_manager import InitializationManager

            init_manager = InitializationManager()

            preset_name = args.preset_name

            BaseCommand.print_info(f"Preset: {preset_name}")
            BaseCommand.print_info("=" * 70)

            # Show preset details
            preset_info = init_manager.show_preset(preset_name)
            if preset_info:
                if isinstance(preset_info, dict):
                    for key, value in preset_info.items():
                        BaseCommand.print_info(f"{key:25s}: {value}")
                else:
                    BaseCommand.print_info(str(preset_info))
            else:
                BaseCommand.print_error(f"Preset '{preset_name}' not found")
                return 1

            return 0

        except Exception as e:
            BaseCommand.print_error(f"Failed to show preset: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return 1

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
            BaseCommand.print_error("No project action specified")
            return 1
