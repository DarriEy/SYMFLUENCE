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
            from symfluence.utils.cli.initialization_manager import InitializationManager

            init_manager = InitializationManager()

            # Build initialization operations dict
            preset_name = args.preset if args.preset else None

            # Validate required parameters if no preset is given
            if not preset_name:
                if not args.domain:
                    import sys
                    sys.stderr.write("Error: --domain is required when not using a preset\n")
                    return 2
                if not args.model:
                    import sys
                    sys.stderr.write("Error: --model is required when not using a preset\n")
                    return 2

            # Validate preset if given
            if preset_name:
                from symfluence.utils.cli.init_presets import get_preset
                try:
                    get_preset(preset_name)
                except ValueError:
                    import sys
                    sys.stderr.write(f"Error: Unknown preset '{preset_name}'\n")
                    return 2

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

            # Generate config
            config = init_manager.generate_config(
                preset_name=preset_name,
                cli_overrides=cli_overrides,
                minimal=minimal,
                comprehensive=comprehensive
            )

            # Determine domain name for output file
            domain_name = config.get('DOMAIN_NAME', 'unnamed_domain')
            output_path = Path(output_dir) / f'config_{domain_name}.yaml'

            # Write config file
            written_path = init_manager.write_config(config, output_path)
            BaseCommand.print_success(f"✅ Created config file: {written_path}")

            # Create scaffold if requested
            if scaffold:
                try:
                    domain_dir = init_manager.create_scaffold(config)
                    BaseCommand.print_success(f"📁 Created project structure: {domain_dir}")
                except Exception as e:
                    BaseCommand.print_warning(f"Failed to create scaffold: {e}")

            # Show next steps
            BaseCommand.print_info("\n📁 To create project structure, run:")
            BaseCommand.print_info(f"   symfluence workflow step setup_project --config {written_path}")
            BaseCommand.print_info("\n✨ Next steps:")
            BaseCommand.print_info("   1. Review and customize the configuration file")
            BaseCommand.print_info("   2. Run setup_project to create directory structure")
            BaseCommand.print_info("   3. Use 'symfluence workflow steps' to see available workflow commands")

            return 0

        except Exception as e:
            BaseCommand.print_error(f"Initialization failed: {e}")
            if args.debug:
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

            from symfluence.utils.project.pour_point_workflow import setup_pour_point_workflow

            result = setup_pour_point_workflow(
                coordinates=args.coordinates,
                domain_def_method=args.domain_def,
                domain_name=args.domain_name,
                bounding_box_coords=getattr(args, 'bounding_box_coords', None),
            )

            if result.used_auto_bounding_box:
                BaseCommand.print_info(
                    f"Auto-calculated bounding box (1-degree buffer): {result.bounding_box_coords}"
                )
            else:
                BaseCommand.print_info(
                    f"User-provided bounding box: {result.bounding_box_coords}"
                )

            BaseCommand.print_success(f"Created config file: {result.config_file}")
            BaseCommand.print_info("Next steps:")
            BaseCommand.print_info(f"  1. Review the generated config file: {result.config_file}")
            BaseCommand.print_info("  2. Run the pour point workflow steps:")
            BaseCommand.print_info(
                "     symfluence workflow step setup_project create_pour_point define_domain discretize_domain "
                f"--config {result.config_file}"
            )
            BaseCommand.print_success("Pour point workflow setup completed")
            return 0

        except Exception as e:
            BaseCommand.print_error(f"Pour point setup failed: {e}")
            if args.debug:
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
            from symfluence.utils.cli.initialization_manager import InitializationManager

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
            if args.debug:
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
            from symfluence.utils.cli.initialization_manager import InitializationManager

            init_manager = InitializationManager()

            preset_name = args.preset_name

            # Show preset details (prints directly, handles errors internally)
            init_manager.show_preset(preset_name)

            return 0

        except Exception as e:
            BaseCommand.print_error(f"Failed to show preset: {e}")
            if args.debug:
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
