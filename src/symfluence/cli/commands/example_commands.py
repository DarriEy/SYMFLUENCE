"""
Example notebook command handlers for SYMFLUENCE CLI.

This module implements handlers for launching and managing example Jupyter notebooks.
"""

import sys
from argparse import Namespace

from .base import BaseCommand


class ExampleCommands(BaseCommand):
    """Handlers for example notebook commands."""

    @staticmethod
    def launch(args: Namespace) -> int:
        """
        Execute: symfluence example launch EXAMPLE_ID

        Launch an example Jupyter notebook.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.cli.notebook_service import NotebookService

            notebook_service = NotebookService()

            example_id = args.example_id
            prefer_lab = not getattr(args, 'notebook', False)  # Default to lab unless --notebook specified

            BaseCommand.print_info(f"📓 Launching example notebook: {example_id}")
            if prefer_lab:
                BaseCommand.print_info("   Using JupyterLab")
            else:
                BaseCommand.print_info("   Using classic Jupyter Notebook")

            # Launch the notebook
            success = notebook_service.launch_example_notebook(
                example_id=example_id,
                prefer_lab=prefer_lab
            )

            if success:
                return 0
            else:
                BaseCommand.print_error("Failed to launch notebook")
                return 1

        except Exception as e:
            BaseCommand.print_error(f"Failed to launch notebook: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def list_examples(args: Namespace) -> int:
        """
        Execute: symfluence example list

        List available example notebooks.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from pathlib import Path

            # Look for example notebooks
            examples_dir = Path(__file__).parent.parent.parent.parent.parent / 'examples'

            BaseCommand.print_info("Available example notebooks:")
            BaseCommand.print_info("=" * 70)

            if examples_dir.exists():
                # Find all notebook files
                notebooks = sorted(examples_dir.rglob('*.ipynb'))

                if notebooks:
                    for i, notebook in enumerate(notebooks, 1):
                        # Extract example ID from path
                        rel_path = notebook.relative_to(examples_dir)
                        BaseCommand.print_info(f"{i:2}. {rel_path}")
                    BaseCommand.print_info("=" * 70)
                    BaseCommand.print_info(f"Total: {len(notebooks)} notebooks")
                else:
                    BaseCommand.print_info("No example notebooks found")
            else:
                BaseCommand.print_info(f"Examples directory not found: {examples_dir}")

            return 0

        except Exception as e:
            BaseCommand.print_error(f"Failed to list examples: {e}")
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
            BaseCommand.print_error("No example action specified")
            return 1
