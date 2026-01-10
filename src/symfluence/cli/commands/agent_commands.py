"""
AI agent command handlers for SYMFLUENCE CLI.

This module implements handlers for the AI agent interface.
"""

import sys
from argparse import Namespace

from .base import BaseCommand


class AgentCommands(BaseCommand):
    """Handlers for AI agent commands."""

    @staticmethod
    def start(args: Namespace) -> int:
        """
        Execute: symfluence agent start

        Start interactive AI agent mode.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.agent.agent_manager import AgentManager

            verbose = getattr(args, 'verbose', False)
            config_path = BaseCommand.get_config_path(args)

            BaseCommand.print_info("🤖 Starting interactive AI agent...")

            # Initialize agent manager
            agent = AgentManager(
                config_path=config_path,
                verbose=verbose
            )

            # Run interactive mode
            return agent.run_interactive_mode()

        except Exception as e:
            BaseCommand.print_error(f"Failed to start agent: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def run(args: Namespace) -> int:
        """
        Execute: symfluence agent run PROMPT

        Execute a single agent prompt.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.agent.agent_manager import AgentManager

            verbose = getattr(args, 'verbose', False)
            config_path = BaseCommand.get_config_path(args)
            prompt = args.prompt

            BaseCommand.print_info(f"🤖 Executing agent prompt: {prompt}")

            # Initialize agent manager
            agent = AgentManager(
                config_path=config_path,
                verbose=verbose
            )

            # Run single prompt
            return agent.run_single_prompt(prompt)

        except Exception as e:
            BaseCommand.print_error(f"Failed to execute prompt: {e}")
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
            BaseCommand.print_error("No agent action specified")
            return 1
