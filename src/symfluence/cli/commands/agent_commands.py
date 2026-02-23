"""
AI agent command handlers for SYMFLUENCE CLI.

This module implements handlers for the AI agent interface.
"""

from argparse import Namespace

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler


class AgentCommands(BaseCommand):
    """Handlers for AI agent commands."""

    @staticmethod
    @cli_exception_handler
    def start(args: Namespace) -> int:
        """
        Execute: symfluence agent start

        Start interactive AI agent mode.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.agent.agent_manager import AgentManager

        verbose = BaseCommand.get_arg(args, 'verbose', False)
        config_path = BaseCommand.get_config_path(args)

        BaseCommand._console.info("Starting interactive AI agent...")

        # Handle connection errors specifically
        try:
            # Initialize agent manager
            agent = AgentManager(
                config_path=config_path,
                verbose=verbose
            )

            # Run interactive mode
            return agent.run_interactive_mode()
        except ConnectionError as e:
            BaseCommand._console.error(f"Failed to connect to AI service: {e}")
            return ExitCode.NETWORK_ERROR

    @staticmethod
    @cli_exception_handler
    def run(args: Namespace) -> int:
        """
        Execute: symfluence agent run PROMPT

        Execute a single agent prompt.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.agent.agent_manager import AgentManager

        verbose = BaseCommand.get_arg(args, 'verbose', False)
        config_path = BaseCommand.get_config_path(args)
        prompt = args.prompt

        BaseCommand._console.info(f"Executing agent prompt: {prompt}")

        # Handle connection and timeout errors specifically
        try:
            # Initialize agent manager
            agent = AgentManager(
                config_path=config_path,
                verbose=verbose
            )

            # Run single prompt
            return agent.run_single_prompt(prompt)
        except ConnectionError as e:
            BaseCommand._console.error(f"Failed to connect to AI service: {e}")
            return ExitCode.NETWORK_ERROR
        except TimeoutError as e:
            BaseCommand._console.error(f"Agent request timed out: {e}")
            return ExitCode.TIMEOUT_ERROR
