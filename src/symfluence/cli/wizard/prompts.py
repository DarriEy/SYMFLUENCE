"""
Rich-based prompt wrappers for SYMFLUENCE project wizard.

This module provides interactive prompt functionality using the rich library,
with support for help text panels, validation retry, and various input types.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .questions import Question, QuestionType
from .state import WizardState


class WizardPrompts:
    """
    Rich-based prompt wrapper for the project wizard.

    Provides type-specific prompts with help panels, validation,
    and consistent styling throughout the wizard.
    """

    # Special input values for navigation
    BACK_COMMAND = 'back'
    HELP_COMMAND = '?'
    QUIT_COMMAND = 'quit'

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize wizard prompts.

        Args:
            console: Rich console instance. Creates new one if not provided.
        """
        self.console = console or Console()

    def ask(
        self,
        question: Question,
        state: WizardState
    ) -> Tuple[Optional[Any], str]:
        """
        Ask a question and return the response.

        Handles help display, validation, and navigation commands.

        Args:
            question: Question definition
            state: Current wizard state for defaults and validation

        Returns:
            Tuple of (answer, action) where action is 'continue', 'back', or 'quit'
        """
        # Show help panel if available
        if question.help_text:
            self._show_help_hint()

        while True:
            try:
                # Get response based on question type
                response = self._dispatch_prompt(question, state)

                # Check for special commands (for text-based inputs)
                if isinstance(response, str):
                    response_lower = response.lower().strip()
                    if response_lower == self.BACK_COMMAND:
                        return None, 'back'
                    if response_lower == self.QUIT_COMMAND:
                        return None, 'quit'
                    if response_lower == self.HELP_COMMAND:
                        self._show_help(question)
                        continue

                # Validate response
                if question.validator:
                    is_valid, error_msg = question.validator(response, state)
                    if not is_valid:
                        self._show_validation_error(error_msg)
                        continue

                return response, 'continue'

            except KeyboardInterrupt:
                self.console.print("\n")
                return None, 'quit'

    def _dispatch_prompt(self, question: Question, state: WizardState) -> Any:
        """
        Dispatch to the appropriate prompt method based on question type.

        Args:
            question: Question definition
            state: Current wizard state

        Returns:
            User's response
        """
        prompt_methods = {
            QuestionType.TEXT: self._ask_text,
            QuestionType.CHOICE: self._ask_choice,
            QuestionType.CONFIRM: self._ask_confirm,
            QuestionType.DATE: self._ask_date,
            QuestionType.COORDINATES: self._ask_coordinates,
            QuestionType.INTEGER: self._ask_integer,
            QuestionType.PATH: self._ask_path,
        }

        method = prompt_methods.get(question.question_type, self._ask_text)
        return method(question, state)

    def _ask_text(self, question: Question, state: WizardState) -> str:
        """Prompt for free-form text input."""
        default = question.get_default(state)
        return Prompt.ask(
            question.prompt,
            console=self.console,
            default=default if default else None,
        )

    def _ask_choice(self, question: Question, state: WizardState) -> str:
        """Prompt for selection from choices."""
        # Display choices
        self.console.print()
        for i, choice in enumerate(question.choices, 1):
            desc = f" - {choice.description}" if choice.description else ""
            self.console.print(f"  [cyan]{i}[/cyan]. {choice.label}{desc}")
        self.console.print()

        # Get default index
        default = question.get_default(state)
        default_idx = None
        if default:
            for i, choice in enumerate(question.choices, 1):
                if choice.value == default:
                    default_idx = str(i)
                    break

        # Prompt for selection
        while True:
            response = Prompt.ask(
                f"Enter choice (1-{len(question.choices)})",
                console=self.console,
                default=default_idx,
            )

            # Check for special commands
            if response.lower().strip() in (self.BACK_COMMAND, self.QUIT_COMMAND, self.HELP_COMMAND):
                return response

            try:
                idx = int(response) - 1
                if 0 <= idx < len(question.choices):
                    return question.choices[idx].value
                self._show_validation_error(f"Please enter a number between 1 and {len(question.choices)}")
            except ValueError:
                # Try matching by label or value
                for choice in question.choices:
                    if response.lower() == choice.value.lower() or response.lower() == choice.label.lower():
                        return choice.value
                self._show_validation_error(f"Invalid choice: {response}")

    def _ask_confirm(self, question: Question, state: WizardState) -> bool:
        """Prompt for yes/no confirmation."""
        default = question.get_default(state)
        if default is None:
            default = False
        return Confirm.ask(
            question.prompt,
            console=self.console,
            default=default,
        )

    def _ask_date(self, question: Question, state: WizardState) -> str:
        """Prompt for date in YYYY-MM-DD format."""
        default = question.get_default(state)

        while True:
            response = Prompt.ask(
                f"{question.prompt} (YYYY-MM-DD)",
                console=self.console,
                default=default,
            )

            # Check for special commands
            if response.lower().strip() in (self.BACK_COMMAND, self.QUIT_COMMAND, self.HELP_COMMAND):
                return response

            # Validate date format
            try:
                datetime.strptime(response, '%Y-%m-%d')
                return response
            except ValueError:
                self._show_validation_error("Invalid date format. Please use YYYY-MM-DD (e.g., 2020-01-15)")

    def _ask_coordinates(self, question: Question, state: WizardState) -> str:
        """Prompt for lat/lon coordinates."""
        default = question.get_default(state)

        while True:
            response = Prompt.ask(
                f"{question.prompt} (lat/lon)",
                console=self.console,
                default=default,
            )

            # Check for special commands
            if response.lower().strip() in (self.BACK_COMMAND, self.QUIT_COMMAND, self.HELP_COMMAND):
                return response

            # Validate coordinate format
            try:
                parts = response.split('/')
                if len(parts) != 2:
                    raise ValueError("Expected format: lat/lon")
                lat, lon = float(parts[0]), float(parts[1])
                if not (-90 <= lat <= 90):
                    raise ValueError(f"Latitude {lat} out of range [-90, 90]")
                if not (-180 <= lon <= 180):
                    raise ValueError(f"Longitude {lon} out of range [-180, 180]")
                return response
            except ValueError as e:
                self._show_validation_error(str(e))

    def _ask_integer(self, question: Question, state: WizardState) -> int:
        """Prompt for integer value."""
        default = question.get_default(state)

        while True:
            try:
                return IntPrompt.ask(
                    question.prompt,
                    console=self.console,
                    default=default if default is not None else None,
                )
            except KeyboardInterrupt:
                raise
            except ValueError:
                self._show_validation_error("Please enter a valid integer")

    def _ask_path(self, question: Question, state: WizardState) -> str:
        """Prompt for file system path."""
        default = question.get_default(state)

        while True:
            response = Prompt.ask(
                question.prompt,
                console=self.console,
                default=default,
            )

            # Check for special commands
            if response.lower().strip() in (self.BACK_COMMAND, self.QUIT_COMMAND, self.HELP_COMMAND):
                return response

            # Expand user home directory
            path = Path(response).expanduser()

            # Path doesn't need to exist, but parent should be writable
            # For now, accept any path
            return str(path)

    def _show_help_hint(self) -> None:
        """Show hint about help availability."""
        self.console.print(
            "[dim]Type '?' for help, 'back' to go back, 'quit' to exit[/dim]",
            style="dim"
        )

    def _show_help(self, question: Question) -> None:
        """Display help panel for a question."""
        if question.help_text:
            panel = Panel(
                question.help_text,
                title="Help",
                border_style="blue",
                padding=(1, 2),
            )
            self.console.print(panel)

    def _show_validation_error(self, message: str) -> None:
        """Display validation error message."""
        self.console.print(f"[red]Error:[/red] {message}")

    def show_welcome(self) -> None:
        """Display wizard welcome message."""
        welcome_text = """
Welcome to the SYMFLUENCE Project Initialization Wizard!

This wizard will guide you through creating a configuration file
for your hydrological modeling project.

You can type:
  • [cyan]?[/cyan]      Show help for the current question
  • [cyan]back[/cyan]   Go back to the previous question
  • [cyan]quit[/cyan]   Exit the wizard (Ctrl+C also works)
"""
        panel = Panel(
            welcome_text.strip(),
            title="SYMFLUENCE Project Wizard",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)
        self.console.print()

    def show_phase_header(self, phase_name: str, description: str) -> None:
        """Display phase header."""
        self.console.print()
        self.console.rule(f"[bold cyan]{phase_name}[/bold cyan]")
        self.console.print(f"[dim]{description}[/dim]")
        self.console.print()

    def show_summary(self, state: WizardState) -> None:
        """Display summary of all wizard answers."""
        self.console.print()
        self.console.rule("[bold cyan]Configuration Summary[/bold cyan]")
        self.console.print()

        # Create summary table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        # Add rows for all answers
        display_order = [
            ('DOMAIN_NAME', 'Domain Name'),
            ('EXPERIMENT_ID', 'Experiment ID'),
            ('EXPERIMENT_TIME_START', 'Start Date'),
            ('EXPERIMENT_TIME_END', 'End Date'),
            ('SPATIAL_EXTENT_TYPE', 'Spatial Extent'),
            ('POUR_POINT_COORDS', 'Pour Point'),
            ('BOUNDING_BOX_COORDS', 'Bounding Box'),
            ('DOMAIN_DEFINITION_METHOD', 'Domain Definition'),
            ('SUB_GRID_DISCRETIZATION', 'Discretization'),
            ('HYDROLOGICAL_MODEL', 'Model'),
            ('FORCING_DATASET', 'Forcing Dataset'),
            ('ENABLE_CALIBRATION', 'Calibration'),
            ('CALIBRATION_PERIOD', 'Calibration Period'),
            ('EVALUATION_PERIOD', 'Evaluation Period'),
            ('OPTIMIZATION_METRIC', 'Optimization Metric'),
            ('NUMBER_OF_ITERATIONS', 'Iterations'),
            ('SYMFLUENCE_DATA_DIR', 'Data Directory'),
            ('SYMFLUENCE_CODE_DIR', 'Code Directory'),
        ]

        for key, label in display_order:
            value = state.get_answer(key)
            if value is not None:
                # Format boolean values
                if isinstance(value, bool):
                    value = "Yes" if value else "No"
                table.add_row(label, str(value))

        self.console.print(table)
        self.console.print()

    def confirm_generate(self) -> bool:
        """Ask user to confirm configuration generation."""
        return Confirm.ask(
            "Generate configuration file with these settings?",
            console=self.console,
            default=True,
        )

    def show_success(self, config_path: str, scaffold_path: Optional[str] = None) -> None:
        """Display success message after config generation."""
        self.console.print()
        self.console.print("[green]Configuration file created successfully![/green]")
        self.console.print(f"  Config: [cyan]{config_path}[/cyan]")
        if scaffold_path:
            self.console.print(f"  Project: [cyan]{scaffold_path}[/cyan]")
        self.console.print()
        self.console.print("Next steps:")
        self.console.print(f"  1. Review the config: [dim]cat {config_path}[/dim]")
        self.console.print(f"  2. Run the workflow: [dim]symfluence workflow run --config {config_path}[/dim]")

    def show_cancelled(self) -> None:
        """Display cancellation message."""
        self.console.print()
        self.console.print("[yellow]Wizard cancelled.[/yellow] No files were created.")
