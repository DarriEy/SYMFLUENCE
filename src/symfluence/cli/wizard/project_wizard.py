"""
Main wizard orchestrator for SYMFLUENCE project initialization.

This module provides the ProjectWizard class that coordinates the interactive
configuration wizard, managing question flow, state, and config generation.
"""

from pathlib import Path
from typing import Optional

from rich.console import Console

from ..exit_codes import ExitCode
from .prompts import WizardPrompts
from .questions import get_questions_for_phase
from .state import WizardPhase, WizardState


# Phase descriptions for display
PHASE_DESCRIPTIONS = {
    WizardPhase.ESSENTIAL: (
        "Essential Settings",
        "Core configuration for your hydrological model"
    ),
    WizardPhase.CALIBRATION: (
        "Calibration Settings",
        "Configure model parameter optimization"
    ),
    WizardPhase.MODEL_SPECIFIC: (
        "Model-Specific Settings",
        "Additional options for your chosen model"
    ),
    WizardPhase.PATHS: (
        "Directory Paths",
        "Where to store data and find SYMFLUENCE code"
    ),
    WizardPhase.SUMMARY: (
        "Summary",
        "Review your configuration before saving"
    ),
}


class ProjectWizard:
    """
    Main orchestrator for the interactive project initialization wizard.

    Coordinates the flow between questions, manages state, and integrates
    with InitializationService for config file generation.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the project wizard.

        Args:
            console: Rich console for output. Creates new one if not provided.
        """
        self.console = console or Console()
        self.prompts = WizardPrompts(self.console)
        self.state = WizardState()

    def run(
        self,
        output_dir: str = './',
        scaffold: bool = False
    ) -> int:
        """
        Execute the wizard flow.

        Args:
            output_dir: Directory for output config file
            scaffold: Whether to create project directory structure

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            # Show welcome message
            self.prompts.show_welcome()

            # Process each phase
            for phase in WizardPhase:
                if phase == WizardPhase.SUMMARY:
                    # Summary phase - show and confirm
                    result = self._handle_summary_phase(output_dir, scaffold)
                    if result is not None:
                        return result
                else:
                    # Question phase
                    result = self._process_phase(phase)
                    if result == 'quit':
                        self.prompts.show_cancelled()
                        return ExitCode.USER_INTERRUPT

            return ExitCode.SUCCESS

        except KeyboardInterrupt:
            self.prompts.show_cancelled()
            return ExitCode.USER_INTERRUPT
        except (OSError, ValueError, RuntimeError, TypeError) as e:
            self.console.print(f"[red]Error:[/red] {e}")
            return ExitCode.GENERAL_ERROR

    def _process_phase(self, phase: WizardPhase) -> Optional[str]:
        """
        Process all questions in a phase.

        Args:
            phase: The wizard phase to process

        Returns:
            'quit' if user wants to exit, None otherwise
        """
        # Get questions for this phase
        questions = get_questions_for_phase(phase, self.state)

        if not questions:
            # Skip empty phases
            return None

        # Show phase header
        name, description = PHASE_DESCRIPTIONS.get(
            phase,
            (phase.name, "")
        )
        self.prompts.show_phase_header(name, description)

        # Update state phase
        self.state.current_phase = phase

        # Process each question
        question_idx = 0
        while question_idx < len(questions):
            question = questions[question_idx]

            # Check if question should be shown (conditions may have changed)
            if not question.should_show(self.state):
                question_idx += 1
                continue

            # Ask the question
            answer, action = self.prompts.ask(question, self.state)

            if action == 'quit':
                return 'quit'
            elif action == 'back':
                # Go back to previous question
                if question_idx > 0:
                    question_idx -= 1
                    # Remove the previous answer
                    prev_question = questions[question_idx]
                    self.state.answers.pop(prev_question.key, None)
                else:
                    # At start of phase - could go to previous phase
                    # For now, just stay at first question
                    self.console.print("[dim]Already at the first question[/dim]")
                continue
            else:
                # Store the answer
                self.state.set_answer(question.key, answer)
                question_idx += 1

        return None

    def _handle_summary_phase(
        self,
        output_dir: str,
        scaffold: bool
    ) -> Optional[int]:
        """
        Handle the summary phase - display summary and generate config.

        Args:
            output_dir: Directory for output config file
            scaffold: Whether to create project directory structure

        Returns:
            Exit code if wizard should end, None to continue
        """
        # Show summary
        self.prompts.show_summary(self.state)

        # Confirm generation
        if not self.prompts.confirm_generate():
            self.prompts.show_cancelled()
            return ExitCode.USER_INTERRUPT

        # Generate config
        try:
            config_path, scaffold_path = self._generate_config(output_dir, scaffold)
            self.prompts.show_success(config_path, scaffold_path)
            return ExitCode.SUCCESS
        except (OSError, ValueError, RuntimeError, TypeError) as e:
            self.console.print(f"[red]Failed to generate config:[/red] {e}")
            return ExitCode.GENERAL_ERROR

    def _generate_config(
        self,
        output_dir: str,
        scaffold: bool
    ) -> tuple:
        """
        Generate configuration file from wizard state.

        Args:
            output_dir: Directory for output config file
            scaffold: Whether to create project directory structure

        Returns:
            Tuple of (config_path, scaffold_path or None)
        """
        from ..services import InitializationService

        # Get config dict from wizard state
        wizard_config = self.state.to_config_dict()

        # Create initialization service
        init_service = InitializationService()

        # Build CLI overrides from wizard answers
        cli_overrides = {
            'domain': wizard_config.get('DOMAIN_NAME'),
            'model': wizard_config.get('HYDROLOGICAL_MODEL'),
            'start_date': self.state.get_answer('EXPERIMENT_TIME_START'),
            'end_date': self.state.get_answer('EXPERIMENT_TIME_END'),
            'forcing': wizard_config.get('FORCING_DATASET'),
            'discretization': wizard_config.get('SUB_GRID_DISCRETIZATION'),
            'definition_method': wizard_config.get('DOMAIN_DEFINITION_METHOD'),
        }

        # Remove None values
        cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

        # Generate config using the service
        config = init_service.generate_config(
            preset_name=None,
            cli_overrides=cli_overrides,
            minimal=False,
            comprehensive=True
        )

        # Merge wizard-specific settings that aren't in CLI overrides
        wizard_specific_keys = [
            'EXPERIMENT_ID',
            'POUR_POINT_LAT',
            'POUR_POINT_LON',
            'BOUNDING_BOX_COORDS',
            'CALIBRATION_PERIOD',
            'EVALUATION_PERIOD',
            'OPTIMIZATION_METRIC',
            'NUMBER_OF_ITERATIONS',
            'SYMFLUENCE_DATA_DIR',
            'SYMFLUENCE_CODE_DIR',
        ]

        for key in wizard_specific_keys:
            if key in wizard_config and wizard_config[key] is not None:
                config[key] = wizard_config[key]

        # Handle pour point coordinates
        pour_point_coords = self.state.get_answer('POUR_POINT_COORDS')
        if pour_point_coords:
            parts = pour_point_coords.split('/')
            if len(parts) == 2:
                config['POUR_POINT_LAT'] = float(parts[0])
                config['POUR_POINT_LON'] = float(parts[1])

        # Model-specific settings
        model = self.state.get_answer('HYDROLOGICAL_MODEL')

        if model == 'SUMMA':
            summa_mode = self.state.get_answer('SUMMA_SPATIAL_MODE')
            if summa_mode:
                config['SUMMA_SPATIAL_MODE'] = summa_mode
            routing = self.state.get_answer('ROUTING_MODEL')
            if routing:
                config['ROUTING_MODEL'] = routing

        elif model == 'FUSE':
            fuse_mode = self.state.get_answer('FUSE_SPATIAL_MODE')
            if fuse_mode:
                config['FUSE_SPATIAL_MODE'] = fuse_mode

        elif model == 'GR':
            gr_type = self.state.get_answer('GR_MODEL_TYPE')
            if gr_type:
                config['GR_MODEL_TYPE'] = gr_type

        # Determine output path
        domain_name = config.get('DOMAIN_NAME', 'unnamed_project')
        output_path = Path(output_dir)
        config_file = output_path / f"config_{domain_name}.yaml"

        # Write config file
        written_path = init_service.write_config(config, config_file)

        # Create scaffold if requested
        scaffold_path = None
        if scaffold:
            scaffold_path = str(init_service.create_scaffold(config))

        return str(written_path), scaffold_path
