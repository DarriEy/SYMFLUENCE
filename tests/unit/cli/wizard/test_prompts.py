"""Unit tests for wizard prompt wrappers."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from symfluence.cli.wizard.prompts import WizardPrompts
from symfluence.cli.wizard.questions import Choice, Question, QuestionType
from symfluence.cli.wizard.state import WizardState


@pytest.fixture
def console():
    """Create a console that writes to StringIO for testing."""
    return Console(file=StringIO(), force_terminal=True, no_color=True)


@pytest.fixture
def prompts(console):
    """Create WizardPrompts instance with test console."""
    return WizardPrompts(console=console)


@pytest.fixture
def state():
    """Create a fresh WizardState."""
    return WizardState()


class TestWizardPromptsInit:
    """Tests for WizardPrompts initialization."""

    def test_init_with_console(self, console):
        """Test initialization with provided console."""
        prompts = WizardPrompts(console=console)
        assert prompts.console is console

    def test_init_without_console(self):
        """Test initialization creates default console."""
        prompts = WizardPrompts()
        assert prompts.console is not None


class TestSpecialCommands:
    """Tests for special navigation commands."""

    def test_back_command_constant(self, prompts):
        """Test back command constant."""
        assert prompts.BACK_COMMAND == 'back'

    def test_help_command_constant(self, prompts):
        """Test help command constant."""
        assert prompts.HELP_COMMAND == '?'

    def test_quit_command_constant(self, prompts):
        """Test quit command constant."""
        assert prompts.QUIT_COMMAND == 'quit'


class TestAskTextQuestion:
    """Tests for text question prompts."""

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_ask_text_returns_value(self, mock_ask, prompts, state):
        """Test asking a text question returns the value."""
        mock_ask.return_value = 'test_value'

        question = Question(
            key='TEST',
            prompt='Enter value:',
            question_type=QuestionType.TEXT,
        )

        answer, action = prompts.ask(question, state)

        assert answer == 'test_value'
        assert action == 'continue'

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_ask_text_back_command(self, mock_ask, prompts, state):
        """Test back command during text input."""
        mock_ask.return_value = 'back'

        question = Question(
            key='TEST',
            prompt='Enter value:',
            question_type=QuestionType.TEXT,
        )

        answer, action = prompts.ask(question, state)

        assert answer is None
        assert action == 'back'

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_ask_text_quit_command(self, mock_ask, prompts, state):
        """Test quit command during text input."""
        mock_ask.return_value = 'quit'

        question = Question(
            key='TEST',
            prompt='Enter value:',
            question_type=QuestionType.TEXT,
        )

        answer, action = prompts.ask(question, state)

        assert answer is None
        assert action == 'quit'


class TestAskChoiceQuestion:
    """Tests for choice question prompts."""

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_ask_choice_by_number(self, mock_ask, prompts, state):
        """Test selecting a choice by number."""
        mock_ask.return_value = '1'

        question = Question(
            key='MODEL',
            prompt='Select model:',
            question_type=QuestionType.CHOICE,
            choices=[
                Choice('summa', 'SUMMA'),
                Choice('fuse', 'FUSE'),
            ],
        )

        answer, action = prompts.ask(question, state)

        assert answer == 'summa'
        assert action == 'continue'

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_ask_choice_by_value(self, mock_ask, prompts, state):
        """Test selecting a choice by value name."""
        mock_ask.return_value = 'fuse'

        question = Question(
            key='MODEL',
            prompt='Select model:',
            question_type=QuestionType.CHOICE,
            choices=[
                Choice('summa', 'SUMMA'),
                Choice('fuse', 'FUSE'),
            ],
        )

        answer, action = prompts.ask(question, state)

        assert answer == 'fuse'
        assert action == 'continue'


class TestAskConfirmQuestion:
    """Tests for confirmation question prompts."""

    @patch('symfluence.cli.wizard.prompts.Confirm.ask')
    def test_ask_confirm_yes(self, mock_ask, prompts, state):
        """Test confirming yes."""
        mock_ask.return_value = True

        question = Question(
            key='ENABLE',
            prompt='Enable feature?',
            question_type=QuestionType.CONFIRM,
        )

        answer, action = prompts.ask(question, state)

        assert answer is True
        assert action == 'continue'

    @patch('symfluence.cli.wizard.prompts.Confirm.ask')
    def test_ask_confirm_no(self, mock_ask, prompts, state):
        """Test confirming no."""
        mock_ask.return_value = False

        question = Question(
            key='ENABLE',
            prompt='Enable feature?',
            question_type=QuestionType.CONFIRM,
        )

        answer, action = prompts.ask(question, state)

        assert answer is False
        assert action == 'continue'


class TestAskDateQuestion:
    """Tests for date question prompts."""

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_ask_date_valid(self, mock_ask, prompts, state):
        """Test entering a valid date."""
        mock_ask.return_value = '2020-01-15'

        question = Question(
            key='DATE',
            prompt='Enter date:',
            question_type=QuestionType.DATE,
        )

        answer, action = prompts.ask(question, state)

        assert answer == '2020-01-15'
        assert action == 'continue'

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_ask_date_invalid_then_valid(self, mock_ask, prompts, state):
        """Test entering invalid date then valid date."""
        # First call returns invalid, second returns valid
        mock_ask.side_effect = ['invalid', '2020-01-15']

        question = Question(
            key='DATE',
            prompt='Enter date:',
            question_type=QuestionType.DATE,
        )

        answer, action = prompts.ask(question, state)

        assert answer == '2020-01-15'
        assert action == 'continue'
        assert mock_ask.call_count == 2


class TestAskCoordinatesQuestion:
    """Tests for coordinates question prompts."""

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_ask_coordinates_valid(self, mock_ask, prompts, state):
        """Test entering valid coordinates."""
        mock_ask.return_value = '51.1722/-115.5717'

        question = Question(
            key='COORDS',
            prompt='Enter coordinates:',
            question_type=QuestionType.COORDINATES,
        )

        answer, action = prompts.ask(question, state)

        assert answer == '51.1722/-115.5717'
        assert action == 'continue'

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_ask_coordinates_invalid_lat(self, mock_ask, prompts, state):
        """Test entering coordinates with invalid latitude."""
        mock_ask.side_effect = ['91.0/-115.0', '51.0/-115.0']

        question = Question(
            key='COORDS',
            prompt='Enter coordinates:',
            question_type=QuestionType.COORDINATES,
        )

        answer, action = prompts.ask(question, state)

        assert answer == '51.0/-115.0'
        assert mock_ask.call_count == 2


class TestAskIntegerQuestion:
    """Tests for integer question prompts."""

    @patch('symfluence.cli.wizard.prompts.IntPrompt.ask')
    def test_ask_integer_valid(self, mock_ask, prompts, state):
        """Test entering a valid integer."""
        mock_ask.return_value = 1000

        question = Question(
            key='ITERATIONS',
            prompt='Enter iterations:',
            question_type=QuestionType.INTEGER,
            default=500,
        )

        answer, action = prompts.ask(question, state)

        assert answer == 1000
        assert action == 'continue'


class TestAskPathQuestion:
    """Tests for path question prompts."""

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_ask_path_valid(self, mock_ask, prompts, state):
        """Test entering a valid path."""
        mock_ask.return_value = '/home/user/data'

        question = Question(
            key='PATH',
            prompt='Enter path:',
            question_type=QuestionType.PATH,
        )

        answer, action = prompts.ask(question, state)

        assert answer == '/home/user/data'
        assert action == 'continue'

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_ask_path_expands_tilde(self, mock_ask, prompts, state):
        """Test that tilde is expanded in paths."""
        mock_ask.return_value = '~/data'

        question = Question(
            key='PATH',
            prompt='Enter path:',
            question_type=QuestionType.PATH,
        )

        answer, action = prompts.ask(question, state)

        # Path should be expanded
        assert '~' not in answer
        assert 'data' in answer


class TestValidation:
    """Tests for question validation."""

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_validation_retry_on_failure(self, mock_ask, prompts, state):
        """Test that invalid input triggers retry."""
        mock_ask.side_effect = ['bad', 'good']

        def validator(value, state):
            if value == 'bad':
                return False, 'Value cannot be "bad"'
            return True, None

        question = Question(
            key='TEST',
            prompt='Enter value:',
            question_type=QuestionType.TEXT,
            validator=validator,
        )

        answer, action = prompts.ask(question, state)

        assert answer == 'good'
        assert mock_ask.call_count == 2


class TestDisplayMethods:
    """Tests for display methods."""

    def test_show_welcome(self, prompts, console):
        """Test welcome message displays."""
        prompts.show_welcome()
        output = console.file.getvalue()
        assert 'Welcome' in output or 'SYMFLUENCE' in output

    def test_show_phase_header(self, prompts, console):
        """Test phase header displays."""
        prompts.show_phase_header('Test Phase', 'Test description')
        output = console.file.getvalue()
        assert 'Test Phase' in output

    def test_show_summary(self, prompts, state, console):
        """Test summary displays answers."""
        state.set_answer('DOMAIN_NAME', 'test_domain')
        state.set_answer('HYDROLOGICAL_MODEL', 'SUMMA')

        prompts.show_summary(state)
        output = console.file.getvalue()

        assert 'test_domain' in output
        assert 'SUMMA' in output

    def test_show_cancelled(self, prompts, console):
        """Test cancelled message displays."""
        prompts.show_cancelled()
        output = console.file.getvalue()
        assert 'cancelled' in output.lower()


class TestKeyboardInterrupt:
    """Tests for keyboard interrupt handling."""

    @patch('symfluence.cli.wizard.prompts.Prompt.ask')
    def test_keyboard_interrupt_returns_quit(self, mock_ask, prompts, state):
        """Test that KeyboardInterrupt returns quit action."""
        mock_ask.side_effect = KeyboardInterrupt()

        question = Question(
            key='TEST',
            prompt='Enter value:',
            question_type=QuestionType.TEXT,
        )

        answer, action = prompts.ask(question, state)

        assert answer is None
        assert action == 'quit'
