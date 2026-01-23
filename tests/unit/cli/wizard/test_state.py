"""Unit tests for wizard state management."""

import pytest

from symfluence.cli.wizard.state import WizardPhase, WizardState


class TestWizardState:
    """Tests for WizardState class."""

    def test_initial_state(self):
        """Test that initial state is correctly set."""
        state = WizardState()

        assert state.answers == {}
        assert state.current_phase == WizardPhase.ESSENTIAL
        assert state.history == []
        assert state.skipped_questions == set()
        assert state.validation_errors == {}

    def test_set_and_get_answer(self):
        """Test setting and retrieving answers."""
        state = WizardState()

        state.set_answer('DOMAIN_NAME', 'test_domain')

        assert state.get_answer('DOMAIN_NAME') == 'test_domain'
        assert state.has_answer('DOMAIN_NAME')
        assert not state.has_answer('NON_EXISTENT')

    def test_get_answer_with_default(self):
        """Test get_answer returns default for missing keys."""
        state = WizardState()

        assert state.get_answer('MISSING', 'default_value') == 'default_value'
        assert state.get_answer('MISSING') is None

    def test_set_answer_records_history(self):
        """Test that setting answers records history."""
        state = WizardState()
        state.current_phase = WizardPhase.ESSENTIAL

        state.set_answer('DOMAIN_NAME', 'test')
        state.set_answer('MODEL', 'SUMMA')

        assert len(state.history) == 2
        assert state.history[0] == (WizardPhase.ESSENTIAL, 'DOMAIN_NAME')
        assert state.history[1] == (WizardPhase.ESSENTIAL, 'MODEL')

    def test_go_back(self):
        """Test navigation back functionality."""
        state = WizardState()
        state.current_phase = WizardPhase.ESSENTIAL

        state.set_answer('DOMAIN_NAME', 'test')
        state.set_answer('MODEL', 'SUMMA')

        # Go back should return previous question
        result = state.go_back()

        assert result == (WizardPhase.ESSENTIAL, 'MODEL')
        assert 'MODEL' not in state.answers
        assert len(state.history) == 1

    def test_go_back_at_start(self):
        """Test go_back returns None at start."""
        state = WizardState()

        result = state.go_back()

        assert result is None

    def test_skip_question(self):
        """Test marking questions as skipped."""
        state = WizardState()

        state.skip_question('OPTIONAL_FIELD')

        assert state.is_skipped('OPTIONAL_FIELD')
        assert not state.is_skipped('OTHER_FIELD')

    def test_validation_errors(self):
        """Test validation error management."""
        state = WizardState()

        state.set_validation_error('DOMAIN_NAME', 'Invalid format')

        assert state.get_validation_error('DOMAIN_NAME') == 'Invalid format'
        assert state.get_validation_error('OTHER') is None

        state.clear_validation_error('DOMAIN_NAME')
        assert state.get_validation_error('DOMAIN_NAME') is None

    def test_set_answer_clears_validation_error(self):
        """Test that setting an answer clears its validation error."""
        state = WizardState()

        state.set_validation_error('DOMAIN_NAME', 'Invalid format')
        state.set_answer('DOMAIN_NAME', 'valid_name')

        assert state.get_validation_error('DOMAIN_NAME') is None

    def test_advance_phase(self):
        """Test phase advancement."""
        state = WizardState()
        state.current_phase = WizardPhase.ESSENTIAL

        result = state.advance_phase()

        assert result is True
        assert state.current_phase == WizardPhase.CALIBRATION

    def test_advance_phase_at_end(self):
        """Test that advance_phase returns False at last phase."""
        state = WizardState()
        state.current_phase = WizardPhase.SUMMARY

        result = state.advance_phase()

        assert result is False
        assert state.current_phase == WizardPhase.SUMMARY

    def test_reset(self):
        """Test state reset."""
        state = WizardState()
        state.set_answer('DOMAIN_NAME', 'test')
        state.current_phase = WizardPhase.CALIBRATION
        state.skip_question('OPTIONAL')
        state.set_validation_error('FIELD', 'error')

        state.reset()

        assert state.answers == {}
        assert state.current_phase == WizardPhase.ESSENTIAL
        assert state.history == []
        assert state.skipped_questions == set()
        assert state.validation_errors == {}

    def test_to_config_dict_basic(self):
        """Test conversion to config dictionary."""
        state = WizardState()
        state.set_answer('DOMAIN_NAME', 'test_domain')
        state.set_answer('HYDROLOGICAL_MODEL', 'SUMMA')
        state.set_answer('EXPERIMENT_TIME_START', '2010-01-01')
        state.set_answer('EXPERIMENT_TIME_END', '2020-12-31')

        config = state.to_config_dict()

        assert config['DOMAIN_NAME'] == 'test_domain'
        assert config['HYDROLOGICAL_MODEL'] == 'SUMMA'
        assert config['EXPERIMENT_TIME_START'] == '2010-01-01 00:00'
        assert config['EXPERIMENT_TIME_END'] == '2020-12-31 23:00'

    def test_to_config_dict_without_calibration(self):
        """Test config dict excludes calibration when disabled."""
        state = WizardState()
        state.set_answer('DOMAIN_NAME', 'test')
        state.set_answer('ENABLE_CALIBRATION', False)
        state.set_answer('CALIBRATION_PERIOD', '2010-01-01 to 2015-12-31')

        config = state.to_config_dict()

        assert 'CALIBRATION_PERIOD' not in config

    def test_to_config_dict_with_calibration(self):
        """Test config dict includes calibration when enabled."""
        state = WizardState()
        state.set_answer('DOMAIN_NAME', 'test')
        state.set_answer('ENABLE_CALIBRATION', True)
        state.set_answer('CALIBRATION_PERIOD', '2010-01-01 to 2015-12-31')

        config = state.to_config_dict()

        assert config.get('CALIBRATION_PERIOD') == '2010-01-01 to 2015-12-31'


class TestWizardPhase:
    """Tests for WizardPhase enum."""

    def test_phase_ordering(self):
        """Test that phases are in correct order."""
        phases = list(WizardPhase)

        assert phases[0] == WizardPhase.ESSENTIAL
        assert phases[1] == WizardPhase.CALIBRATION
        assert phases[2] == WizardPhase.MODEL_SPECIFIC
        assert phases[3] == WizardPhase.PATHS
        assert phases[4] == WizardPhase.SUMMARY

    def test_phase_count(self):
        """Test the number of phases."""
        assert len(WizardPhase) == 5
