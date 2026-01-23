"""Unit tests for wizard question definitions."""

import pytest

from symfluence.cli.wizard.questions import (
    ALL_QUESTIONS,
    CALIBRATION_QUESTIONS,
    ESSENTIAL_QUESTIONS,
    FUSE_QUESTIONS,
    GR_QUESTIONS,
    PATHS_QUESTIONS,
    SUMMA_QUESTIONS,
    Choice,
    Question,
    QuestionType,
    get_all_questions,
    get_questions_for_phase,
)
from symfluence.cli.wizard.state import WizardPhase, WizardState


class TestQuestionType:
    """Tests for QuestionType enum."""

    def test_all_types_exist(self):
        """Test that all expected question types exist."""
        expected_types = ['TEXT', 'CHOICE', 'CONFIRM', 'DATE', 'COORDINATES', 'INTEGER', 'PATH']

        for type_name in expected_types:
            assert hasattr(QuestionType, type_name)

    def test_type_count(self):
        """Test the number of question types."""
        assert len(QuestionType) == 7


class TestChoice:
    """Tests for Choice dataclass."""

    def test_choice_creation(self):
        """Test creating a choice."""
        choice = Choice(
            value='test_value',
            label='Test Label',
            description='A test description'
        )

        assert choice.value == 'test_value'
        assert choice.label == 'Test Label'
        assert choice.description == 'A test description'

    def test_choice_without_description(self):
        """Test creating a choice without description."""
        choice = Choice(value='val', label='Label')

        assert choice.value == 'val'
        assert choice.label == 'Label'
        assert choice.description is None


class TestQuestion:
    """Tests for Question dataclass."""

    def test_question_creation(self):
        """Test creating a basic question."""
        question = Question(
            key='TEST_KEY',
            prompt='Test prompt?',
            question_type=QuestionType.TEXT,
        )

        assert question.key == 'TEST_KEY'
        assert question.prompt == 'Test prompt?'
        assert question.question_type == QuestionType.TEXT
        assert question.phase == WizardPhase.ESSENTIAL

    def test_question_with_help_text(self):
        """Test question with help text."""
        question = Question(
            key='TEST',
            prompt='Test?',
            question_type=QuestionType.TEXT,
            help_text='This is help text',
        )

        assert question.help_text == 'This is help text'

    def test_get_default_static(self):
        """Test getting static default value."""
        question = Question(
            key='TEST',
            prompt='Test?',
            question_type=QuestionType.TEXT,
            default='static_default',
        )

        state = WizardState()
        assert question.get_default(state) == 'static_default'

    def test_get_default_callable(self):
        """Test getting dynamic default value."""
        def dynamic_default(state):
            return f"dynamic_{state.get_answer('PREFIX', 'none')}"

        question = Question(
            key='TEST',
            prompt='Test?',
            question_type=QuestionType.TEXT,
            default=dynamic_default,
        )

        state = WizardState()
        state.set_answer('PREFIX', 'test')

        assert question.get_default(state) == 'dynamic_test'

    def test_should_show_without_condition(self):
        """Test that questions without conditions always show."""
        question = Question(
            key='TEST',
            prompt='Test?',
            question_type=QuestionType.TEXT,
        )

        state = WizardState()
        assert question.should_show(state) is True

    def test_should_show_with_condition_true(self):
        """Test conditional question that should show."""
        def condition(state):
            return state.get_answer('ENABLE') is True

        question = Question(
            key='TEST',
            prompt='Test?',
            question_type=QuestionType.TEXT,
            condition=condition,
        )

        state = WizardState()
        state.set_answer('ENABLE', True)

        assert question.should_show(state) is True

    def test_should_show_with_condition_false(self):
        """Test conditional question that should not show."""
        def condition(state):
            return state.get_answer('ENABLE') is True

        question = Question(
            key='TEST',
            prompt='Test?',
            question_type=QuestionType.TEXT,
            condition=condition,
        )

        state = WizardState()
        state.set_answer('ENABLE', False)

        assert question.should_show(state) is False

    def test_question_with_choices(self):
        """Test question with choice options."""
        question = Question(
            key='MODEL',
            prompt='Select model?',
            question_type=QuestionType.CHOICE,
            choices=[
                Choice('summa', 'SUMMA', 'SUMMA model'),
                Choice('fuse', 'FUSE', 'FUSE model'),
            ],
        )

        assert len(question.choices) == 2
        assert question.choices[0].value == 'summa'
        assert question.choices[1].label == 'FUSE'


class TestQuestionLists:
    """Tests for predefined question lists."""

    def test_essential_questions_exist(self):
        """Test that essential questions are defined."""
        assert len(ESSENTIAL_QUESTIONS) > 0

    def test_essential_questions_have_domain_name(self):
        """Test that DOMAIN_NAME is in essential questions."""
        keys = [q.key for q in ESSENTIAL_QUESTIONS]
        assert 'DOMAIN_NAME' in keys

    def test_essential_questions_have_model(self):
        """Test that model selection is in essential questions."""
        keys = [q.key for q in ESSENTIAL_QUESTIONS]
        assert 'HYDROLOGICAL_MODEL' in keys

    def test_calibration_questions_exist(self):
        """Test that calibration questions are defined."""
        assert len(CALIBRATION_QUESTIONS) > 0

    def test_calibration_enable_question(self):
        """Test that calibration enable question exists."""
        keys = [q.key for q in CALIBRATION_QUESTIONS]
        assert 'ENABLE_CALIBRATION' in keys

    def test_calibration_questions_are_conditional(self):
        """Test that calibration detail questions are conditional."""
        for question in CALIBRATION_QUESTIONS:
            if question.key != 'ENABLE_CALIBRATION':
                assert question.condition is not None

    def test_model_specific_questions_are_conditional(self):
        """Test that model-specific questions have conditions."""
        for question in SUMMA_QUESTIONS + FUSE_QUESTIONS + GR_QUESTIONS:
            assert question.condition is not None

    def test_paths_questions_exist(self):
        """Test that path questions are defined."""
        assert len(PATHS_QUESTIONS) > 0

    def test_paths_questions_have_data_dir(self):
        """Test that data directory question exists."""
        keys = [q.key for q in PATHS_QUESTIONS]
        assert 'SYMFLUENCE_DATA_DIR' in keys

    def test_all_questions_dict_structure(self):
        """Test that ALL_QUESTIONS has correct structure."""
        assert isinstance(ALL_QUESTIONS, dict)
        assert WizardPhase.ESSENTIAL in ALL_QUESTIONS
        assert WizardPhase.CALIBRATION in ALL_QUESTIONS
        assert WizardPhase.MODEL_SPECIFIC in ALL_QUESTIONS
        assert WizardPhase.PATHS in ALL_QUESTIONS
        assert WizardPhase.SUMMARY in ALL_QUESTIONS


class TestGetQuestionsForPhase:
    """Tests for get_questions_for_phase function."""

    def test_get_essential_questions(self):
        """Test getting essential phase questions."""
        state = WizardState()
        questions = get_questions_for_phase(WizardPhase.ESSENTIAL, state)

        assert len(questions) > 0
        # All essential questions without conditions should be included
        unconditional = [q for q in ESSENTIAL_QUESTIONS if q.condition is None]
        for q in unconditional:
            assert q in questions

    def test_get_calibration_questions_disabled(self):
        """Test calibration questions when calibration is disabled."""
        state = WizardState()
        state.set_answer('ENABLE_CALIBRATION', False)

        questions = get_questions_for_phase(WizardPhase.CALIBRATION, state)

        # Should only have the enable question
        keys = [q.key for q in questions]
        assert 'ENABLE_CALIBRATION' in keys
        # Detail questions should not appear
        assert 'CALIBRATION_PERIOD' not in keys

    def test_get_calibration_questions_enabled(self):
        """Test calibration questions when calibration is enabled."""
        state = WizardState()
        state.set_answer('ENABLE_CALIBRATION', True)

        questions = get_questions_for_phase(WizardPhase.CALIBRATION, state)

        keys = [q.key for q in questions]
        assert 'ENABLE_CALIBRATION' in keys
        # Detail questions should appear
        assert 'OPTIMIZATION_METRIC' in keys

    def test_get_model_specific_summa(self):
        """Test model-specific questions for SUMMA."""
        state = WizardState()
        state.set_answer('HYDROLOGICAL_MODEL', 'SUMMA')

        questions = get_questions_for_phase(WizardPhase.MODEL_SPECIFIC, state)

        keys = [q.key for q in questions]
        # SUMMA-specific questions should appear
        assert any('SUMMA' in key or 'ROUTING' in key for key in keys)
        # FUSE-specific questions should not
        fuse_keys = [q.key for q in FUSE_QUESTIONS]
        for fuse_key in fuse_keys:
            assert fuse_key not in keys

    def test_get_model_specific_fuse(self):
        """Test model-specific questions for FUSE."""
        state = WizardState()
        state.set_answer('HYDROLOGICAL_MODEL', 'FUSE')

        questions = get_questions_for_phase(WizardPhase.MODEL_SPECIFIC, state)

        keys = [q.key for q in questions]
        # FUSE-specific questions should appear
        assert 'FUSE_SPATIAL_MODE' in keys

    def test_get_summary_questions(self):
        """Test that summary phase has no questions."""
        state = WizardState()
        questions = get_questions_for_phase(WizardPhase.SUMMARY, state)

        assert len(questions) == 0


class TestGetAllQuestions:
    """Tests for get_all_questions function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        state = WizardState()
        questions = get_all_questions(state)

        assert isinstance(questions, list)

    def test_includes_essential_questions(self):
        """Test that essential questions are included."""
        state = WizardState()
        questions = get_all_questions(state)

        keys = [q.key for q in questions]
        assert 'DOMAIN_NAME' in keys

    def test_filters_by_conditions(self):
        """Test that conditional questions are filtered."""
        state = WizardState()
        state.set_answer('ENABLE_CALIBRATION', False)
        state.set_answer('HYDROLOGICAL_MODEL', 'SUMMA')

        questions = get_all_questions(state)

        keys = [q.key for q in questions]
        # FUSE questions should not appear for SUMMA model
        assert 'FUSE_SPATIAL_MODE' not in keys


class TestConditionFunctions:
    """Tests for question condition functions."""

    def test_pour_point_condition(self):
        """Test pour point coordinate condition."""
        from symfluence.cli.wizard.questions import _is_pour_point

        state = WizardState()

        state.set_answer('SPATIAL_EXTENT_TYPE', 'pour_point')
        assert _is_pour_point(state) is True

        state.set_answer('SPATIAL_EXTENT_TYPE', 'bounding_box')
        assert _is_pour_point(state) is False

    def test_bounding_box_condition(self):
        """Test bounding box coordinate condition."""
        from symfluence.cli.wizard.questions import _is_bounding_box

        state = WizardState()

        state.set_answer('SPATIAL_EXTENT_TYPE', 'bounding_box')
        assert _is_bounding_box(state) is True

        state.set_answer('SPATIAL_EXTENT_TYPE', 'pour_point')
        assert _is_bounding_box(state) is False

    def test_calibration_enabled_condition(self):
        """Test calibration enabled condition."""
        from symfluence.cli.wizard.questions import _is_calibration_enabled

        state = WizardState()

        state.set_answer('ENABLE_CALIBRATION', True)
        assert _is_calibration_enabled(state) is True

        state.set_answer('ENABLE_CALIBRATION', False)
        assert _is_calibration_enabled(state) is False

    def test_model_conditions(self):
        """Test model-specific conditions."""
        from symfluence.cli.wizard.questions import _is_fuse, _is_gr, _is_summa

        state = WizardState()

        state.set_answer('HYDROLOGICAL_MODEL', 'SUMMA')
        assert _is_summa(state) is True
        assert _is_fuse(state) is False
        assert _is_gr(state) is False

        state.set_answer('HYDROLOGICAL_MODEL', 'FUSE')
        assert _is_summa(state) is False
        assert _is_fuse(state) is True

        state.set_answer('HYDROLOGICAL_MODEL', 'GR')
        assert _is_gr(state) is True
