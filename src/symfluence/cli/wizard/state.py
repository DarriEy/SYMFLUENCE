"""
Wizard state management for SYMFLUENCE project initialization.

This module provides state tracking for the interactive wizard,
including answer storage, phase management, and navigation history.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class WizardPhase(Enum):
    """
    Wizard phases representing different stages of configuration.

    The wizard progresses through these phases sequentially,
    with some phases being conditional on previous answers.
    """
    ESSENTIAL = auto()       # Required settings: domain, model, dates, coordinates
    CALIBRATION = auto()     # Calibration settings (conditional)
    MODEL_SPECIFIC = auto()  # Model-specific parameters (conditional)
    PATHS = auto()           # Data and code directories
    SUMMARY = auto()         # Review and confirm


@dataclass
class WizardState:
    """
    Manages wizard state including answers, navigation, and validation.

    Tracks all user responses, maintains navigation history for back
    functionality, and provides utilities for answer retrieval.

    Attributes:
        answers: Dictionary mapping question keys to user responses
        current_phase: Current wizard phase
        history: Stack of (phase, key) tuples for back navigation
        skipped_questions: Set of question keys that were skipped
        validation_errors: Dictionary of validation errors by question key
    """

    answers: Dict[str, Any] = field(default_factory=dict)
    current_phase: WizardPhase = WizardPhase.ESSENTIAL
    history: List[tuple] = field(default_factory=list)
    skipped_questions: set = field(default_factory=set)
    validation_errors: Dict[str, str] = field(default_factory=dict)

    def set_answer(self, key: str, value: Any) -> None:
        """
        Store an answer and record in history for back navigation.

        Args:
            key: Question identifier
            value: User's response
        """
        self.answers[key] = value
        self.history.append((self.current_phase, key))
        # Clear any previous validation error for this key
        self.validation_errors.pop(key, None)

    def get_answer(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a stored answer.

        Args:
            key: Question identifier
            default: Value to return if key not found

        Returns:
            The stored answer or default value
        """
        return self.answers.get(key, default)

    def has_answer(self, key: str) -> bool:
        """
        Check if a question has been answered.

        Args:
            key: Question identifier

        Returns:
            True if the question has an answer
        """
        return key in self.answers and self.answers[key] is not None

    def go_back(self) -> Optional[tuple]:
        """
        Navigate back to previous question.

        Returns:
            Tuple of (phase, key) for the previous question, or None if at start
        """
        if not self.history:
            return None

        # Remove current question from history
        phase, key = self.history.pop()

        # Remove the answer for the previous question
        self.answers.pop(key, None)

        # Update current phase
        self.current_phase = phase

        return (phase, key)

    def skip_question(self, key: str) -> None:
        """
        Mark a question as skipped.

        Args:
            key: Question identifier
        """
        self.skipped_questions.add(key)

    def is_skipped(self, key: str) -> bool:
        """
        Check if a question was skipped.

        Args:
            key: Question identifier

        Returns:
            True if the question was skipped
        """
        return key in self.skipped_questions

    def set_validation_error(self, key: str, error: str) -> None:
        """
        Store a validation error for a question.

        Args:
            key: Question identifier
            error: Error message
        """
        self.validation_errors[key] = error

    def get_validation_error(self, key: str) -> Optional[str]:
        """
        Get validation error for a question.

        Args:
            key: Question identifier

        Returns:
            Error message or None
        """
        return self.validation_errors.get(key)

    def clear_validation_error(self, key: str) -> None:
        """
        Clear validation error for a question.

        Args:
            key: Question identifier
        """
        self.validation_errors.pop(key, None)

    def advance_phase(self) -> bool:
        """
        Advance to the next wizard phase.

        Returns:
            True if advanced successfully, False if already at last phase
        """
        phases = list(WizardPhase)
        current_idx = phases.index(self.current_phase)

        if current_idx < len(phases) - 1:
            self.current_phase = phases[current_idx + 1]
            return True
        return False

    def to_config_dict(self) -> Dict[str, Any]:
        """
        Convert wizard answers to configuration dictionary format.

        Maps wizard question keys to SYMFLUENCE configuration keys.

        Returns:
            Dictionary suitable for config file generation
        """
        config = {}

        # Map wizard keys to config keys
        key_mapping = {
            'DOMAIN_NAME': 'DOMAIN_NAME',
            'EXPERIMENT_ID': 'EXPERIMENT_ID',
            'EXPERIMENT_TIME_START': 'EXPERIMENT_TIME_START',
            'EXPERIMENT_TIME_END': 'EXPERIMENT_TIME_END',
            'POUR_POINT_LAT': 'POUR_POINT_LAT',
            'POUR_POINT_LON': 'POUR_POINT_LON',
            'BOUNDING_BOX_COORDS': 'BOUNDING_BOX_COORDS',
            'DOMAIN_DEFINITION_METHOD': 'DOMAIN_DEFINITION_METHOD',
            'SUB_GRID_DISCRETIZATION': 'SUB_GRID_DISCRETIZATION',
            'HYDROLOGICAL_MODEL': 'HYDROLOGICAL_MODEL',
            'FORCING_DATASET': 'FORCING_DATASET',
            'CALIBRATION_PERIOD': 'CALIBRATION_PERIOD',
            'EVALUATION_PERIOD': 'EVALUATION_PERIOD',
            'OPTIMIZATION_METRIC': 'OPTIMIZATION_METRIC',
            'NUMBER_OF_ITERATIONS': 'NUMBER_OF_ITERATIONS',
            'SYMFLUENCE_DATA_DIR': 'SYMFLUENCE_DATA_DIR',
            'SYMFLUENCE_CODE_DIR': 'SYMFLUENCE_CODE_DIR',
        }

        for wizard_key, config_key in key_mapping.items():
            if wizard_key in self.answers and self.answers[wizard_key] is not None:
                value = self.answers[wizard_key]

                # Format dates
                if wizard_key in ('EXPERIMENT_TIME_START',):
                    value = f"{value} 00:00"
                elif wizard_key in ('EXPERIMENT_TIME_END',):
                    value = f"{value} 23:00"

                config[config_key] = value

        # Handle calibration enable flag
        if self.get_answer('ENABLE_CALIBRATION', False):
            # Calibration is enabled, keep calibration-related settings
            pass
        else:
            # Remove calibration settings if disabled
            for key in ['CALIBRATION_PERIOD', 'EVALUATION_PERIOD',
                       'OPTIMIZATION_METRIC', 'NUMBER_OF_ITERATIONS']:
                config.pop(key, None)

        return config

    def reset(self) -> None:
        """Reset wizard state to initial values."""
        self.answers.clear()
        self.current_phase = WizardPhase.ESSENTIAL
        self.history.clear()
        self.skipped_questions.clear()
        self.validation_errors.clear()
