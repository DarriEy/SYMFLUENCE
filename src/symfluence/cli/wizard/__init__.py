"""
Interactive project initialization wizard for SYMFLUENCE.

This package provides an interactive command-line wizard for configuring
new SYMFLUENCE projects. It guides users through essential configuration
options with help text, validation, and a summary before generating files.

Usage:
    symfluence project init --interactive

Example:
    >>> from symfluence.cli.wizard import ProjectWizard
    >>> wizard = ProjectWizard()
    >>> exit_code = wizard.run(output_dir='./', scaffold=True)
"""

from .project_wizard import ProjectWizard
from .prompts import WizardPrompts
from .questions import (
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
from .state import WizardPhase, WizardState

__all__ = [
    # Main classes
    'ProjectWizard',
    'WizardPrompts',
    'WizardState',
    # Enums
    'WizardPhase',
    'QuestionType',
    # Question classes
    'Question',
    'Choice',
    # Question lists
    'ALL_QUESTIONS',
    'ESSENTIAL_QUESTIONS',
    'CALIBRATION_QUESTIONS',
    'SUMMA_QUESTIONS',
    'FUSE_QUESTIONS',
    'GR_QUESTIONS',
    'PATHS_QUESTIONS',
    # Functions
    'get_questions_for_phase',
    'get_all_questions',
]
