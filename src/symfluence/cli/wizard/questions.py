"""
Question definitions for SYMFLUENCE project initialization wizard.

This module defines all questions presented during the interactive wizard,
including their types, validation rules, and conditional display logic.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

from .state import WizardPhase, WizardState


class QuestionType(Enum):
    """Types of questions the wizard can present."""
    TEXT = auto()         # Free-form text input
    CHOICE = auto()       # Select from predefined options
    CONFIRM = auto()      # Yes/no confirmation
    DATE = auto()         # Date in YYYY-MM-DD format
    COORDINATES = auto()  # Lat/lon coordinate pair
    INTEGER = auto()      # Integer value
    PATH = auto()         # File system path


@dataclass
class Choice:
    """A single choice option for CHOICE type questions."""
    value: str
    label: str
    description: Optional[str] = None


@dataclass
class Question:
    """
    Definition of a wizard question.

    Attributes:
        key: Unique identifier for storing the answer
        prompt: The question text shown to the user
        question_type: Type of input expected
        help_text: Optional detailed help shown on request
        default: Default value (can be callable for dynamic defaults)
        choices: List of choices for CHOICE type questions
        validator: Optional validation function returning (bool, error_message)
        condition: Optional function to determine if question should be shown
        phase: Wizard phase this question belongs to
    """
    key: str
    prompt: str
    question_type: QuestionType
    help_text: Optional[str] = None
    default: Optional[Union[Any, Callable[[WizardState], Any]]] = None
    choices: List[Choice] = field(default_factory=list)
    validator: Optional[Callable[[Any, WizardState], tuple]] = None
    condition: Optional[Callable[[WizardState], bool]] = None
    phase: WizardPhase = WizardPhase.ESSENTIAL

    def get_default(self, state: WizardState) -> Any:
        """
        Get the default value, evaluating callable defaults.

        Args:
            state: Current wizard state for dynamic defaults

        Returns:
            The default value for this question
        """
        if callable(self.default):
            return self.default(state)
        return self.default

    def should_show(self, state: WizardState) -> bool:
        """
        Determine if this question should be displayed.

        Args:
            state: Current wizard state for conditional logic

        Returns:
            True if question should be shown
        """
        if self.condition is None:
            return True
        return self.condition(state)


# =============================================================================
# Condition Functions
# =============================================================================

def _is_pour_point(state: WizardState) -> bool:
    """Show pour point question if spatial extent type is pour_point."""
    return state.get_answer('SPATIAL_EXTENT_TYPE') == 'pour_point'


def _is_bounding_box(state: WizardState) -> bool:
    """Show bounding box question if spatial extent type is bounding_box."""
    return state.get_answer('SPATIAL_EXTENT_TYPE') == 'bounding_box'


def _is_calibration_enabled(state: WizardState) -> bool:
    """Show calibration questions if calibration is enabled."""
    return state.get_answer('ENABLE_CALIBRATION', False)


def _is_summa(state: WizardState) -> bool:
    """Show SUMMA-specific questions."""
    return state.get_answer('HYDROLOGICAL_MODEL') == 'SUMMA'


def _is_fuse(state: WizardState) -> bool:
    """Show FUSE-specific questions."""
    return state.get_answer('HYDROLOGICAL_MODEL') == 'FUSE'


def _is_gr(state: WizardState) -> bool:
    """Show GR-specific questions."""
    return state.get_answer('HYDROLOGICAL_MODEL') == 'GR'


def _needs_discretization(state: WizardState) -> bool:
    """Show discretization if domain method supports it."""
    method = state.get_answer('DOMAIN_DEFINITION_METHOD')
    return method in ('delineate', 'subset')


# =============================================================================
# Dynamic Default Functions
# =============================================================================

def _default_experiment_id(state: WizardState) -> str:
    """Generate default experiment ID from domain name."""
    domain = state.get_answer('DOMAIN_NAME', 'unnamed')
    return f"{domain}_run1"


def _default_data_dir(state: WizardState) -> str:
    """Detect or suggest default data directory."""
    import os
    from pathlib import Path

    # Check environment variable first
    env_dir = os.environ.get('SYMFLUENCE_DATA_DIR')
    if env_dir and Path(env_dir).exists():
        return env_dir

    # Check common locations
    home = Path.home()
    common_locations = [
        home / 'symfluence_data',
        home / 'data' / 'symfluence',
        Path('/data') / 'symfluence',
    ]

    for loc in common_locations:
        if loc.exists():
            return str(loc)

    # Default to home directory
    return str(home / 'symfluence_data')


def _default_code_dir(state: WizardState) -> str:
    """Detect SYMFLUENCE code directory."""
    from pathlib import Path

    # Try to detect from current installation
    try:
        import symfluence
        pkg_path = Path(symfluence.__file__).resolve()
        # Go up: __init__.py -> symfluence -> src -> SYMFLUENCE
        repo_root = pkg_path.parent.parent.parent
        if (repo_root / '.git').exists():
            return str(repo_root)
    except (ImportError, AttributeError, OSError):
        pass

    return str(Path.home() / 'SYMFLUENCE')


# =============================================================================
# Question Lists by Phase
# =============================================================================

ESSENTIAL_QUESTIONS: List[Question] = [
    Question(
        key='DOMAIN_NAME',
        prompt='What is the name of your study domain?',
        question_type=QuestionType.TEXT,
        help_text='A short identifier for your study area (e.g., "bow_river", "provo_utah"). '
                  'Use lowercase letters, numbers, and underscores only.',
        phase=WizardPhase.ESSENTIAL,
    ),
    Question(
        key='EXPERIMENT_ID',
        prompt='What is the experiment identifier?',
        question_type=QuestionType.TEXT,
        help_text='A unique name for this experiment run (e.g., "calibration_v1", "baseline_run"). '
                  'Helps distinguish different configurations for the same domain.',
        default=_default_experiment_id,
        phase=WizardPhase.ESSENTIAL,
    ),
    Question(
        key='EXPERIMENT_TIME_START',
        prompt='What is the simulation start date?',
        question_type=QuestionType.DATE,
        help_text='The beginning of your simulation period in YYYY-MM-DD format.',
        default='2010-01-01',
        phase=WizardPhase.ESSENTIAL,
    ),
    Question(
        key='EXPERIMENT_TIME_END',
        prompt='What is the simulation end date?',
        question_type=QuestionType.DATE,
        help_text='The end of your simulation period in YYYY-MM-DD format.',
        default='2020-12-31',
        phase=WizardPhase.ESSENTIAL,
    ),
    Question(
        key='SPATIAL_EXTENT_TYPE',
        prompt='How do you want to define the spatial extent?',
        question_type=QuestionType.CHOICE,
        help_text='Choose whether to specify a single pour point (watershed outlet) '
                  'or a bounding box for your study area.',
        choices=[
            Choice('pour_point', 'Pour Point', 'Single outlet point for watershed delineation'),
            Choice('bounding_box', 'Bounding Box', 'Rectangular region defined by coordinates'),
        ],
        default='pour_point',
        phase=WizardPhase.ESSENTIAL,
    ),
    Question(
        key='POUR_POINT_COORDS',
        prompt='Enter the pour point coordinates (lat/lon):',
        question_type=QuestionType.COORDINATES,
        help_text='The latitude and longitude of your watershed outlet point. '
                  'Format: latitude/longitude (e.g., 51.1722/-115.5717)',
        condition=_is_pour_point,
        phase=WizardPhase.ESSENTIAL,
    ),
    Question(
        key='BOUNDING_BOX_COORDS',
        prompt='Enter the bounding box coordinates (lat_max/lon_min/lat_min/lon_max):',
        question_type=QuestionType.TEXT,
        help_text='The corners of your study region. '
                  'Format: lat_max/lon_min/lat_min/lon_max (e.g., 55.0/10.0/45.0/20.0)',
        condition=_is_bounding_box,
        phase=WizardPhase.ESSENTIAL,
    ),
    Question(
        key='DOMAIN_DEFINITION_METHOD',
        prompt='How should the domain be defined?',
        question_type=QuestionType.CHOICE,
        help_text='Method for defining the hydrological model domain.',
        choices=[
            Choice('lumped', 'Lumped', 'Single unit - simplest approach'),
            Choice('delineate', 'Delineate', 'Automatically delineate watershed from DEM'),
            Choice('subset', 'Subset', 'Extract from existing geofabric'),
            Choice('point', 'Point', 'Single point simulation'),
        ],
        default='lumped',
        phase=WizardPhase.ESSENTIAL,
    ),
    Question(
        key='SUB_GRID_DISCRETIZATION',
        prompt='What sub-grid discretization method should be used?',
        question_type=QuestionType.CHOICE,
        help_text='How to divide the domain into computational units.',
        choices=[
            Choice('lumped', 'Lumped', 'Single unit for the entire domain'),
            Choice('elevation', 'Elevation Bands', 'Divide by elevation zones'),
            Choice('GRU', 'GRUs', 'Grouped Response Units'),
            Choice('HRU', 'HRUs', 'Hydrological Response Units'),
        ],
        default='lumped',
        condition=_needs_discretization,
        phase=WizardPhase.ESSENTIAL,
    ),
    Question(
        key='HYDROLOGICAL_MODEL',
        prompt='Which hydrological model do you want to use?',
        question_type=QuestionType.CHOICE,
        help_text='The hydrological model for simulating water processes.',
        choices=[
            Choice('SUMMA', 'SUMMA', 'Structure for Unifying Multiple Modeling Alternatives'),
            Choice('FUSE', 'FUSE', 'Framework for Understanding Structural Errors'),
            Choice('GR', 'GR', 'GR4J/GR6J rainfall-runoff models'),
            Choice('HYPE', 'HYPE', 'Hydrological Predictions for the Environment'),
            Choice('RHESSys', 'RHESSys', 'Regional Hydro-Ecologic Simulation System'),
            Choice('NGEN', 'NGEN', 'Next Generation Water Resources Modeling Framework'),
        ],
        default='SUMMA',
        phase=WizardPhase.ESSENTIAL,
    ),
    Question(
        key='FORCING_DATASET',
        prompt='Which meteorological forcing dataset should be used?',
        question_type=QuestionType.CHOICE,
        help_text='Source of meteorological input data (precipitation, temperature, etc.).',
        choices=[
            Choice('ERA5', 'ERA5', 'ECMWF Reanalysis v5 - global coverage'),
            Choice('RDRS', 'RDRS', 'Regional Deterministic Reanalysis System - Canada'),
            Choice('NLDAS', 'NLDAS', 'North American Land Data Assimilation System - USA'),
            Choice('CONUS404', 'CONUS404', 'Continental US 4km WRF reanalysis'),
        ],
        default='ERA5',
        phase=WizardPhase.ESSENTIAL,
    ),
]


CALIBRATION_QUESTIONS: List[Question] = [
    Question(
        key='ENABLE_CALIBRATION',
        prompt='Do you want to enable model calibration?',
        question_type=QuestionType.CONFIRM,
        help_text='Calibration optimizes model parameters using observed data.',
        default=False,
        phase=WizardPhase.CALIBRATION,
    ),
    Question(
        key='CALIBRATION_PERIOD',
        prompt='What is the calibration period? (YYYY-MM-DD to YYYY-MM-DD)',
        question_type=QuestionType.TEXT,
        help_text='Date range for parameter optimization. Should be a subset of simulation period.',
        condition=_is_calibration_enabled,
        phase=WizardPhase.CALIBRATION,
    ),
    Question(
        key='EVALUATION_PERIOD',
        prompt='What is the evaluation period? (YYYY-MM-DD to YYYY-MM-DD)',
        question_type=QuestionType.TEXT,
        help_text='Date range for independent model evaluation. Should not overlap with calibration.',
        condition=_is_calibration_enabled,
        phase=WizardPhase.CALIBRATION,
    ),
    Question(
        key='OPTIMIZATION_METRIC',
        prompt='Which optimization metric should be used?',
        question_type=QuestionType.CHOICE,
        help_text='Objective function for parameter optimization.',
        choices=[
            Choice('KGE', 'KGE', 'Kling-Gupta Efficiency - recommended'),
            Choice('NSE', 'NSE', 'Nash-Sutcliffe Efficiency'),
            Choice('RMSE', 'RMSE', 'Root Mean Square Error'),
            Choice('MAE', 'MAE', 'Mean Absolute Error'),
        ],
        default='KGE',
        condition=_is_calibration_enabled,
        phase=WizardPhase.CALIBRATION,
    ),
    Question(
        key='NUMBER_OF_ITERATIONS',
        prompt='How many optimization iterations?',
        question_type=QuestionType.INTEGER,
        help_text='Number of iterations for the optimization algorithm. '
                  'More iterations may improve results but increase runtime.',
        default=1000,
        condition=_is_calibration_enabled,
        phase=WizardPhase.CALIBRATION,
    ),
]


SUMMA_QUESTIONS: List[Question] = [
    Question(
        key='SUMMA_SPATIAL_MODE',
        prompt='What spatial mode should SUMMA use?',
        question_type=QuestionType.CHOICE,
        help_text='SUMMA spatial configuration.',
        choices=[
            Choice('lumped', 'Lumped', 'Single computational unit'),
            Choice('semi_distributed', 'Semi-distributed', 'Multiple GRUs/HRUs'),
            Choice('distributed', 'Distributed', 'Fully distributed grid'),
        ],
        default='lumped',
        condition=_is_summa,
        phase=WizardPhase.MODEL_SPECIFIC,
    ),
    Question(
        key='ROUTING_MODEL',
        prompt='Which routing model should be used with SUMMA?',
        question_type=QuestionType.CHOICE,
        help_text='Model for routing water through the river network.',
        choices=[
            Choice('mizuRoute', 'mizuRoute', 'Mizukami routing model - recommended'),
            Choice('none', 'None', 'No routing - lumped output only'),
        ],
        default='mizuRoute',
        condition=_is_summa,
        phase=WizardPhase.MODEL_SPECIFIC,
    ),
]


FUSE_QUESTIONS: List[Question] = [
    Question(
        key='FUSE_SPATIAL_MODE',
        prompt='What spatial mode should FUSE use?',
        question_type=QuestionType.CHOICE,
        help_text='FUSE spatial configuration.',
        choices=[
            Choice('lumped', 'Lumped', 'Single computational unit'),
            Choice('semi_distributed', 'Semi-distributed', 'Multiple elevation bands'),
            Choice('distributed', 'Distributed', 'Grid-based'),
        ],
        default='lumped',
        condition=_is_fuse,
        phase=WizardPhase.MODEL_SPECIFIC,
    ),
]


GR_QUESTIONS: List[Question] = [
    Question(
        key='GR_MODEL_TYPE',
        prompt='Which GR model variant should be used?',
        question_type=QuestionType.CHOICE,
        help_text='GR model family variant.',
        choices=[
            Choice('GR4J', 'GR4J', '4-parameter daily model'),
            Choice('GR6J', 'GR6J', '6-parameter daily model with groundwater'),
        ],
        default='GR4J',
        condition=_is_gr,
        phase=WizardPhase.MODEL_SPECIFIC,
    ),
]


PATHS_QUESTIONS: List[Question] = [
    Question(
        key='SYMFLUENCE_DATA_DIR',
        prompt='Where should SYMFLUENCE store data?',
        question_type=QuestionType.PATH,
        help_text='Root directory for SYMFLUENCE domain data, forcing files, and outputs. '
                  'Will be created if it does not exist.',
        default=_default_data_dir,
        phase=WizardPhase.PATHS,
    ),
    Question(
        key='SYMFLUENCE_CODE_DIR',
        prompt='Where is the SYMFLUENCE code directory?',
        question_type=QuestionType.PATH,
        help_text='Path to SYMFLUENCE repository root. Auto-detected if installed as package.',
        default=_default_code_dir,
        phase=WizardPhase.PATHS,
    ),
]


# All questions organized by phase
ALL_QUESTIONS: Dict[WizardPhase, List[Question]] = {
    WizardPhase.ESSENTIAL: ESSENTIAL_QUESTIONS,
    WizardPhase.CALIBRATION: CALIBRATION_QUESTIONS,
    WizardPhase.MODEL_SPECIFIC: SUMMA_QUESTIONS + FUSE_QUESTIONS + GR_QUESTIONS,
    WizardPhase.PATHS: PATHS_QUESTIONS,
    WizardPhase.SUMMARY: [],  # Summary phase has no questions
}


def get_questions_for_phase(
    phase: WizardPhase,
    state: WizardState
) -> List[Question]:
    """
    Get questions for a phase, filtered by conditions.

    Args:
        phase: The wizard phase
        state: Current wizard state for condition evaluation

    Returns:
        List of questions that should be shown for this phase
    """
    questions = ALL_QUESTIONS.get(phase, [])
    return [q for q in questions if q.should_show(state)]


def get_all_questions(state: WizardState) -> List[Question]:
    """
    Get all questions that should be shown based on current state.

    Args:
        state: Current wizard state

    Returns:
        Ordered list of all applicable questions
    """
    all_q = []
    for phase in WizardPhase:
        all_q.extend(get_questions_for_phase(phase, state))
    return all_q
