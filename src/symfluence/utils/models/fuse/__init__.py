"""
FUSE model module.

This module provides a clean, modular structure for the FUSE (Framework for
Understanding Structural Errors) hydrological model integration.

The module has been refactored from a single large file (fuse_utils.py, 3094 lines)
into separate components for better maintainability and organization.

Components:
    - FUSEPreProcessor: Handles FUSE preprocessing and setup (preprocessor.py, 1375 lines)
    - FUSERunner: Executes FUSE model simulations (runner.py, 1194 lines)
    - FUSEPostprocessor: Processes FUSE model outputs (postprocessor.py, 95 lines)
    - FuseDecisionAnalyzer: Analyzes FUSE model decisions (decision_analyzer.py, 393 lines)

For backward compatibility, all classes are re-exported at the package level,
so existing code using `from symfluence.utils.models.fuse_utils import FUSEPreProcessor`
will continue to work.

Refactored: 2025-12-31
"""

# Import all classes from their respective modules
from .preprocessor import FUSEPreProcessor
from .runner import FUSERunner
from .postprocessor import FUSEPostprocessor
from .decision_analyzer import FuseDecisionAnalyzer

__all__ = [
    'FUSEPreProcessor',
    'FUSERunner',
    'FUSEPostprocessor',
    'FuseDecisionAnalyzer',
]
