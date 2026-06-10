# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Analysis Registry for SYMFLUENCE

Central registry for model-specific analysis components including sensitivity
analyzers and decision/structure analyzers. Enables dynamic discovery and
instantiation of analysis components without hardcoding model checks.

Architecture:
    The AnalysisRegistry enables extensible analysis workflows:

    1. Component Types (Self-Registering):
       - Sensitivity Analyzers: Parameter importance and uncertainty analysis
       - Decision Analyzers: Model structure/decision comparison analysis

    2. Registration Mechanism (Decorator Pattern):
       Each model registers its analyzers via the unified registry:

       @R.sensitivity_analyzers.add('SUMMA')
       class SummaSensitivityAnalyzer: ...

       @R.decision_analyzers.add('SUMMA')
       class SummaStructureAnalyzer: ...

    3. Discovery and Instantiation (Factory Pattern):
       AnalysisRegistry acts as factory for component creation:
       - Lookup by model name: AnalysisRegistry.get_sensitivity_analyzer('SUMMA')
       - Returns class (not instance) for flexible instantiation

Benefits:
    - Loose coupling: AnalysisManager doesn't need model-specific imports
    - Easy extension: New models register without framework changes
    - Graceful fallback: Missing analyzers return None (vs hard error)
    - Testing: Mock analyzers can replace production implementations

Example Registration:
    # In models/summa/__init__.py
    from symfluence.core.registries import R

    @R.decision_analyzers.add('SUMMA')
    class SummaStructureAnalyzer(BaseStructureEnsembleAnalyzer):
        def run_full_analysis(self):
            ...

Example Lookup:
    # In evaluation/analysis_manager.py
    analyzer_cls = AnalysisRegistry.get_decision_analyzer('SUMMA')
    if analyzer_cls:
        analyzer = analyzer_cls(config, logger, reporting_manager)
        results = analyzer.run_full_analysis()

See Also:
    - ModelRegistry: Similar pattern for preprocessors/runners/postprocessors
    - EvaluationRegistry: Registry for variable-type evaluators
"""
from __future__ import annotations

from typing import Dict, List, Optional, Type

from symfluence.core.registries import R


class AnalysisRegistry:
    """Central registry for model-specific analysis components (Registry Pattern).

    Implements the Registry Pattern to enable dynamic analysis component discovery
    and extensibility without tight coupling. Model-specific analyzers self-register
    via ``R.sensitivity_analyzers.add()`` / ``R.decision_analyzers.add()`` /
    ``R.koopman_analyzers.add()``, allowing the framework to instantiate
    appropriate components based on model configuration.

    The registry stores two types of analysis components:
    1. Sensitivity Analyzers: Parameter importance/uncertainty analysis
    2. Decision Analyzers: Model structure/decision comparison (ensemble analysis)

    Component Discovery:
        AnalysisManager queries registry by model name and retrieves class
        references for instantiation. Returns None for unregistered models
        (graceful fallback vs hard error).

    Example:
        >>> # Register a decision analyzer
        >>> @R.decision_analyzers.add('SUMMA')
        ... class SummaStructureAnalyzer:
        ...     def run_full_analysis(self): ...

        >>> # Query the registry
        >>> analyzer_cls = AnalysisRegistry.get_decision_analyzer('SUMMA')
        >>> if analyzer_cls:
        ...     analyzer = analyzer_cls(config, logger)
        ...     results = analyzer.run_full_analysis()
    """

    @classmethod
    def get_sensitivity_analyzer(cls, model_name: str) -> Optional[Type]:
        """Get the sensitivity analyzer class for a model.

        Args:
            model_name: Model identifier (e.g., 'SUMMA', 'FUSE')

        Returns:
            Analyzer class if registered, None otherwise
        """
        return R.sensitivity_analyzers.get(model_name.upper())

    @classmethod
    def get_decision_analyzer(cls, model_name: str) -> Optional[Type]:
        """Get the decision/structure analyzer class for a model.

        Args:
            model_name: Model identifier (e.g., 'SUMMA', 'FUSE')

        Returns:
            Analyzer class if registered, None otherwise
        """
        return R.decision_analyzers.get(model_name.upper())

    @classmethod
    def list_sensitivity_analyzers(cls) -> List[str]:
        """List all models with registered sensitivity analyzers.

        Returns:
            Sorted list of model names with sensitivity analyzers
        """
        return R.sensitivity_analyzers.keys()

    @classmethod
    def list_decision_analyzers(cls) -> List[str]:
        """List all models with registered decision analyzers.

        Returns:
            Sorted list of model names with decision analyzers
        """
        return R.decision_analyzers.keys()

    @classmethod
    def list_all_analyzers(cls) -> Dict[str, List[str]]:
        """List all registered analyzers by type.

        Returns:
            Dictionary with 'sensitivity' and 'decision' keys containing
            lists of registered model names
        """
        return {
            'sensitivity': cls.list_sensitivity_analyzers(),
            'decision': cls.list_decision_analyzers(),
            'koopman': cls.list_koopman_analyzers(),
        }

    @classmethod
    def has_sensitivity_analyzer(cls, model_name: str) -> bool:
        """Check if a model has a registered sensitivity analyzer.

        Args:
            model_name: Model identifier

        Returns:
            True if analyzer is registered, False otherwise
        """
        return model_name.upper() in R.sensitivity_analyzers

    @classmethod
    def has_decision_analyzer(cls, model_name: str) -> bool:
        """Check if a model has a registered decision analyzer.

        Args:
            model_name: Model identifier

        Returns:
            True if analyzer is registered, False otherwise
        """
        return model_name.upper() in R.decision_analyzers

    # ------------------------------------------------------------------
    # Koopman analyzers
    # ------------------------------------------------------------------

    @classmethod
    def get_koopman_analyzer(cls, name: str = "DEFAULT") -> Optional[Type]:
        """Get a registered Koopman analyzer class.

        Args:
            name: Analyzer identifier (default "DEFAULT")

        Returns:
            Analyzer class if registered, None otherwise
        """
        return R.koopman_analyzers.get(name.upper())

    @classmethod
    def list_koopman_analyzers(cls) -> List[str]:
        """List all registered Koopman analyzers.

        Returns:
            Sorted list of registered Koopman analyzer names
        """
        return R.koopman_analyzers.keys()

    @classmethod
    def has_koopman_analyzer(cls, name: str = "DEFAULT") -> bool:
        """Check if a Koopman analyzer is registered.

        Args:
            name: Analyzer identifier

        Returns:
            True if analyzer is registered, False otherwise
        """
        return name.upper() in R.koopman_analyzers
