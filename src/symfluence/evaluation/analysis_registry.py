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
       Each model registers its analyzers using class decorators:

       @AnalysisRegistry.register_sensitivity_analyzer('SUMMA')
       class SummaSensitivityAnalyzer: ...

       @AnalysisRegistry.register_decision_analyzer('SUMMA')
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
    from symfluence.evaluation.analysis_registry import AnalysisRegistry

    @AnalysisRegistry.register_decision_analyzer('SUMMA')
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

from typing import Dict, Type, Optional, List
import logging


class AnalysisRegistry:
    """Central registry for model-specific analysis components (Registry Pattern).

    Implements the Registry Pattern to enable dynamic analysis component discovery
    and extensibility without tight coupling. Model-specific analyzers self-register
    via decorators, allowing the framework to instantiate appropriate components
    based on model configuration.

    The registry stores two types of analysis components:
    1. Sensitivity Analyzers: Parameter importance/uncertainty analysis
    2. Decision Analyzers: Model structure/decision comparison (ensemble analysis)

    Component Discovery:
        AnalysisManager queries registry by model name and retrieves class
        references for instantiation. Returns None for unregistered models
        (graceful fallback vs hard error).

    Attributes:
        _sensitivity_analyzers: Dict[model_name] -> sensitivity_analyzer_class
        _decision_analyzers: Dict[model_name] -> decision_analyzer_class

    Example:
        >>> # Register a decision analyzer
        >>> @AnalysisRegistry.register_decision_analyzer('SUMMA')
        ... class SummaStructureAnalyzer:
        ...     def run_full_analysis(self): ...

        >>> # Query the registry
        >>> analyzer_cls = AnalysisRegistry.get_decision_analyzer('SUMMA')
        >>> if analyzer_cls:
        ...     analyzer = analyzer_cls(config, logger)
        ...     results = analyzer.run_full_analysis()
    """

    _sensitivity_analyzers: Dict[str, Type] = {}
    _decision_analyzers: Dict[str, Type] = {}

    @classmethod
    def register_sensitivity_analyzer(cls, model_name: str):
        """Decorator to register a sensitivity analyzer for a model.

        The analyzer should implement a `run_sensitivity_analysis(results_file)` method
        that returns sensitivity analysis results (typically a Dict).

        Args:
            model_name: Model identifier (e.g., 'SUMMA', 'FUSE', 'GR')

        Returns:
            Decorator function that registers the class

        Example:
            @AnalysisRegistry.register_sensitivity_analyzer('SUMMA')
            class SummaSensitivityAnalyzer:
                def __init__(self, config, logger, reporting_manager=None):
                    ...
                def run_sensitivity_analysis(self, results_file):
                    ...
        """
        def decorator(analyzer_cls: Type) -> Type:
            cls._sensitivity_analyzers[model_name.upper()] = analyzer_cls
            return analyzer_cls
        return decorator

    @classmethod
    def register_decision_analyzer(cls, model_name: str):
        """Decorator to register a decision/structure analyzer for a model.

        The analyzer should extend BaseStructureEnsembleAnalyzer or implement
        a compatible interface with `run_full_analysis()` method that returns
        a tuple of (results_file_path, best_combinations_dict).

        Args:
            model_name: Model identifier (e.g., 'SUMMA', 'FUSE', 'GR')

        Returns:
            Decorator function that registers the class

        Example:
            @AnalysisRegistry.register_decision_analyzer('SUMMA')
            class SummaStructureAnalyzer(BaseStructureEnsembleAnalyzer):
                def run_full_analysis(self):
                    return results_file, best_combinations
        """
        def decorator(analyzer_cls: Type) -> Type:
            cls._decision_analyzers[model_name.upper()] = analyzer_cls
            return analyzer_cls
        return decorator

    @classmethod
    def get_sensitivity_analyzer(cls, model_name: str) -> Optional[Type]:
        """Get the sensitivity analyzer class for a model.

        Args:
            model_name: Model identifier (e.g., 'SUMMA', 'FUSE')

        Returns:
            Analyzer class if registered, None otherwise
        """
        return cls._sensitivity_analyzers.get(model_name.upper())

    @classmethod
    def get_decision_analyzer(cls, model_name: str) -> Optional[Type]:
        """Get the decision/structure analyzer class for a model.

        Args:
            model_name: Model identifier (e.g., 'SUMMA', 'FUSE')

        Returns:
            Analyzer class if registered, None otherwise
        """
        return cls._decision_analyzers.get(model_name.upper())

    @classmethod
    def list_sensitivity_analyzers(cls) -> List[str]:
        """List all models with registered sensitivity analyzers.

        Returns:
            Sorted list of model names with sensitivity analyzers
        """
        return sorted(list(cls._sensitivity_analyzers.keys()))

    @classmethod
    def list_decision_analyzers(cls) -> List[str]:
        """List all models with registered decision analyzers.

        Returns:
            Sorted list of model names with decision analyzers
        """
        return sorted(list(cls._decision_analyzers.keys()))

    @classmethod
    def list_all_analyzers(cls) -> Dict[str, List[str]]:
        """List all registered analyzers by type.

        Returns:
            Dictionary with 'sensitivity' and 'decision' keys containing
            lists of registered model names
        """
        return {
            'sensitivity': cls.list_sensitivity_analyzers(),
            'decision': cls.list_decision_analyzers()
        }

    @classmethod
    def has_sensitivity_analyzer(cls, model_name: str) -> bool:
        """Check if a model has a registered sensitivity analyzer.

        Args:
            model_name: Model identifier

        Returns:
            True if analyzer is registered, False otherwise
        """
        return model_name.upper() in cls._sensitivity_analyzers

    @classmethod
    def has_decision_analyzer(cls, model_name: str) -> bool:
        """Check if a model has a registered decision analyzer.

        Args:
            model_name: Model identifier

        Returns:
            True if analyzer is registered, False otherwise
        """
        return model_name.upper() in cls._decision_analyzers
