"""
Plotter Registry for SYMFLUENCE

Central registry for model-specific plotting components. Enables dynamic discovery
and instantiation of plotters without hardcoding model checks in ReportingManager.

Architecture:
    The PlotterRegistry enables extensible visualization workflows:

    1. Component Types (Self-Registering):
       - Model Plotters: Model-specific output visualization
       - Results Plotters: Model results and comparison visualization

    2. Registration Mechanism (Decorator Pattern):
       Each model registers its plotter using class decorators:

       @PlotterRegistry.register_plotter('SUMMA')
       class SUMMAPlotter(BasePlotter):
           def plot_results(self, **kwargs): ...

    3. Discovery and Instantiation (Factory Pattern):
       PlotterRegistry acts as factory for plotter creation:
       - Lookup by model name: PlotterRegistry.get_plotter('SUMMA')
       - Returns class (not instance) for flexible instantiation

Benefits:
    - Loose coupling: ReportingManager doesn't need model-specific imports
    - Easy extension: New models register without framework changes
    - Graceful fallback: Missing plotters return None (vs hard error)
    - Eliminates code duplication between plotters

Example Registration:
    # In models/summa/plotter.py
    from symfluence.reporting.plotter_registry import PlotterRegistry

    @PlotterRegistry.register_plotter('SUMMA')
    class SUMMAPlotter(BasePlotter):
        def plot_results(self, experiment_id: str, **kwargs):
            # SUMMA-specific visualization
            pass

Example Lookup:
    # In reporting/reporting_manager.py
    plotter_cls = PlotterRegistry.get_plotter('SUMMA')
    if plotter_cls:
        plotter = plotter_cls(config, logger, plot_config)
        plotter.plot_results(experiment_id=experiment_id)

See Also:
    - ModelRegistry: Similar pattern for preprocessors/runners/postprocessors
    - AnalysisRegistry: Registry for analysis components
"""

from typing import Any, Callable, Dict, List, Optional, Type


class PlotterRegistry:
    """Central registry for model-specific plotting components (Registry Pattern).

    Implements the Registry Pattern to enable dynamic plotter discovery
    and extensibility without tight coupling. Model-specific plotters self-register
    via decorators, allowing the framework to instantiate appropriate plotters
    based on model configuration.

    The registry supports:
    1. Model Plotters: Classes that handle model-specific output visualization
    2. Visualization Functions: Simple functions for specific visualization tasks

    Component Discovery:
        ReportingManager queries registry by model name and retrieves class/function
        references for instantiation. Returns None for unregistered models
        (graceful fallback vs hard error).

    Attributes:
        _plotters: Dict[model_name] -> plotter_class
        _visualization_funcs: Dict[(model_name, viz_type)] -> visualization_function

    Example:
        >>> # Register a plotter
        >>> @PlotterRegistry.register_plotter('SUMMA')
        ... class SUMMAPlotter:
        ...     def plot_results(self, **kwargs): ...

        >>> # Query the registry
        >>> plotter_cls = PlotterRegistry.get_plotter('SUMMA')
        >>> if plotter_cls:
        ...     plotter = plotter_cls(config, logger, plot_config)
        ...     plotter.plot_results(experiment_id='test')
    """

    _plotters: Dict[str, Type] = {}
    _visualization_funcs: Dict[str, Callable] = {}

    @classmethod
    def register_plotter(cls, model_name: str):
        """Decorator to register a plotter class for a model.

        The plotter should implement plotting methods for model-specific
        visualization (e.g., plot_results, plot_outputs, plot_streamflow).

        Args:
            model_name: Model identifier (e.g., 'SUMMA', 'FUSE', 'GR')

        Returns:
            Decorator function that registers the class

        Example:
            @PlotterRegistry.register_plotter('SUMMA')
            class SUMMAPlotter(BasePlotter):
                def __init__(self, config, logger, plot_config):
                    ...
                def plot_results(self, experiment_id, **kwargs):
                    ...
        """
        def decorator(plotter_cls: Type) -> Type:
            cls._plotters[model_name.upper()] = plotter_cls
            return plotter_cls
        return decorator

    @classmethod
    def register_visualization(cls, model_name: str, viz_type: str):
        """Decorator to register a visualization function for a specific model and type.

        Allows fine-grained registration of individual visualization functions
        for cases where a full plotter class is not needed.

        Args:
            model_name: Model identifier (e.g., 'SUMMA', 'FUSE')
            viz_type: Visualization type (e.g., 'streamflow', 'snow', 'outputs')

        Returns:
            Decorator function that registers the visualization function

        Example:
            @PlotterRegistry.register_visualization('SUMMA', 'outputs')
            def plot_summa_outputs(config, logger, experiment_id, **kwargs):
                ...
        """
        def decorator(viz_func: Callable) -> Callable:
            key = f"{model_name.upper()}_{viz_type.upper()}"
            cls._visualization_funcs[key] = viz_func
            return viz_func
        return decorator

    @classmethod
    def get_plotter(cls, model_name: str) -> Optional[Type]:
        """Get the plotter class for a model.

        Args:
            model_name: Model identifier (e.g., 'SUMMA', 'FUSE')

        Returns:
            Plotter class if registered, None otherwise
        """
        return cls._plotters.get(model_name.upper())

    @classmethod
    def get_visualization(cls, model_name: str, viz_type: str) -> Optional[Callable]:
        """Get a visualization function for a model and type.

        Args:
            model_name: Model identifier (e.g., 'SUMMA', 'FUSE')
            viz_type: Visualization type (e.g., 'streamflow', 'outputs')

        Returns:
            Visualization function if registered, None otherwise
        """
        key = f"{model_name.upper()}_{viz_type.upper()}"
        return cls._visualization_funcs.get(key)

    @classmethod
    def list_plotters(cls) -> List[str]:
        """List all models with registered plotters.

        Returns:
            Sorted list of model names with plotters
        """
        return sorted(list(cls._plotters.keys()))

    @classmethod
    def list_visualizations(cls) -> List[str]:
        """List all registered visualization function keys.

        Returns:
            Sorted list of 'MODEL_VIZTYPE' keys
        """
        return sorted(list(cls._visualization_funcs.keys()))

    @classmethod
    def has_plotter(cls, model_name: str) -> bool:
        """Check if a model has a registered plotter.

        Args:
            model_name: Model identifier

        Returns:
            True if plotter is registered, False otherwise
        """
        return model_name.upper() in cls._plotters

    @classmethod
    def has_visualization(cls, model_name: str, viz_type: str) -> bool:
        """Check if a model has a registered visualization function.

        Args:
            model_name: Model identifier
            viz_type: Visualization type

        Returns:
            True if visualization is registered, False otherwise
        """
        key = f"{model_name.upper()}_{viz_type.upper()}"
        return key in cls._visualization_funcs

    @classmethod
    def summary(cls) -> Dict[str, Any]:
        """Get a summary of all registered components.

        Returns:
            Dictionary with plotter and visualization information
        """
        return {
            'plotters': cls.list_plotters(),
            'visualizations': cls.list_visualizations(),
            'plotter_count': len(cls._plotters),
            'visualization_count': len(cls._visualization_funcs)
        }

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations. Primarily for testing."""
        cls._plotters.clear()
        cls._visualization_funcs.clear()
