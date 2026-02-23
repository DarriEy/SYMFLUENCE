"""Component Registry

Registry for hydrological model execution components including preprocessors,
runners, postprocessors, and visualizers. Implements the Registry Pattern to
decouple model implementations from the framework orchestration layer.

Component Types:
    - Preprocessors: Input data preparation (forcing, attributes, settings)
    - Runners: Model executable invocation
    - Postprocessors: Output file processing and result extraction
    - Visualizers: Model-specific diagnostic plots

Registration Pattern:
    Each model registers its components using class decorators:

    >>> @ComponentRegistry.register_preprocessor('SUMMA')
    ... class SUMMAPreprocessor: ...

    >>> @ComponentRegistry.register_runner('SUMMA', method_name='run_summa')
    ... class SUMMARunner: ...

    Registration happens at module import time (in models/__init__.py)

Discovery and Instantiation:
    ComponentRegistry acts as factory for component creation:
    - Lookup by model name: ComponentRegistry.get_preprocessor('SUMMA')
    - Returns class (not instance) for flexible instantiation
    - Allows downstream code to customize initialization
"""

import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Registry for hydrological model execution components.

    Implements the Registry Pattern to enable dynamic model discovery and
    extensibility without tight coupling. Model components self-register via
    decorators, allowing the framework to instantiate appropriate components
    based on configuration.

    The registry stores four types of model components:
    1. Preprocessors: Input preparation (forcing, attributes, parameters)
    2. Runners: Model executable execution
    3. Postprocessors: Output file processing and metric extraction
    4. Visualizers: Diagnostic plots and visualizations

    Attributes:
        _preprocessors: Dict[model_name] -> preprocessor_class
        _runners: Dict[model_name] -> runner_class
        _postprocessors: Dict[model_name] -> postprocessor_class
        _visualizers: Dict[model_name] -> visualizer_function
        _runner_methods: Dict[model_name] -> method_name (e.g., 'run', 'run_summa')

    Example Component Registration::

        @ComponentRegistry.register_preprocessor('SUMMA')
        class SUMMAPreprocessor(BaseModelPreProcessor):
            def run_preprocessing(self):
                pass

        @ComponentRegistry.register_runner('SUMMA', method_name='run_summa')
        class SUMMARunner:
            def run_summa(self):
                pass

    Example Component Lookup::

        preprocessor_cls = ComponentRegistry.get_preprocessor('SUMMA')
        if preprocessor_cls:
            preprocessor = preprocessor_cls(config, logger)
            preprocessor.run_preprocessing()
    """

    _preprocessors: Dict[str, Type] = {}
    _runners: Dict[str, Type] = {}
    _postprocessors: Dict[str, Type] = {}
    _visualizers: Dict[str, Callable] = {}
    _runner_methods: Dict[str, str] = {}

    # Mapping from component kind to (Protocol class, required attributes/methods)
    _PROTOCOL_CHECKS = {
        'preprocessor': (
            'symfluence.models.base.protocols', 'ModelPreProcessor',
            ['MODEL_NAME', 'run_preprocessing'],
        ),
        'runner': (
            'symfluence.models.base.protocols', 'ModelRunner',
            ['MODEL_NAME', 'run'],
        ),
        'postprocessor': (
            'symfluence.models.base.protocols', 'ModelPostProcessor',
            ['MODEL_NAME', 'extract_streamflow'],
        ),
    }

    @classmethod
    def _validate_protocol_conformance(cls, component_cls: Type, kind: str, model_name: str) -> None:
        """Advisory check that a registered class conforms to its Protocol.

        Verifies the class has the expected attributes/methods via ``hasattr``
        on the class itself (instantiation-free). Logs a warning if the class
        appears non-conformant but does **not** raise — registration still
        succeeds. This is advisory for the paper release, not a hard gate.
        """
        entry = cls._PROTOCOL_CHECKS.get(kind)
        if entry is None:
            return

        _module, _cls_name, required_attrs = entry

        missing = [attr for attr in required_attrs if not hasattr(component_cls, attr)]
        if missing:
            logger.warning(
                f"Registered {kind} '{component_cls.__name__}' for model "
                f"'{model_name}' is missing expected attributes/methods: "
                f"{missing}. It may not conform to the {_cls_name} Protocol."
            )

    @classmethod
    def register_preprocessor(cls, model_name: str) -> Callable[[Type], Type]:
        """Register a preprocessor class for a model.

        Args:
            model_name: Model name (e.g., 'SUMMA', 'FUSE')

        Returns:
            Decorator function that registers the class

        Example:
            >>> @ComponentRegistry.register_preprocessor('MYMODEL')
            ... class MyPreprocessor:
            ...     def run_preprocessing(self): ...
        """
        def decorator(preprocessor_cls: Type) -> Type:
            cls._validate_protocol_conformance(preprocessor_cls, 'preprocessor', model_name)
            cls._preprocessors[model_name.upper()] = preprocessor_cls
            return preprocessor_cls
        return decorator

    @classmethod
    def register_runner(
        cls, model_name: str, method_name: str = "run"
    ) -> Callable[[Type], Type]:
        """Register a runner class for a model.

        Args:
            model_name: Model name (e.g., 'SUMMA', 'FUSE')
            method_name: Name of the method to invoke for running the model

        Returns:
            Decorator function that registers the class

        Example:
            >>> @ComponentRegistry.register_runner('MYMODEL', method_name='execute')
            ... class MyRunner:
            ...     def execute(self): ...
        """
        def decorator(runner_cls: Type) -> Type:
            cls._validate_protocol_conformance(runner_cls, 'runner', model_name)
            cls._runners[model_name.upper()] = runner_cls
            cls._runner_methods[model_name.upper()] = method_name
            return runner_cls
        return decorator

    @classmethod
    def register_postprocessor(cls, model_name: str) -> Callable[[Type], Type]:
        """Register a postprocessor class for a model.

        Args:
            model_name: Model name (e.g., 'SUMMA', 'FUSE')

        Returns:
            Decorator function that registers the class

        Example:
            >>> @ComponentRegistry.register_postprocessor('MYMODEL')
            ... class MyPostprocessor:
            ...     def extract_streamflow(self): ...
        """
        def decorator(postprocessor_cls: Type) -> Type:
            cls._validate_protocol_conformance(postprocessor_cls, 'postprocessor', model_name)
            cls._postprocessors[model_name.upper()] = postprocessor_cls
            return postprocessor_cls
        return decorator

    @classmethod
    def register_visualizer(cls, model_name: str) -> Callable[[Callable], Callable]:
        """Register a visualization function for a model.

        The visualizer should be a callable with signature:
        (reporting_manager, config, project_dir, experiment_id, workflow)

        Args:
            model_name: Model name (e.g., 'SUMMA', 'FUSE')

        Returns:
            Decorator function that registers the visualizer

        Example:
            >>> @ComponentRegistry.register_visualizer('MYMODEL')
            ... def visualize_mymodel(reporting_manager, config, project_dir, ...):
            ...     pass
        """
        def decorator(visualizer_func: Callable) -> Callable:
            cls._visualizers[model_name.upper()] = visualizer_func
            return visualizer_func
        return decorator

    @classmethod
    def get_preprocessor(cls, model_name: str) -> Optional[Type]:
        """Get preprocessor class for a model.

        Args:
            model_name: Model name (case-insensitive)

        Returns:
            Preprocessor class or None if not registered
        """
        return cls._preprocessors.get(model_name.upper())

    @classmethod
    def get_runner(cls, model_name: str) -> Optional[Type]:
        """Get runner class for a model.

        Args:
            model_name: Model name (case-insensitive)

        Returns:
            Runner class or None if not registered
        """
        return cls._runners.get(model_name.upper())

    @classmethod
    def get_postprocessor(cls, model_name: str) -> Optional[Type]:
        """Get postprocessor class for a model.

        Args:
            model_name: Model name (case-insensitive)

        Returns:
            Postprocessor class or None if not registered
        """
        return cls._postprocessors.get(model_name.upper())

    @classmethod
    def get_visualizer(cls, model_name: str) -> Optional[Callable]:
        """Get visualizer function for a model.

        Args:
            model_name: Model name (case-insensitive)

        Returns:
            Visualizer function or None if not registered
        """
        return cls._visualizers.get(model_name.upper())

    @classmethod
    def get_runner_method(cls, model_name: str) -> str:
        """Get the runner method name for a model.

        Args:
            model_name: Model name (case-insensitive)

        Returns:
            Method name string (defaults to 'run' if not specified)
        """
        return cls._runner_methods.get(model_name.upper(), "run")

    @classmethod
    def list_models(cls) -> list[str]:
        """List all models with registered components.

        Returns:
            Sorted list of model names that have either a runner or preprocessor
        """
        return sorted(list(set(cls._runners.keys()) | set(cls._preprocessors.keys())))

    @classmethod
    def get_model_components(cls, model_name: str) -> dict:
        """Get all registered component classes for a model.

        Useful for debugging and introspection of model registrations.

        Args:
            model_name: Name of the model (e.g., 'SUMMA', 'GNN')

        Returns:
            Dict mapping component type to class (or None if not registered):
                - preprocessor: Preprocessor class or None
                - runner: Runner class or None
                - postprocessor: Postprocessor class or None
                - visualizer: Visualizer function or None
                - runner_method: Name of the run method (str)
        """
        model_name = model_name.upper()
        return {
            'preprocessor': cls._preprocessors.get(model_name),
            'runner': cls._runners.get(model_name),
            'postprocessor': cls._postprocessors.get(model_name),
            'visualizer': cls._visualizers.get(model_name),
            'runner_method': cls._runner_methods.get(model_name, 'run'),
        }

    @classmethod
    def validate_model_registration(
        cls,
        model_name: str,
        require_all: bool = False
    ) -> dict:
        """Validate that a model has all required components registered.

        Checks for the presence of preprocessor, runner, and postprocessor.
        Visualizer is considered optional.

        Args:
            model_name: Name of the model to validate (e.g., 'SUMMA', 'GNN')
            require_all: If True, raises ValueError when required components
                are missing. If False (default), returns validation status.

        Returns:
            Dict with keys:
                - valid: bool indicating if all required components present
                - model_name: the model name validated
                - components: dict of component_type -> class or None
                - missing: list of missing required component types
                - optional_missing: list of missing optional component types

        Raises:
            ValueError: If require_all=True and required components are missing
        """
        model_name = model_name.upper()
        components = {
            'preprocessor': cls._preprocessors.get(model_name),
            'runner': cls._runners.get(model_name),
            'postprocessor': cls._postprocessors.get(model_name),
            'visualizer': cls._visualizers.get(model_name),
        }

        required = ['preprocessor', 'runner', 'postprocessor']
        optional = ['visualizer']

        missing = [comp for comp in required if components[comp] is None]
        optional_missing = [comp for comp in optional if components[comp] is None]

        valid = len(missing) == 0

        result = {
            'valid': valid,
            'model_name': model_name,
            'components': components,
            'missing': missing,
            'optional_missing': optional_missing,
        }

        if require_all and not valid:
            raise ValueError(
                f"Model '{model_name}' has incomplete registration. "
                f"Missing required components: {missing}"
            )

        return result

    @classmethod
    def validate_all_models(
        cls,
        require_all: bool = False,
        logger: logging.Logger = None
    ) -> dict:
        """Validate registration status of all registered models.

        Checks each model returned by list_models() and reports their
        registration completeness.

        Args:
            require_all: If True, raises ValueError on first incomplete model.
                If False (default), returns status for all models.
            logger: Optional logger for warnings about incomplete registrations.

        Returns:
            Dict mapping model_name -> validation result

        Raises:
            ValueError: If require_all=True and any model is incomplete
        """
        log = logger or globals().get('logger')
        results = {}

        for model_name in cls.list_models():
            status = cls.validate_model_registration(model_name, require_all=False)
            results[model_name] = status

            if not status['valid'] and log:
                log.warning(
                    f"Model '{model_name}' has incomplete registration. "
                    f"Missing: {status['missing']}"
                )

            if require_all and not status['valid']:
                raise ValueError(
                    f"Model '{model_name}' has incomplete registration. "
                    f"Missing required components: {status['missing']}"
                )

        return results
