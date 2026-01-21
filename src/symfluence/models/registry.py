"""Model Registry

Central plugin registry system enabling dynamic model component discovery and
instantiation. Implements the Registry Pattern to decouple model implementations
from the framework orchestration layer, allowing new models to be added without
modifying core code.

Architecture:
    The ModelRegistry enables SYMFLUENCE's extensible model architecture:

    1. Component Types (Self-Registering):
       - Preprocessors: Input data preparation (forcing, attributes, settings)
       - Runners: Model executable invocation
       - Postprocessors: Output file processing and result extraction
       - Visualizers: Model-specific diagnostic plots

    2. Registration Mechanism (Decorator Pattern):
       Each model registers its components using class decorators:
       @ModelRegistry.register_preprocessor('SUMMA')
       class SUMMAPreprocessor: ...

       @ModelRegistry.register_runner('SUMMA', method_name='run_summa')
       class SUMMARunner: ...

       Registration happens at module import time (in models/__init__.py)

    3. Discovery and Instantiation (Factory Pattern):
       ModelRegistry acts as factory for component creation:
       - Lookup by model name: ModelRegistry.get_preprocessor('SUMMA')
       - Returns class (not instance) for flexible instantiation
       - Allows downstream code to customize initialization

    4. Registration via Module Imports:
       Model components self-register when their modules are imported.
       The optimization layer depends on models (not vice versa):
       optimization.model_optimizers → models.{model_name}.runner

Supported Models:
    Primary hydrological models:
    - SUMMA: Land surface model with distributed discretization
    - FUSE: Modular flexible rainfall-runoff
    - GR: GR4J/GR6J lumped conceptual models
    - HYPE: Semi-distributed with internal routing
    - RHESSys: Ecosystem-hydrological model
    - MESH: Pan-Arctic model
    - NGEN: NextGen modular framework

    Data-driven/routing models:
    - LSTM: Neural network surrogate
    - MIZUROUTE: Streamflow routing (auto-added dependency)

Component Registration Pattern:
    1. Preprocessor Registration:
       @ModelRegistry.register_preprocessor('MYMODEL')
       class MyPreprocessor:
           def run_preprocessing(self): ...

       Purpose: Convert ERA5 → model forcing, apply parameter files, etc.

    2. Runner Registration:
       @ModelRegistry.register_runner('MYMODEL', method_name='execute')
       class MyRunner:
           def execute(self): ...

       Purpose: Invoke model executable with configured inputs

    3. Postprocessor Registration:
       @ModelRegistry.register_postprocessor('MYMODEL')
       class MyPostprocessor:
           def extract_streamflow(self): ...

       Purpose: Parse model outputs, extract metrics, standardize formats

    4. Visualizer Registration:
       @ModelRegistry.register_visualizer('MYMODEL')
       def visualize_mymodel(reporting_manager, config, project_dir, ...): ...

       Purpose: Generate diagnostic plots and timeseries visualizations

Registry Storage (Class-Level):
    _preprocessors: Dict[model_name: str] → preprocessor_class
    _runners: Dict[model_name: str] → runner_class
    _postprocessors: Dict[model_name: str] → postprocessor_class
    _visualizers: Dict[model_name: str] → visualizer_function
    _runner_methods: Dict[model_name: str] → method_name (e.g., 'run_summa')

Lifecycle:
    1. Framework startup: Model modules imported, components registered
    2. Workflow execution: ModelManager queries registry by model name
    3. Component lookup: get_preprocessor('SUMMA') returns SUMMAPreprocessor class
    4. Instantiation: preprocessor = preprocessor_cls(config, logger)
    5. Execution: preprocessor.run_preprocessing()

Benefits:
    - Loose coupling: Framework doesn't need to import specific model modules
    - Easy extension: New models register without framework changes
    - Third-party models: External libraries can register components
    - Testing: Mock components can replace production implementations
    - Fallback gracefully: Missing components return None (vs hard error)

Examples:
    >>> # Query registry
    >>> from symfluence.models.registry import ModelRegistry
    >>> preproc_cls = ModelRegistry.get_preprocessor('SUMMA')
    >>> runner_cls = ModelRegistry.get_runner('FUSE')
    >>> method_name = ModelRegistry.get_runner_method('GR')

    >>> # List all registered models
    >>> models = ModelRegistry.list_models()
    >>> print(f"Supported models: {models}")

    >>> # Register custom model at runtime
    >>> @ModelRegistry.register_preprocessor('MYMODEL')
    ... class MyPreprocessor: ...

References:
    - Registry Pattern: Gang of Four design patterns
    - Factory Pattern: Creational design pattern for object creation
    - Python decorators: PEP 318
"""

import logging
from typing import Dict, Any, Type, Tuple

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for hydrological model components (Registry Pattern).

    Implements the Registry Pattern to enable dynamic model discovery and
    extensibility without tight coupling. Model components self-register via
    decorators, allowing the framework to instantiate appropriate components
    based on configuration.

    The registry stores four types of model components:
    1. Preprocessors: Input preparation (forcing, attributes, parameters)
    2. Runners: Model executable execution
    3. Postprocessors: Output file processing and metric extraction
    4. Visualizers: Diagnostic plots and visualizations

    Component Discovery:
        ModelManager queries registry by model name and retrieves class
        references for instantiation. Returns None for unregistered models
        (graceful fallback vs hard error).

    Registration Flow:
        1. Model module defines components with @ModelRegistry decorators
        2. At import time, decorators register classes in static dicts
        3. ModelManager imports models (or imports triggered elsewhere)
        4. Registry populated with all registered components
        5. Workflow execution queries registry by model name

    Example Component Registration:
        # In models/summa/preprocessor.py
        @ModelRegistry.register_preprocessor('SUMMA')
        class SUMMAPreprocessor(BaseModelPreProcessor):
            def run_preprocessing(self):
                # SUMMA-specific input preparation
                pass

        # In models/summa/runner.py
        @ModelRegistry.register_runner('SUMMA', method_name='run_summa')
        class SUMMARunner:
            def run_summa(self):
                # Invoke SUMMA executable
                pass

    Example Component Lookup:
        # In workflow orchestration
        preprocessor_cls = ModelRegistry.get_preprocessor('SUMMA')
        if preprocessor_cls:
            preprocessor = preprocessor_cls(config, logger)
            preprocessor.run_preprocessing()
        else:
            logger.warning("No preprocessor for SUMMA")

    Attributes:
        _preprocessors: Dict[model_name] → preprocessor_class
        _runners: Dict[model_name] → runner_class
        _postprocessors: Dict[model_name] → postprocessor_class
        _visualizers: Dict[model_name] → visualizer_function
        _runner_methods: Dict[model_name] → method_name (e.g., 'run', 'run_summa')

    Supported Models:
        SUMMA, FUSE, GR, HYPE, NGEN, MESH, LSTM, RHESSys, MIZUROUTE, and others
        registered via the decorator pattern.

    Design Patterns:
        - Registry Pattern: Centralized component storage
        - Factory Pattern: Component creation via get_*() methods
        - Decorator Pattern: Registration via @register_* decorators
        - Lazy Initialization: Components imported on-demand

    See Also:
        ModelManager: Uses registry to discover and invoke components
        optimization.model_optimizers: Depend on model components for calibration
    """

    _preprocessors: Dict[str, Type] = {}
    _runners: Dict[str, Type] = {}
    _postprocessors: Dict[str, Type] = {}
    _visualizers: Dict[str, Type] = {}
    _runner_methods: Dict[str, str] = {}
    # Config management registries
    _config_adapters: Dict[str, Any] = {}
    _config_schemas: Dict[str, Type] = {}
    _config_defaults: Dict[str, Dict[str, Any]] = {}
    _config_transformers: Dict[str, Dict[str, Tuple[str, ...]]] = {}
    _config_validators: Dict[str, Any] = {}
    # Result extraction registry
    _result_extractors: Dict[str, Any] = {}

    @classmethod
    def register_preprocessor(cls, model_name):
        def decorator(preprocessor_cls):
            cls._preprocessors[model_name] = preprocessor_cls
            return preprocessor_cls
        return decorator

    @classmethod
    def register_runner(cls, model_name, method_name="run"):
        def decorator(runner_cls):
            cls._runners[model_name] = runner_cls
            cls._runner_methods[model_name] = method_name
            return runner_cls
        return decorator

    @classmethod
    def register_postprocessor(cls, model_name):
        def decorator(postprocessor_cls):
            cls._postprocessors[model_name] = postprocessor_cls
            return postprocessor_cls
        return decorator

    @classmethod
    def register_visualizer(cls, model_name):
        """
        Register a visualization function for a model.

        The visualizer should be a callable with signature:
        (reporting_manager, config, project_dir, experiment_id, workflow)
        """
        def decorator(visualizer_func):
            cls._visualizers[model_name] = visualizer_func
            return visualizer_func
        return decorator

    @classmethod
    def get_preprocessor(cls, model_name):
        # Try exact match first, then uppercase for case-insensitive lookup
        result = cls._preprocessors.get(model_name)
        if result is None:
            result = cls._preprocessors.get(model_name.upper())
        return result

    @classmethod
    def get_runner(cls, model_name):
        # Try exact match first, then uppercase for case-insensitive lookup
        result = cls._runners.get(model_name)
        if result is None:
            result = cls._runners.get(model_name.upper())
        return result

    @classmethod
    def get_postprocessor(cls, model_name):
        # Try exact match first, then uppercase for case-insensitive lookup
        result = cls._postprocessors.get(model_name)
        if result is None:
            result = cls._postprocessors.get(model_name.upper())
        return result

    @classmethod
    def get_visualizer(cls, model_name):
        # Try exact match first, then uppercase for case-insensitive lookup
        result = cls._visualizers.get(model_name)
        if result is None:
            result = cls._visualizers.get(model_name.upper())
        return result

    @classmethod
    def get_runner_method(cls, model_name):
        # Try exact match first, then uppercase for case-insensitive lookup
        result = cls._runner_methods.get(model_name)
        if result is None:
            result = cls._runner_methods.get(model_name.upper())
        return result if result is not None else "run"

    @classmethod
    def list_models(cls):
        return sorted(list(set(cls._runners.keys()) | set(cls._preprocessors.keys())))

    # =========================================================================
    # Config Management Registration (New in Refactoring Phase 1)
    # =========================================================================

    @classmethod
    def register_config_adapter(cls, model_name):
        """
        Register a complete config adapter for a model.

        The adapter provides schema, defaults, transformers, and validation.

        Args:
            model_name: Model name (e.g., 'SUMMA', 'FUSE')

        Example:
            >>> @ModelRegistry.register_config_adapter('SUMMA')
            ... class SUMMAConfigAdapter(ModelConfigAdapter):
            ...     def get_config_schema(self):
            ...         return SUMMAConfig
        """
        def decorator(adapter_cls):
            cls._config_adapters[model_name.upper()] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def register_config_schema(cls, model_name, schema):
        """
        Register Pydantic config schema for a model.

        Args:
            model_name: Model name
            schema: Pydantic BaseModel class
        """
        cls._config_schemas[model_name.upper()] = schema
        return schema

    @classmethod
    def register_config_defaults(cls, model_name, defaults):
        """
        Register default configuration values for a model.

        Args:
            model_name: Model name
            defaults: Dictionary of default values
        """
        cls._config_defaults[model_name.upper()] = defaults
        return defaults

    @classmethod
    def register_config_transformers(cls, model_name, transformers):
        """
        Register flat-to-nested field transformers for a model.

        Args:
            model_name: Model name
            transformers: Dictionary mapping flat keys to nested paths
        """
        cls._config_transformers[model_name.upper()] = transformers
        return transformers

    @classmethod
    def register_config_validator(cls, model_name, validator):
        """
        Register custom validation function for a model.

        Args:
            model_name: Model name
            validator: Callable that takes config dict and raises on validation error
        """
        cls._config_validators[model_name.upper()] = validator
        return validator

    @classmethod
    def get_config_adapter(cls, model_name):
        """Get config adapter instance for a model."""
        adapter_cls = cls._config_adapters.get(model_name.upper())
        return adapter_cls(model_name) if adapter_cls else None

    @classmethod
    def get_config_schema(cls, model_name):
        """Get Pydantic config schema for a model."""
        # Try adapter first
        adapter = cls.get_config_adapter(model_name)
        if adapter:
            return adapter.get_config_schema()
        # Fall back to direct registration
        return cls._config_schemas.get(model_name.upper())

    @classmethod
    def get_config_defaults(cls, model_name):
        """Get default configuration for a model."""
        # Try adapter first
        adapter = cls.get_config_adapter(model_name)
        if adapter:
            return adapter.get_defaults()
        # Fall back to direct registration
        return cls._config_defaults.get(model_name.upper(), {})

    @classmethod
    def get_config_transformers(cls, model_name):
        """Get flat-to-nested transformers for a model."""
        # Try adapter first
        adapter = cls.get_config_adapter(model_name)
        if adapter:
            return adapter.get_field_transformers()
        # Fall back to direct registration
        return cls._config_transformers.get(model_name.upper(), {})

    @classmethod
    def get_config_validator(cls, model_name):
        """Get config validator function for a model."""
        # Try adapter first
        adapter = cls.get_config_adapter(model_name)
        if adapter:
            return adapter.validate
        # Fall back to direct registration
        return cls._config_validators.get(model_name.upper())

    @classmethod
    def validate_model_config(cls, model_name, config):
        """
        Validate model configuration using registered validator.

        Args:
            model_name: Model name
            config: Configuration dictionary

        Raises:
            ConfigValidationError: If validation fails
        """
        validator = cls.get_config_validator(model_name)
        if validator:
            validator(config)

    # =========================================================================
    # Result Extraction Registry Methods
    # =========================================================================

    @classmethod
    def register_result_extractor(cls, model_name):
        """
        Register a result extractor for a model.

        The extractor handles model-specific output file location and
        variable extraction logic.

        Args:
            model_name: Model name (e.g., 'SUMMA', 'NGEN')

        Example:
            >>> @ModelRegistry.register_result_extractor('SUMMA')
            ... class SUMMAResultExtractor(ModelResultExtractor):
            ...     def extract_variable(self, output_file, variable_type):
            ...         # SUMMA-specific extraction logic
            ...         pass
        """
        def decorator(extractor_cls):
            cls._result_extractors[model_name.upper()] = extractor_cls
            return extractor_cls
        return decorator

    @classmethod
    def get_result_extractor(cls, model_name):
        """Get result extractor instance for a model.

        Args:
            model_name: Model name

        Returns:
            ModelResultExtractor instance or None if not registered
        """
        extractor_cls = cls._result_extractors.get(model_name.upper())
        return extractor_cls(model_name) if extractor_cls else None

    @classmethod
    def has_result_extractor(cls, model_name):
        """Check if a model has a registered result extractor.

        Args:
            model_name: Model name

        Returns:
            bool: True if extractor is registered
        """
        return model_name.upper() in cls._result_extractors

    @classmethod
    def list_result_extractors(cls):
        """List all models with registered result extractors.

        Returns:
            List of model names with result extractors
        """
        return sorted(list(cls._result_extractors.keys()))

    # =========================================================================
    # Forcing Adapter Registry Methods (Delegates to ForcingAdapterRegistry)
    # =========================================================================

    @classmethod
    def get_forcing_adapter(cls, model_name, config, logger=None):
        """Get forcing adapter instance for a model.

        This method delegates to ForcingAdapterRegistry for backward compatibility.

        Args:
            model_name: Model name
            config: Configuration dictionary
            logger: Optional logger instance

        Returns:
            ForcingAdapter instance or None if not registered
        """
        try:
            from symfluence.models.adapters import ForcingAdapterRegistry
            return ForcingAdapterRegistry.get_adapter(model_name, config, logger)
        except (ImportError, ValueError):
            return None

    @classmethod
    def has_forcing_adapter(cls, model_name):
        """Check if a model has a registered forcing adapter.

        Args:
            model_name: Model name

        Returns:
            bool: True if adapter is registered
        """
        try:
            from symfluence.models.adapters import ForcingAdapterRegistry
            return ForcingAdapterRegistry.is_registered(model_name)
        except ImportError:
            return False

    @classmethod
    def list_forcing_adapters(cls):
        """List all models with registered forcing adapters.

        Returns:
            List of model names with forcing adapters
        """
        try:
            from symfluence.models.adapters import ForcingAdapterRegistry
            return ForcingAdapterRegistry.get_registered_models()
        except ImportError:
            return []

    # =========================================================================
    # Model Registration Validation Methods
    # =========================================================================

    @classmethod
    def validate_model_registration(
        cls,
        model_name: str,
        require_all: bool = False
    ) -> Dict[str, Any]:
        """
        Validate that a model has all required components registered.

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

        Example:
            >>> status = ModelRegistry.validate_model_registration('GNN')
            >>> if not status['valid']:
            ...     print(f"Missing: {status['missing']}")
        """
        components = {
            'preprocessor': cls._preprocessors.get(model_name),
            'runner': cls._runners.get(model_name),
            'postprocessor': cls._postprocessors.get(model_name),
            'visualizer': cls._visualizers.get(model_name),
        }

        # Required components (visualizer is optional)
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
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate registration status of all registered models.

        Checks each model returned by list_models() and reports their
        registration completeness.

        Args:
            require_all: If True, raises ValueError on first incomplete model.
                If False (default), returns status for all models.
            logger: Optional logger for warnings about incomplete registrations.
                If None, uses module-level logger.

        Returns:
            Dict mapping model_name -> validation result from
            validate_model_registration()

        Raises:
            ValueError: If require_all=True and any model is incomplete

        Example:
            >>> results = ModelRegistry.validate_all_models(logger=logger)
            >>> for model, status in results.items():
            ...     if not status['valid']:
            ...         print(f"{model}: missing {status['missing']}")
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

    @classmethod
    def get_model_components(cls, model_name: str) -> Dict[str, Any]:
        """
        Get all registered component classes for a model.

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

        Example:
            >>> components = ModelRegistry.get_model_components('SUMMA')
            >>> print(f"Runner: {components['runner']}")
            >>> print(f"Run method: {components['runner_method']}")
        """
        return {
            'preprocessor': cls._preprocessors.get(model_name),
            'runner': cls._runners.get(model_name),
            'postprocessor': cls._postprocessors.get(model_name),
            'visualizer': cls._visualizers.get(model_name),
            'runner_method': cls._runner_methods.get(model_name, 'run'),
        }
