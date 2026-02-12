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

    1. Preprocessor Registration::

        @ModelRegistry.register_preprocessor('MYMODEL')
        class MyPreprocessor:
            def run_preprocessing(self): ...

       Purpose: Convert ERA5 to model forcing, apply parameter files, etc.

    2. Runner Registration::

        @ModelRegistry.register_runner('MYMODEL', method_name='execute')
        class MyRunner:
            def execute(self): ...

       Purpose: Invoke model executable with configured inputs

    3. Postprocessor Registration::

        @ModelRegistry.register_postprocessor('MYMODEL')
        class MyPostprocessor:
            def extract_streamflow(self): ...

       Purpose: Parse model outputs, extract metrics, standardize formats

    4. Visualizer Registration::

        @ModelRegistry.register_visualizer('MYMODEL')
        def visualize_mymodel(reporting_manager, config, project_dir, ...): ...

       Purpose: Generate diagnostic plots and timeseries visualizations

Registry Architecture:
    This module provides a unified facade over specialized sub-registries:
    - ComponentRegistry: preprocessors, runners, postprocessors, visualizers
    - ConfigRegistry: config adapters, schemas, defaults, transformers, validators
    - ResultExtractorRegistry: model result extractors

    The facade pattern maintains backward compatibility while enabling
    focused, maintainable sub-registries.

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
from typing import Any, Callable, Dict, Optional, Tuple, Type

from symfluence.models.registries.component_registry import ComponentRegistry
from symfluence.models.registries.config_registry import ConfigRegistry
from symfluence.models.registries.result_registry import ResultExtractorRegistry

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Central registry for hydrological model components (Registry Pattern).

    This class is a facade that delegates to specialized sub-registries:
    - ComponentRegistry: preprocessors, runners, postprocessors, visualizers
    - ConfigRegistry: config adapters, schemas, defaults, transformers, validators
    - ResultExtractorRegistry: model result extractors

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

    Example Component Registration::

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

    Example Component Lookup::

        # In workflow orchestration
        preprocessor_cls = ModelRegistry.get_preprocessor('SUMMA')
        if preprocessor_cls:
            preprocessor = preprocessor_cls(config, logger)
            preprocessor.run_preprocessing()
        else:
            logger.warning("No preprocessor for SUMMA")

    Attributes:
        _preprocessors: Delegated to ComponentRegistry
        _runners: Delegated to ComponentRegistry
        _postprocessors: Delegated to ComponentRegistry
        _visualizers: Delegated to ComponentRegistry
        _runner_methods: Delegated to ComponentRegistry
        _config_adapters: Delegated to ConfigRegistry
        _config_schemas: Delegated to ConfigRegistry
        _config_defaults: Delegated to ConfigRegistry
        _config_transformers: Delegated to ConfigRegistry
        _config_validators: Delegated to ConfigRegistry
        _result_extractors: Delegated to ResultExtractorRegistry

    Supported Models:
        SUMMA, FUSE, GR, HYPE, NGEN, MESH, LSTM, RHESSys, MIZUROUTE, and others
        registered via the decorator pattern.

    Design Patterns:
        - Facade Pattern: Unified interface over sub-registries
        - Registry Pattern: Centralized component storage
        - Factory Pattern: Component creation via get_*() methods
        - Decorator Pattern: Registration via @register_* decorators
        - Lazy Initialization: Components imported on-demand

    See Also:
        ComponentRegistry: Core component registration
        ConfigRegistry: Configuration management
        ResultExtractorRegistry: Result extraction
        ModelManager: Uses registry to discover and invoke components
        optimization.model_optimizers: Depend on model components for calibration
    """

    # =========================================================================
    # Class-level attributes for backward compatibility
    # These are aliases to the underlying registry dictionaries
    # =========================================================================

    # Component registry attributes
    _preprocessors = ComponentRegistry._preprocessors
    _runners = ComponentRegistry._runners
    _postprocessors = ComponentRegistry._postprocessors
    _visualizers = ComponentRegistry._visualizers
    _runner_methods = ComponentRegistry._runner_methods

    # Config registry attributes
    _config_adapters = ConfigRegistry._config_adapters
    _config_schemas = ConfigRegistry._config_schemas
    _config_defaults = ConfigRegistry._config_defaults
    _config_transformers = ConfigRegistry._config_transformers
    _config_validators = ConfigRegistry._config_validators

    # Result extractor registry attributes
    _result_extractors = ResultExtractorRegistry._result_extractors

    # =========================================================================
    # Component Registration (Delegates to ComponentRegistry)
    # =========================================================================

    @classmethod
    def register_preprocessor(cls, model_name: str) -> Callable[[Type], Type]:
        """Register a preprocessor class for a model.

        Delegates to ComponentRegistry.register_preprocessor().
        """
        return ComponentRegistry.register_preprocessor(model_name)

    @classmethod
    def register_runner(
        cls, model_name: str, method_name: str = "run"
    ) -> Callable[[Type], Type]:
        """Register a runner class for a model.

        Delegates to ComponentRegistry.register_runner().
        """
        return ComponentRegistry.register_runner(model_name, method_name)

    @classmethod
    def register_postprocessor(cls, model_name: str) -> Callable[[Type], Type]:
        """Register a postprocessor class for a model.

        Delegates to ComponentRegistry.register_postprocessor().
        """
        return ComponentRegistry.register_postprocessor(model_name)

    @classmethod
    def register_visualizer(cls, model_name: str) -> Callable[[Callable], Callable]:
        """Register a visualization function for a model.

        Delegates to ComponentRegistry.register_visualizer().
        """
        return ComponentRegistry.register_visualizer(model_name)

    # =========================================================================
    # Component Retrieval (Delegates to ComponentRegistry)
    # =========================================================================

    @classmethod
    def get_preprocessor(cls, model_name: str) -> Optional[Type]:
        """Get preprocessor class for a model.

        Delegates to ComponentRegistry.get_preprocessor().
        """
        return ComponentRegistry.get_preprocessor(model_name)

    @classmethod
    def get_runner(cls, model_name: str) -> Optional[Type]:
        """Get runner class for a model.

        Delegates to ComponentRegistry.get_runner().
        """
        return ComponentRegistry.get_runner(model_name)

    @classmethod
    def get_postprocessor(cls, model_name: str) -> Optional[Type]:
        """Get postprocessor class for a model.

        Delegates to ComponentRegistry.get_postprocessor().
        """
        return ComponentRegistry.get_postprocessor(model_name)

    @classmethod
    def get_visualizer(cls, model_name: str) -> Optional[Callable]:
        """Get visualizer function for a model.

        Delegates to ComponentRegistry.get_visualizer().
        """
        return ComponentRegistry.get_visualizer(model_name)

    @classmethod
    def get_runner_method(cls, model_name: str) -> str:
        """Get the runner method name for a model.

        Delegates to ComponentRegistry.get_runner_method().
        """
        return ComponentRegistry.get_runner_method(model_name)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all models with registered components.

        Delegates to ComponentRegistry.list_models().
        """
        return ComponentRegistry.list_models()

    @classmethod
    def get_model_components(cls, model_name: str) -> Dict[str, Any]:
        """Get all registered component classes for a model.

        Delegates to ComponentRegistry.get_model_components().
        """
        return ComponentRegistry.get_model_components(model_name)

    @classmethod
    def validate_model_registration(
        cls,
        model_name: str,
        require_all: bool = False
    ) -> Dict[str, Any]:
        """Validate that a model has all required components registered.

        Delegates to ComponentRegistry.validate_model_registration().
        """
        return ComponentRegistry.validate_model_registration(model_name, require_all)

    @classmethod
    def validate_all_models(
        cls,
        require_all: bool = False,
        logger: logging.Logger = None
    ) -> Dict[str, Dict[str, Any]]:
        """Validate registration status of all registered models.

        Delegates to ComponentRegistry.validate_all_models().
        """
        return ComponentRegistry.validate_all_models(require_all, logger)

    # =========================================================================
    # Config Management Registration (Delegates to ConfigRegistry)
    # =========================================================================

    @classmethod
    def register_config_adapter(cls, model_name: str) -> Callable[[Type], Type]:
        """Register a complete config adapter for a model.

        Delegates to ConfigRegistry.register_config_adapter().
        """
        return ConfigRegistry.register_config_adapter(model_name)

    @classmethod
    def register_config_schema(cls, model_name: str, schema: Type) -> Type:
        """Register Pydantic config schema for a model.

        Delegates to ConfigRegistry.register_config_schema().
        """
        return ConfigRegistry.register_config_schema(model_name, schema)

    @classmethod
    def register_config_defaults(
        cls, model_name: str, defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Register default configuration values for a model.

        Delegates to ConfigRegistry.register_config_defaults().
        """
        return ConfigRegistry.register_config_defaults(model_name, defaults)

    @classmethod
    def register_config_transformers(
        cls, model_name: str, transformers: Dict[str, Tuple[str, ...]]
    ) -> Dict[str, Tuple[str, ...]]:
        """Register flat-to-nested field transformers for a model.

        Delegates to ConfigRegistry.register_config_transformers().
        """
        return ConfigRegistry.register_config_transformers(model_name, transformers)

    @classmethod
    def register_config_validator(cls, model_name: str, validator: Callable) -> Callable:
        """Register custom validation function for a model.

        Delegates to ConfigRegistry.register_config_validator().
        """
        return ConfigRegistry.register_config_validator(model_name, validator)

    # =========================================================================
    # Config Management Retrieval (Delegates to ConfigRegistry)
    # =========================================================================

    @classmethod
    def get_config_adapter(cls, model_name: str) -> Optional[Any]:
        """Get config adapter instance for a model.

        Delegates to ConfigRegistry.get_config_adapter().
        """
        return ConfigRegistry.get_config_adapter(model_name)

    @classmethod
    def get_config_schema(cls, model_name: str) -> Optional[Type]:
        """Get Pydantic config schema for a model.

        Delegates to ConfigRegistry.get_config_schema().
        """
        return ConfigRegistry.get_config_schema(model_name)

    @classmethod
    def get_config_defaults(cls, model_name: str) -> Dict[str, Any]:
        """Get default configuration for a model.

        Delegates to ConfigRegistry.get_config_defaults().
        """
        return ConfigRegistry.get_config_defaults(model_name)

    @classmethod
    def get_config_transformers(
        cls, model_name: str
    ) -> Dict[str, Tuple[str, ...]]:
        """Get flat-to-nested transformers for a model.

        Delegates to ConfigRegistry.get_config_transformers().
        """
        return ConfigRegistry.get_config_transformers(model_name)

    @classmethod
    def get_config_validator(cls, model_name: str) -> Optional[Callable]:
        """Get config validator function for a model.

        Delegates to ConfigRegistry.get_config_validator().
        """
        return ConfigRegistry.get_config_validator(model_name)

    @classmethod
    def validate_model_config(cls, model_name: str, config: Dict[str, Any]) -> None:
        """Validate model configuration using registered validator.

        Delegates to ConfigRegistry.validate_model_config().
        """
        ConfigRegistry.validate_model_config(model_name, config)

    # =========================================================================
    # Result Extraction Registry Methods (Delegates to ResultExtractorRegistry)
    # =========================================================================

    @classmethod
    def register_result_extractor(cls, model_name: str) -> Callable[[Type], Type]:
        """Register a result extractor for a model.

        Delegates to ResultExtractorRegistry.register_result_extractor().
        """
        return ResultExtractorRegistry.register_result_extractor(model_name)

    @classmethod
    def get_result_extractor(cls, model_name: str) -> Optional[Any]:
        """Get result extractor instance for a model.

        Delegates to ResultExtractorRegistry.get_result_extractor().
        """
        return ResultExtractorRegistry.get_result_extractor(model_name)

    @classmethod
    def has_result_extractor(cls, model_name: str) -> bool:
        """Check if a model has a registered result extractor.

        Delegates to ResultExtractorRegistry.has_result_extractor().
        """
        return ResultExtractorRegistry.has_result_extractor(model_name)

    @classmethod
    def list_result_extractors(cls) -> list[str]:
        """List all models with registered result extractors.

        Delegates to ResultExtractorRegistry.list_result_extractors().
        """
        return ResultExtractorRegistry.list_result_extractors()

    # =========================================================================
    # Forcing Adapter Registry Methods (Delegates to ForcingAdapterRegistry)
    # =========================================================================

    @classmethod
    def get_forcing_adapter(cls, model_name: str, config: Dict, logger=None) -> Optional[Any]:
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
    def has_forcing_adapter(cls, model_name: str) -> bool:
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
    def list_forcing_adapters(cls) -> list[str]:
        """List all models with registered forcing adapters.

        Returns:
            List of model names with forcing adapters
        """
        try:
            from symfluence.models.adapters import ForcingAdapterRegistry
            return ForcingAdapterRegistry.get_registered_models()
        except ImportError:
            return []
