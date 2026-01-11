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

    4. Lazy Loading via Model Optimizers:
       Model-specific optimizers auto-import model modules:
       from symfluence.optimization import model_optimizers  # Triggers registration

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
        models/model_optimizers: Auto-registers optimizer components
    """

    _preprocessors = {}
    _runners = {}
    _postprocessors = {}
    _visualizers = {}
    _runner_methods = {}

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
        return cls._preprocessors.get(model_name)

    @classmethod
    def get_runner(cls, model_name):
        return cls._runners.get(model_name)

    @classmethod
    def get_postprocessor(cls, model_name):
        return cls._postprocessors.get(model_name)

    @classmethod
    def get_visualizer(cls, model_name):
        return cls._visualizers.get(model_name)

    @classmethod
    def get_runner_method(cls, model_name):
        return cls._runner_methods.get(model_name, "run")

    @classmethod
    def list_models(cls):
        return sorted(list(set(cls._runners.keys()) | set(cls._preprocessors.keys())))
