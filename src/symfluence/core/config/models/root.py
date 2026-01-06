"""
Root configuration model for SYMFLUENCE.

Contains SymfluenceConfig - the main configuration class that orchestrates
all other config models and provides validation, factory methods, and
backward compatibility.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path
from pydantic import BaseModel, Field, model_validator, ConfigDict
from functools import cached_property
import pandas as pd
import warnings
import logging

logger = logging.getLogger(__name__)

from .base import FROZEN_CONFIG
from .system import SystemConfig
from .domain import DomainConfig, DelineationConfig
from .forcing import ForcingConfig, NexConfig, EMEarthConfig
from .model_configs import (
    ModelConfig, SUMMAConfig, FUSEConfig, GRConfig, HYPEConfig,
    NGENConfig, MESHConfig, MizuRouteConfig, LSTMConfig
)
from .optimization import (
    OptimizationConfig, PSOConfig, DEConfig, DDSConfig, SCEUAConfig,
    NSGA2Config, DPEConfig, LargeDomainConfig, EmulationConfig
)
from .evaluation import (
    EvaluationConfig, StreamflowConfig, SNOTELConfig, FluxNetConfig,
    USGSGWConfig, SMAPConfig, GRACEConfig, MODISSnowConfig, AttributesConfig
)
from .paths import PathsConfig, ShapefilePathConfig


class SymfluenceConfig(BaseModel):
    """
    Hierarchical root configuration model for SYMFLUENCE.

    Organizes 346+ configuration parameters into logical nested sections:
    - system: System settings (paths, logging, MPI)
    - domain: Domain definition (timing, spatial extent, discretization)
    - forcing: Meteorological forcing data
    - model: Hydrological model configurations
    - optimization: Calibration and optimization settings
    - evaluation: Evaluation data and analysis
    - paths: File paths and directories

    Features:
    - Type-safe hierarchical access: config.domain.name vs config['DOMAIN_NAME']
    - Factory methods: from_preset(), from_minimal(), from_file()
    - Backward compatibility: to_dict(), get(), __getitem__()
    - Immutable after creation (frozen=True) to prevent mutation bugs
    - All validation logic preserved from original flat model
    """

    # Immutable config to prevent mutation bugs
    model_config = ConfigDict(
        extra='allow',  # Allow extra keys for extensibility
        frozen=True,   # Immutable configs for thread safety and caching
        populate_by_name=True,  # Allow both field names and aliases for backward compat
    )

    # ========================================
    # NESTED CONFIGURATION SECTIONS
    # ========================================

    system: SystemConfig = Field(default_factory=lambda: SystemConfig(
        SYMFLUENCE_DATA_DIR=Path.cwd() / 'data',
        SYMFLUENCE_CODE_DIR=Path.cwd()
    ))
    domain: DomainConfig = Field(default_factory=lambda: DomainConfig(
        DOMAIN_NAME='unnamed_domain',
        EXPERIMENT_ID='run_1',
        EXPERIMENT_TIME_START='2010-01-01 00:00',
        EXPERIMENT_TIME_END='2020-12-31 23:00',
        DOMAIN_DEFINITION_METHOD='lumped',
        DOMAIN_DISCRETIZATION='lumped'
    ))
    forcing: ForcingConfig = Field(default_factory=lambda: ForcingConfig(FORCING_DATASET='ERA5'))
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(HYDROLOGICAL_MODEL='SUMMA'))
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    # ========================================
    # CROSS-FIELD VALIDATORS (from original model)
    # ========================================

    @model_validator(mode='before')
    @classmethod
    def normalize_flat_config(cls, values):
        """Normalize flat config inputs into nested sections with required checks."""
        if isinstance(values, cls):
            return values
        if not isinstance(values, dict):
            return values

        from symfluence.core.config.transformers import FLAT_TO_NESTED_MAP, transform_flat_to_nested

        flat_keys = {key for key in values if key in FLAT_TO_NESTED_MAP}
        if not flat_keys:
            return values

        required_flat = {
            'SYMFLUENCE_DATA_DIR',
            'SYMFLUENCE_CODE_DIR',
            'DOMAIN_NAME',
            'EXPERIMENT_ID',
            'EXPERIMENT_TIME_START',
            'EXPERIMENT_TIME_END',
            'DOMAIN_DEFINITION_METHOD',
            'DOMAIN_DISCRETIZATION',
            'HYDROLOGICAL_MODEL',
            'FORCING_DATASET',
        }

        if 'DOMAIN_DEFINITION_METHOD' in values:
            allowed_definition_methods = {
                'lumped',
                'discretized',
                'distributed',
                'subset',
                'point',
                'delineate',
            }
            if values.get('DOMAIN_DEFINITION_METHOD') not in allowed_definition_methods:
                raise ValueError(
                    "DOMAIN_DEFINITION_METHOD must be one of "
                    + ", ".join(sorted(allowed_definition_methods))
                )

        def has_nested_value(data, path):
            current = data
            for part in path:
                if not isinstance(current, dict) or part not in current:
                    return False
                current = current[part]
            return current is not None and current != ''

        missing = []
        for key in required_flat:
            if key in values and values[key] not in (None, ''):
                continue
            path = FLAT_TO_NESTED_MAP.get(key)
            if path and has_nested_value(values, path):
                continue
            missing.append(key)

        if missing:
            raise ValueError(
                "Missing required configuration keys: " + ", ".join(sorted(missing))
            )

        nested = transform_flat_to_nested({key: values[key] for key in flat_keys})

        sections = ('system', 'domain', 'forcing', 'model', 'optimization', 'evaluation', 'paths')
        for section in sections:
            if section in values and isinstance(values[section], dict):
                nested.setdefault(section, {}).update(values[section])
            if not nested.get(section):
                nested.pop(section, None)

        for key, value in values.items():
            if key in flat_keys or key in sections:
                continue
            nested[key] = value

        return nested

    @model_validator(mode='after')
    def validate_time_periods(self):
        """Validate that time periods make logical sense"""
        from symfluence.core.exceptions import ConfigurationError

        try:
            start = pd.to_datetime(self.domain.time_start)
            end = pd.to_datetime(self.domain.time_end)

            if start >= end:
                raise ConfigurationError(
                    f"EXPERIMENT_TIME_START ({start}) must be before EXPERIMENT_TIME_END ({end})"
                )

            # Validate calibration period is within experiment period
            if self.domain.calibration_period:
                cal_start, cal_end = self._parse_period(self.domain.calibration_period)
                if cal_start < start or cal_end > end:
                    raise ConfigurationError(
                        f"CALIBRATION_PERIOD ({self.domain.calibration_period}) must be within "
                        f"EXPERIMENT_TIME_START ({start}) and EXPERIMENT_TIME_END ({end})"
                    )

            # Validate evaluation period is within experiment period
            if self.domain.evaluation_period:
                eval_start, eval_end = self._parse_period(self.domain.evaluation_period)
                if eval_start < start or eval_end > end:
                    raise ConfigurationError(
                        f"EVALUATION_PERIOD ({self.domain.evaluation_period}) must be within "
                        f"EXPERIMENT_TIME_START ({start}) and EXPERIMENT_TIME_END ({end})"
                    )

        except Exception as e:
            if "ConfigurationError" in str(type(e)):
                raise
            raise ConfigurationError(f"Invalid date format: {e}")

        return self

    @model_validator(mode='after')
    def validate_coordinates(self):
        """Validate coordinate formats and bounds"""
        from symfluence.core.exceptions import ConfigurationError

        # Validate pour point coordinates
        if self.domain.pour_point_coords:
            try:
                lat, lon = self.domain.pour_point_coords.split('/')
                lat_f, lon_f = float(lat), float(lon)

                if not (-90 <= lat_f <= 90):
                    raise ConfigurationError(
                        f"POUR_POINT_COORDS latitude {lat_f} out of range [-90, 90]"
                    )
                if not (-180 <= lon_f <= 180):
                    raise ConfigurationError(
                        f"POUR_POINT_COORDS longitude {lon_f} out of range [-180, 180]"
                    )
            except ValueError as e:
                raise ConfigurationError(
                    f"POUR_POINT_COORDS must be 'lat/lon' format, got '{self.domain.pour_point_coords}'"
                )

        # Validate bounding box coordinates
        if self.domain.bounding_box_coords:
            try:
                north, west, south, east = self.domain.bounding_box_coords.split('/')
                north_f, west_f, south_f, east_f = float(north), float(west), float(south), float(east)

                if not (-90 <= south_f <= north_f <= 90):
                    raise ConfigurationError(
                        f"BOUNDING_BOX_COORDS invalid latitude range: south={south_f}, north={north_f}"
                    )
                if not (-180 <= west_f <= 180 and -180 <= east_f <= 180):
                    raise ConfigurationError(
                        f"BOUNDING_BOX_COORDS longitude out of range [-180, 180]"
                    )
                if south_f >= north_f:
                    raise ConfigurationError(
                        f"BOUNDING_BOX_COORDS: south ({south_f}) must be < north ({north_f})"
                    )
            except ValueError as e:
                raise ConfigurationError(
                    f"BOUNDING_BOX_COORDS must be 'north/west/south/east' format, got '{self.domain.bounding_box_coords}'"
                )

        return self

    @model_validator(mode='after')
    def validate_model_requirements(self):
        """Validate model-specific required fields based on HYDROLOGICAL_MODEL"""
        from symfluence.core.exceptions import ConfigurationError

        models = self._parse_models()
        missing_fields = []

        # Helper to check if value is unset
        def is_unset(value):
            return value is None or (isinstance(value, str) and value in ['', 'None'])

        # SUMMA requirements - validate only if summa config provided
        if 'SUMMA' in models and self.model.summa:
            summa_required = {
                'SUMMA_EXE': self.model.summa.exe,
                'SETTINGS_SUMMA_PATH': self.model.summa.settings_path,
            }
            for field, value in summa_required.items():
                if is_unset(value):
                    missing_fields.append(f"{field} (required for SUMMA)")

        # FUSE requirements - validate only if fuse config provided
        if 'FUSE' in models and self.model.fuse:
            fuse_required = {
                'FUSE_EXE': self.model.fuse.exe,
                'SETTINGS_FUSE_PATH': self.model.fuse.settings_path,
            }
            for field, value in fuse_required.items():
                if is_unset(value):
                    missing_fields.append(f"{field} (required for FUSE)")

        # GR requirements - validate only if gr config provided
        if 'GR' in models and self.model.gr:
            gr_required = {
                'GR_EXE': self.model.gr.exe,
                'SETTINGS_GR_PATH': self.model.gr.settings_path,
            }
            for field, value in gr_required.items():
                if is_unset(value):
                    missing_fields.append(f"{field} (required for GR)")

        # HYPE requirements - validate only if hype config provided
        if 'HYPE' in models and self.model.hype:
            hype_required = {
                'SETTINGS_HYPE_PATH': self.model.hype.settings_path,
            }
            for field, value in hype_required.items():
                if is_unset(value):
                    missing_fields.append(f"{field} (required for HYPE)")

        # NGEN requirements - validate only if ngen config provided
        if 'NGEN' in models and self.model.ngen:
            ngen_required = {
                'NGEN_EXE': self.model.ngen.exe,
                'NGEN_INSTALL_PATH': self.model.ngen.install_path,
            }
            for field, value in ngen_required.items():
                if is_unset(value):
                    missing_fields.append(f"{field} (required for NGEN)")

        # MESH requirements - validate only if mesh config provided
        if 'MESH' in models and self.model.mesh:
            mesh_required = {
                'MESH_EXE': self.model.mesh.exe,
                'SETTINGS_MESH_PATH': self.model.mesh.settings_path,
            }
            for field, value in mesh_required.items():
                if is_unset(value):
                    missing_fields.append(f"{field} (required for MESH)")

        # Routing model requirements - validate only if mizuroute config provided
        if self.model.routing_model:
            routing_model = self.model.routing_model.upper()
            if routing_model == 'MIZUROUTE' and self.model.mizuroute:
                mizu_required = {
                    'EXE_NAME_MIZUROUTE': self.model.mizuroute.exe,
                    'INSTALL_PATH_MIZUROUTE': self.model.mizuroute.install_path,
                }
                for field, value in mizu_required.items():
                    if is_unset(value):
                        missing_fields.append(f"{field} (required for mizuRoute)")

        if missing_fields:
            raise ConfigurationError(
                f"Model-specific configuration incomplete:\n"
                + "\n".join(f"  • {field}" for field in missing_fields)
                + f"\n\nSelected models: {', '.join(models)}"
                + (f"\nRouting model: {self.model.routing_model}" if self.model.routing_model else "")
            )

        return self

    @model_validator(mode='after')
    def validate_spatial_mode_consistency(self):
        """Validate and auto-align spatial modes with domain definition"""

        models = self._parse_models()
        issues = []

        # Check FUSE spatial mode
        if 'FUSE' in models and self.model.fuse:
            if self.domain.definition_method == 'lumped' and self.model.fuse.spatial_mode != 'lumped':
                issues.append(
                    f"FUSE_SPATIAL_MODE is '{self.model.fuse.spatial_mode}' but DOMAIN_DEFINITION_METHOD is 'lumped'. "
                    f"Consider setting FUSE_SPATIAL_MODE to 'lumped'."
                )

        # Check GR spatial mode
        if 'GR' in models and self.model.gr:
            if self.domain.definition_method in ['distributed', 'discretized'] and self.model.gr.spatial_mode == 'lumped':
                issues.append(
                    f"GR_SPATIAL_MODE is 'lumped' but DOMAIN_DEFINITION_METHOD is '{self.domain.definition_method}'. "
                    f"Consider setting GR_SPATIAL_MODE to 'distributed'."
                )

        # Log alignment actions as info, other issues as warnings
        if issues:
            for issue in issues:
                if 'Auto-aligned' in issue:
                    logger.info(f"Spatial mode configuration: {issue}")
                else:
                    warnings.warn(f"Spatial mode configuration: {issue}", UserWarning)

        return self

    @model_validator(mode='after')
    def validate_optimization_configuration(self):
        """Validate optimization algorithm and parameter settings"""
        from symfluence.core.exceptions import ConfigurationError

        # Check if optimization is enabled
        optimization_methods = self.optimization.methods
        if isinstance(optimization_methods, str):
            optimization_methods = [m.strip() for m in optimization_methods.split(',') if m.strip()]

        if not optimization_methods or len(optimization_methods) == 0:
            # No optimization configured, skip validation
            return self

        errors = []

        # Validate optimization algorithm
        valid_algorithms = [
            'PSO', 'DE', 'DDS', 'ASYNC-DDS', 'ASYNCDDS', 'ASYNC_DDS',
            'SCE-UA', 'SCEUA', 'NSGA-II', 'NSGA2',
            'ADAM', 'LBFGS', 'GD', 'SGD'
        ]
        if self.optimization.algorithm not in valid_algorithms:
            errors.append(
                f"ITERATIVE_OPTIMIZATION_ALGORITHM '{self.optimization.algorithm}' not recognized. "
                f"Valid algorithms: {', '.join(valid_algorithms)}"
            )

        # Validate optimization metric
        valid_metrics = ['KGE', 'KGEp', 'NSE', 'RMSE', 'MAE', 'PBIAS', 'R2']
        if self.optimization.metric not in valid_metrics:
            errors.append(
                f"OPTIMIZATION_METRIC '{self.optimization.metric}' not recognized. "
                f"Valid metrics: {', '.join(valid_metrics)}"
            )

        # Validate iterations and population size
        if self.optimization.iterations < 1:
            errors.append(f"NUMBER_OF_ITERATIONS must be >= 1, got {self.optimization.iterations}")

        if self.optimization.population_size < 1:
            errors.append(f"POPULATION_SIZE must be >= 1, got {self.optimization.population_size}")

        # Algorithm-specific validation
        if self.optimization.algorithm in ['DE', 'Differential Evolution'] and self.optimization.de:
            if not (0 <= self.optimization.de.scaling_factor <= 2):
                errors.append(f"DE_SCALING_FACTOR should be in [0, 2], got {self.optimization.de.scaling_factor}")
            if not (0 <= self.optimization.de.crossover_rate <= 1):
                errors.append(f"DE_CROSSOVER_RATE should be in [0, 1], got {self.optimization.de.crossover_rate}")

        if self.optimization.algorithm == 'DDS' and self.optimization.dds:
            if not (0 < self.optimization.dds.r <= 1):
                errors.append(f"DDS_R should be in (0, 1], got {self.optimization.dds.r}")

        if self.optimization.algorithm == 'PSO' and self.optimization.pso:
            if self.optimization.pso.cognitive_param < 0:
                errors.append(f"PSO_COGNITIVE_PARAM should be >= 0, got {self.optimization.pso.cognitive_param}")
            if self.optimization.pso.social_param < 0:
                errors.append(f"PSO_SOCIAL_PARAM should be >= 0, got {self.optimization.pso.social_param}")
            if not (0 <= self.optimization.pso.inertia_weight <= 1):
                errors.append(f"PSO_INERTIA_WEIGHT should be in [0, 1], got {self.optimization.pso.inertia_weight}")

        if self.optimization.algorithm in ['NSGA-II', 'NSGA2'] and self.optimization.nsga2:
            if not (0 <= self.optimization.nsga2.crossover_rate <= 1):
                errors.append(f"NSGA2_CROSSOVER_RATE should be in [0, 1], got {self.optimization.nsga2.crossover_rate}")
            if not (0 <= self.optimization.nsga2.mutation_rate <= 1):
                errors.append(f"NSGA2_MUTATION_RATE should be in [0, 1], got {self.optimization.nsga2.mutation_rate}")

        # Validate DPE settings if differentiable optimization is used
        if 'differentiable' in optimization_methods or 'emulator' in optimization_methods:
            if self.optimization.dpe:
                if self.optimization.dpe.learning_rate <= 0:
                    errors.append(f"DPE_LEARNING_RATE must be > 0, got {self.optimization.dpe.learning_rate}")
                if self.optimization.dpe.epochs < 1:
                    errors.append(f"DPE_EPOCHS must be >= 1, got {self.optimization.dpe.epochs}")
                if not (0 <= self.optimization.dpe.iterate_convergence_tol <= 1):
                    errors.append(f"DPE_ITERATE_CONVERGENCE_TOL should be in [0, 1], got {self.optimization.dpe.iterate_convergence_tol}")

        # Validate emulation settings
        if self.optimization.large_domain and self.optimization.large_domain.enabled:
            if self.optimization.large_domain.training_epochs < 1:
                errors.append(f"LARGE_DOMAIN_TRAINING_EPOCHS must be >= 1, got {self.optimization.large_domain.training_epochs}")

            # Validate objective weights sum to 1.0
            total_weight = (
                self.optimization.large_domain.streamflow_weight +
                self.optimization.large_domain.smap_weight +
                self.optimization.large_domain.grace_weight +
                self.optimization.large_domain.modis_weight
            )
            if abs(total_weight - 1.0) > 0.01:
                errors.append(
                    f"Large domain emulator objective weights should sum to 1.0, got {total_weight:.3f}. "
                    f"(Streamflow: {self.optimization.large_domain.streamflow_weight}, "
                    f"SMAP: {self.optimization.large_domain.smap_weight}, "
                    f"GRACE: {self.optimization.large_domain.grace_weight}, "
                    f"MODIS: {self.optimization.large_domain.modis_weight})"
                )

        if errors:
            raise ConfigurationError(
                f"Optimization configuration invalid:\n"
                + "\n".join(f"  • {error}" for error in errors)
            )

        return self

    # ========================================
    # HELPER METHODS (from original model)
    # ========================================

    def _parse_period(self, period_str: str):
        """Parse period string 'YYYY-MM-DD, YYYY-MM-DD' into start/end dates"""
        start_str, end_str = period_str.split(',')
        return pd.to_datetime(start_str.strip()), pd.to_datetime(end_str.strip())

    def _parse_models(self):
        """Parse HYDROLOGICAL_MODEL into list of model names"""
        if isinstance(self.model.hydrological_model, str):
            return [m.strip().upper() for m in self.model.hydrological_model.split(',')]
        return [str(m).upper() for m in self.model.hydrological_model]

    # ========================================
    # BACKWARD COMPATIBILITY LAYER
    # ========================================

    @cached_property
    def _flattened_dict_cache(self) -> Dict[str, Any]:
        """
        Cached flattened dictionary for performance optimization.

        Since the config is frozen (immutable), we can safely cache this.
        This significantly improves performance of get() and __getitem__() methods.
        """
        from symfluence.core.config.transformers import flatten_nested_config
        return flatten_nested_config(self)

    def to_dict(self, flatten: bool = True) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Args:
            flatten: If True, returns flat dict with uppercase keys (legacy format)
                    If False, returns nested dict structure

        Returns:
            Configuration as dictionary

        Example:
            >>> config = SymfluenceConfig.from_preset('fuse-basic')
            >>> flat_dict = config.to_dict(flatten=True)
            >>> flat_dict['DOMAIN_NAME']
            'my_basin'
        """
        if not flatten:
            # Return nested structure
            return self.model_dump(by_alias=False)

        # Use cached flattened dict for performance (config is immutable)
        return self._flattened_dict_cache

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dict-like get method for backward compatibility.

        Supports both flat keys ('DOMAIN_NAME') and dotted paths ('domain.name').

        Args:
            key: Configuration key (uppercase) or dotted path
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get('DOMAIN_NAME')
            'my_basin'
            >>> config.get('NONEXISTENT', 'fallback')
            'fallback'
        """
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    def __getitem__(self, key: str) -> Any:
        """
        Dict-like bracket access for backward compatibility.

        Args:
            key: Configuration key (uppercase)

        Returns:
            Configuration value

        Raises:
            KeyError: If key not found

        Example:
            >>> config['DOMAIN_NAME']
            'my_basin'
        """
        # Use cached flat dict for performance (config is immutable)
        if key in self._flattened_dict_cache:
            return self._flattened_dict_cache[key]

        raise KeyError(f"Configuration key not found: {key}")

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __getattr__(self, name: str) -> Any:
        """Provide attribute-style access for legacy flat keys."""
        from symfluence.core.config.transformers import FLAT_TO_NESTED_MAP

        path = FLAT_TO_NESTED_MAP.get(name)
        if path:
            value = self
            for part in path:
                value = getattr(value, part)
            return value

        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    # ========================================
    # FACTORY METHODS
    # ========================================

    @classmethod
    def from_file(
        cls,
        path: Path,
        overrides: Optional[Dict[str, Any]] = None,
        *,
        use_env: bool = True,
        validate: bool = True
    ) -> 'SymfluenceConfig':
        """
        Load configuration from YAML file with full 5-layer hierarchy.

        Loading precedence (highest to lowest):
        1. CLI overrides (programmatic)
        2. Environment variables (SYMFLUENCE_*)
        3. Config file (YAML)
        4. Defaults from nested models

        Args:
            path: Path to configuration YAML file
            overrides: Dictionary of CLI/programmatic overrides
            use_env: Whether to load environment variables (default: True)
            validate: Whether to validate using Pydantic (default: True)

        Returns:
            Validated SymfluenceConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
            FileNotFoundError: If config file is missing

        Example:
            >>> config = SymfluenceConfig.from_file(
            ...     'config.yaml',
            ...     overrides={'DEBUG_MODE': True}
            ... )
        """
        # Import here to avoid circular dependency
        from symfluence.core.config.factories import from_file_factory
        return from_file_factory(cls, path, overrides, use_env=use_env, validate=validate)

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> 'SymfluenceConfig':
        """
        Create configuration from a named preset.

        Args:
            preset_name: Name of preset ('fuse-provo', 'summa-basic', etc.)
            **overrides: Additional overrides to apply on top of preset

        Returns:
            Fully validated SymfluenceConfig instance

        Example:
            >>> config = SymfluenceConfig.from_preset(
            ...     'fuse-provo',
            ...     DOMAIN_NAME='my_basin',
            ...     EXPERIMENT_TIME_START='2020-01-01 00:00'
            ... )
        """
        # Import here to avoid circular dependency
        from symfluence.core.config.factories import from_preset_factory
        return from_preset_factory(cls, preset_name, **overrides)

    @classmethod
    def from_minimal(
        cls,
        domain_name: str,
        model: str,
        forcing_dataset: str = 'ERA5',
        **overrides
    ) -> 'SymfluenceConfig':
        """
        Create minimal viable configuration for quick setup.

        Automatically applies sensible defaults based on model choice.

        Args:
            domain_name: Name for the domain/basin
            model: Hydrological model ('SUMMA', 'FUSE', 'GR', etc.)
            forcing_dataset: Forcing data source (default: 'ERA5')
            **overrides: Additional configuration overrides

        Returns:
            Validated SymfluenceConfig with minimal required fields

        Example:
            >>> config = SymfluenceConfig.from_minimal(
            ...     domain_name='test_basin',
            ...     model='SUMMA',
            ...     POUR_POINT_COORDS='51.17/-115.57',
            ...     EXPERIMENT_TIME_START='2020-01-01 00:00',
            ...     EXPERIMENT_TIME_END='2020-12-31 23:00'
            ... )
        """
        # Import here to avoid circular dependency
        from symfluence.core.config.factories import from_minimal_factory
        return from_minimal_factory(cls, domain_name, model, forcing_dataset, **overrides)
