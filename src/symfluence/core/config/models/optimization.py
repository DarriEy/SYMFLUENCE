"""
Optimization configuration models.

Contains configuration classes for calibration algorithms:
PSOConfig, DEConfig, DDSConfig, SCEUAConfig, NSGA2Config, DPEConfig,
EmulationConfig, and the parent OptimizationConfig.
"""

from typing import List, Literal, Optional, Dict, Union
from pydantic import BaseModel, Field, field_validator

from .base import FROZEN_CONFIG

# Supported optimization algorithms
OptimizationAlgorithmType = Literal[
    'PSO', 'DE', 'DDS', 'ASYNC-DDS', 'SCE-UA', 'NSGA-II', 'DPE', 'GA'
]

# Supported optimization metrics
OptimizationMetricType = Literal[
    'KGE', 'KGEp', 'NSE', 'RMSE', 'MAE', 'PBIAS', 'R2', 'correlation'
]

# Supported sampling methods
SamplingMethodType = Literal['lhs', 'random', 'sobol', 'halton']


class PSOConfig(BaseModel):
    """Particle Swarm Optimization algorithm settings"""
    model_config = FROZEN_CONFIG

    swrmsize: int = Field(default=20, alias='SWRMSIZE', ge=2, le=10000)
    cognitive_param: float = Field(default=1.5, alias='PSO_COGNITIVE_PARAM', ge=0, le=4.0)
    social_param: float = Field(default=1.5, alias='PSO_SOCIAL_PARAM', ge=0, le=4.0)
    inertia_weight: float = Field(default=0.7, alias='PSO_INERTIA_WEIGHT', ge=0, le=1.0)
    inertia_reduction_rate: float = Field(default=0.99, alias='PSO_INERTIA_REDUCTION_RATE', ge=0, le=1.0)
    inertia_schedule: str = Field(default='LINEAR', alias='INERTIA_SCHEDULE')


class DEConfig(BaseModel):
    """Differential Evolution algorithm settings"""
    model_config = FROZEN_CONFIG

    scaling_factor: float = Field(default=0.5, alias='DE_SCALING_FACTOR', ge=0, le=2.0)
    crossover_rate: float = Field(default=0.9, alias='DE_CROSSOVER_RATE', ge=0, le=1.0)


class DDSConfig(BaseModel):
    """Dynamically Dimensioned Search algorithm settings"""
    model_config = FROZEN_CONFIG

    r: float = Field(default=0.2, alias='DDS_R', gt=0, le=1.0)
    async_pool_size: int = Field(default=10, alias='ASYNC_DDS_POOL_SIZE', ge=1)
    async_batch_size: int = Field(default=10, alias='ASYNC_DDS_BATCH_SIZE', ge=1)
    max_stagnation_batches: int = Field(default=10, alias='MAX_STAGNATION_BATCHES', ge=1)


class SCEUAConfig(BaseModel):
    """Shuffled Complex Evolution - University of Arizona algorithm settings"""
    model_config = FROZEN_CONFIG

    number_of_complexes: int = Field(default=2, alias='NUMBER_OF_COMPLEXES')
    points_per_subcomplex: int = Field(default=5, alias='POINTS_PER_SUBCOMPLEX')
    number_of_evolution_steps: int = Field(default=20, alias='NUMBER_OF_EVOLUTION_STEPS')
    evolution_stagnation: int = Field(default=5, alias='EVOLUTION_STAGNATION')
    percent_change_threshold: float = Field(default=0.01, alias='PERCENT_CHANGE_THRESHOLD')


class NSGA2Config(BaseModel):
    """Non-dominated Sorting Genetic Algorithm II settings"""
    model_config = FROZEN_CONFIG

    multi_target: bool = Field(default=False, alias='NSGA2_MULTI_TARGET')
    primary_target: str = Field(default='streamflow', alias='NSGA2_PRIMARY_TARGET')
    secondary_target: str = Field(default='gw_depth', alias='NSGA2_SECONDARY_TARGET')
    primary_metric: str = Field(default='KGE', alias='NSGA2_PRIMARY_METRIC')
    secondary_metric: str = Field(default='KGE', alias='NSGA2_SECONDARY_METRIC')
    crossover_rate: float = Field(default=0.9, alias='NSGA2_CROSSOVER_RATE', ge=0, le=1.0)
    mutation_rate: float = Field(default=0.1, alias='NSGA2_MUTATION_RATE', ge=0, le=1.0)
    eta_c: int = Field(default=20, alias='NSGA2_ETA_C', ge=1)
    eta_m: int = Field(default=20, alias='NSGA2_ETA_M', ge=1)


class DPEConfig(BaseModel):
    """Differentiable Parameter Estimation settings"""
    model_config = FROZEN_CONFIG

    training_cache: str = Field(default='default', alias='DPE_TRAINING_CACHE')
    hidden_dims: Optional[List[int]] = Field(default_factory=lambda: [256, 128, 64], alias='DPE_HIDDEN_DIMS')
    training_samples: int = Field(default=500, alias='DPE_TRAINING_SAMPLES', ge=1)
    validation_samples: int = Field(default=100, alias='DPE_VALIDATION_SAMPLES', ge=1)
    epochs: int = Field(default=300, alias='DPE_EPOCHS', ge=1, le=10000)
    learning_rate: float = Field(default=1e-3, alias='DPE_LEARNING_RATE', gt=0, le=1.0)
    optimization_lr: float = Field(default=1e-2, alias='DPE_OPTIMIZATION_LR', gt=0, le=1.0)
    optimization_steps: int = Field(default=200, alias='DPE_OPTIMIZATION_STEPS', ge=1)
    optimizer: str = Field(default='ADAM', alias='DPE_OPTIMIZER')
    objective_weights: Optional[Dict[str, float]] = Field(default_factory=lambda: {'KGE': 1.0}, alias='DPE_OBJECTIVE_WEIGHTS')
    emulator_iterate: bool = Field(default=True, alias='DPE_EMULATOR_ITERATE')
    iterate_max_iterations: int = Field(default=5, alias='DPE_ITERATE_MAX_ITERATIONS', ge=1)
    iterate_samples_per_cycle: int = Field(default=100, alias='DPE_ITERATE_SAMPLES_PER_CYCLE', ge=1)
    iterate_sampling_radius: float = Field(default=0.1, alias='DPE_ITERATE_SAMPLING_RADIUS', gt=0, le=1.0)
    iterate_convergence_tol: float = Field(default=1e-4, alias='DPE_ITERATE_CONVERGENCE_TOL', gt=0)
    iterate_min_improvement: float = Field(default=1e-6, alias='DPE_ITERATE_MIN_IMPROVEMENT', ge=0)
    iterate_sampling_method: str = Field(default='gaussian', alias='DPE_ITERATE_SAMPLING_METHOD')
    use_nn_head: bool = Field(default=True, alias='DPE_USE_NN_HEAD')
    pretrain_nn_head: bool = Field(default=False, alias='DPE_PRETRAIN_NN_HEAD')
    use_sundials: bool = Field(default=True, alias='DPE_USE_SUNDIALS')
    autodiff_steps: int = Field(default=100, alias='DPE_AUTODIFF_STEPS', ge=1)
    autodiff_lr: float = Field(default=0.001, alias='DPE_AUTODIFF_LR', gt=0, le=1.0)
    fd_step: float = Field(default=0.005, alias='DPE_FD_STEP', gt=0)
    gd_step_size: float = Field(default=0.1, alias='DPE_GD_STEP_SIZE', gt=0, le=1.0)


class EmulationConfig(BaseModel):
    """Model emulation settings"""
    model_config = FROZEN_CONFIG

    num_samples: int = Field(default=100, alias='EMULATION_NUM_SAMPLES', ge=1)
    seed: int = Field(default=22, alias='EMULATION_SEED')
    sampling_method: SamplingMethodType = Field(default='lhs', alias='EMULATION_SAMPLING_METHOD')
    parallel_ensemble: bool = Field(default=False, alias='EMULATION_PARALLEL_ENSEMBLE')
    max_parallel_jobs: int = Field(default=100, alias='EMULATION_MAX_PARALLEL_JOBS', ge=1)
    skip_mizuroute: bool = Field(default=False, alias='EMULATION_SKIP_MIZUROUTE')
    use_attributes: bool = Field(default=False, alias='EMULATION_USE_ATTRIBUTES')
    max_iterations: int = Field(default=3, alias='EMULATION_MAX_ITERATIONS', ge=1)


class OptimizationConfig(BaseModel):
    """Calibration and optimization configuration"""
    model_config = FROZEN_CONFIG

    # General optimization settings
    methods: Union[List[str], str] = Field(default_factory=list, alias='OPTIMIZATION_METHODS')
    target: str = Field(default='streamflow', alias='OPTIMIZATION_TARGET')
    calibration_variable: str = Field(default='streamflow', alias='CALIBRATION_VARIABLE')
    calibration_timestep: str = Field(default='daily', alias='CALIBRATION_TIMESTEP')
    algorithm: OptimizationAlgorithmType = Field(default='PSO', alias='ITERATIVE_OPTIMIZATION_ALGORITHM')
    metric: OptimizationMetricType = Field(default='KGE', alias='OPTIMIZATION_METRIC')
    iterations: int = Field(default=1000, alias='NUMBER_OF_ITERATIONS', ge=1)
    population_size: int = Field(default=50, alias='POPULATION_SIZE', ge=2, le=10000)
    final_evaluation_numerical_method: str = Field(default='ida', alias='FINAL_EVALUATION_NUMERICAL_METHOD')
    cleanup_parallel_dirs: bool = Field(default=True, alias='CLEANUP_PARALLEL_DIRS')

    @field_validator('algorithm', mode='before')
    @classmethod
    def normalize_algorithm(cls, v):
        """Normalize algorithm name to uppercase for case-insensitive matching"""
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator('metric', mode='before')
    @classmethod
    def normalize_metric(cls, v):
        """Normalize metric name to uppercase for case-insensitive matching"""
        if isinstance(v, str):
            return v.upper()
        return v

    # Error logging and debugging options
    params_keep_trials: bool = Field(
        default=False,
        alias='PARAMS_KEEP_TRIALS',
        description="Convenience flag: enables ERROR_LOGGING_MODE='failures' to save "
                    "parameter files and logs from failed runs for debugging"
    )
    error_logging_mode: str = Field(
        default='none',
        alias='ERROR_LOGGING_MODE',
        description="Error artifact capture mode: 'none' (disabled), 'failures' "
                    "(save artifacts from failed runs), 'all' (save all runs)"
    )
    stop_on_model_failure: bool = Field(
        default=False,
        alias='STOP_ON_MODEL_FAILURE',
        description="Stop optimization immediately when a model run fails"
    )
    error_log_dir: str = Field(
        default='error_logs',
        alias='ERROR_LOG_DIR',
        description="Subdirectory name for error artifacts within the output directory"
    )

    # Algorithm-specific settings
    pso: Optional[PSOConfig] = Field(default_factory=PSOConfig)
    de: Optional[DEConfig] = Field(default_factory=DEConfig)
    dds: Optional[DDSConfig] = Field(default_factory=DDSConfig)
    sce_ua: Optional[SCEUAConfig] = Field(default_factory=SCEUAConfig)
    nsga2: Optional[NSGA2Config] = Field(default_factory=NSGA2Config)
    dpe: Optional[DPEConfig] = Field(default_factory=DPEConfig)
    emulation: Optional[EmulationConfig] = Field(default_factory=EmulationConfig)

    @field_validator('methods', mode='before')
    @classmethod
    def validate_list_fields(cls, v):
        """Normalize string lists"""
        if v is None:
            return []
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v
