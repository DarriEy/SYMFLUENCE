"""
Optimization configuration models.

Contains configuration classes for calibration algorithms:
PSOConfig, DEConfig, DDSConfig, SCEUAConfig, NSGA2Config,
EmulationConfig, and the parent OptimizationConfig.
"""

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator

from .base import FROZEN_CONFIG

# Supported optimization algorithms
OptimizationAlgorithmType = Literal[
    'PSO', 'DE', 'DDS', 'ASYNC-DDS', 'SCE-UA', 'NSGA-II',
    'ADAM', 'LBFGS', 'CMA-ES', 'CMAES', 'DREAM', 'GLUE',
    'BASIN-HOPPING', 'BASINHOPPING', 'BH',
    'NELDER-MEAD', 'NELDERMEAD', 'NM', 'SIMPLEX', 'GA',
    'BAYESIAN-OPT', 'BAYESIAN_OPT', 'BAYESIAN', 'BO',
    'MOEAD', 'MOEA-D', 'MOEA_D',
    'SIMULATED-ANNEALING', 'SIMULATED_ANNEALING', 'SA', 'ANNEALING',
    'ABC', 'ABC-SMC', 'ABC_SMC', 'APPROXIMATE-BAYESIAN'
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


class ABCConfig(BaseModel):
    """Approximate Bayesian Computation (ABC-SMC) algorithm settings.

    ABC-SMC uses Sequential Monte Carlo to efficiently sample from the
    approximate posterior distribution when the likelihood is intractable.

    Reference:
        Beaumont, M.A., Cornuet, J.M., Marin, J.M., and Robert, C.P. (2009).
        Adaptive approximate Bayesian computation. Biometrika, 96(4), 983-990.
    """
    model_config = FROZEN_CONFIG

    # Population settings
    n_particles: int = Field(
        default=100,
        alias='ABC_PARTICLES',
        ge=10,
        le=10000,
        description="Number of particles in the population. More particles = better posterior "
                    "approximation but higher computational cost. Recommend 100-500 for hydrology."
    )

    # Tolerance schedule
    n_generations: int = Field(
        default=20,
        alias='ABC_GENERATIONS',
        ge=3,
        le=100,
        description="Maximum number of SMC generations. Each generation reduces the tolerance."
    )
    initial_tolerance: float = Field(
        default=0.5,
        alias='ABC_INITIAL_TOLERANCE',
        gt=0,
        le=10.0,
        description="Initial acceptance tolerance (distance threshold). For KGE, this is (1-KGE), "
                    "so 0.5 means accepting KGE >= 0.5."
    )
    final_tolerance: float = Field(
        default=0.05,
        alias='ABC_FINAL_TOLERANCE',
        gt=0,
        le=1.0,
        description="Target final tolerance. For KGE, 0.05 means targeting KGE >= 0.95."
    )
    tolerance_quantile: float = Field(
        default=0.75,
        alias='ABC_TOLERANCE_QUANTILE',
        gt=0.1,
        le=0.95,
        description="Quantile of accepted distances for adaptive tolerance schedule. "
                    "Higher values (0.7-0.9) give more gradual tolerance decrease."
    )
    tolerance_decay: float = Field(
        default=0.9,
        alias='ABC_TOLERANCE_DECAY',
        gt=0.5,
        le=0.99,
        description="Maximum tolerance reduction per generation (geometric decay factor). "
                    "Values closer to 1.0 give slower, more stable convergence."
    )

    # Perturbation kernel settings
    perturbation_scale: float = Field(
        default=2.0,
        alias='ABC_PERTURBATION_SCALE',
        gt=0,
        le=10.0,
        description="Scaling factor for perturbation kernel bandwidth. Higher = more exploration."
    )
    kernel_type: Literal['gaussian', 'uniform', 'component_wise'] = Field(
        default='component_wise',
        alias='ABC_KERNEL_TYPE',
        description="Type of perturbation kernel: 'gaussian' (full covariance), "
                    "'uniform' (box kernel), 'component_wise' (independent Gaussians)."
    )
    use_olcm: bool = Field(
        default=True,
        alias='ABC_USE_OLCM',
        description="Use Optimal Local Covariance Matrix adaptation for perturbation kernel."
    )

    # Acceptance settings
    min_acceptance_rate: float = Field(
        default=0.05,
        alias='ABC_MIN_ACCEPTANCE_RATE',
        gt=0.001,
        le=0.5,
        description="Minimum acceptance rate before stopping. Lower allows more generations."
    )
    min_ess_ratio: float = Field(
        default=0.5,
        alias='ABC_MIN_ESS_RATIO',
        gt=0.1,
        le=1.0,
        description="Minimum effective sample size ratio before resampling."
    )

    # Convergence settings
    convergence_threshold: float = Field(
        default=0.001,
        alias='ABC_CONVERGENCE_THRESHOLD',
        gt=0,
        le=0.1,
        description="Stop if tolerance improvement < this fraction."
    )
    min_generations: int = Field(
        default=5,
        alias='ABC_MIN_GENERATIONS',
        ge=1,
        description="Minimum generations to run before checking convergence."
    )


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

    # Gradient-based optimization settings (Adam, L-BFGS)
    gradient_mode: Literal['auto', 'native', 'finite_difference'] = Field(
        default='auto',
        alias='GRADIENT_MODE',
        description="Gradient computation method for Adam/L-BFGS: "
                    "'auto' uses native gradients if available (e.g., JAX autodiff for HBV), "
                    "'native' requires native gradients (error if unavailable), "
                    "'finite_difference' always uses FD (useful for comparison)"
    )
    gradient_epsilon: float = Field(
        default=1e-4,
        alias='GRADIENT_EPSILON',
        gt=0,
        le=0.1,
        description="Perturbation size for finite-difference gradient approximation"
    )
    gradient_clip_value: float = Field(
        default=1.0,
        alias='GRADIENT_CLIP_VALUE',
        gt=0,
        description="Maximum gradient L2 norm (prevents exploding gradients)"
    )

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
    abc: Optional[ABCConfig] = Field(default_factory=ABCConfig)
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
