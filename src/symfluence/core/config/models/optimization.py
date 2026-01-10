"""
Optimization configuration models.

Contains configuration classes for calibration algorithms:
PSOConfig, DEConfig, DDSConfig, SCEUAConfig, NSGA2Config, DPEConfig,
LargeDomainConfig, EmulationConfig, and the parent OptimizationConfig.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .base import FROZEN_CONFIG


class PSOConfig(BaseModel):
    """Particle Swarm Optimization algorithm settings"""
    model_config = FROZEN_CONFIG

    swrmsize: int = Field(default=20, alias='SWRMSIZE')
    cognitive_param: float = Field(default=1.5, alias='PSO_COGNITIVE_PARAM')
    social_param: float = Field(default=1.5, alias='PSO_SOCIAL_PARAM')
    inertia_weight: float = Field(default=0.7, alias='PSO_INERTIA_WEIGHT')
    inertia_reduction_rate: float = Field(default=0.99, alias='PSO_INERTIA_REDUCTION_RATE')
    inertia_schedule: str = Field(default='LINEAR', alias='INERTIA_SCHEDULE')


class DEConfig(BaseModel):
    """Differential Evolution algorithm settings"""
    model_config = FROZEN_CONFIG

    scaling_factor: float = Field(default=0.5, alias='DE_SCALING_FACTOR')
    crossover_rate: float = Field(default=0.9, alias='DE_CROSSOVER_RATE')


class DDSConfig(BaseModel):
    """Dynamically Dimensioned Search algorithm settings"""
    model_config = FROZEN_CONFIG

    r: float = Field(default=0.2, alias='DDS_R')
    async_pool_size: int = Field(default=10, alias='ASYNC_DDS_POOL_SIZE')
    async_batch_size: int = Field(default=10, alias='ASYNC_DDS_BATCH_SIZE')
    max_stagnation_batches: int = Field(default=10, alias='MAX_STAGNATION_BATCHES')


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
    crossover_rate: float = Field(default=0.9, alias='NSGA2_CROSSOVER_RATE')
    mutation_rate: float = Field(default=0.1, alias='NSGA2_MUTATION_RATE')
    eta_c: int = Field(default=20, alias='NSGA2_ETA_C')
    eta_m: int = Field(default=20, alias='NSGA2_ETA_M')


class DPEConfig(BaseModel):
    """Differentiable Parameter Estimation settings"""
    model_config = FROZEN_CONFIG

    training_cache: str = Field(default='default', alias='DPE_TRAINING_CACHE')
    hidden_dims: Optional[List[int]] = Field(default_factory=lambda: [256, 128, 64], alias='DPE_HIDDEN_DIMS')
    training_samples: int = Field(default=500, alias='DPE_TRAINING_SAMPLES')
    validation_samples: int = Field(default=100, alias='DPE_VALIDATION_SAMPLES')
    epochs: int = Field(default=300, alias='DPE_EPOCHS')
    learning_rate: float = Field(default=1e-3, alias='DPE_LEARNING_RATE')
    optimization_lr: float = Field(default=1e-2, alias='DPE_OPTIMIZATION_LR')
    optimization_steps: int = Field(default=200, alias='DPE_OPTIMIZATION_STEPS')
    optimizer: str = Field(default='ADAM', alias='DPE_OPTIMIZER')
    objective_weights: Optional[Dict[str, float]] = Field(default_factory=lambda: {'KGE': 1.0}, alias='DPE_OBJECTIVE_WEIGHTS')
    emulator_iterate: bool = Field(default=True, alias='DPE_EMULATOR_ITERATE')
    iterate_max_iterations: int = Field(default=5, alias='DPE_ITERATE_MAX_ITERATIONS')
    iterate_samples_per_cycle: int = Field(default=100, alias='DPE_ITERATE_SAMPLES_PER_CYCLE')
    iterate_sampling_radius: float = Field(default=0.1, alias='DPE_ITERATE_SAMPLING_RADIUS')
    iterate_convergence_tol: float = Field(default=1e-4, alias='DPE_ITERATE_CONVERGENCE_TOL')
    iterate_min_improvement: float = Field(default=1e-6, alias='DPE_ITERATE_MIN_IMPROVEMENT')
    iterate_sampling_method: str = Field(default='gaussian', alias='DPE_ITERATE_SAMPLING_METHOD')
    use_nn_head: bool = Field(default=True, alias='DPE_USE_NN_HEAD')
    pretrain_nn_head: bool = Field(default=False, alias='DPE_PRETRAIN_NN_HEAD')
    use_sundials: bool = Field(default=True, alias='DPE_USE_SUNDIALS')
    autodiff_steps: int = Field(default=100, alias='DPE_AUTODIFF_STEPS')
    autodiff_lr: float = Field(default=0.001, alias='DPE_AUTODIFF_LR')
    fd_step: float = Field(default=0.005, alias='DPE_FD_STEP')
    gd_step_size: float = Field(default=0.1, alias='DPE_GD_STEP_SIZE')


class LargeDomainConfig(BaseModel):
    """Large domain emulation settings"""
    model_config = FROZEN_CONFIG

    enabled: bool = Field(default=True, alias='LARGE_DOMAIN_EMULATION_ENABLED')
    emulator_setting: str = Field(default='SUMMA_AUTODIFF_FD', alias='EMULATOR_SETTING')
    mode: str = Field(default='SUMMA_AUTODIFF', alias='LARGE_DOMAIN_EMULATOR_MODE')
    optimizer: str = Field(default='Adam', alias='LARGE_DOMAIN_EMULATOR_OPTIMIZER')
    training_epochs: int = Field(default=100, alias='LARGE_DOMAIN_TRAINING_EPOCHS')
    parameter_ensemble_size: int = Field(default=1000, alias='LARGE_DOMAIN_PARAMETER_ENSEMBLE_SIZE')
    batch_size: int = Field(default=32, alias='LARGE_DOMAIN_BATCH_SIZE')
    validation_split: Optional[float] = Field(default=None, alias='LARGE_DOMAIN_VALIDATION_SPLIT')
    pretrain_nn_head: bool = Field(default=True, alias='LARGE_DOMAIN_EMULATOR_PRETRAIN_NN_HEAD')
    use_nn_head: bool = Field(default=True, alias='LARGE_DOMAIN_EMULATOR_USE_NN_HEAD')
    training_samples: int = Field(default=500, alias='LARGE_DOMAIN_EMULATOR_TRAINING_SAMPLES')
    epochs: int = Field(default=50, alias='LARGE_DOMAIN_EMULATOR_EPOCHS')
    autodiff_steps: int = Field(default=100, alias='LARGE_DOMAIN_EMULATOR_AUTODIFF_STEPS')
    streamflow_weight: float = Field(default=0.5, alias='LARGE_DOMAIN_EMULATOR_STREAMFLOW_WEIGHT')
    smap_weight: float = Field(default=0.2, alias='LARGE_DOMAIN_EMULATOR_SMAP_WEIGHT')
    grace_weight: float = Field(default=0.15, alias='LARGE_DOMAIN_EMULATOR_GRACE_WEIGHT')
    modis_weight: float = Field(default=0.15, alias='LARGE_DOMAIN_EMULATOR_MODIS_WEIGHT')


class EmulationConfig(BaseModel):
    """Model emulation settings"""
    model_config = FROZEN_CONFIG

    num_samples: int = Field(default=100, alias='EMULATION_NUM_SAMPLES')
    seed: int = Field(default=22, alias='EMULATION_SEED')
    sampling_method: str = Field(default='lhs', alias='EMULATION_SAMPLING_METHOD')
    parallel_ensemble: bool = Field(default=False, alias='EMULATION_PARALLEL_ENSEMBLE')
    max_parallel_jobs: int = Field(default=100, alias='EMULATION_MAX_PARALLEL_JOBS')
    skip_mizuroute: bool = Field(default=False, alias='EMULATION_SKIP_MIZUROUTE')
    use_attributes: bool = Field(default=False, alias='EMULATION_USE_ATTRIBUTES')
    max_iterations: int = Field(default=3, alias='EMULATION_MAX_ITERATIONS')


class OptimizationConfig(BaseModel):
    """Calibration and optimization configuration"""
    model_config = FROZEN_CONFIG

    # General optimization settings
    methods: Union[List[str], str] = Field(default_factory=list, alias='OPTIMIZATION_METHODS')
    target: str = Field(default='streamflow', alias='OPTIMIZATION_TARGET')
    calibration_variable: str = Field(default='streamflow', alias='CALIBRATION_VARIABLE')
    calibration_timestep: str = Field(default='daily', alias='CALIBRATION_TIMESTEP')
    algorithm: str = Field(default='PSO', alias='ITERATIVE_OPTIMIZATION_ALGORITHM')
    metric: str = Field(default='KGE', alias='OPTIMIZATION_METRIC')
    iterations: int = Field(default=1000, alias='NUMBER_OF_ITERATIONS')
    population_size: int = Field(default=50, alias='POPULATION_SIZE')
    final_evaluation_numerical_method: str = Field(default='ida', alias='FINAL_EVALUATION_NUMERICAL_METHOD')
    cleanup_parallel_dirs: bool = Field(default=True, alias='CLEANUP_PARALLEL_DIRS')

    # Algorithm-specific settings
    pso: Optional[PSOConfig] = Field(default_factory=PSOConfig)
    de: Optional[DEConfig] = Field(default_factory=DEConfig)
    dds: Optional[DDSConfig] = Field(default_factory=DDSConfig)
    sce_ua: Optional[SCEUAConfig] = Field(default_factory=SCEUAConfig)
    nsga2: Optional[NSGA2Config] = Field(default_factory=NSGA2Config)
    dpe: Optional[DPEConfig] = Field(default_factory=DPEConfig)
    large_domain: Optional[LargeDomainConfig] = Field(default_factory=LargeDomainConfig)
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
