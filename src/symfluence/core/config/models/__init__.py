"""
Hierarchical configuration models for SYMFLUENCE.

This package defines a nested configuration structure that organizes the 346+
configuration parameters into logical sections while maintaining backward
compatibility through factory methods and dict-like access.

Key design features:
- Type-safe hierarchical structure (config.domain.name vs config['DOMAIN_NAME'])
- Factory methods: from_preset(), from_minimal(), from_file()
- Backward compatibility: to_dict(), get(), __getitem__()
- All validation logic preserved from original flat model
- Immutable configs (frozen=True) to prevent mutation bugs
"""

# Root config - the main entry point
# FEWS adapter config
from symfluence.fews.config import FEWSConfig

# Data configs
from .data import (
    DataConfig,
    GeospatialConfig,
    MODISLandcoverConfig,
    NASADEMConfig,
    NLCDConfig,
    SoilGridsConfig,
)

# Domain configs
from .domain import DelineationConfig, DomainConfig

# Evaluation configs
from .evaluation import (
    AttributesConfig,
    EvaluationConfig,
    FluxNetConfig,
    GRACEConfig,
    ISMNConfig,
    MODISETConfig,
    MODISSnowConfig,
    SMAPConfig,
    SNOTELConfig,
    StreamflowConfig,
    USGSGWConfig,
)

# Forcing configs
from .forcing import EMEarthConfig, ERA5Config, ForcingConfig, NexConfig

# Model configs
from .model_configs import (
    FUSEConfig,
    GNNConfig,
    GRConfig,
    HYPEConfig,
    LSTMConfig,
    MESHConfig,
    MizuRouteConfig,
    ModelConfig,
    NGENConfig,
    RHESSysConfig,
    SUMMAConfig,
)

# Optimization configs
from .optimization import (
    DDSConfig,
    DEConfig,
    EmulationConfig,
    NSGA2Config,
    OptimizationConfig,
    PSOConfig,
    SCEUAConfig,
)

# Paths configs
from .paths import PathsConfig, ShapefilePathConfig
from .root import SymfluenceConfig

# State management and data assimilation configs
from .state_config import DataAssimilationConfig, EnKFConfig, StateConfig

# System config
from .system import SystemConfig

__all__ = [
    # Root
    "SymfluenceConfig",
    # System
    "SystemConfig",
    # Domain
    "DomainConfig",
    "DelineationConfig",
    # Data
    "DataConfig",
    "GeospatialConfig",
    "SoilGridsConfig",
    "MODISLandcoverConfig",
    "NLCDConfig",
    "NASADEMConfig",
    # Forcing
    "ForcingConfig",
    "NexConfig",
    "EMEarthConfig",
    "ERA5Config",
    # Models
    "ModelConfig",
    "SUMMAConfig",
    "FUSEConfig",
    "GRConfig",
    "HYPEConfig",
    "NGENConfig",
    "MESHConfig",
    "MizuRouteConfig",
    "LSTMConfig",
    "RHESSysConfig",
    "GNNConfig",
    # Optimization
    "OptimizationConfig",
    "PSOConfig",
    "DEConfig",
    "DDSConfig",
    "SCEUAConfig",
    "NSGA2Config",
    "EmulationConfig",
    # Evaluation
    "EvaluationConfig",
    "StreamflowConfig",
    "SNOTELConfig",
    "FluxNetConfig",
    "USGSGWConfig",
    "SMAPConfig",
    "ISMNConfig",
    "GRACEConfig",
    "MODISSnowConfig",
    "MODISETConfig",
    "AttributesConfig",
    # Paths
    "PathsConfig",
    "ShapefilePathConfig",
    # State management and data assimilation
    "StateConfig",
    "EnKFConfig",
    "DataAssimilationConfig",
    # FEWS
    "FEWSConfig",
]
