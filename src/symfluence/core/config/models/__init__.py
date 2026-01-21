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
from .root import SymfluenceConfig

# System config
from .system import SystemConfig

# Domain configs
from .domain import DomainConfig, DelineationConfig

# Data configs
from .data import (
    DataConfig,
    GeospatialConfig,
    SoilGridsConfig,
    MODISLandcoverConfig,
    NLCDConfig,
    NASADEMConfig,
)

# Forcing configs
from .forcing import ForcingConfig, NexConfig, EMEarthConfig, ERA5Config

# Model configs
from .model_configs import (
    ModelConfig,
    SUMMAConfig,
    FUSEConfig,
    GRConfig,
    HYPEConfig,
    NGENConfig,
    MESHConfig,
    MizuRouteConfig,
    LSTMConfig,
    RHESSysConfig,
    GNNConfig,
)

# Optimization configs
from .optimization import (
    OptimizationConfig,
    PSOConfig,
    DEConfig,
    DDSConfig,
    SCEUAConfig,
    NSGA2Config,
    EmulationConfig,
)

# Evaluation configs
from .evaluation import (
    EvaluationConfig,
    StreamflowConfig,
    SNOTELConfig,
    FluxNetConfig,
    USGSGWConfig,
    SMAPConfig,
    ISMNConfig,
    GRACEConfig,
    MODISSnowConfig,
    MODISETConfig,
    AttributesConfig,
)

# Paths configs
from .paths import PathsConfig, ShapefilePathConfig


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
]
