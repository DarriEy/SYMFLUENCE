"""
Legacy optimization modules.

This package contains deprecated optimization components that are maintained
for backward compatibility but will be removed in a future version.

Deprecated modules:
    - differentiable_parameter_emulator: Use gradient-based methods (run_adam, run_lbfgs)
      in model-specific optimizers instead.
    - large_domain_emulator: Use unified model optimizers with gradient-based methods instead.

Migration guide:
    Instead of:
        from symfluence.utils.optimization.legacy import DifferentiableParameterOptimizer
        optimizer = DifferentiableParameterOptimizer(config, logger)

    Use:
        from symfluence.utils.optimization import OptimizerRegistry
        optimizer_cls = OptimizerRegistry.get_optimizer('SUMMA')  # or 'FUSE', 'NGEN'
        optimizer = optimizer_cls(config, logger)
        results = optimizer.run_adam()  # or run_lbfgs()
"""

from .differentiable_parameter_emulator import (
    DifferentiableParameterOptimizer,
    EmulatorConfig,
    ObjectiveHead,
)
from .large_domain_emulator import LargeDomainEmulator

__all__ = [
    "DifferentiableParameterOptimizer",
    "EmulatorConfig",
    "ObjectiveHead",
    "LargeDomainEmulator",
]
