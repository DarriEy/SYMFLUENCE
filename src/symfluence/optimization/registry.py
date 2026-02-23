"""Central registry for model-specific optimizers, workers, and parameter managers."""

import logging
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class OptimizerRegistry:
    """
    Central registry for optimization components.

    Allows registration and lookup of:
    - Model-specific optimizers (SUMMA, FUSE, NGEN, etc.)
    - Model-specific workers
    - Parameter managers
    - Calibration targets

    Usage:
        @OptimizerRegistry.register_optimizer('FUSE')
        class FUSEOptimizer(BaseModelOptimizer):
            ...

        # Later, look up the optimizer
        optimizer_cls = OptimizerRegistry.get_optimizer('FUSE')
    """

    _optimizers: Dict[str, Type] = {}
    _workers: Dict[str, Type] = {}
    _parameter_managers: Dict[str, Type] = {}
    _calibration_targets: Dict[str, Type] = {}

    @classmethod
    def register_optimizer(cls, model_name: str):
        """Decorator to register a model-specific optimizer."""
        def decorator(optimizer_cls):
            key = model_name.upper()
            logger.debug(f"Registering optimizer for {key}: {optimizer_cls}")
            cls._optimizers[key] = optimizer_cls
            return optimizer_cls
        return decorator

    @classmethod
    def register_worker(cls, model_name: str):
        """Decorator to register a model-specific worker."""
        def decorator(worker_cls):
            key = model_name.upper()
            logger.debug(f"Registering worker for {key}: {worker_cls}")
            cls._workers[key] = worker_cls
            return worker_cls
        return decorator

    @classmethod
    def register_parameter_manager(cls, model_name: str):
        """Decorator for registering a model-specific parameter manager."""
        def decorator(param_manager_cls):
            logger.debug(f"Registering parameter manager for {model_name}: {param_manager_cls}")
            cls._parameter_managers[model_name.upper()] = param_manager_cls
            return param_manager_cls
        return decorator

    @classmethod
    def register_calibration_target(cls, model_name: str, target_type: str = 'streamflow'):
        """Decorator to register a model-specific calibration target."""
        def decorator(target_cls):
            key = f"{model_name.upper()}_{target_type.upper()}"
            cls._calibration_targets[key] = target_cls
            # logger.debug(f"Registered calibration target: {key}")
            return target_cls
        return decorator

    # =========================================================================
    # Lookup methods
    # =========================================================================

    @classmethod
    def get_optimizer(cls, model_name: str) -> Optional[Type]:
        """Get registered optimizer class for a model, or None."""
        return cls._optimizers.get(model_name.upper())

    @classmethod
    def get_worker(cls, model_name: str) -> Optional[Type]:
        """Get registered worker class for a model, or None."""
        return cls._workers.get(model_name.upper())

    @classmethod
    def get_parameter_manager(cls, model_name: str):
        """Get the parameter manager class for a given model."""
        name_upper = model_name.upper()
        logger.debug(f"Getting parameter manager for {name_upper}. Registered: {list(cls._parameter_managers.keys())}")
        if name_upper not in cls._parameter_managers:
            return None
        return cls._parameter_managers[name_upper]

    @classmethod
    def get_calibration_target(
        cls,
        model_name: str,
        target_type: str = 'streamflow'
    ) -> Optional[Type]:
        """Get registered calibration target class, or None."""
        key = f"{model_name.upper()}_{target_type.upper()}"
        return cls._calibration_targets.get(key)

    # =========================================================================
    # Discovery methods
    # =========================================================================

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return sorted(cls._optimizers.keys())

    @classmethod
    def list_optimizers(cls) -> List[str]:
        """List all registered optimizer model names."""
        return sorted(cls._optimizers.keys())

    @classmethod
    def list_workers(cls) -> List[str]:
        """List all registered worker model names."""
        return sorted(cls._workers.keys())

    @classmethod
    def list_calibration_targets(cls) -> List[str]:
        """List all registered calibration target keys."""
        return sorted(cls._calibration_targets.keys())

    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """Check if a model has an optimizer registered."""
        return model_name.upper() in cls._optimizers

    # =========================================================================
    # Utility methods
    # =========================================================================

    @classmethod
    def get_available_algorithms(cls, model_name: str) -> List[str]:
        """Get available optimization algorithms for a model."""
        optimizer_cls = cls.get_optimizer(model_name)
        if optimizer_cls is None:
            return []

        # Check for run_* methods
        algorithms = []
        for attr_name in dir(optimizer_cls):
            if attr_name.startswith('run_') and callable(getattr(optimizer_cls, attr_name, None)):
                algo_name = attr_name[4:].upper()  # Remove 'run_' prefix
                algorithms.append(algo_name)

        return sorted(algorithms)

    @classmethod
    def clear(cls):
        """Clear all registrations (for testing)."""
        cls._optimizers.clear()
        cls._workers.clear()
        cls._parameter_managers.clear()
        cls._calibration_targets.clear()
        logger.debug("Cleared all optimizer registrations")

    @classmethod
    def summary(cls) -> Dict[str, Any]:
        """Get summary of all registered components."""
        return {
            'optimizers': list(cls._optimizers.keys()),
            'workers': list(cls._workers.keys()),
            'parameter_managers': list(cls._parameter_managers.keys()),
            'calibration_targets': list(cls._calibration_targets.keys()),
            'total_models': len(cls._optimizers),
        }
