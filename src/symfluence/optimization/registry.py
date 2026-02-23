"""Central registry for model-specific optimizers, workers, and parameter managers.

.. deprecated::
    This registry is a thin delegation shim around
    :pydata:`symfluence.core.registries.R`.  Prefer ``R.optimizers``,
    ``R.workers``, ``R.parameter_managers``, ``R.calibration_targets`` directly.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Type

from symfluence.core.registries import R
from symfluence.core.registry import _RegistryProxy

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

    .. deprecated::
        Use ``R.optimizers``, ``R.workers``, ``R.parameter_managers``,
        ``R.calibration_targets`` from :mod:`symfluence.core.registries` instead.
    """

    # Backward-compat proxies: read-only views into R.* so that code
    # accessing e.g. ``OptimizerRegistry._optimizers`` still works.
    _optimizers: Dict[str, Type] = _RegistryProxy(R.optimizers)
    _workers: Dict[str, Type] = _RegistryProxy(R.workers)
    _parameter_managers: Dict[str, Type] = _RegistryProxy(R.parameter_managers)
    _calibration_targets: Dict[str, Type] = _RegistryProxy(R.calibration_targets)

    @classmethod
    def register_optimizer(cls, model_name: str):
        """Decorator to register a model-specific optimizer.

        .. deprecated::
            Use ``R.optimizers.add()`` or ``model_manifest()`` instead.
        """
        def decorator(optimizer_cls):
            warnings.warn(
                "OptimizerRegistry.register_optimizer() is deprecated; "
                "use R.optimizers.add() or model_manifest() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            key = model_name.upper()
            logger.debug(f"Registering optimizer for {key}: {optimizer_cls}")
            R.optimizers.add(model_name, optimizer_cls)
            return optimizer_cls
        return decorator

    @classmethod
    def register_worker(cls, model_name: str):
        """Decorator to register a model-specific worker.

        .. deprecated::
            Use ``R.workers.add()`` or ``model_manifest()`` instead.
        """
        def decorator(worker_cls):
            warnings.warn(
                "OptimizerRegistry.register_worker() is deprecated; "
                "use R.workers.add() or model_manifest() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            key = model_name.upper()
            logger.debug(f"Registering worker for {key}: {worker_cls}")
            R.workers.add(model_name, worker_cls)
            return worker_cls
        return decorator

    @classmethod
    def register_parameter_manager(cls, model_name: str):
        """Decorator for registering a model-specific parameter manager.

        .. deprecated::
            Use ``R.parameter_managers.add()`` or ``model_manifest()`` instead.
        """
        def decorator(param_manager_cls):
            warnings.warn(
                "OptimizerRegistry.register_parameter_manager() is deprecated; "
                "use R.parameter_managers.add() or model_manifest() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.debug(f"Registering parameter manager for {model_name}: {param_manager_cls}")
            R.parameter_managers.add(model_name, param_manager_cls)
            return param_manager_cls
        return decorator

    @classmethod
    def register_calibration_target(cls, model_name: str, target_type: str = 'streamflow'):
        """Decorator to register a model-specific calibration target.

        .. deprecated::
            Use ``R.calibration_targets.add()`` or ``model_manifest()`` instead.
        """
        def decorator(target_cls):
            warnings.warn(
                "OptimizerRegistry.register_calibration_target() is deprecated; "
                "use R.calibration_targets.add() or model_manifest() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            key = f"{model_name.upper()}_{target_type.upper()}"
            R.calibration_targets.add(key, target_cls)
            return target_cls
        return decorator

    # =========================================================================
    # Lookup methods
    # =========================================================================

    @classmethod
    def get_optimizer(cls, model_name: str) -> Optional[Type]:
        """Get registered optimizer class for a model, or None."""
        return R.optimizers.get(model_name.upper())

    @classmethod
    def get_worker(cls, model_name: str) -> Optional[Type]:
        """Get registered worker class for a model, or None."""
        return R.workers.get(model_name.upper())

    @classmethod
    def get_parameter_manager(cls, model_name: str):
        """Get the parameter manager class for a given model."""
        name_upper = model_name.upper()
        logger.debug(f"Getting parameter manager for {name_upper}. Registered: {R.parameter_managers.keys()}")
        return R.parameter_managers.get(name_upper)

    @classmethod
    def get_calibration_target(
        cls,
        model_name: str,
        target_type: str = 'streamflow'
    ) -> Optional[Type]:
        """Get registered calibration target class, or None."""
        key = f"{model_name.upper()}_{target_type.upper()}"
        return R.calibration_targets.get(key)

    # =========================================================================
    # Discovery methods
    # =========================================================================

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return sorted(R.optimizers.keys())

    @classmethod
    def list_optimizers(cls) -> List[str]:
        """List all registered optimizer model names."""
        return sorted(R.optimizers.keys())

    @classmethod
    def list_workers(cls) -> List[str]:
        """List all registered worker model names."""
        return sorted(R.workers.keys())

    @classmethod
    def list_calibration_targets(cls) -> List[str]:
        """List all registered calibration target keys."""
        return sorted(R.calibration_targets.keys())

    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """Check if a model has an optimizer registered."""
        return model_name.upper() in R.optimizers

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
        R.optimizers.clear()
        R.workers.clear()
        R.parameter_managers.clear()
        R.calibration_targets.clear()
        logger.debug("Cleared all optimizer registrations")

    @classmethod
    def summary(cls) -> Dict[str, Any]:
        """Get summary of all registered components."""
        return {
            'optimizers': list(R.optimizers.keys()),
            'workers': list(R.workers.keys()),
            'parameter_managers': list(R.parameter_managers.keys()),
            'calibration_targets': list(R.calibration_targets.keys()),
            'total_models': len(R.optimizers),
        }
