"""Objective Registry for SYMFLUENCE

Provides a central registry for objective functions used in calibration.

This module implements a plugin pattern for objective functions. Objective classes
register themselves using the @ObjectiveRegistry.register() decorator, enabling
dynamic instantiation by type string from configuration without hardcoded imports.

This design allows users to select different objective functions (single-variable,
multi-variable, custom) via configuration and enables straightforward addition of
new objectives without modifying the registry code.

Example:
    Register a custom objective:

    >>> @ObjectiveRegistry.register('CUSTOM_OBJECTIVE')
    ... class CustomObjective(BaseObjective):
    ...     def calculate(self, evaluation_results): ...

    Use in calibration:

    >>> config = {'OBJECTIVE_FUNCTION': 'CUSTOM_OBJECTIVE', ...}
    >>> objective = ObjectiveRegistry.get_objective('CUSTOM_OBJECTIVE', config, logger)
    >>> score = objective.calculate(eval_results)
"""
import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional

from symfluence.core.registries import R

if TYPE_CHECKING:
    from .base import BaseObjective

class ObjectiveRegistry:
    """Plugin registry for objective function implementations.

    Manages objective function classes using a plugin pattern. Objectives register
    themselves using @register() decorator and are dynamically instantiated via
    get_objective() based on a type string from configuration.

    Class Attributes:
        _handlers (dict): Maps objective type strings (uppercase) to objective classes.
    """

    @classmethod
    def register(cls, objective_type: str):
        """Decorator to register an objective function class.

        Registers an objective class for a given type string. The type is converted
        to uppercase for case-insensitive lookups. The decorated class must implement
        the BaseObjective interface (calculate method).

        Args:
            objective_type: Case-insensitive type identifier for the objective
                (e.g., 'MULTIVARIATE', 'SINGLE_VARIABLE'). Will be stored in uppercase.

        Returns:
            Decorator function that registers the class and returns it unchanged.

        Example:
            >>> @ObjectiveRegistry.register('MULTIVARIATE')
            ... class MultivariateObjective(BaseObjective):
            ...     pass
        """
        def decorator(handler_class):
            warnings.warn(
                "ObjectiveRegistry.register() is deprecated; "
                "use R.objectives.add() or model_manifest() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            R.objectives.add(objective_type, handler_class)
            return handler_class
        return decorator

    @classmethod
    def get_objective(
        cls,
        objective_type: str,
        config: Dict[str, Any],
        logger
    ) -> Optional['BaseObjective']:
        """Get an instance of the appropriate objective handler.

        Instantiates and returns an objective function of the specified type.
        The objective is configured with the provided config dict and logger.

        Args:
            objective_type: Case-insensitive objective type (e.g., 'MULTIVARIATE').
                Must match a registered objective type.
            config: Configuration dictionary containing objective settings
                (e.g., OBJECTIVE_WEIGHTS, OBJECTIVE_METRICS).
            logger: Python logger instance for diagnostic messages.

        Returns:
            BaseObjective: Initialized objective instance, or None if the type
            is not registered.

        Raises:
            TypeError: If the registered class doesn't implement BaseObjective.

        Example:
            >>> objective = ObjectiveRegistry.get_objective('MULTIVARIATE', config, logger)
            >>> if objective is None:
            ...     raise ValueError("Objective not found")
        """
        handler_class = R.objectives.get(objective_type.upper())
        if handler_class is None:
            return None
        return handler_class(config, logger)

    @classmethod
    def list_objectives(cls) -> list:
        """Get sorted list of all registered objective types.

        Returns:
            list: Registered objective type strings in uppercase, sorted alphabetically.

        Example:
            >>> ObjectiveRegistry.list_objectives()
            ['MULTIVARIATE', 'SINGLE_VARIABLE']
        """
        return R.objectives.keys()
