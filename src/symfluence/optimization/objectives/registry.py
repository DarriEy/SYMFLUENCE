# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Objective Registry for SYMFLUENCE

Provides a central registry for objective functions used in calibration.

This module implements a plugin pattern for objective functions. Objective classes
register themselves via the unified registry (``@R.objectives.add()``), enabling
dynamic instantiation by type string from configuration without hardcoded imports.
``ObjectiveRegistry`` is a lookup facade over ``R.objectives``.

This design allows users to select different objective functions (single-variable,
multi-variable, custom) via configuration and enables straightforward addition of
new objectives without modifying the registry code.

Example:
    Register a custom objective:

    >>> @R.objectives.add('CUSTOM_OBJECTIVE')
    ... class CustomObjective(BaseObjective):
    ...     def calculate(self, evaluation_results): ...

    Use in calibration:

    >>> config = {'OBJECTIVE_FUNCTION': 'CUSTOM_OBJECTIVE', ...}
    >>> objective = ObjectiveRegistry.get_objective('CUSTOM_OBJECTIVE', config, logger)
    >>> score = objective.calculate(eval_results)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from symfluence.core.registries import R

if TYPE_CHECKING:
    from .base import BaseObjective

class ObjectiveRegistry:
    """Lookup facade for objective function implementations.

    Objectives register themselves via ``@R.objectives.add()`` and are
    dynamically instantiated via get_objective() based on a type string
    from configuration.
    """

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
